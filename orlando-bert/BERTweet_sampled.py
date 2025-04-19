import os
import glob
import sys
import argparse
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
nltk.download('stopwords')

# Thresholds for filtering tweets
MIN_LIKES = 10
MIN_RETWEETS = 0

# Sampling parameters
SAMPLE_FRAC = 0.10  # 10% overall
SAMPLE_SEED = 42     # Seed for reproducibility

# Combine custom stop words with NLTK stop words
custom_stop_words = [
    "presidential", "dems", "republican", "trump2024", "democratic", "republicans",
    "democrat", "democrats", "the", "biden", "trump", "harris", "kamala", "vote",
    "maga", "MAGA", "president", "donald", "gop", "joe", "rnc", "dnc", "election",
    "voters", "tweet", "retweet", "follow", "campaign", "nominee", "presidency",
    "liberals", "liberal", "realdonaltrump", "conservatives", "conservative", "political",
    "bidenharris2024", "2024", "joebiden", "potus"
]
nltk_stop_words = stopwords.words('english')
custom_stop_words = list(set(custom_stop_words) | set(nltk_stop_words))

# Initialize embedding model components
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModel.from_pretrained("vinai/bertweet-base")

def bertweet_embedding(texts):
    """Generate embeddings for a list of texts using vinai/bertweet-base."""
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def remove_links(text):
    """Remove words containing 'https' from text."""
    return " ".join(word for word in text.split() if "https" not in word)


def filter_data(df):
    """Filter and process the DataFrame containing tweet data."""
    print(f"Total tweets prefiltering: {df.shape[0]}")
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date_parsed'].notna()]

    df['likeCount'] = pd.to_numeric(df['likeCount'], errors='coerce')
    df['retweetCount'] = pd.to_numeric(df['retweetCount'], errors='coerce')
    df = df[(df['likeCount'] >= MIN_LIKES) & (df['retweetCount'] >= MIN_RETWEETS)]
    print(f"Tweets after thresholding: {df.shape[0]}")

    df = df.sort_values('epoch').drop_duplicates(subset=['conversationId'], keep='first')
    print(f"Tweets after selecting first per conversation: {df.shape[0]}")

    df = df[(df['lang'] == 'en') & df['text'].notna()]
    df['processed_text'] = df['text'].apply(remove_links)
    df = df.dropna(subset=['processed_text'])
    df = df[df['processed_text'].str.strip() != ""].reset_index(drop=True)
    print(f"Tweets after processing: {df.shape[0]}")
    return df


def load_data_stratified(root_folder, sample_frac=SAMPLE_FRAC, seed=SAMPLE_SEED):
    """
    Load all CSVs, sample 10% overall but equally per week,
    then filter only the sampled data.
    """
    # 1) Load all raw data
    part_folders = sorted(glob.glob(os.path.join(root_folder, "part_*")))
    raw_dfs = []
    for folder in part_folders:
        files = glob.glob(os.path.join(folder, "*.csv.gz"))
        for f in files:
            raw_dfs.append(pd.read_csv(f, compression="gzip"))
    raw = pd.concat(raw_dfs, ignore_index=True)

    # 2) Parse weeks and drop invalid dates
    raw['date_parsed'] = pd.to_datetime(raw['date'], errors='coerce')
    raw = raw[raw['date_parsed'].notna()]
    raw['week'] = raw['date_parsed'].dt.to_period('W').apply(lambda r: r.start_time)

    # 3) Compute per-week sample size
    total_n = len(raw)
    target_n = int(total_n * sample_frac)
    weeks = raw['week'].unique()
    per_week = max(1, target_n // len(weeks))

    # 4) Stratified sampling by week
    sampled = (
        raw
        .groupby('week', group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_week), random_state=seed))
    )
    print(f"Sampled {len(sampled)} tweets across {len(weeks)} weeks (~{sample_frac*100}% overall)")

    # 5) Filter sampled subset with existing logic
    return filter_data(sampled)


def initialize_topic_model(embedding_model):
    """Initialize BERTopic with UMAP and HDBSCAN."""
    representation_model = KeyBERTInspired()
    umap_model = UMAP(n_neighbors=15, n_components=5,
                      metric='cosine', random_state=42)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=50)
    vectorizer_model = CountVectorizer(
        stop_words=custom_stop_words,
        ngram_range=(1,3),
        min_df=3
    )
    return BERTopic(
        representation_model=representation_model,
        embedding_model=bertweet_embedding,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="english",
        nr_topics=None,
        calculate_probabilities=False,
        verbose=True
    )


def process_data_sampled(main_dir, embedding_model):
    print("Loading and stratified-sampling data...")
    sampled_df = load_data_stratified(main_dir)
    docs = sampled_df['processed_text'].tolist()
    print(f"Total sampled documents: {len(docs)}")

    embeddings = embedding_model.encode(
        docs, show_progress_bar=True
    ).astype('float64')

    topic_model = initialize_topic_model(embedding_model)
    topics, _ = topic_model.fit_transform(
        documents=docs, embeddings=embeddings
    )
    sampled_df['topic'] = topics
    return topic_model, sampled_df


def save_model(save_dir, topic_model: BERTopic, df: pd.DataFrame):
    """Save the trained BERTopic model and a CSV of topics."""
    columns_to_save = ['id','topic','date_parsed']
    topic_model.save(os.path.join(save_dir, 'bertopic_model'), serialization='pytorch', save_ctfidf=True, save_embedding_model=embedding_model)
    df[columns_to_save].to_csv(
        os.path.join(save_dir, 'topics.csv.gz'),
        index=False, compression='gzip'
    )


if __name__ == '__main__':
    # Paths for HPC
    base_save_dir = "/user/home/sv22482/work/ADS-US-Election/orlando-bert/ADS"
    main_dir = "/user/work/sv22482/usc-x-24-us-election"

    parser = argparse.ArgumentParser(
        description="BERTopic Batch Training or Loading a Saved Model"
    )
    parser.add_argument(
        '--load_model', type=str, default=None,
        help='Path to a saved model directory'
    )
    args = parser.parse_args()

    if args.load_model:
        print(f"Loading model from {args.load_model}")
        try:
            topic_model = BERTopic.load(
                os.path.join(args.load_model,'bertopic_model')
            )
        except Exception as e:
            print('Error loading model:', e)
            sys.exit(1)
        version_str = os.path.basename(os.path.normpath(args.load_model))
        combined_df = load_data_stratified(main_dir, SAMPLE_FRAC, SAMPLE_SEED)
    else:
        # Determine new version folder
        version_folders = [f for f in os.listdir(base_save_dir)
                           if f.startswith('version_') and f[8:].isdigit()]
        new_num = max([int(f[8:]) for f in version_folders], default=0) + 1
        new_version = f'version_{new_num}'
        save_dir = os.path.join(base_save_dir, new_version)
        os.makedirs(save_dir, exist_ok=True)
        version_str = new_version

        embedding_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2', device='cuda'
        )
        topic_model, combined_df = process_data_sampled(
            main_dir, embedding_model
        )
        topic_model.topics_ = combined_df['topic'].tolist()
        save_model(save_dir, topic_model, combined_df)

    # Visualization (unchanged)
    docs = combined_df['processed_text'].tolist()
    timestamps = combined_df['date_parsed'].tolist()

    info = topic_model.get_topic_info()
    fig = topic_model.visualize_barchart(top_n_topics=info.shape[0])
    fig.write_html(os.path.join(base_save_dir, f'barchart_{version_str}.html'))

    fig = topic_model.visualize_topics()
    fig.write_html(os.path.join(base_save_dir, f'topics_{version_str}.html'))

    fig = topic_model.visualize_heatmap()
    fig.write_html(os.path.join(base_save_dir, f'heatmap_{version_str}.html'))

    cutoff_date = pd.Timestamp('2024-01-01')
    filtered_df = combined_df[combined_df['date_parsed'] >= cutoff_date]
    docs_ft = filtered_df['processed_text'].tolist()
    times_ft = filtered_df['date_parsed'].tolist()

    topics_over_time = topic_model.topics_over_time(docs_ft, times_ft, nr_bins=20)
    fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=8)
    fig.write_html(os.path.join(base_save_dir, f'topics_over_time_{version_str}.html'))
