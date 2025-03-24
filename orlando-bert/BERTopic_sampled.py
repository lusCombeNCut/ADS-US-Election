import os
import glob
import sys
import argparse
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Thresholds for filtering tweets
MIN_LIKES = 10
MIN_RETWEETS = 0

# Sampling parameters
SAMPLE_FRAC = 0.01  # Adjust the fraction as needed
SAMPLE_SEED = 42   # Fixed seed for reproducibility

# Combine custom stop words with NLTK stop words
custom_stop_words = [
    "presidential", "dems", "republican", "trump2024", "democratic", "republicans",
    "democrat", "democrats", "the", "biden", "trump", "harris", "kamala", "vote",
    "maga", "MAGA", "president", "donald", "gop", "joe", "rnc", "dnc", "election",
    "voters", "vote", "tweet", "retweet", "follow", "campaign", "nominee", "presidency",
    "liberals", "liberal", "realdonaltrump", "conservatives", "conservative", "political",
    "bidenharris2024", "2024", "joebiden", "potus"
]
nltk_stop_words = stopwords.words('english')
custom_stop_words = list(set(custom_stop_words) | set(nltk_stop_words))


def remove_links(text):
    """Remove words containing 'https' from text."""
    return " ".join(word for word in text.split() if "https" not in word)


def filter_data(df):
    """Filter and process the DataFrame containing tweet data."""
    print(f"Total tweets prefiltering: {df.shape[0]}")
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date_parsed'].notna()]
    
    # Convert likeCount and retweetCount to numeric values
    df['likeCount'] = pd.to_numeric(df['likeCount'], errors='coerce')
    df['retweetCount'] = pd.to_numeric(df['retweetCount'], errors='coerce')
    df = df[(df['likeCount'] >= MIN_LIKES) & (df['retweetCount'] >= MIN_RETWEETS)]
    print(f"Tweets after thresholding: {df.shape[0]}")
    
    # Keep only the first tweet per conversation based on 'epoch'
    df = df.sort_values('epoch').drop_duplicates(subset=['conversationId'], keep='first')
    print(f"Tweets after selecting first per conversation: {df.shape[0]}")
    
    # Only keep English tweets and process text
    df = df[(df['lang'] == 'en') & (df["text"].notna())]
    df["processed_text"] = df["text"].apply(remove_links)
    df = df.dropna(subset=["processed_text"])
    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)
    print(f"Tweets after processing: {df.shape[0]}")
    return df


def load_data_sampled(root_folder, sample_frac=SAMPLE_FRAC, seed=SAMPLE_SEED):
    """
    Load CSV files from each 'part_*' folder, sample a fixed fraction from the raw data,
    and then filter the sampled data for reproducibility and efficiency.
    """
    part_folders = sorted(glob.glob(os.path.join(root_folder, "part_*")))
    sampled_dfs = []
    
    for folder in part_folders:
        file_pattern = os.path.join(folder, "*.csv.gz")
        files = glob.glob(file_pattern)
        if not files:
            continue
        dfs = [pd.read_csv(f, compression="gzip") for f in files]
        combined_df = pd.concat(dfs, ignore_index=True)
        # Sample a fraction from the raw data using a fixed seed
        sampled_df = combined_df.sample(frac=sample_frac, random_state=seed)
        # Now filter only the sampled data
        sampled_df = filter_data(sampled_df)
        sampled_dfs.append(sampled_df)

    full_df = pd.concat(sampled_dfs, ignore_index=True)
    return full_df


def initialize_topic_model(embedding_model):
    """Initialize BERTopic with standard UMAP and HDBSCAN."""
    representation_model = KeyBERTInspired()
    
    # Standard UMAP settings
    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
    # Standard HDBSCAN settings
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=50, metric='cosine', cluster_selection_method='eom')
    custom_vectorizer = CountVectorizer(stop_words=custom_stop_words, ngram_range=(1, 3), min_df=3)
    
    topic_model = BERTopic(
        representation_model=representation_model,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=custom_vectorizer,
        language="english",
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=True,
    )
    return topic_model


def process_data_sampled(main_dir, embedding_model, sample_frac=SAMPLE_FRAC, seed=SAMPLE_SEED):
    print("Loading and sampling data from parts...")
    combined_df = load_data_sampled(main_dir, sample_frac, seed)
    docs = combined_df["processed_text"].tolist()
    print(f"Total documents after sampling: {len(docs)}")
    
    embeddings = embedding_model.encode(docs, show_progress_bar=True).astype("float64")
    topic_model = initialize_topic_model(embedding_model)
    topics, _ = topic_model.fit_transform(documents=docs, embeddings=embeddings)
    combined_df['topic'] = topics
    return topic_model, combined_df


def save_model(save_dir, topic_model: BERTopic, df):
    """Save the trained BERTopic model and a CSV of topics."""
    columns_to_save = ["id", "topic", "date_parsed"]
    topic_model.save(os.path.join(save_dir, "bertopic_model"), serialization="pytorch")
    df[columns_to_save].to_csv(os.path.join(save_dir, "topics.csv.gz"), index=False, compression="gzip")


if __name__ == "__main__":
    # HPC paths
    base_save_dir = "/user/home/sv22482/work/ADS-US-Election/orlando-bert/ADS"
    main_dir = "/user/work/sv22482/usc-x-24-us-election"

    # # Local paths for testing (uncomment if needed)
    # base_save_dir = r"C:\Users\Orlan\Documents\Applied-Data-Science"
    # main_dir = r"C:\Users\Orlan\Documents\usc-x-24-us-election-main"

    parser = argparse.ArgumentParser(description="BERTopic Batch Training or Loading a Saved Model")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to the previously saved model directory")
    args = parser.parse_args()

    if args.load_model:
        print(f"Loading model from {args.load_model}")
        try:
            topic_model = BERTopic.load(os.path.join(args.load_model, "bertopic_model"))
        except Exception as e:
            print("Error loading model:", e)
            sys.exit(1)
        version_str = os.path.basename(os.path.normpath(args.load_model))
        # Reload data using the sampling function
        combined_df = load_data_sampled(main_dir, SAMPLE_FRAC, SAMPLE_SEED)
    else:
        version_folders = [folder for folder in os.listdir(base_save_dir)
                           if folder.startswith("version_") and folder[8:].isdigit()]
        new_version_number = max([int(folder[8:]) for folder in version_folders], default=0) + 1
        new_version = "version_" + str(new_version_number)
        save_dir = os.path.join(base_save_dir, new_version)
        os.makedirs(save_dir, exist_ok=True)
        version_str = new_version

        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
        topic_model, combined_df = process_data_sampled(main_dir, embedding_model, SAMPLE_FRAC, SAMPLE_SEED)
        topic_model.topics_ = combined_df['topic'].tolist()
        save_model(save_dir, topic_model, combined_df)

    # Visualization: Extract docs and timestamps
    docs = combined_df["processed_text"].tolist()
    timestamps = combined_df["date_parsed"].tolist()

    info = topic_model.get_topic_info()
    fig = topic_model.visualize_barchart(top_n_topics=info.shape[0])
    fig.write_html(os.path.join(base_save_dir, f"barchart_{version_str}.html"))

    fig = topic_model.visualize_topics()
    fig.write_html(os.path.join(base_save_dir, f"topics_{version_str}.html"))

    fig = topic_model.visualize_heatmap()
    fig.write_html(os.path.join(base_save_dir, f"heatmap_{version_str}.html"))

    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
    fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=8)
    fig.write_html(os.path.join(base_save_dir, f"topics_over_time_{version_str}.html"))
