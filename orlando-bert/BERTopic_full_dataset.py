import os
import glob
import sys
import argparse
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

MIN_LIKES = 20
MIN_RETWEETS = 3

custom_stop_words = ["presidentail", "dems","republican", "trump2024", "democratic", "reupublicans",  "democrat", "democrats",
 "the", "biden", "trump", "harris", "kamala", "vote", "maga", "MAGA", "president", "donald", "gop", "joe",
 "rnc", "dnc", "election", "voters", "vote", "tweet", "retweet", "follow", "campaign"]
nltk_stop_words = stopwords.words('english')
custom_stop_words = list(set(custom_stop_words) | set(nltk_stop_words))

def remove_links(text):
    return " ".join(word for word in text.split() if "https" not in word)

def filter_data(df):
    print(f"Total tweets prefiltering: {df.shape[0]}")
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    # Filter out rows where date parsing failed
    df = df[df['date_parsed'].notna()]
    
    # Convert likeCount and retweetCount to numeric values
    df['likeCount'] = pd.to_numeric(df['likeCount'], errors='coerce')
    df['retweetCount'] = pd.to_numeric(df['retweetCount'], errors='coerce')
    df = df[(df['likeCount'] >= MIN_LIKES) & (df['retweetCount'] >= MIN_RETWEETS)]
    print(f"Tweets after thresholding: {df.shape[0]}")
    
    df = df.sort_values('epoch').drop_duplicates(subset=['conversationId'], keep='first')
    print(f"Tweets after selecting first per conversation: {df.shape[0]}")
    
    df = df[(df['lang'] == 'en') & (df["text"].notna())]
    # Copy text to processed_text and remove links
    df["processed_text"] = df["text"].apply(remove_links)
    df = df.dropna(subset=["processed_text"])
    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)
    print(f"Tweets after processing: {df.shape[0]}")
    # df = df.head(50)
    return df

def load_data_batches(root_folder):
    part_folders = sorted(glob.glob(os.path.join(root_folder, "part_*")))
    for folder in part_folders:
        file_pattern = os.path.join(folder, "*.csv.gz")
        files = glob.glob(file_pattern)
        if not files:
            continue
        dfs = [pd.read_csv(f, compression="gzip") for f in files]
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = filter_data(combined_df)
        docs = combined_df["processed_text"].tolist()
        yield combined_df, docs

def initialize_topic_model(embedding_model):
    representation_model = KeyBERTInspired()
    online_dim_reducer = IncrementalPCA(n_components=5)
    online_cluster = MiniBatchKMeans(n_clusters=20, batch_size=100)
    custom_vectorizer = OnlineCountVectorizer(stop_words=custom_stop_words, ngram_range=(1, 1), min_df=2)
    
    topic_model = BERTopic(
        representation_model=representation_model,
        embedding_model=embedding_model,
        umap_model=online_dim_reducer,       # Replacing UMAP with IncrementalPCA
        hdbscan_model=online_cluster,         # Replacing HDBSCAN with MiniBatchKMeans
        vectorizer_model=custom_vectorizer,   # Use custom vectorizer
        language="english",
        nr_topics=None,
        calculate_probabilities=False,       # No probabilities needed
        verbose=True,
    )
    return topic_model

def process_batches(main_dir, embedding_model):
    topic_model = None
    all_dfs = []
    
    for i, (df_batch, docs_batch) in enumerate(load_data_batches(main_dir)):
        print(f"\nProcessing batch {i+1} with {len(docs_batch)} documents")
        embeddings_batch = embedding_model.encode(docs_batch, show_progress_bar=True).astype("float64")
        if i == 0:
            topic_model = initialize_topic_model(embedding_model)
            topics = topic_model.fit_transform(documents=docs_batch, embeddings=embeddings_batch)[0]
        else:
            topic_model.partial_fit(documents=docs_batch, embeddings=embeddings_batch)
            topics = topic_model.transform(docs_batch)[0]
        df_batch['topic'] = topics
        all_dfs.append(df_batch)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return topic_model, combined_df

def save_model(save_dir, topic_model: BERTopic, df):
    columns_to_save = ["id", "topic", "date_parsed"]

    topic_model.save(os.path.join(save_dir, "bertopic_model"), serialization="pytorch")
    df[columns_to_save].to_csv(os.path.join(save_dir , "topics.csv.gz"), index=False, compression="gzip")

if __name__ == "__main__":
    
    # HPC paths
    base_save_dir = "/user/home/sv22482/work/ADS-US-Election/orlando-bert/ADS"
    main_dir = '/user/work/sv22482/usc-x-24-us-election'

    # # Local paths for testing
    # base_save_dir = r"C:\Users\Orlan\Documents\Applied-Data-Science\ADS"
    # main_dir = r"C:\Users\Orlan\Documents\usc-x-24-us-election-main"

    parser = argparse.ArgumentParser(description="BERTopic Incremental Training or Loading a Saved Model")
    parser.add_argument("--load_model", type=str, default=None, help="Path to the previously saved model directory") #/user/home/sv22482/work/ADS-US-Election/orlando-bert/ADS/version_21 
    args = parser.parse_args()
    
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        try:
            topic_model = BERTopic.load(os.path.join(args.load_model, "bertopic_model"))
        except Exception as e:
            print("Error loading model:", e)
            sys.exit(1)
        version_str = os.path.basename(os.path.normpath(args.load_model))
        combined_df = pd.concat([df for df, _ in load_data_batches(main_dir)], ignore_index=True)
 
    else:
        
        version_folders = [folder for folder in os.listdir(base_save_dir) 
                           if folder.startswith("version_") and folder[8:].isdigit()]
        new_version_number = max([int(folder[8:]) for folder in version_folders], default=0) + 1
        new_version = "version_" + str(new_version_number)
        save_dir = os.path.join(base_save_dir, new_version)
        os.makedirs(save_dir, exist_ok=True)
        version_str = new_version
        
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
        topic_model, combined_df = process_batches(main_dir, embedding_model)
        topic_model.topics_ = combined_df['topic'].tolist()
        save_model(save_dir, topic_model, combined_df)

    # Extract docs and timestamps from the filtered DataFrame
    docs = combined_df["processed_text"].tolist()
    timestamps = combined_df["date_parsed"].tolist()
    # Visualization: Save figures with version_str in filename
    info = topic_model.get_topic_info()

    fig = topic_model.visualize_barchart(top_n_topics=info.shape[0])
    fig.write_html(os.path.join(base_save_dir, f"barchart_{version_str}.html"))

    fig = topic_model.visualize_topics()
    fig.write_html(os.path.join(base_save_dir, f"topics_{version_str}.html"))

    fig = topic_model.visualize_heatmap()
    fig.write_html(os.path.join(base_save_dir, f"heatmap_{version_str}.html"))

    fig = topic_model.visualize_hierarchy()
    fig.write_html(os.path.join(base_save_dir, f"hierarchy_{version_str}.html"))

    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)

    fig = topic_model.visualize_topics_over_time(docs, timestamps, top_n_topics=8)
    fig.write_html(os.path.join(base_save_dir, f"topics_over_time_{version_str}.html"))

