import os
import re
import glob
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans

# Thresholds
MIN_LIKES = 5
MIN_RETWEETS = 0

def filter_data(df):
    print(f"Total tweets prefiltering: {df.shape[0]}")
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[(df['likeCount'] >= MIN_LIKES) & (df['retweetCount'] >= MIN_RETWEETS)]
    print(f"Tweets after thresholding: {df.shape[0]}")
    df = df.sort_values('epoch').drop_duplicates(subset=['conversationId'], keep='first')
    print(f"Tweets after selecting first per conversation: {df.shape[0]}")
    df = df[(df['lang'] == 'en') & (df["text"].notna())]
    df["processed_text"] = df["text"]
    df = df.dropna(subset=["processed_text"])
    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)
    print(f"Tweets after processing: {df.shape[0]}")
    return df

def load_data_batches(root_folder):
    """Generator that yields a (df, docs) tuple per batch (per part_ folder)."""
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
    """Initialize a BERTopic model with online-compatible components."""
    representation_model = KeyBERTInspired()
    # Use online variants for dimensionality reduction and clustering:
    online_dim_reducer = IncrementalPCA(n_components=5)
    online_cluster = MiniBatchKMeans(n_clusters=20, batch_size=100)
    online_vectorizer = OnlineCountVectorizer()
    
    topic_model = BERTopic(
        representation_model=representation_model,
        embedding_model=embedding_model,
        umap_model=online_dim_reducer,       # Replacing UMAP with IncrementalPCA
        hdbscan_model=online_cluster,         # Replacing HDBSCAN with MiniBatchKMeans
        vectorizer_model=online_vectorizer,   # Online count vectorizer for updating vocabulary
        language="english",
        nr_topics="auto",
        calculate_probabilities=False,       # No probabilities needed
        verbose=True,
    )
    return topic_model

def process_batches(main_dir, embedding_model):
    """Process data in batches and update the BERTopic model incrementally.
    Returns the combined dataframe with topic assignments.
    """
    topic_model = None
    all_dfs = []
    
    for i, (df_batch, docs_batch) in enumerate(load_data_batches(main_dir)):
        print(f"\nProcessing batch {i+1} with {len(docs_batch)} documents")
        embeddings_batch = embedding_model.encode(docs_batch, show_progress_bar=True)
        if i == 0:
            topic_model = initialize_topic_model(embedding_model)
            topics, _ = topic_model.fit_transform(documents=docs_batch, embeddings=embeddings_batch)
        else:
            topics, _ = topic_model.partial_fit(documents=docs_batch, embeddings=embeddings_batch)
        df_batch['topic'] = topics
        all_dfs.append(df_batch)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return topic_model, combined_df

def save_model(save_dir, model: BERTopic, df):
    """Save the model and topic assignments (without probabilities)."""
    # Save only "id" (if available) and "topic" columns.
    columns_to_save = ["id", "topic"] if "id" in df.columns else ["topic"]
    model.save(f"{save_dir}/bertopic_model", serialization="pytorch")
    df[columns_to_save].to_csv(f'{save_dir}/topics.csv.gz', index=False, compression="gzip")

if __name__ == "__main__":
    base_save_dir = "/user/work/sv22482/ADS"
    main_dir = '/user/work/sv22482/usc-x-24-us-election'
    
    # Filter for folders that match the expected version pattern "version_<number>"
    version_folders = [folder for folder in os.listdir(base_save_dir) 
                       if folder.startswith("version_") and folder[8:].isdigit()]
    new_version_number = max([int(folder[8:]) for folder in version_folders], default=0) + 1
    new_version = "version_" + str(new_version_number)
    save_dir = os.path.join(base_save_dir, new_version)
    os.makedirs(save_dir, exist_ok=True)
    
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    
    topic_model, combined_df = process_batches(main_dir, embedding_model)
    
    # Save the model and combined dataframe (only topics)
    save_model(save_dir, topic_model, combined_df)
    
    # Visualization (using combined data)
    timestamps = combined_df['date_parsed'].tolist()
    info = topic_model.get_topic_info()
    
    fig = topic_model.visualize_barchart(top_n_topics=info.shape[0])
    fig.write_html("barchart.html")
    
    topics_over_time = topic_model.topics_over_time(combined_df["processed_text"].tolist(), timestamps)
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("topics_over_time.html")
    
    fig = topic_model.visualize_topics()
    fig.write_html("topics.html")
