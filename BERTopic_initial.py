import os
import sys
import glob
import pickle
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired

MAX_NUM_FILES = 20
CUSTOM_IGNORE_WORDS = {"rt", "amp"}
MIN_LIKES = 5
MIN_RETWEETS = 0

# File paths for saving/loading
MODEL_PATH = "saved_bertopic_model"
EMBEDDINGS_PATH = "saved_embeddings.pkl"

def load_data(folder_path, max_files=MAX_NUM_FILES):
    file_pattern = os.path.join(folder_path, "*.csv.gz")
    file_list = glob.glob(file_pattern)
    dfs = [pd.read_csv(file_list[i], compression="gzip") for i in range(min(len(file_list), max_files))]
    return pd.concat(dfs, ignore_index=True)

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

def run_topic_model_fitting(docs, embedding_model):
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    representation_model = KeyBERTInspired()

    custom_umap = umap.UMAP(n_neighbors=10, n_components=5, metric="cosine", init="random", random_state=42)
    custom_hdbscan = HDBSCAN(min_samples=20, gen_min_span_tree=True, prediction_data=True, core_dist_n_jobs=5)

    topic_model = BERTopic(
        representation_model=representation_model,
        embedding_model=embedding_model,
        umap_model=custom_umap,
        hdbscan_model=custom_hdbscan,
        language="english",
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )
    
    topics, probs = topic_model.fit_transform(documents=docs, embeddings=embeddings)

    topic_model.save(MODEL_PATH)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    
    return topic_model, embeddings, topics, probs
def get_docs():
    folder_path = r"C:\Users\Orlan\Documents\Applied-Data-Science\part_1"
    df = load_data(folder_path)
    df = filter_data(df)
    docs  = df["processed_text"].tolist()
    return docs, df

def main():
    if '-y' not in sys.argv:
        user_input = input("Continue analysis with these tweets? (yes/no): ")
        if user_input.lower().strip() not in ['yes', 'y']:
            print("Exiting analysis.")
            return

    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

    if os.path.exists(MODEL_PATH) and os.path.exists(EMBEDDINGS_PATH):
        load_choice = input("Saved topic model and embeddings found. Load them? (yes/no): ")
        if load_choice.lower().strip() in ['yes', 'y']:
            print("Loading saved embeddings and topic model...")
            with open(EMBEDDINGS_PATH, "rb") as f:
                embeddings = pickle.load(f)
            topic_model = BERTopic.load(MODEL_PATH)
        else:
            print("Re-running the fitting process...")
            docs, df = get_docs()
            topic_model, embeddings, topics, probs = run_topic_model_fitting(docs, embedding_model)
    else:
        print("No saved model/embeddings found. Running the fitting process...")
        docs, df = get_docs()
        topic_model, embeddings, topics, probs = run_topic_model_fitting(docs, embedding_model)

    info = topic_model.get_topic_info()
    print(info)

    # fig = topic_model.visualize_documents(docs, embeddings=embeddings, sample=100)
    # fig.show()

    fig = topic_model.visualize_barchart(top_n_topics=info.shape[0])
    fig.show()

    timestamps = df['date_parsed'].tolist()
    topics_over_time = topic_model.topics_over_time(docs, timestamps)

    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.show()

    fig = topic_model.visualize_topics()
    fig.show()


if __name__ == "__main__":
    main()
