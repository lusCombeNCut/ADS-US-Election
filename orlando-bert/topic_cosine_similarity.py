import os
import glob
import pandas as pd
import numpy as np
import torch
import nltk
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Monkey-patch hf_hub_download to read local files only
import huggingface_hub
_orig_hf_download = huggingface_hub.hf_hub_download
def _local_hf_download(repo_id, filename, revision=None, repo_type=None, subfolder=None, **kwargs):
    return os.path.join(repo_id, filename)
huggingface_hub.hf_hub_download = _local_hf_download

nltk.download('stopwords')

# === CONFIG ===
MODEL_PATH     = "/user/home/sv22482/work/ADS-US-Election/orlando-bert/ADS/version_5"
DATA_FOLDER    = "/user/home/sv22482/work/usc-x-24-us-election"
OUTPUT_FOLDER  = "/user/home/sv22482/work/ADS-US-Election/orlando-bert/inference-output"
BATCH_SIZE     = 100_000
EMB_BATCH      = 512
MIN_LIKES      = 10
MIN_RETWEETS   = 0
TOP_K          = 5  # Top-k most similar topics to return
# ==============

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mpnet_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device=device
)

def remove_links(text: str) -> str:
    return " ".join(word for word in text.split() if "https" not in word)

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"    • Total tweets prefiltering: {df.shape[0]}")
    df = df.copy()
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date_parsed'].notna()]

    df['likeCount']    = pd.to_numeric(df['likeCount'], errors='coerce')
    df['retweetCount'] = pd.to_numeric(df['retweetCount'], errors='coerce')
    df = df[(df['likeCount'] >= MIN_LIKES) & (df['retweetCount'] >= MIN_RETWEETS)]
    print(f"    • After thresholds: {df.shape[0]}")

    df = df.sort_values('epoch').drop_duplicates(subset=['conversationId'], keep='first')
    print(f"    • First per conversation: {df.shape[0]}")

    df = df[(df['lang'] == 'en') & df['text'].notna()]
    df['processed_text'] = df['text'].apply(remove_links)
    df = df.dropna(subset=['processed_text'])
    df = df[df['processed_text'].str.strip() != ""].reset_index(drop=True)
    print(f"    • After processing: {df.shape[0]}")
    return df

def batch_embed(texts: list[str]) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), EMB_BATCH):
        chunk = texts[i : i + EMB_BATCH]
        embs = mpnet_model.encode(
            chunk,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        all_embs.append(embs)
    return np.vstack(all_embs)

# Load the BERTopic model
topic_model = BERTopic.load(os.path.join(MODEL_PATH, "bertopic_model"))
huggingface_hub.hf_hub_download = _orig_hf_download

# Get topic embeddings
topic_embs = topic_model.topic_embeddings_
topic_ids  = list(topic_model.get_topics().keys())

if topic_embs is None:
    raise ValueError("Topic embeddings are not available in the loaded BERTopic model.")

# Inference with similarity computation
for part_dir in sorted(glob.glob(os.path.join(DATA_FOLDER, "part_*"))):
    print(f"Processing folder: {part_dir}")
    for in_file in sorted(glob.glob(os.path.join(part_dir, "*.csv.gz"))):
        print(f"  File: {os.path.basename(in_file)}")
        reader = pd.read_csv(in_file, compression="gzip", chunksize=BATCH_SIZE)
        base = os.path.splitext(os.path.splitext(os.path.basename(in_file))[0])[0]

        for batch_idx, chunk in enumerate(reader):
            df_chunk = filter_data(chunk)
            if df_chunk.empty:
                print(f"    Batch {batch_idx}: no valid rows, skipping")
                continue

            docs = df_chunk['processed_text'].tolist()
            embeddings = batch_embed(docs)

            # Compute cosine similarity between each doc embedding and topic embeddings
            sims = cosine_similarity(embeddings, topic_embs)  # shape (n_docs, n_topics)

            # Get top-k topic indices and their scores
            topk_idx = np.argsort(-sims, axis=1)[:, :TOP_K]  # highest scores first
            topk_sims = np.take_along_axis(sims, topk_idx, axis=1)

            # Store as list of (topic_id, similarity_score) tuples
            topk_topics = [
                [(topic_ids[idx], score) for idx, score in zip(row_idx, row_sim)]
                for row_idx, row_sim in zip(topk_idx, topk_sims)
            ]

            df_chunk['top_topics'] = topk_topics

            out_path = os.path.join(OUTPUT_FOLDER, f"{base}_batch{batch_idx}_similarity.csv.gz")
            df_chunk[['id', 'top_topics', 'date_parsed']].to_csv(
                out_path, index=False, compression="gzip"
            )
            print(f"    Saved batch {batch_idx} ({len(df_chunk)} rows) → {out_path}")

print("Inference complete (with similarity scores).")
