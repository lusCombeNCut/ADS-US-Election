import os
import pandas as pd
import numpy as np
import torch
import nltk
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Monkey-patch hf_hub_download to read local files only
import huggingface_hub
_orig_hf_download = huggingface_hub.hf_hub_download
def _local_hf_download(repo_id, filename, revision=None, repo_type=None, subfolder=None, **kwargs):
    return os.path.join(repo_id, filename)
huggingface_hub.hf_hub_download = _local_hf_download

nltk.download('stopwords')

# === CONFIG ===
MODEL_PATH     = "HPC-output-dir/version_5"
DATA_FILE      = "results-data/weekly_headlines_full.csv"
OUTPUT_FILE    = "results-data/news-desc-topic-assignment.csv"
EMB_BATCH      = 1024
# ==============

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mpnet_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device=device
)

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

# Load dataset
print(f"Loading data from {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

docs = df['description'].tolist()
embeddings = batch_embed(docs)

# Assign the best topic for each document
topics, probs = topic_model.transform(docs, embeddings=embeddings)

# Add topics to DataFrame
df['assigned_topic'] = topics
df['topic_probability'] = probs

# Save the results
df[['week_start', 'description', 'assigned_topic', 'topic_probability']].to_csv(
    OUTPUT_FILE, index=False
)

print(f"Saved topic assignments → {OUTPUT_FILE}")
print("Done.")
