import os
import glob
import pandas as pd
import torch
import nltk
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModel
from torch.nn import DataParallel

# 0) Monkey‑patch hf_hub_download to read local files only
import huggingface_hub
_orig_hf_download = huggingface_hub.hf_hub_download
def _local_hf_download(repo_id, filename, revision=None, repo_type=None, subfolder=None, **kwargs):
    return os.path.join(repo_id, filename)
huggingface_hub.hf_hub_download = _local_hf_download

# Ensure NLTK stopwords (if you use them elsewhere)
nltk.download('stopwords')

# === CONFIG ===
MODEL_PATH     = "/user/home/sv22482/work/ADS-US-Election/orlando-bert/ADS/version_5"
DATA_FOLDER    = "/user/home/sv22482/work/usc-x-24-us-election"
OUTPUT_FOLDER  = "/user/home/sv22482/work/ADS-US-Election/orlando-bert/inference-output"
BATCH_SIZE     = 100_000
EMB_BATCH      = 512
MAXLEN         = 128
MIN_LIKES      = 10
MIN_RETWEETS   = 0
# ==============

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1) GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load tokenizer & model onto GPU
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
bertweet_model = AutoModel.from_pretrained("vinai/bertweet-base").to(device)

# 3) If multiple GPUs, wrap in DataParallel
if torch.cuda.device_count() > 1:
    bertweet_model = DataParallel(bertweet_model)

def remove_links(text: str) -> str:
    """Remove words containing 'https' from text."""
    return " ".join(word for word in text.split() if "https" not in word)

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and process the DataFrame containing tweet data."""
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

def batch_embed(texts: list[str]) -> torch.Tensor:
    """Tokenize & embed in sub-batches on GPU, return CPU tensor of CLS embeddings."""
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), EMB_BATCH):
            batch = texts[i:i+EMB_BATCH]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAXLEN,
                return_tensors="pt"
            ).to(device)
            out = bertweet_model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu()
            all_embs.append(cls)
    return torch.cat(all_embs, dim=0)

# 4) Load your BERTopic model from the local PyTorch serialization
topic_model = BERTopic.load(os.path.join(MODEL_PATH, "bertopic_model"))

# (optional) restore original hf_hub_download to avoid side-effects elsewhere
huggingface_hub.hf_hub_download = _orig_hf_download

# 5) Batched inference with filtering
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
            topics, _ = topic_model.transform(docs, embeddings=embeddings.numpy())
            df_chunk['topic'] = topics

            out_path = os.path.join(OUTPUT_FOLDER, f"{base}_batch{batch_idx}.csv.gz")
            df_chunk[['id', 'topic', 'date_parsed']].to_csv(
                out_path, index=False, compression="gzip"
            )
            print(f"    Saved batch {batch_idx} ({len(df_chunk)} rows) → {out_path}")

print("Inference complete.")
