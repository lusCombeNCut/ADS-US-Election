import pandas as pd
from glob import glob

TOPICS_PATH = "./orlando_BERTopic_results_v3.csv"
CHUNKS_PATH = "../cleaned"

chunks = glob(f"{CHUNKS_PATH}/*.csv")
topics = pd.read_csv(TOPICS_PATH)
chunks_df = [pd.read_csv(f) for f in chunks]
print("ALL CHUNKS READ")
dset = pd.concat(chunks_df, ignore_index=True)
print("ALL CHUNKS CONCAT")
merged = pd.merge(topics, dset, on="id", how='left')
merged.to_csv("./BERTTopic_sentiment_irony.csv")
