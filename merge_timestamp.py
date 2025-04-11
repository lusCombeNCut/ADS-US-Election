import pandas as pd
from glob import glob

TOPICS_PATH = "./v3_results.csv"
CHUNKS_PATH = "./sentiment_results"

chunks = glob(f"{CHUNKS_PATH}/*.csv")
topics = pd.read_csv(TOPICS_PATH, dtype={'id': str})[['id', 'topic']]
chunks_df = [pd.read_csv(f, dtype={'id': str}) for f in chunks]
print("ALL CHUNKS READ")
dset = pd.concat(chunks_df, ignore_index=True)
print("ALL CHUNKS CONCAT")
merged = pd.merge(topics, dset, on="id", how='inner')
merged.to_csv("./BERTTopic_sentiment_irony.csv", index=False)
