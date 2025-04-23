import pandas as pd

import glob


TOPICS_PATH = "orlando-bert/inference-output-low-filtering/topic-inference-results.csv"

dfs = [pd.read_csv(f) for f in glob.glob("sentiment_results/*.csv")]

df = pd.concat(dfs, ignore_index=True)
print("DF CONCAT")

df = df.drop_duplicates(subset=['id'])
print("DROPPED DUPES", df.shape)

topics = pd.read_csv(TOPICS_PATH)

print(f"TOPICS SHAPE {topics.shape}")
print(f"SENT SHAPE {df.shape}")
merged = pd.merge(topics, df, on='id', how='inner')[['id', 'date', 'topic', 'sentiment', 'irony']]
print(f"MERGED SHAPE {merged.shape}")

merged.to_csv('new_merged_sentiment_topic.csv')
