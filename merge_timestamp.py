import pandas as pd
from glob import glob

TOPICS_PATH = "./orlando-bert/topic-inference-results.csv"
CHUNKS_PATH = "./sentiment_results"

# I want to merge the topic inference results with the sentiment results using the 'id' column.
# Read the topic inference results
topic_inference_df = pd.read_csv(TOPICS_PATH, dtype={'id': str})

# Read all sentiment results CSV files and concatenate them into a single DataFrame
sentiment_chunks = glob(f"{CHUNKS_PATH}/*.csv")
sentiment_df = [pd.read_csv(f, dtype={'id': str}) for f in sentiment_chunks]
sentiment_df = pd.concat(sentiment_df)  # Combine all sentiment data into a single DataFrame

# Merge the topic inference results with the sentiment results using the 'id' do not use date colum from sentiment_df
# Merge the DataFrames on the 'id' column
# and drop the 'date' column from sentiment_df to avoid duplication

sentiment_df.drop(columns=['date'], inplace=True, errors='ignore')
merged_df = pd.merge(topic_inference_df, sentiment_df, on='id', how='inner')
# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_results_topicDates.csv', index=False)






