#!/usr/bin/env python

import os
import glob
import argparse
import pandas as pd
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Download stopwords if not already available
nltk.download('stopwords')

# Sampling parameters
SAMPLE_FRAC = 0.1  # Adjust fraction as needed
SAMPLE_SEED = 42    # Fixed seed for reproducibility

# Combine custom stop words with NLTK stop words
custom_stop_words = [
    "presidential", "dems", "republican", "trump2024", "democratic", "republicans",
    "democrat", "democrats", "the", "biden", "trump", "harris", "kamala", "vote",
    "maga", "MAGA", "president", "donald", "gop", "joe", "rnc", "dnc", "election",
    "voters", "vote", "tweet", "retweet", "follow", "campaign", "nominee", "presidency",
    "liberals", "liberal", "realdonaltrump", "conservatives", "conservative", "political",
    "bidenharris2024", "2024", "joebiden", "potus"
]
nltk_stop_words = stopwords.words('english')
custom_stop_words = list(set(custom_stop_words) | set(nltk_stop_words))


def remove_links(text):
    """Remove words containing 'https' from text."""
    return " ".join(word for word in text.split() if "https" not in word)


def filter_data(df):
    """Filter and process the DataFrame containing tweet data."""
    print(f"Total tweets prefiltering: {df.shape[0]}")
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date_parsed'].notna()]
    
    # Convert likeCount and retweetCount to numeric values
    df['likeCount'] = pd.to_numeric(df['likeCount'], errors='coerce')
    df['retweetCount'] = pd.to_numeric(df['retweetCount'], errors='coerce')
    
    # Apply thresholds (you can adjust these if needed)
    df = df[(df['likeCount'] >= 10) & (df['retweetCount'] >= 0)]
    print(f"Tweets after thresholding: {df.shape[0]}")
    
    # Keep only the first tweet per conversation based on 'epoch'
    df = df.sort_values('epoch').drop_duplicates(subset=['conversationId'], keep='first')
    print(f"Tweets after selecting first per conversation: {df.shape[0]}")
    
    # Only keep English tweets and process text
    df = df[(df['lang'] == 'en') & (df["text"].notna())]
    df["processed_text"] = df["text"].apply(remove_links)
    df = df.dropna(subset=["processed_text"])
    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)
    print(f"Tweets after processing: {df.shape[0]}")
    return df


def load_data_sampled(root_folder, sample_frac=SAMPLE_FRAC, seed=SAMPLE_SEED):
    """
    Load CSV files from each 'part_*' folder, sample a fixed fraction from the raw data,
    and then filter the sampled data.
    """
    part_folders = sorted(glob.glob(os.path.join(root_folder, "part_*")))
    sampled_dfs = []
    
    for folder in part_folders:
        file_pattern = os.path.join(folder, "*.csv.gz")
        files = glob.glob(file_pattern)
        if not files:
            continue
        dfs = [pd.read_csv(f, compression="gzip") for f in files]
        combined_df = pd.concat(dfs, ignore_index=True)
        sampled_df = combined_df.sample(frac=sample_frac, random_state=seed)
        sampled_df = filter_data(sampled_df)
        sampled_dfs.append(sampled_df)
    
    if not sampled_dfs:
        print("No data found in the specified root folder.")
        exit(1)
    
    full_df = pd.concat(sampled_dfs, ignore_index=True)
    return full_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract tweet IDs assigned to a specified topic using a saved BERTopic model."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the saved BERTopic model directory (the directory containing 'bertopic_model')"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing the part_* folders with tweet CSVs"
    )
    parser.add_argument(
        "--topic", type=int, required=True,
        help="Topic number for which to extract tweet IDs"
    )
    parser.add_argument(
        "--output", type=str, default="tweet_ids.csv",
        help="Output CSV file for the tweet IDs"
    )
    args = parser.parse_args()
    
    # Load the saved BERTopic model
    model_file = os.path.join(args.model_path, "bertopic_model")
    print(f"Loading model from {model_file}")
    try:
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
        topic_model = BERTopic.load(model_file, embedding_model=embedding_model)
    except Exception as e:
        print("Error loading model:", e)
        exit(1)
    
    # Load and sample the data
    print(f"Loading data from {args.data_dir}")
    df = load_data_sampled(args.data_dir, SAMPLE_FRAC, SAMPLE_SEED)
    
    # Make sure the processed text column is available
    if "processed_text" not in df.columns:
        print("Processed text column not found in data.")
        exit(1)
    
    docs = df["processed_text"].tolist()
    
    # Use the loaded model to assign topics to the documents
    topics, _ = topic_model.transform(docs)
    df["topic"] = topics
    
    # Extract tweet IDs for rows where the topic matches the specified value
    selected_ids = df.loc[df["topic"] == args.topic, "id"].tolist()
    
    print(f"Found {len(selected_ids)} tweets assigned to topic {args.topic}")
    
    # Save the tweet IDs to a CSV file
    output_df = pd.DataFrame({"id": selected_ids})
    output_df.to_csv(args.output, index=False)
    print(f"Tweet IDs saved to {args.output}")


if __name__ == "__main__":
    main()
