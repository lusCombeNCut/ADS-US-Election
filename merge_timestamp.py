import pandas as pd
from glob import glob
import os
import concurrent.futures
from tqdm import tqdm
import multiprocessing

# Function to safely format ID with error handling
def format_id(x):
    try:
        # Handle NaN values
        if pd.isna(x):
            return None
            
        # Convert to string if it's a float
        if isinstance(x, float):
            return str(int(x))
            
        # Handle URLs and other invalid formats
        if isinstance(x, str) and ('twitter.com' in x or 'http' in x):
            # print(f"Invalid ID (URL): {x} - skipping this entry")
            return None
            
        # Now proceed with normal string processing
        if isinstance(x, str) and '.' in x:
            return str(int(float(x)))
            
        return str(x)
    except (ValueError, TypeError) as e:
        # print(f"Invalid ID found: {x} (Error: {e}) - skipping this entry")
        return None  # Will be filtered out later

# Must define format_chunk outside of parallel_format_ids to make it picklable
def format_chunk(chunk):
    chunk_copy = chunk.copy()
    chunk_copy['id'] = chunk_copy['id'].apply(format_id)
    chunk_copy = chunk_copy[chunk_copy['id'].notna()]
    chunk_copy['id'] = chunk_copy['id'].apply(lambda x: x.zfill(19) if len(x) < 19 else x)
    return chunk_copy

# Apply ID formatting in parallel to chunks of dataframe
def parallel_format_ids(df, num_workers=4):
    # Split dataframe into chunks
    chunk_size = max(1, len(df) // num_workers)
    df_chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(format_chunk, df_chunks), 
                           total=len(df_chunks), 
                           desc="Formatting IDs"))
    
    # Combine results
    return pd.concat(results, ignore_index=True)

# Parallelize reading sentiment chunks
def read_chunk(chunk_path):
    try:
        df = pd.read_csv(chunk_path, dtype={'id': str})
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        print(f"Successfully loaded {chunk_path}")
        return df
    except Exception as e:
        print(f"Error loading {chunk_path}: {e}")
        return None

def main():
    TOPICS_PATH = r"orlando-bert\inference-output-low-filtering\topic-inference-results.csv"
    topic_inference_df = pd.read_csv(TOPICS_PATH, dtype={'id': str})

    # Save original data for comparison
    original_topic_df = topic_inference_df.copy()

    # Fix ID formatting for topic inference data
    topic_inference_df = parallel_format_ids(topic_inference_df, num_workers=4)

    if not os.path.exists('sentiment_results.csv.gz'):
        CHUNKS_PATH = "./sentiment_results"
        sentiment_chunks = glob(f"{CHUNKS_PATH}/*.csv")
        
        # Read chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            dfs = list(tqdm(executor.map(read_chunk, sentiment_chunks), 
                        total=len(sentiment_chunks), 
                        desc="Loading chunks"))
        
        # Filter out None results (from failed loads)
        dfs = [df for df in dfs if df is not None]
        
        sentiment_df = pd.concat(dfs)
        sentiment_df = sentiment_df.sort_values('date').drop_duplicates(subset='id', keep='first')
        
        # Save original sentiment data for comparison
        original_sentiment_df = sentiment_df.copy()
        
        # Process sentiment data in parallel
        sentiment_df = parallel_format_ids(sentiment_df, num_workers=4)
        
        sentiment_df.to_csv('sentiment_results.csv.gz', index=False, compression='gzip')
    else:
        # Same approach for the cached version
        sentiment_df = pd.read_csv('sentiment_results.csv.gz', compression='gzip', dtype={'id': str})
        sentiment_df = sentiment_df.sort_values('date').drop_duplicates(subset='id', keep='first')
        
        original_sentiment_df = sentiment_df.copy()
        
        # Process sentiment data in parallel
        sentiment_df = parallel_format_ids(sentiment_df, num_workers=4)

    # Check merge results before ID formatting
    original_merged_df = pd.merge(original_topic_df, original_sentiment_df, on='id', how='inner')
    original_match_count = len(original_merged_df)

    # Check merge results after ID formatting
    merged_df = pd.merge(topic_inference_df, sentiment_df, on='id', how='inner')
    new_match_count = len(merged_df)

    # Find IDs from topic inference that don't exist in sentiment results
    missing_ids = topic_inference_df[~topic_inference_df['id'].isin(sentiment_df['id'])]
    print(f"Number of tweet IDs not found in sentiment results: {len(missing_ids)}")
    
    # Save missing IDs to CSV
    missing_ids.to_csv('missing_tweet_ids.csv', index=False)
    print(f"Missing tweet IDs saved to 'missing_tweet_ids.csv'")

    # Print comparison
    print(f"Original number of matches: {original_match_count}")
    print(f"Number of matches after ID formatting: {new_match_count}")
    print(f"Difference: {new_match_count - original_match_count} additional matches")
    if original_match_count > 0:
        print(f"Improvement: {((new_match_count - original_match_count) / original_match_count * 100):.2f}% more matches")
    else:
        print("No original matches to calculate improvement percentage.")

    # Save the merged data
    merged_df.to_csv('merged_low_filter.csv', index=False)

if __name__ == "__main__":
    # This is required on Windows when using multiprocessing
    multiprocessing.freeze_support()
    main()






