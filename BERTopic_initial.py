import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import umap.umap_ as umap_module
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sys
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import geopandas as gpd
from shapely.geometry import Point

MAX_NUM_FILES = 20  # Limit to a subset of files for this run

# Attempt to import cuML's UMAP and HDBSCAN for GPU acceleration
try:
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    gpu_available = True
except ImportError:
    from hdbscan import HDBSCAN
    gpu_available = False

# Patch UMAP's spectral_layout
orig_spectral_layout = umap_module.spectral_layout
def patched_spectral_layout(data, graph, n_components, random_state, **kwargs):
    N = graph.shape[0]
    if (n_components + 1) >= N:
        A = graph.toarray() if hasattr(graph, "toarray") else graph
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(A)
        return eigenvectors[:, :n_components]
    else:
        return orig_spectral_layout(data, graph, n_components, random_state, **kwargs)
umap_module.spectral_layout = patched_spectral_layout

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")    

# Define stopwords and lemmatizer
CUSTOM_IGNORE_WORDS = {"rt", "amp", "ni", "el", "la", "nada", "de", "para", "al", "con", "le", "ver", "hay", "eu", "en", "se", "va", "trump", "biden", "joe", "donald"}
stop_words = set(stopwords.words("english")).union(CUSTOM_IGNORE_WORDS)
stop_words = list(stop_words)
lemmatizer = WordNetLemmatizer()

def is_english_word(word):
    """Check if a word exists in WordNet."""
    return bool(wordnet.synsets(word))

def preprocess_text(text):
    """Preprocess a tweet: remove URLs, mentions, hashtags, and apply text cleaning."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)      # Remove mentions
    text = re.sub(r"#\w+", "", text)      # Remove hashtags
    text = text.lower()                   # Lowercase

    # Tokenize and keep only alphabetic tokens
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if is_english_word(token)]
    
    processed_text = " ".join(tokens)
    return processed_text if processed_text.strip() else None


# ~~~ Open all csv files in part 1 ~~~
folder_path = r"C:\Users\Orlan\Documents\Applied-Data-Science\part_1"
file_pattern = os.path.join(folder_path, "*.csv.gz")
file_list = glob.glob(file_pattern)

print("Num files found:", file_list)

dfs = []  # List to store dataframes

for file_idx in range(0, min(len(file_list), MAX_NUM_FILES)):
    temp_df = pd.read_csv(file_list[file_idx], compression="gzip")
    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True)
# print(df.head())
print(f"Total tweets prefiltering: {df.shape[0]}")

# ~~~ Data exploration plots ~~~
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')

# if 'likeCount' in df.columns:
#     plt.figure(figsize=(10,6))
#     plt.hist(df['likeCount'], bins=30)
#     plt.xscale('log')
#     plt.xlabel('Like Count (log scale)')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Like Counts')
#     plt.tight_layout()
#     plt.show()

# if 'retweetCount' in df.columns:
#     plt.figure(figsize=(10,6))
#     plt.hist(df['retweetCount'], bins=50)
#     plt.xscale('log')
#     plt.xlabel("Retweet Count")
#     plt.ylabel("Frequency")
#     plt.title("Distribution of Retweets")
#     plt.tight_layout()
#     plt.show()

# Engagement thresholding
min_like_threshold =5
min_retweet_threshold = 0
df = df[(df['likeCount'] >= min_like_threshold) & (df['retweetCount'] >= min_retweet_threshold)]
print(f"Number of tweets after thresholding: {df.shape[0]}")

# Keep only the first tweet of each unique conversationId
df = df.sort_values('epoch').drop_duplicates(subset=['conversationId'], keep='first')
print(f"Number of tweets after selecting first tweet per conversation: {df.shape[0]}")

df = df[df["text"].notna()]
df["processed_text"] = df["text"].apply(preprocess_text)
df = df.dropna(subset=["processed_text"]) 
df = df[df["processed_text"].str.strip() != ""] 
df = df.reset_index(drop=True)
processed_tweets = df["processed_text"].tolist()
print(f"After removing null {len(processed_tweets)}.")


embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

# Allow user confirmation to continue analysis (or use -y flag to skip prompt)
if '-y' not in sys.argv:
    user_input = input("Continue analysis with these tweets? (yes/no): ")
    if user_input.lower().strip() not in ['yes', 'y']:
        print("Exiting analysis.")
        exit(0)

# --- Topic Modeling with BERTopic ---
if gpu_available:
    print("Using GPU-accelerated UMAP and HDBSCAN.")
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    custom_umap = cumlUMAP(n_neighbors=30, n_components=5, metric="cosine", init="random", random_state=42)
    custom_hdbscan = cumlHDBSCAN(min_samples=20, gen_min_span_tree=True, prediction_data=True)
else:
    print("Using CPU-based UMAP and HDBSCAN.")
    custom_umap = umap.UMAP(n_neighbors=30, n_components=5, metric="cosine", init="random", random_state=42)
    custom_hdbscan = HDBSCAN(min_samples=20, gen_min_span_tree=True, prediction_data=True)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=custom_umap,
    hdbscan_model=custom_hdbscan,
    language="english",
    nr_topics=10,
    calculate_probabilities=True,
    verbose=True,
)

start_time = time.perf_counter()
topics, probabilities = topic_model.fit_transform(processed_tweets)
end_time = time.perf_counter()
total_time = end_time - start_time
avg_time_per_tweet = total_time / len(processed_tweets)
print(f"Total topic modeling time: {total_time:.2f} seconds")
print(f"Average time per tweet: {avg_time_per_tweet:.4f} seconds")

unique_topics = np.unique(topics)
if len(unique_topics) <= 1:
    print("Too few topics detected! Try adjusting parameters or using more diverse data.")
    exit(1)

topic_info_df = topic_model.get_topic_info()
print(topic_info_df)

for topic_id in topic_info_df['Topic'].unique():
    if topic_id != -1:
        print(f"Topic {topic_id}:")
        print(topic_model.get_topic(topic_id))
        print("\n")

# Visualize topics
fig = topic_model.visualize_topics()
fig.show()

# # --- Plotting Tweet Locations on a Map ---
# # We'll attempt to parse the 'location' column as "lat,lon" if available.
# loc_df = df[df['location'].apply(lambda x: isinstance(x, str) and x.strip() != "")]

# if not loc_df.empty:
#     def parse_location(loc):
#         try:
#             parts = loc.split(',')
#             lat = float(parts[0].strip())
#             lon = float(parts[1].strip())
#             return lat, lon
#         except Exception:
#             return None
#     coords = loc_df['location'].apply(parse_location)
#     valid_coords = coords.dropna()
#     if not valid_coords.empty:
#         # Create new columns for lat and lon
#         loc_df = loc_df.loc[valid_coords.index].copy()
#         loc_df['lat'] = valid_coords.apply(lambda x: x[0])
#         loc_df['lon'] = valid_coords.apply(lambda x: x[1])
#         # Save locations to CSV for debugging
#         loc_csv_path = "tweet_locations.csv"
#         loc_df.to_csv(loc_csv_path, index=False)
#         print(f"Saved tweet locations to {loc_csv_path}")
#         # Create a GeoDataFrame and plot on a world map
#         gdf = gpd.GeoDataFrame(loc_df, geometry=gpd.points_from_xy(loc_df['lon'], loc_df['lat']))
#         world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#         ax = world.plot(figsize=(15, 10), color='lightgray', edgecolor='white')
#         gdf.plot(ax=ax, color='red', markersize=5)
#         plt.title("Tweet Locations")
#         plt.show()
#     else:
#         print("No valid coordinate data found in the location column.")
# else:
#     print("No location data available.")

# # --- Deduplication based on same-user cosine similarity ---
# print("Similar tweets from the same users will be removed based on cosine similarity of the embeddings.")
# duplicate_threshold = 0.9
# embeddings = embedding_model.encode(processed_tweets, show_progress_bar=True, device="cuda")
# remove_indices = []

# for user, group in df.groupby("user"):
#     indices = group.index.tolist()
#     if len(indices) < 2:
#         continue 
#     rep_indices = [] 
#     for idx in indices:
#         if not rep_indices:
#             rep_indices.append(idx)
#         else:
#             current_embedding = embeddings[idx].reshape(1, -1)
#             rep_embeddings = [embeddings[r] for r in rep_indices]
#             rep_embeddings = np.vstack(rep_embeddings)
#             sims = cosine_similarity(current_embedding, rep_embeddings)[0]
#             if sims.max() > duplicate_threshold:
#                 remove_indices.append(idx)
#             else:
#                 rep_indices.append(idx)

# print(f"Found {len(remove_indices)} duplicate tweets to remove based on cosine similarity.")

# removed_df = df.loc[remove_indices]
# removed_csv_path = "removed_duplicates.csv"
# removed_df.to_csv(removed_csv_path, index=False)
# print(f"Saved removed tweets to {removed_csv_path}")

# df = df.drop(index=remove_indices).reset_index(drop=True)
# processed_tweets = df["processed_text"].tolist()
# print(f"After deduplication, {df.shape[0]} tweets remain for topic modeling.")