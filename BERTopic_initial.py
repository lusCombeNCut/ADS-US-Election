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
from bertopic.vectorizers import ClassTfidfTransformer

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
CUSTOM_IGNORE_WORDS = {"rt", "amp", "ni", "el", "la", "nada", "de", "para", "al", "con", "le", "ver", "hay", "eu", "en", "se", "va"}
stop_words = set(stopwords.words("english")).union(CUSTOM_IGNORE_WORDS)
stop_words = list(stop_words)
lemmatizer = WordNetLemmatizer()

def is_english_word(word):
    """Check if a word exists in WordNet."""
    return bool(wordnet.synsets(word))

def preprocess_text(text):
    """Preprocess a tweet: remove URLs, mentions, hashtags, and apply text cleaning."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = text.lower()  # Lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [word for word in tokens if is_english_word(word)]
    processed_text = " ".join(tokens)
    return processed_text if processed_text.strip() else None

# Load tweets dataset
csv_path = "may_july_chunk_1.csv.gz"  # Change to your file name
df = pd.read_csv(csv_path, compression="gzip")

df["processed_text"] = df["text"].dropna().apply(preprocess_text)
df = df.dropna(subset=["processed_text"])  # Drop rows where processed_text is None
df = df[df["processed_text"].str.strip() != ""]  # Remove empty strings
df = df.head(1000)

processed_tweets = df["processed_text"].tolist()
print(f"Loaded and preprocessed {len(processed_tweets)} tweets.")  # Debugging output

# Check if we have enough unique tweets
if len(set(processed_tweets)) < 10:
    print("Not enough unique tweets for meaningful topic modeling.")
    exit(1)

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda" if gpu_available else "cpu")

if gpu_available:
    print("Using GPU-accelerated UMAP and HDBSCAN.")
    custom_umap = cumlUMAP(n_neighbors=15, n_components=5, metric="cosine", init="random", random_state=42)
    custom_hdbscan = cumlHDBSCAN(min_samples=5, gen_min_span_tree=True, prediction_data=True)
else:
    print("Using CPU-based UMAP and HDBSCAN.")
    custom_umap = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine", init="random", random_state=42)
    custom_hdbscan = HDBSCAN(min_samples=5, gen_min_span_tree=True, prediction_data=True)

vectorizer_model = CountVectorizer(stop_words=stop_words)

ctfidf_model = ClassTfidfTransformer(seed_words = ["biden", "joe", "democrat", "trump", "donald", "republican"], seed_multiplier = 2)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=custom_umap,
    hdbscan_model=custom_hdbscan,
    vectorizer_model=vectorizer_model,
    language="english",
    nr_topics=5,
    calculate_probabilities=True,
    verbose=True,
    ctfidf_model=ctfidf_model  # Add the seed topics here
)

topics, probabilities = topic_model.fit_transform(processed_tweets)

unique_topics = np.unique(topics)
if len(unique_topics) <= 1:
    print("Too few topics detected! Try adjusting parameters or using more diverse data.")
    exit(1)

# Visualize the topics
topics, probabilities = topic_model.fit_transform(processed_tweets)

unique_topics = np.unique(topics)
if len(unique_topics) <= 1:
    print("Too few topics detected! Try adjusting parameters or using more diverse data.")
    exit(1)

df["Topic"] = topics
df.to_csv("tweet_topics_500.csv", index=False)
print("Saved topic information to tweet_topics_500.csv.")
topic_info_df = topic_model.get_topic_info()
print(topic_info_df)


for topic_id in topic_info_df['Topic'].unique():
    if topic_id != -1:
        print(f"Topic {topic_id}:")
        print(topic_model.get_topic(topic_id))
        print("\n")

fig = topic_model.visualize_topics()
fig.show()