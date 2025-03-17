import os
import glob
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired

MIN_LIKES = 20
MIN_RETWEETS = 3

def load_data(root_folder):
    all_files = []
    
    for part_folder in glob.glob(os.path.join(root_folder, "part_*")):
        file_pattern = os.path.join(part_folder, "*.csv.gz")
        all_files.extend(glob.glob(file_pattern))
    
    print("number of files", len(all_files))

    dfs = [pd.read_csv(f, compression="gzip") for f in all_files]

    # # ~~~ For local testing load the first file only ~~~
    # f = all_files[0]
    # dfs = [pd.read_csv(f, compression="gzip")]

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = filter_data(combined_df)
    docs = combined_df["processed_text"].tolist()

    return combined_df, docs

def filter_data(df):
    print(f"Total tweets prefiltering: {df.shape[0]}")
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[(df['likeCount'] >= MIN_LIKES) & (df['retweetCount'] >= MIN_RETWEETS)]
    print(f"Tweets after thresholding: {df.shape[0]}")

    df = df.sort_values('epoch').drop_duplicates(subset=['conversationId'], keep='first')
    print(f"Tweets after selecting first per conversation: {df.shape[0]}")

    df = df[(df['lang'] == 'en') & (df["text"].notna())]
    df["processed_text"] = df["text"]
    df = df.dropna(subset=["processed_text"])
    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)
    print(f"Tweets after processing: {df.shape[0]}")

    return df

def save_model(save_dir, model: BERTopic, df):
    columns_to_save = ["id", "topic"] + [f"topic_prob_{i}" for i in range(probs.shape[1])]
    model.save(f"{save_dir}/bertopic_model", serialization="pytorch")
    df[columns_to_save].to_csv(f'{save_dir}/topics.csv.gz', index=False, compression="gzip")
    
def run_topic_model_fitting(docs, embedding_model):
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    representation_model = KeyBERTInspired()

    custom_umap = umap.UMAP(n_neighbors=10, n_components=5, metric="cosine", init="random", random_state=42)
    custom_hdbscan = HDBSCAN(min_samples=20, gen_min_span_tree=True, prediction_data=True, core_dist_n_jobs=5)

    topic_model = BERTopic(
        representation_model=representation_model,
        embedding_model=embedding_model,
        umap_model=custom_umap,
        hdbscan_model=custom_hdbscan,
        language="english",
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )
    
    topics, probs = topic_model.fit_transform(documents=docs, embeddings=embeddings)
    
    return topic_model, embeddings, topics, probs

if __name__ == "__main__":
    
    # # ~~~ Local paths ~~~
    # main_dir = r'C:\Users\Orlan\Documents\usc-x-24-us-election'
    # save_dir = r"C:\Users\Orlan\Documents\Applied-Data-Science\orlando-bert"

    # ~~~ Bluecrystal paths ~~~
    save_dir = "/user/work/sv22482/ADS"
    main_dir = '/user/work/sv22482/usc-x-24-us-election'

    new_version = "version_" + \
        str(max([int(folder[8:])
                 for folder in os.listdir(save_dir)], default=0) + 1)

    save_dir += f'/{new_version}'

    os.makedirs(save_dir, exist_ok=True)
    df, docs = load_data(main_dir)
    
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    topic_model, embeddings, topics, probs = run_topic_model_fitting(docs, embedding_model)


    df['topic'] = topics
    for i in range(probs.shape[1]):
        df[f'topic_prob_{i}'] = probs[:, i]

    save_model(save_dir, topic_model, df)

    timestamps = df['date_parsed'].tolist()

    info = topic_model.get_topic_info()

    fig = topic_model.visualize_barchart(top_n_topics=info.shape[0])
    fig.write_html("barchart.html")

    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    fig.write_html("topics_over_time.html")

    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("topics_over_time.html")

    fig = topic_model.visualize_topics()
    fig.write_html("topics.html")

