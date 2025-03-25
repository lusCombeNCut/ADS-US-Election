import os
import pandas as pd
from bertopic import BERTopic

# ----- Settings: Update these paths as needed -----
model_path = "/path/to/your/saved/bertopic_model"      # e.g., "/user/home/sv22482/work/ADS-US-Election/orlando-bert/ADS/version_1/bertopic_model"
topics_path = "/path/to/your/topics.csv.gz"              # e.g., "/user/home/sv22482/work/ADS-US-Election/orlando-bert/ADS/version_1/topics.csv.gz"

# ----- Load the model and topics data -----
print("Loading BERTopic model...")
topic_model = BERTopic.load(model_path)

print("Loading topics data...")
df_topics = pd.read_csv(topics_path, compression="gzip")
df_topics["date_parsed"] = pd.to_datetime(df_topics["date_parsed"], errors="coerce")

# ----- Filter out data points before 2024 -----
cutoff_date = pd.Timestamp("2024-01-01")
filtered_df = df_topics[df_topics["date_parsed"] >= cutoff_date]

# Because topics.csv.gz does not include original texts, create a dummy docs list using the "id" column.
# (If you need the actual text, consider saving it as part of your data.)
docs_filtered = filtered_df["id"].astype(str).tolist()
timestamps_filtered = filtered_df["date_parsed"].tolist()

# ----- Generate plots -----
# Bar chart of topics (uses the model's internal topic info)
fig_bar = topic_model.visualize_barchart(top_n_topics=topic_model.get_topic_info().shape[0])
fig_bar.show()

# 2D topics visualization
fig_topics = topic_model.visualize_topics()
fig_topics.show()

# Heatmap of topic similarities
fig_heatmap = topic_model.visualize_heatmap()
fig_heatmap.show()

# Topics over time visualization using the filtered dummy docs and timestamps
topics_over_time = topic_model.topics_over_time(docs_filtered, timestamps_filtered, nr_bins=20)
fig_topics_over_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=8)
fig_topics_over_time.show()
