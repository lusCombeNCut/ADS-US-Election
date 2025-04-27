import os
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic

# ----- Settings: Update these paths as needed -----
model_path = "./orlando-bert/version_3/bertopic_model"
topics_path = "./orlando-bert/version_3/topics.csv.gz"

# ----- Load the model and topics data -----
print("Loading BERTopic model...")
topic_model = BERTopic.load(model_path)

print("Loading topics data...")
df_topics = pd.read_csv(topics_path, compression="gzip")
df_topics["date_parsed"] = pd.to_datetime(df_topics["date_parsed"], errors="coerce")

# ----- Filter out data points before 2024 -----
cutoff_date = pd.Timestamp("2024-01-01")
filtered_df = df_topics[df_topics["date_parsed"] >= cutoff_date]

# # ----- Plot other visualizations (if desired) -----
# fig_bar = topic_model.visualize_barchart(top_n_topics=topic_model.get_topic_info().shape[0], autoscale=True, width=400, height=400)
# fig_bar.show()

# fig_topics = topic_model.visualize_topics()
# fig_topics.show()

# fig_heatmap = topic_model.visualize_heatmap()
# fig_heatmap.show()

# ----- Custom Plot: Top 5 Topics Over Time (Scatter Plot with Logarithmic Y-Axis) -----

# Determine the top 5 topics by frequency in the filtered data
top_topics = filtered_df['topic'].value_counts().nlargest(5).index.tolist()
top_topics = [1]  # Overriding for demonstration purposes; remove if not needed

# Retrieve topic info from the model (includes topic labels)
topic_info = topic_model.get_topic_info()

# Create a mapping from topic id to its label (or "Name" from the topic info)
topic_label_mapping = {}
for topic in top_topics:
    # Find the label for the topic; if not available, default to the topic id
    label_row = topic_info[topic_info.Topic == topic]
    if not label_row.empty:
        topic_label = label_row["Name"].values[0]
    else:
        topic_label = str(topic)
    topic_label_mapping[topic] = topic_label

# Filter the DataFrame to include only the top topics
filtered_top = filtered_df[filtered_df['topic'].isin(top_topics)].copy()

# Set the 'date_parsed' column as the index for resampling
filtered_top.set_index('date_parsed', inplace=True)

# Resample the data by month (change 'M' to another frequency like 'W' for weekly if needed)
topic_over_time = filtered_top.groupby('topic').resample('M').size().reset_index(name='counts')

# Create the scatter plot
fig, ax = plt.subplots(figsize=(12, 6))
for topic in top_topics:
    topic_data = topic_over_time[topic_over_time['topic'] == topic]
    ax.scatter(topic_data['date_parsed'], topic_data['counts'],
               label=f"Topic {topic}: {topic_label_mapping[topic]}")

ax.set_title("Number of Tweets With the Assassination Topic")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Tweets")
ax.set_yscale('log')  # Set logarithmic y-axis
ax.legend(title="Topic")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

