import json
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from glob import glob

# =============================================================================
# 1. Load Topic Representations from JSON
# =============================================================================
topics_path = r'HPC-output-dir\version_5\bertopic_model\topics.json'
infer_low_path = r'orlando-bert\inference-output-low-filtering\topic-inference-results.csv'
infer_high_path = r'orlando-bert\inference-output-high-filtering\topic-inference-results.csv'
merged_path = 'merged_low_filter.csv'
polling_path = 'presidential_primary_averages_2024.csv'
sentiment_path = 'sentiment_results'

with open(topics_path, 'r') as f:
    topics_data = json.load(f)

# Build a dictionary mapping each topic code to a descriptive label.
topics_dict = {}
for topic_code, reps in topics_data.get("topic_representations", {}).items():
    if reps:
        topics_dict[topic_code] = reps[0][0]  # top word as label
    else:
        topics_dict[topic_code] = 'Unknown'

# =============================================================================
# 2. Load and Explore the Tweet Data
# =============================================================================
merged_df = pd.read_csv(merged_path)
merged_df['date_parsed'] = pd.to_datetime(merged_df['date_parsed'])
merged_df['topic_label'] = merged_df['topic'].astype(str).map(topics_dict)
merged_df['week'] = merged_df['date_parsed'].dt.to_period('W').astype(str)  # Convert to string
weekly_tweet_volume = merged_df.groupby(merged_df['week']).size().reset_index(name='tweet_count')
weekly_tweet_volume['source'] = 'Merged With sentiment and irony'

infer_low_df = pd.read_csv(infer_low_path)
infer_low_df['date_parsed'] = pd.to_datetime(infer_low_df['date_parsed'])
infer_low_df['week'] = infer_low_df['date_parsed'].dt.to_period('W').astype(str)  # Convert to string
infer_low_weekly_tweet_volume = infer_low_df.groupby(infer_low_df['week']).size().reset_index(name='tweet_count')
infer_low_weekly_tweet_volume['source'] = 'Isolated Topic Inference with low filtering'

# Combine the two dataframes
combined_df = pd.concat([weekly_tweet_volume, infer_low_weekly_tweet_volume], ignore_index=True)

# remove all dates before 2024-01-01
combined_df = combined_df[combined_df['week'] >= '2022-01-01']
combined_df['week'] = combined_df['week'].astype(str)  # Convert to string for plotting

# Plot the histogram with different colors for each source
fig7 = px.bar(combined_df, x='week', y='tweet_count', color='source', barmode='group', title="Number of Tweets per Week")
fig7.update_xaxes(tickangle=45)
fig7.update_layout(title_text="Number of Tweets per Week (Combined)")
fig7.show()


# =============================================================================
# 4. Visualizations
# =============================================================================
# -- 1. Most Common Topic Each Week --
weekly_topic_counts = merged_df.groupby(['week', 'topic_label']).size().reset_index(name='tweet_count')

fig1 = px.line(weekly_topic_counts, x='week', y='tweet_count', color='topic_label', title='Most Common Topic Each Week')
fig1.update_xaxes(tickangle=45)
fig1.show()

# -- 2. Sentiment for Each Topic Over Time (Top 5 Topics) --
top_topics = merged_df['topic_label'].value_counts().head(5).index
filtered_df = merged_df[merged_df['topic_label'].isin(top_topics)]
weekly_sentiment = filtered_df.groupby([filtered_df['date_parsed'].dt.to_period('W').astype(str), 'topic_label', 'sentiment']).size().reset_index(name='tweet_count')

fig2 = px.line(weekly_sentiment, x='date_parsed', y='tweet_count', color='sentiment', facet_col='topic_label', title="Sentiment Over Time for Top 5 Topics")
fig2.update_xaxes(tickangle=45)
fig2.show()

# -- 3. Correlation Between Polling Changes and Topics --
poll_df = pd.read_csv(polling_path)
poll_df['date'] = pd.to_datetime(poll_df['date'])

# Filter for 2024 and selected candidates (e.g., Biden and Trump)
poll_df = poll_df[(poll_df['cycle'] == 2024) & (poll_df['candidate'].isin(['Trump', 'Biden']))]

# Calculate weekly delta in polling for Biden
biden_poll = poll_df[poll_df['candidate'] == "Biden"]
biden_poll['week'] = biden_poll['date'].dt.to_period('W').astype(str)  # Convert to string
biden_poll['poll_delta'] = biden_poll['pct_estimate'].diff()

# Calculate weekly topic proportions
topic_proportions = merged_df.groupby([merged_df['date_parsed'].dt.to_period('W').astype(str), 'topic_label']).size().unstack().fillna(0)
topic_proportions = topic_proportions.div(topic_proportions.sum(axis=1), axis=0)  # Normalize

# Merge polling data with topic proportions
merged_df = pd.merge(biden_poll, topic_proportions, left_on='week', right_index=True)

# Calculate correlation between polling change and each topic
for topic in topic_proportions.columns:
    # Align data for Pearson correlation
    aligned_data = merged_df[['poll_delta', topic]].dropna()
    
    # Calculate Pearson correlation
    if aligned_data.shape[0] > 1:  # Ensure there are at least two data points
        correlation = pearsonr(aligned_data['poll_delta'], aligned_data[topic])
        print(f"Correlation between polling change and {topic}: {correlation[0]}")
    else:
        print(f"Not enough data to calculate correlation for {topic}.")

# -- 4. Correlation Between Polling and Sentiment --
weekly_sentiment_change = merged_df.groupby([merged_df['date_parsed'].dt.to_period('W').astype(str), 'sentiment']).size().unstack().fillna(0)
weekly_sentiment_change = weekly_sentiment_change.div(weekly_sentiment_change.sum(axis=1), axis=0)

# Merge with polling data
merged_sentiment_poll = pd.merge(biden_poll, weekly_sentiment_change, left_on='week', right_index=True)

# Calculate correlation between polling change and sentiment
for sentiment in ['POS', 'NEU', 'NEG']:
    # Align data for Pearson correlation
    aligned_sentiment = merged_sentiment_poll[['poll_delta', sentiment]].dropna()
    
    if aligned_sentiment.shape[0] > 1:  # Ensure there are at least two data points
        correlation = pearsonr(aligned_sentiment['poll_delta'], aligned_sentiment[sentiment])
        print(f"Correlation between polling change and {sentiment} sentiment: {correlation[0]}")
    else:
        print(f"Not enough data to calculate correlation for {sentiment} sentiment.")

# -- 5. Correlation Between Topics and Irony --
weekly_irony = merged_df.groupby([merged_df['date_parsed'].dt.to_period('W').astype(str), 'irony']).size().unstack().fillna(0)
weekly_irony = weekly_irony.div(weekly_irony.sum(axis=1), axis=0)

# Merge irony distribution with topic data
merged_irony_topic = pd.merge(weekly_irony, topic_proportions, left_index=True, right_index=True)

# Calculate correlation between irony 'not ironic' and topic 0
correlation = pearsonr(merged_irony_topic['not ironic'].dropna(), merged_irony_topic[0].dropna())
print(f"Correlation between irony and topic 0: {correlation[0]}")

# =============================================================================
# 6. Dual Y-Axis Plot: Social Media Positive Sentiment vs. Polling Trends (For Trump)
# =============================================================================
selected_candidate_for_plot = "Trump"
poll_candidate = poll_df[poll_df['candidate'] == selected_candidate_for_plot]
daily_poll = poll_candidate.groupby(poll_candidate['date'].dt.date)['pct_estimate'].mean()

fig3, ax1 = plt.subplots(figsize=(12, 6))
if 'POS' in weekly_sentiment_change.columns:
    color1 = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Positive Sentiment (EMA)', color=color1)
    ax1.plot(weekly_sentiment_change.index, weekly_sentiment_change['POS'], marker='o', color=color1, label='Positive Sentiment (EMA)')
    ax1.tick_params(axis='y', labelcolor=color1)
else:
    print("No 'POS' sentiment column found in the tweet data.")

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Polling Estimate (%)', color=color2)
ax2.plot(daily_poll.index, daily_poll.values, marker='s', linestyle='--', color=color2, label=f'Polling Estimate: {selected_candidate_for_plot}')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title("Social Media Positive Sentiment vs. Polling Trends\nCandidate: " + selected_candidate_for_plot)
fig3.tight_layout()
plt.xticks(rotation=45)
plt.show()

# =============================================================================
# 7. Interactive Plot for Political Events
# =============================================================================
# Add a political event for illustration (e.g., assassination attempt)
events = pd.DataFrame({
    'date': ['2024-05-15'],
    'event': ['Assassination Attempt'],
})

# Plot with political event annotations
fig4 = px.line(weekly_topic_counts, x='week', y='tweet_count_x', color='topic_label', title='Most Common Topic Each Week')

# Add event annotation
for _, event in events.iterrows():
    fig4.add_vline(x=event['date'], line=dict(color='red', dash='dash'))
    fig4.add_annotation(x=event['date'], y=10, text=event['event'], showarrow=True, arrowhead=2)

fig4.update_xaxes(tickangle=45)
fig4.show()
