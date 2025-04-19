import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# =============================================================================
# 1. Load Topic Representations from JSON
# =============================================================================
topics_path = 'orlando-bert/version_3/bertopic_model/topics.json'
results_path = 'BERTTopic_sentiment_irony.csv'
polling_path = 'presidential_primary_averages_2024.csv'

with open(topics_path, 'r') as f:
    topics_data = json.load(f)

# Build a dictionary mapping each topic code to a descriptive label.
# We use the top term (first string) for each topic.
topics_dict = {}
for topic_code, reps in topics_data.get("topic_representations", {}).items():
    if reps:
        topics_dict[topic_code] = reps[0][0]  # top word as label
    else:
        topics_dict[topic_code] = 'Unknown'

# =============================================================================
# 2. Load and Explore the Tweet Data
# =============================================================================
tweets_df = pd.read_csv(results_path)
tweets_df['date'] = pd.to_datetime(tweets_df['date'])

print("=== Tweet Data Summary ===")
print("Total number of tweet entries:", len(tweets_df))
print("Tweet Data Columns:", tweets_df.columns.tolist())
print("\nSample of Tweet Data:")
print(tweets_df.head())

# Map tweet topic codes to descriptive labels.
tweets_df['topic_label'] = tweets_df['topic'].astype(str).map(topics_dict)

print("\n=== Sentiment Distribution ===")
print(tweets_df['sentiment'].value_counts())
print("\n=== Irony Distribution ===")
print(tweets_df['irony'].value_counts())
print("\n=== Topic Distribution (by code) ===")
print(tweets_df['topic'].value_counts())

# =============================================================================
# 3. Visualize the Tweet Data
# =============================================================================
# # -- Daily Tweet Volume --
# daily_tweet_volume = tweets_df.groupby(tweets_df['date'].dt.date).size()
# plt.figure(figsize=(10, 5))
# plt.plot(daily_tweet_volume.index, daily_tweet_volume.values, marker='o')
# plt.title("Daily Tweet Volume")
# plt.xlabel("Date")
# plt.ylabel("Number of Tweets")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# -- EMA Smoothed Sentiment Proportions --
daily_sentiments = tweets_df.groupby(tweets_df['date'].dt.date)['sentiment'].value_counts(normalize=True).unstack().fillna(0)
ema_span = 7  # 7-day exponential moving average
daily_sentiments_ema = daily_sentiments.ewm(span=ema_span).mean()

# plt.figure(figsize=(10, 5))
# for sentiment in daily_sentiments_ema.columns:
#     plt.plot(daily_sentiments_ema.index, daily_sentiments_ema[sentiment], marker='o', label=sentiment)
# plt.title("EMA Smoothed Sentiment Proportions")
# plt.xlabel("Date")
# plt.ylabel("Proportion of Tweets")
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # -- Topic Distribution Bar Plot (Top N Topics) --
# topic_counts = tweets_df['topic'].value_counts()
# top_n = 10
# top_topics = topic_counts.sort_values(ascending=False).head(top_n)

# plt.figure(figsize=(10, 5))
# plt.bar(top_topics.index.astype(str), top_topics.values)
# plt.title(f"Top {top_n} Topics by Tweet Count")
# plt.xlabel("Topic Code")
# plt.ylabel("Number of Tweets")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# =============================================================================
# 4. Additional Analyses Ideas (Printed)
# =============================================================================
print("\n=== Additional Analyses Ideas ===")
print("""
1. Time Series by Topic:
   - Group tweets by 'topic' or 'topic_label' and plot sentiment trends over time.
2. Event-Centric Analysis:
   - Overlay key political events (e.g., debates, election day) on time series plots.
3. Irony vs. Sentiment:
   - Explore the frequency and impact of ironic tweets across sentiment classes.
4. Correlation with External Data:
   - Overlay traditional polling data or election odds with social media sentiment.
5. Advanced Visualizations:
   - Create interactive dashboards (using Plotly or Dash) for multifaceted analysis.
6. Demographic Inference:
   - Integrate demographic data (if available) to study group-specific political engagement.
""")

# =============================================================================
# 5. Load and Summarize the Polling Data (2024 Only & Selected Candidates)
# =============================================================================
poll_df = pd.read_csv(polling_path)
poll_df['date'] = pd.to_datetime(poll_df['date'])

# Filter to 2024 only (if 'cycle' exists)
if 'cycle' in poll_df.columns:
    poll_df = poll_df[poll_df['cycle'] == 2024]

# IMPORTANT: Filter only for 'national' in the state column.
if 'state' in poll_df.columns:
    poll_df = poll_df[poll_df['state'].str.lower() == 'national']
    
print("\n=== Polling Data After Filtering (2024, state='national') ===")
print("Total polling rows after filtering:", poll_df.shape[0])
print("Polling Data Columns:", poll_df.columns.tolist())
print("\nSample of Polling Data:")
print(poll_df.head())
print("\nPolling Data Summary:")
print(poll_df.describe(include='all'))

# Limit polling data to the selected candidates.
selected_candidates = ["Trump", "Biden"]
poll_df = poll_df[poll_df['candidate'].isin(selected_candidates)]

print("\n=== Polling Data Summary (Selected Candidates) ===")
for candidate in poll_df['candidate'].unique():
    candidate_poll_df = poll_df[poll_df['candidate'] == candidate]
    print(f"\nCandidate: {candidate}")
    print(candidate_poll_df.describe(include='all'))
    print(candidate_poll_df.head())

# =============================================================================
# 6. Dual Y-Axis Plot: Social Media Positive Sentiment vs. Polling Trends
# =============================================================================
# For demonstration, we plot data for one selected candidate; adjust as needed.
selected_candidate_for_plot = "Biden"
poll_candidate = poll_df[poll_df['candidate'] == selected_candidate_for_plot]
daily_poll = poll_candidate.groupby(poll_candidate['date'].dt.date)['pct_estimate'].mean()

fig, ax1 = plt.subplots(figsize=(12, 6))
if 'POS' in daily_sentiments_ema.columns:
    color1 = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Positive Sentiment (EMA)', color=color1)
    ax1.plot(daily_sentiments_ema.index, daily_sentiments_ema['POS'], marker='o', color=color1,
             label='Positive Sentiment (EMA)')
    ax1.tick_params(axis='y', labelcolor=color1)
else:
    print("No 'POS' sentiment column found in the tweet data.")

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Polling Estimate (%)', color=color2)
ax2.plot(daily_poll.index, daily_poll.values, marker='s', linestyle='--', color=color2,
         label=f'Polling Estimate: {selected_candidate_for_plot}')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title("Social Media Positive Sentiment vs. Polling Trends\nCandidate: " + selected_candidate_for_plot)
fig.tight_layout()
plt.xticks(rotation=45)
plt.show()

# =============================================================================
# 7. Build Correlation Matrix: Topic & Candidate Changes (Selected Candidates Only)
# =============================================================================
# For each topic (from the tweet data) and for each candidate (from the polling data),
# compute the Pearson correlation between the day-to-day change in the proportion of tweets
# with 'POS' sentiment for that topic and the day-to-day change in the candidate's polling estimates.
unique_topics = tweets_df['topic'].astype(str).unique()
unique_candidates = poll_df['candidate'].unique()

# Create an empty dictionary to store correlation coefficients.
corr_data = {cand: {} for cand in unique_candidates}

for topic in unique_topics:
    topic_df = tweets_df[tweets_df['topic'].astype(str) == topic]
    # Daily proportion of 'POS' tweets for the topic.
    daily_topic_sent = topic_df.groupby(topic_df['date'].dt.date)['sentiment'].apply(lambda x: np.mean(x == 'POS'))
    daily_topic_sent_change = daily_topic_sent.diff().dropna()
    
    for cand in unique_candidates:
        cand_poll_df = poll_df[poll_df['candidate'] == cand]
        daily_cand_poll = cand_poll_df.groupby(cand_poll_df['date'].dt.date)['pct_estimate'].mean()
        daily_cand_poll_change = daily_cand_poll.diff().dropna()
        
        common_dates = daily_topic_sent_change.index.intersection(daily_cand_poll_change.index)
        print(f"Candidate {cand}, Topic {topic}: {len(common_dates)} overlapping dates")
        if len(common_dates) >= 2:
            topic_changes = daily_topic_sent_change.loc[common_dates]
            poll_changes = daily_cand_poll_change.loc[common_dates]
            # Check if either series is constant.
            if np.allclose(topic_changes, topic_changes.iloc[0]) or np.allclose(poll_changes, poll_changes.iloc[0]):
                corr_data[cand][topic] = np.nan
            else:
                corr_coef, _ = pearsonr(topic_changes, poll_changes)
                corr_data[cand][topic] = corr_coef
        else:
            corr_data[cand][topic] = np.nan

corr_matrix = pd.DataFrame(corr_data).T  # Rows: candidates, Columns: topics

print("\n=== Correlation Matrix Preview ===")
print(corr_matrix.head())

# =============================================================================
# 8. Visualize the Correlation Matrix as a Single Heatmap
# =============================================================================
def plot_heatmap(matrix, title):
    # Convert matrix to masked array to handle NaNs.
    data = np.ma.masked_invalid(matrix.values)
    
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color='gray')
    
    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    
    xtick_labels = [topics_dict.get(str(topic), str(topic)) for topic in matrix.columns]
    ytick_labels = matrix.index.tolist()
    
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation Coefficient')
    
    plt.title(title)
    plt.xlabel("Topic (Descriptive Label)")
    plt.ylabel("Candidate")
    plt.tight_layout()
    plt.show()

plot_heatmap(corr_matrix, "Correlation between Daily Changes in Topic Positive Sentiment and Polling Changes\n(Selected Candidates)")
