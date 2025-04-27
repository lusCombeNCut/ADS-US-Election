import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# set font size for all plots
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.titlesize'] = 16
# set fonts to times new roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = 'Times New Roman'

# === CONFIG ===
NEWS_INPUT_FILE   = "results-data/news-headline-topic-similarity.csv"
TWEETS_INPUT_FILE = "results-data/new_merged_sentiment_topic.csv"
OUTPUT_FILE       = "results-data/weekly_topic_comparison.csv"
# ==============

# --- load & parse news ---
news_df = pd.read_csv(NEWS_INPUT_FILE)

def clean_numpy_format(s):
    s2 = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    try:
        return ast.literal_eval(s2)
    except:
        return []

news_df['top_topics'] = news_df['top_topics'].apply(clean_numpy_format)
news_df = (
    news_df.assign(
        top_topic=lambda df: df['top_topics'].apply(lambda t: t[0][0] if t else None),
        top_similarity=lambda df: df['top_topics'].apply(lambda t: t[0][1] if t else None),
    )
    .groupby('week_start')
    .apply(lambda g: g.loc[g['top_similarity'].idxmax()])
    .reset_index(drop=True)[['week_start','top_topics','top_topic']]
    .rename(columns={'top_topic':'news_top_topic'})
)

# --- load & filter tweets ---
tweets_df = pd.read_csv(TWEETS_INPUT_FILE, parse_dates=['date'])
tweets_df = tweets_df[(tweets_df.date >= '2024-06-01') & (tweets_df.date <= '2024-11-23')]

# align weeks to Saturday
tweets_df['week_start'] = (
    tweets_df['date']
      - pd.to_timedelta((tweets_df['date'].dt.weekday - 5) % 7, unit='d')
).dt.strftime('%Y-%m-%d')

# drop the “unknown” and topic 0 categories
tweets_df = tweets_df[(tweets_df.topic != -1) & (tweets_df.topic != 0)]

# compute counts & proportions
weekly_topic_counts = (
    tweets_df.groupby(['week_start','topic'])
             .size()
             .reset_index(name='count')
)
weekly_totals = tweets_df.groupby('week_start').size().reset_index(name='total')
weekly_props = (
    weekly_topic_counts
      .merge(weekly_totals, on='week_start')
      .assign(proportion=lambda df: df['count']/df['total'])
)

# build for each week the FULL ranked list of tweet topics
tweet_ranked = (
    weekly_props
      .sort_values(['week_start','proportion'], ascending=[True, False])
      .groupby('week_start')
      .agg(tweet_topics_sorted=('topic', list))
      .reset_index()
)

# merge with news
comp = news_df.merge(tweet_ranked, on='week_start', how='inner')

# extract the single top tweet topic
comp['tweets_top_topic'] = comp['tweet_topics_sorted'].apply(lambda lst: lst[0] if lst else None)

# print table
comp['top_10_tweet_topics'] = comp['tweet_topics_sorted'].apply(
    lambda lst: ', '.join(str(t) for t in lst[:10])
)
table_df = comp[['week_start', 'news_top_topic', 'top_10_tweet_topics']]
print("\nTop News Topic and Top 10 Tweet Topics by Week\n")
print(table_df.to_string(index=False))

# compute P(news_top_topic in top N tweet topics)
max_tweet_N = comp['tweet_topics_sorted'].apply(len).max()
Ns_tweet, probs_tweet = [], []
for N in range(1, max_tweet_N+1):
    hits = [
        news in lst[:N]
        for news, lst in zip(comp['news_top_topic'], comp['tweet_topics_sorted'])
    ]
    Ns_tweet.append(N)
    probs_tweet.append(sum(hits) / len(hits))
plot_df_tweet = pd.DataFrame({'N': Ns_tweet, 'probability': probs_tweet})

# compute P(top_tweet_topic in top N news topics)
comp['news_topics_sorted'] = comp['top_topics'].apply(lambda lst: [t[0] for t in lst])
max_news_N = comp['news_topics_sorted'].apply(len).max()
Ns_news, probs_news = [], []
for N in range(1, max_news_N+1):
    hits = [
        tweet in news_list[:N]
        for tweet, news_list in zip(comp['tweets_top_topic'], comp['news_topics_sorted'])
    ]
    Ns_news.append(N)
    probs_news.append(sum(hits) / len(hits))
plot_df_news = pd.DataFrame({'N': Ns_news, 'probability': probs_news})

plt.figure(figsize=(6,4))
plt.plot(plot_df_tweet['N'], plot_df_tweet['probability'], marker='o')
plt.xlabel('N (Top N Tweet Topics)')
plt.ylabel('P (Top News Topic ' + r'$\epsilon$' + ' Top N Tweet Topics)')
plt.title('News vs. Tweet Topics')
plt.xticks([n for n in Ns_tweet if n % 5 == 0])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
slope = 0.5 / 43.0
plt.gca().axline((0, 0), (43, 0.5), linestyle=':', color='gray', linewidth=1, label='Expected for independence')
plt.xlim(0, max_tweet_N+1)
plt.ylim(0, 1.05)
plt.legend()
plt.show()

# Compute ranks

def get_rank(tweet_list, news_topic):
    try:
        return tweet_list.index(news_topic) + 1
    except ValueError:
        return None

comp['news_in_tweet_rank'] = comp.apply(
    lambda row: get_rank(row['tweet_topics_sorted'], row['news_top_topic']),
    axis=1
)

ranks = comp['news_in_tweet_rank'].dropna().astype(int)
plt.figure(figsize=(6,4))
bins = list(range(1, ranks.max()+2))
plt.hist(ranks, bins=bins, align='left', edgecolor='black')
plt.xlabel('Rank of News Top Topic among Tweet Topics')
plt.ylabel('Number of Weeks')
plt.title('Histogram of News Topic Rank in Tweets')
plt.xticks(bins[:-1])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# # Regular Chi-squared Test
# contingency_table = pd.crosstab(comp['tweets_top_topic'], comp['news_top_topic'])
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# print("\n=== Chi-Squared Test ===")
# print(f"Chi-squared statistic: {chi2:.2f}")
# print(f"Degrees of freedom: {dof}")
# print(f"P-value: {p:.4f}")
# if p < 0.05:
#     print("=> Significant association between tweets_top_topic and news_top_topic (reject H0)")
# else:
#     print("=> No significant association detected (fail to reject H0)")

# plt.figure(figsize=(12,8))
# sns.heatmap(contingency_table, annot=True, fmt='d', cmap='viridis', cbar_kws={'label': 'Count'})
# plt.title('Contingency Table: Tweets Top Topic vs News Top Topic')
# plt.xlabel('News Top Topic')
# plt.ylabel('Tweets Top Topic')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()

# --- Simplified Chi-squared Test ---
# Focus on top 10 tweet and news topics

# Get top 10 topics
top_tweet_topics = comp['tweets_top_topic'].value_counts().nlargest(10).index
top_news_topics = comp['news_top_topic'].value_counts().nlargest(10).index

# # Simplify
# comp['tweets_top_topic_simplified'] = comp['tweets_top_topic'].apply(
#     lambda x: x if x in top_tweet_topics else -999
# )
# comp['news_top_topic_simplified'] = comp['news_top_topic'].apply(
#     lambda x: x if x in top_news_topics else -999
# )

# # New contingency table
# contingency_table_simplified = pd.crosstab(
#     comp['tweets_top_topic_simplified'],
#     comp['news_top_topic_simplified']
# )

# # Chi-squared test
# chi2_s, p_s, dof_s, expected_s = chi2_contingency(contingency_table_simplified)
# print("\n=== Simplified Chi-Squared Test ===")
# print(f"Chi-squared statistic: {chi2_s:.2f}")
# print(f"Degrees of freedom: {dof_s}")
# print(f"P-value: {p_s:.4f}")
# if p_s < 0.05:
#     print("=> Significant association (reject H0)")
# else:
#     print("=> No significant association (fail to reject H0)")

# # Plot simplified heatmap
# plt.figure(figsize=(7,5))
# sns.heatmap(contingency_table_simplified, annot=True, fmt='d', cmap='coolwarm', cbar_kws={'label': 'Count'})
# plt.title('Simplified Contingency Table')
# plt.xlabel('News Top Topic (Simplified)')
# plt.ylabel('Tweets Top Topic (Simplified)')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()

# --- Save final output ---
comp.to_csv(OUTPUT_FILE, index=False)
print(f"\nWrote {len(comp)} rows to {OUTPUT_FILE}\n")