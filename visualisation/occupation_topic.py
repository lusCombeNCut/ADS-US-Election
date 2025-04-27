import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from scipy.stats import chi2_contingency
import itertools
import statsmodels.api as sm
import patsy

# Set plot aesthetics
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'font.family': 'Times New Roman'
})

# Load data
topics = pd.read_csv(
    './results-data/new_merged_sentiment_topic.csv',
    usecols=['id','topic'],
    dtype={'id': str}
)
occupations = pd.read_csv(
    './results-data/tweet_occupations.csv',
    dtype={'tweet_id': str}
)

topic_labels = json.load(open(
    r'HPC-output-dir\version_5\bertopic_model\topics.json', 'r'
)).get('topic_labels', {})
topic_labels = {int(k): v for k, v in topic_labels.items()}

topics['topic_label'] = (
    topics['topic']
    .map(topic_labels)
    .fillna(topics['topic'])
    .astype(str)
    .str.replace('_', ' ')
    .str.replace(r'^(\d+)\s+', r'\1 - ', regex=True)
)

def normalize_id(col):
    return (
        col
        .astype(float)
        .round()
        .astype(int)
        .astype(str)
        .str.zfill(19)
    )

topics['id'] = normalize_id(topics['id'])
occupations['tweet_id'] = normalize_id(occupations['tweet_id'])

MIN_OCCUPATION = 100

df = pd.merge(
    occupations,
    topics.rename(columns={'id':'tweet_id'}),
    on='tweet_id',
    how='inner'
)
print(f"Total matched tweets: {len(df)}")

df_filtered = df.groupby('inferred_occupation').filter(lambda g: len(g) > MIN_OCCUPATION)

# Build count matrix
counts = (
    df_filtered
    .groupby(['inferred_occupation','topic_label'])
    .size()
    .unstack(fill_value=0)
)

totals = counts.sum(axis=1)
# Filter rare topics
global_props = counts.sum(axis=0) / counts.values.sum()
keep_topics = global_props[global_props > 0.018].index
filtered_counts = counts[keep_topics]

top10 = totals.nlargest(10).index
df_filtered = df_filtered[df_filtered['inferred_occupation'].isin(top10)]
filtered_counts = filtered_counts.loc[top10]

# Plot heatmap of proportions
proportions = filtered_counts.div(totals, axis=0).loc[top10]
proportions.columns = proportions.columns.str.split().str[:4].str.join(' ')

plt.figure(figsize=(8,8))
sns.heatmap(
    proportions,
    cmap="YlGnBu",
    linewidths=0.5,
    annot=True,
    fmt=".1%",
    cbar=False
)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Occupation', fontsize=16)
plt.title('Topic Proportion Heatmap', fontsize=18)
plt.xticks(rotation=60, ha='right', fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# # 1) Chi-Square Test of Independence
# chi2, p, dof, expected = chi2_contingency(filtered_counts)
# print(f"Chi2 statistic: {chi2:.2f}, dof: {dof}, p-value: {p:.4f}")
# if p < 0.05:
#     print("Significant difference: topic distribution depends on occupation.")
# else:
#     print("No significant difference in topic distributions across occupations.")

# # Residuals heatmap
# residuals = (filtered_counts - expected) / np.sqrt(expected)
# plt.figure(figsize=(8,6))
# sns.heatmap(
#     residuals,
#     cmap="RdBu",
#     center=0,
#     annot=True,
#     fmt=".2f"
# )
# plt.title('Chi-Square Residuals', fontsize=18)
# plt.xlabel('Topic', fontsize=16)
# plt.ylabel('Occupation', fontsize=16)
# plt.xticks(rotation=60, ha='right', fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.show()

# # 2) Pairwise Chi-Square Tests
# occupations_list = filtered_counts.index.tolist()
# pval_matrix = pd.DataFrame(
#     np.ones((len(occupations_list), len(occupations_list))),
#     index=occupations_list,
#     columns=occupations_list
# )
# for occ1, occ2 in itertools.combinations(occupations_list, 2):
#     table = filtered_counts.loc[[occ1, occ2]]
#     _, p_pc, _, _ = chi2_contingency(table)
#     pval_matrix.loc[occ1, occ2] = p_pc
#     pval_matrix.loc[occ2, occ1] = p_pc

# print("\nPairwise Chi-Square p-values:")
# print(pval_matrix)

# # 3) Multinomial Logistic Regression
# # Prepare tweet-level data for top10 occupations and topics
# df_reg = df_filtered.copy()
# df_reg['topic_short'] = df_reg['topic_label'].str.split().str[:3].str.join(' ')
# keep_short = proportions.columns.tolist()
# df_reg = df_reg[
#     df_reg['topic_short'].isin(keep_short)
# ]
# df_reg = df_reg[df_reg['inferred_occupation'].isin(top10)]
# df_reg['occupation_cat'] = df_reg['inferred_occupation'].astype('category')
# df_reg['topic_cat'] = df_reg['topic_short'].astype('category')
# # Design matrices for multinomial logit
# y, X = patsy.dmatrices('topic_cat ~ C(occupation_cat)', df_reg, return_type='dataframe')
# model = sm.MNLogit(y, X)
# res = model.fit(method='newton', maxiter=100, disp=False)
# print(res.summary())



# # 2) Bubble Plot
# fig, ax = plt.subplots(figsize=(12,6))
# x = np.arange(len(proportions.columns))
# for i, occ in enumerate(proportions.index):
#     y = [i]*len(proportions.columns)
#     sizes = proportions.loc[occ] * 2000  # scale factor
#     ax.scatter(x, y, s=sizes, alpha=0.6, edgecolors='w')

# ax.set_xticks(x)
# ax.set_xticklabels(proportions.columns, rotation=45, ha='right')
# ax.set_yticks(range(len(proportions.index)))
# ax.set_yticklabels(proportions.index)
# ax.set_title('Topic Proportion Bubble Plot (Top 10 Occupations)')
# plt.tight_layout()
# plt.show()

# # 3) Horizontal 100% Stacked Bars
# order = proportions.max(axis=1).sort_values(ascending=False).index
# prop_sorted = proportions.loc[order]

# ax = prop_sorted.plot(kind='barh', stacked=True, figsize=(10,8))
# ax.set_xlabel('Proportion of Tweets')
# ax.set_ylabel('Occupation')
# ax.set_title('Topic Distribution by Occupation (Horizontal) — Top 10')
# ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
# plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
