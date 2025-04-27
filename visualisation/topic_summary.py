import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

# set font size and style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'font.family': 'Times New Roman'
})


topics = pd.read_csv('./results-data/new_merged_sentiment_topic.csv', 
                     usecols=['id','topic'], 
                     dtype={'id': str})

topic_counts = topics['topic'].value_counts().sort_index()


plt.figure(figsize=(5, 3))

bar_colors = ['#2164dc' if ((i-1) % 5 != 0) else '#00d0ee' for i in range(len(topic_counts))]

sns.barplot(x=topic_counts.index, y=topic_counts.values, palette=bar_colors)
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('Topic ID')
plt.ylabel('Number of Tweets (log scale)')
plt.title('Number of Tweets per Topic')

xticks = topic_counts.index
xtick_labels = [label if (i-1) % 5 == 0 else '' for i, label in enumerate(xticks)]
plt.xticks(ticks=range(len(xticks)), labels=xtick_labels)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.5)  # Add minor y-axis grid lines
plt.tight_layout()
plt.show()

# # Calculate percentage of tweets per topic
# topic_percentages = (topic_counts / topic_counts.sum()) * 100

# plt.figure(figsize=(5, 3))

# sns.barplot(x=topic_percentages.index, y=topic_percentages.values, palette=bar_colors)
# plt.xlabel('Topic ID')
# plt.ylabel('Percentage of Tweets')
# plt.title('Percentage of Tweets per Topic')

# xtick_labels = [label if (i-1) % 5 == 0 else '' for i, label in enumerate(topic_percentages.index)]
# plt.xticks(ticks=range(len(topic_percentages.index)), labels=xtick_labels)

# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()