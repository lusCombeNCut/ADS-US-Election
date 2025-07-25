import pandas as pd

import matplotlib.pyplot as plt

# Load the data
file_path = 'results-data/tweet_counts.csv'
data = pd.read_csv(file_path)

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Plot configuration
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'font.family': 'Times New Roman'
})

# Create the plot
plt.figure(figsize=(7, 4))
plt.plot(data['date'], data['count'], marker='None', linestyle='-', color='b', label='Tweet Count')
plt.title('Number of Tweets Per Day')
plt.xlabel('Date')

# add minor grid lines
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.5)

plt.ylabel('Tweet Count')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.ylim(0, data['count'].max() * 1.1)  # Add some space above the max count
plt.xlim(data['date'].min(), data['date'].max())  # Set x-axis limits to the date range
# Show the plot
plt.show()