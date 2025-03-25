import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert the date_parsed column to datetime format
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
    
    # Group by date and topic, counting the number of tweets per group
    df_grouped = df.groupby(['date_parsed', 'topic']).size().reset_index(name='tweet_count')
    
    # Pivot the DataFrame so that each topic becomes a column with tweet counts
    pivot_df = df_grouped.pivot(index='date_parsed', columns='topic', values='tweet_count').fillna(0)
    
    # Plotting the time series graph
    plt.figure(figsize=(10, 6))
    
    # Plot a line for each topic
    for topic in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[topic], marker='o', label=f"Topic {topic}")
    
    plt.xlabel("Time")
    plt.ylabel("Number of Tweets")
    plt.title("Number of Tweets per Topic Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

path = ""
main(path)
