import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# Set visual style consistent with crypto_sentiment_vis.py
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = 'Times New Roman'

def visualize_topic_keywords(topic_id=3):
    """
    Visualize word importance weights for a specific topic.
    
    Parameters:
    -----------
    topic_id : int
        ID of the topic to visualize
    """
    # Load the topics JSON file
    topics_path = r'HPC-output-dir\version_5\bertopic_model\topics.json'
    with open(topics_path, 'r') as f:
        topics_data = json.load(f)
    
    # Print number of tweets for the selected topic
    topic_sizes = topics_data.get("topic_sizes", {})
    num_tweets = topic_sizes.get(str(topic_id), 0)
    print(f"Number of tweets for topic {topic_id}: {num_tweets}")

    # Extract keyword data for the selected topic
    topic_words = topics_data.get("topic_representations", {}).get(str(topic_id), [])

    topic_label = topics_data.get("topic_labels", {}).get(str(topic_id), f"Topic {topic_id}")
    
    if not topic_words:
        print(f"No keywords found for topic {topic_id}")
        return
    
    # Extract words and weights
    if isinstance(topic_words[0], list) or isinstance(topic_words[0], tuple):
        words = [item[0] for item in topic_words]
        weights = [item[1] for item in topic_words]
    else:
        words = topic_words
        weights = list(range(len(words), 0, -1))  # Assign descending weights if not provided
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'word': words,
        'weight': weights
    })
    
    # Take top 15 words for visualization
    df = df.head(15)
    
    # Sort in ascending order for better visualization
    df = df.sort_values('weight')
    
    # Create horizontal bar chart
    plt.figure(figsize=(7, 6))
    bars = plt.barh(df['word'], df['weight'], color='steelblue')
    
    # Add weight labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', 
                 va='center')
    
    plt.title(f'Keyword Weights for the Cryptocurrency Topic')
    plt.xlabel('Weight')
    plt.ylabel('Keywords')
    plt.tight_layout()
    plt.yticks(rotation=45)
    plt.savefig(f'topic_{topic_id}_keywords.png', dpi=300, bbox_inches='tight')
    # add more white space at the end of bars so the labels are not cut off
    plt.xlim(0, df['weight'].max() * 1.2)
    plt.show()

if __name__ == "__main__":
    # If topic keywords are available in your JSON:
    # Visualize the keyword weights for topic ID 3
    # Change this number to visualize a different topic
    visualize_topic_keywords(topic_id=3)