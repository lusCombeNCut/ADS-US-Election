import matplotlib.pyplot as plt
import pandas as pd
import glob
from datetime import datetime
import json
import yfinance as yf
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# set font size for all plots
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.titlesize'] = 16

# set fonts to times new roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = 'Times New Roman'


def plot_all_sentiment():
    dfs = [pd.read_csv(f) for f in glob.glob("sentiment_results/*.csv")]
    print("ALL RES READ")
    df = pd.concat(dfs, ignore_index=True)
    print("DF CONCAT")
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.sort_values('date')
    df = df[df['date'] >= datetime(2024, 5, 1)]
    df = df[df['date'] <= datetime(2024, 11, 20)]
    df.set_index('date', inplace=True)
    print("DATE SORTED")

    for df in [df[df['irony'] == 'ironic'], df[df['irony'] == 'not ironic']]:
        sentiment_counts = pd.DataFrame()

        for sentiment in ['POS', 'NEG', 'NEU']:
            sentiment_counts[sentiment] = (
                df['sentiment'].eq(sentiment)
                .resample('D')
                .sum()
            )
        sentiment_counts['total'] = sentiment_counts.sum(axis=1)
        sentiment_counts = sentiment_counts[sentiment_counts['total'] > 100]
        print("SENTIMENT COUNTS DONE")

        exp_alpha = 0.25

        sentiment_counts['pos_smooth'] = sentiment_counts['POS'].ewm(alpha=exp_alpha, adjust=False).mean()
        sentiment_counts['neu_smooth'] = sentiment_counts['NEU'].ewm(alpha=exp_alpha, adjust=False).mean()
        sentiment_counts['neg_smooth'] = sentiment_counts['NEG'].ewm(alpha=exp_alpha, adjust=False).mean()

        sentiment_props = pd.DataFrame()
        sentiment_props['POS'] = sentiment_counts['POS'] / sentiment_counts['total']
        sentiment_props['NEG'] = sentiment_counts['NEG'] / sentiment_counts['total']
        sentiment_props['NEU'] = sentiment_counts['NEU'] / sentiment_counts['total']

        sentiment_props = sentiment_props.interpolate()

        sentiment_props['pos_smooth'] = sentiment_props['POS'].ewm(alpha=exp_alpha, adjust=False).mean()
        sentiment_props['neu_smooth'] = sentiment_props['NEU'].ewm(alpha=exp_alpha, adjust=False).mean()
        sentiment_props['neg_smooth'] = sentiment_props['NEG'].ewm(alpha=exp_alpha, adjust=False).mean()

        """
        fig, ax = plt.subplots(2, 2)
        ax[0][0].plot(sentiment_counts.index, sentiment_counts['POS'], 'g-', label='Positive')
        ax[0][0].plot(sentiment_counts.index, sentiment_counts['NEG'], 'r-', label='Negative')
        ax[0][0].plot(sentiment_counts.index, sentiment_counts['NEU'], 'b-', label='Neutral')
        ax[0][0].set_title('counts unsmoothed')
        ax[0][0].legend()

        ax[0][1].plot(sentiment_counts.index, sentiment_counts['pos_smooth'], 'g-', label='Positive')
        ax[0][1].plot(sentiment_counts.index, sentiment_counts['neg_smooth'], 'r-', label='Negative')
        ax[0][1].plot(sentiment_counts.index, sentiment_counts['neu_smooth'], 'b-', label='Neutral')
        ax[0][1].set_title('counts smoothed')
        ax[0][1].legend()

        ax[1][0].plot(sentiment_counts.index, sentiment_props['POS'], 'g-', label='Positive')
        ax[1][0].plot(sentiment_counts.index, sentiment_props['NEG'], 'r-', label='Negative')
        ax[1][0].plot(sentiment_counts.index, sentiment_props['NEU'], 'b-', label='Neutral')
        ax[1][0].set_title('props unsmoothed')
        ax[1][0].legend()

        """

        events = {
            "Election Date": pd.Timestamp('2024-11-05')
        }

        plt.figure(figsize=(7, 6))

        for event, dt in events.items():
            plt.axvline(x=dt, color='black', linestyle='--', linewidth=2, label=event)

        # Add different markers for each line
        plt.plot(sentiment_counts.index, sentiment_props['pos_smooth'], 'g-o', markersize=5, markevery=5, label='Positive')
        plt.plot(sentiment_counts.index, sentiment_props['neg_smooth'], 'r-s', markersize=5, markevery=5, label='Negative')
        plt.plot(sentiment_counts.index, sentiment_props['neu_smooth'], 'b-^', markersize=5, markevery=5, label='Neutral')
        plt.title('Sentiment as a Proportion of Overall Tweets')
        plt.legend()
        plt.show()


def plot_topics(exp_alpha=0.3):
    df = pd.read_csv("merged_low_filter.csv")

    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.sort_values('date')
    df = df[df['date'] >= datetime(2024, 5, 1)]
    df = df[df['date'] <= datetime(2024, 11, 20)]

    # Get the topic labels from HPC-output-dir\version_5\bertopic_model\topics.json
    topics_path = r'HPC-output-dir\version_5\bertopic_model\topics.json'
    # topics_path = "HPC-output-dir/version_5/bertopic_model/topics.json"
    with open(topics_path, 'r') as f:
        topics_data = json.load(f)

    # user selects topic number, script finds label in field "topic_labels: " of the json file
    topic = 3  # 3 - crypto topic
    topic_label = topics_data.get("topic_labels", {}).get(str(topic), None)

    sub_df = df[df['topic'] == topic].copy()
    sub_df.set_index('date', inplace=True)

    sentiment_counts = pd.DataFrame()

    for sentiment in ['POS', 'NEG', 'NEU']:
        sentiment_counts[sentiment] = (
            sub_df['sentiment'].eq(sentiment)
            .resample('D')
            .sum()
        )
    sentiment_counts['total'] = sentiment_counts[['POS', 'NEG', 'NEU']].sum(axis=1)

    sentiment_props = pd.DataFrame()
    sentiment_props['POS'] = sentiment_counts['POS'] / sentiment_counts['total']
    sentiment_props['NEG'] = sentiment_counts['NEG'] / sentiment_counts['total']
    sentiment_props['NEU'] = sentiment_counts['NEU'] / sentiment_counts['total']

    sentiment_props = sentiment_props.interpolate()

    """
    sentiment_props['pos_smooth'] = sentiment_props['POS'].rolling(14, center=True).mean()
    sentiment_props['neu_smooth'] = sentiment_props['NEU'].rolling(14, center=True).mean()
    sentiment_props['neg_smooth'] = sentiment_props['NEG'].rolling(14, center=True).mean()
    """

    sentiment_counts['pos_smooth'] = sentiment_counts['POS'].ewm(alpha=exp_alpha, adjust=False).mean()
    sentiment_counts['neu_smooth'] = sentiment_counts['NEU'].ewm(alpha=exp_alpha, adjust=False).mean()
    sentiment_counts['neg_smooth'] = sentiment_counts['NEG'].ewm(alpha=exp_alpha, adjust=False).mean()

    sentiment_props['pos_smooth'] = sentiment_props['POS'].ewm(alpha=exp_alpha, adjust=False).mean()
    sentiment_props['neu_smooth'] = sentiment_props['NEU'].ewm(alpha=exp_alpha, adjust=False).mean()
    sentiment_props['neg_smooth'] = sentiment_props['NEG'].ewm(alpha=exp_alpha, adjust=False).mean()

    """
    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_counts.index, sentiment_counts['pos_smooth'], 'g-', label='Positive')
    plt.plot(sentiment_counts.index, sentiment_counts['neg_smooth'], 'r-', label='Negative')
    plt.plot(sentiment_counts.index, sentiment_counts['neu_smooth'], 'b-', label='Neutral')

    plt.title(f"{topic[1]} : {sub_df.shape[0]} Tweets")
    plt.legend()

    """
    # Add Yahoo Finance historical crypto/USD price for the same period
    crypto_ticker = "BTC-USD"  # Example: Bitcoin to USD
    crypto_data = yf.download(crypto_ticker, start="2024-05-01", end="2024-11-20")
    crypto_data['Close'] = crypto_data['Close'] / crypto_data['Close'].max()  # Normalize for comparison

    # Plot the original simple sentiment graph
    plt.figure(figsize=(7, 6))
    plt.plot(sentiment_counts.index, sentiment_props['pos_smooth'], 'g-o', markersize=5, markevery=5, label='Positive Sentiment')
    plt.plot(sentiment_counts.index, sentiment_props['neg_smooth'], 'r-s', markersize=5, markevery=5, label='Negative Sentiment')
    plt.plot(sentiment_counts.index, sentiment_props['neu_smooth'], 'b-^', markersize=5, markevery=5, label='Neutral Sentiment')
    plt.title(f"{topic_label} : {sub_df.shape[0]} Tweets - Sentiment Graph")
    plt.legend()
    plt.show()

    # Calculate the ratio of positive to negative sentiment with safeguards
    sentiment_props['pos_neg_ratio'] = sentiment_props['POS'] / (sentiment_props['NEG'] + 1e-10)  # Add small value to prevent division by zero
    sentiment_props['inv_negative'] = 1 / (sentiment_props['NEG'] + 1e-10)  # Inverse of negative sentiment

    # Replace infinite values with NaN for all metrics
    sentiment_props['pos_neg_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    sentiment_props['inv_negative'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Resample crypto data to daily frequency
    crypto_daily = crypto_data['Close'].resample('D').last().ffill()

    start_date = max(sentiment_props.index.min(), crypto_daily.index.min())
    end_date = min(sentiment_props.index.max(), crypto_daily.index.max())

    sentiment_aligned = sentiment_props.loc[start_date:end_date]
    crypto_aligned = crypto_daily.loc[start_date:end_date]

    # Define which metrics to analyze
    sentiment_metrics = {
        'pos_neg_ratio': 'Positive/Negative Ratio',
        'pos_smooth': 'Positive Sentiment',
        'NEU': 'Neutral Sentiment',
        'inv_negative': 'Inverse Negative'
    }

    # Maximum number of days to look back
    max_delay = 30
    all_correlation_results = {}

    # Calculate correlations for each metric
    for metric_name, metric_label in sentiment_metrics.items():
        correlation_results = []

        for delay in range(max_delay + 1):
            # Shift sentiment data and align with crypto data
            shifted_sentiment = sentiment_aligned[metric_name].shift(delay)

            crypto_series = crypto_aligned
            if isinstance(crypto_aligned, pd.DataFrame):
                crypto_series = crypto_aligned.iloc[:, 0]

            valid_data = pd.DataFrame({
                'sentiment': shifted_sentiment,
                'price': crypto_series
            }).dropna()

            if (len(valid_data) > 5 and
                valid_data['sentiment'].nunique() > 1 and
                    valid_data['price'].nunique() > 1):
                try:
                    correlation = valid_data['sentiment'].corr(valid_data['price'])
                    correlation_results.append((delay, correlation))
                except Exception as e:
                    print(f"Error calculating correlation for {metric_name} with delay {delay}: {e}")
                    correlation_results.append((delay, float('nan')))
            else:
                correlation_results.append((delay, float('nan')))

        all_correlation_results[metric_name] = pd.DataFrame(correlation_results, columns=['Delay', 'Correlation'])

    # Plot correlation coefficients for all metrics
    plt.figure(figsize=(7, 6))
    # Use different marker shapes for each metric
    markers = {'pos_neg_ratio': 'o', 'pos_smooth': 's', 'NEU': '^', 'inv_negative': 'D'}
    for metric_name, metric_label in sentiment_metrics.items():
        corr_df = all_correlation_results[metric_name]
        plt.plot(corr_df['Delay'], corr_df['Correlation'], '-', marker=markers[metric_name], label=metric_label)

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Correlation of {crypto_ticker} Price with \n Multiple Sentiment Metrics")
    plt.xlabel("Time Delay (Days)")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=corr_df['Delay'][corr_df['Correlation'].idxmax()], color='red', linestyle=':', label='Max Correlation Delay')
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.tight_layout()
    plt.show()

    # Plot sentiment and normalized crypto price for reference
    plt.figure(figsize=(7, 6))
    plt.plot(sentiment_counts.index, sentiment_props['pos_smooth'], 'g-o', markersize=5, markevery=5, label='Positive Sentiment')
    plt.plot(sentiment_counts.index, sentiment_props['neg_smooth'], 'r-s', markersize=5, markevery=5, label='Negative Sentiment')
    plt.plot(sentiment_counts.index, sentiment_props['neu_smooth'], 'b-^', markersize=5, markevery=5, label='Neutral Sentiment')
    plt.plot(crypto_data.index, crypto_data['Close'], 'k--D', markersize=5, markevery=5, label=f'{crypto_ticker} Price (Normalized)')
    plt.title(f"Crypto Price Versus Sentiment for the Crypto Topic")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    """

    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_counts.index, sentiment_counts['POS'], 'g-', label='Positive')
    plt.plot(sentiment_counts.index, sentiment_counts['NEG'], 'r-', label='Negative')
    plt.plot(sentiment_counts.index, sentiment_counts['NEU'], 'b-', label='Neutral')

    plt.title(f"{topic_label} : {sub_df.shape[0]} Tweets")

    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_counts.index, sentiment_counts['pos_smooth'], 'g-', label='Positive')
    plt.plot(sentiment_counts.index, sentiment_counts['neg_smooth'], 'r-', label='Negative')
    plt.plot(sentiment_counts.index, sentiment_counts['neu_smooth'], 'b-', label='Neutral')

    plt.title(f"{topic_label} : {sub_df.shape[0]} Tweets 7 day sma")
    plt.legend()
    plt.show()
    """


# plot_all_sentiment()
plot_topics(0.1)

def load_crypto(ticker='BTC-USD'):
    data = yf.download(ticker, start="2024-05-01", end="2024-11-20")['Close']
    daily = data.resample('D').last().ffill()
    returns = daily.pct_change().dropna()
    return daily, returns

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
    print(f"Number of tweets for topic {topic_id}: {num_tweets:,}")

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
    
    # Add subtitle with number of tweets
    plt.title(f'Keyword Weights for the {topic_label}\n{num_tweets:,} tweets')
    plt.xlabel('Weight')
    plt.ylabel('Keywords')
    plt.tight_layout()
    plt.yticks(rotation=45)
    # Add more white space at the end of bars so the labels are not cut off
    plt.xlim(0, df['weight'].max() * 1.2)
    plt.savefig(f'topic_{topic_id}_keywords.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlate_sentiment_with_crypto(topic_id=3, exp_alpha=0.1):
    """
    Analyze correlation between sentiment for a topic and crypto returns
    """
    # Load topic and sentiment data
    df = pd.read_csv("merged_low_filter.csv")
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.sort_values('date')
    df = df[df['date'] >= datetime(2024, 5, 1)]
    df = df[df['date'] <= datetime(2024, 11, 20)]

    # Get topic information
    topics_path = r'HPC-output-dir\version_5\bertopic_model\topics.json'
    with open(topics_path, 'r') as f:
        topics_data = json.load(f)
    
    topic_label = topics_data.get("topic_labels", {}).get(str(topic_id), f"Topic {topic_id}")
    topic_sizes = topics_data.get("topic_sizes", {})
    num_tweets = topic_sizes.get(str(topic_id), 0)

    # Filter for selected topic
    sub_df = df[df['topic'] == topic_id].copy()
    sub_df.set_index('date', inplace=True)
    
    # Calculate sentiment counts and proportions
    sentiment_counts = pd.DataFrame()
    for sentiment in ['POS', 'NEG', 'NEU']:
        sentiment_counts[sentiment] = (
            sub_df['sentiment'].eq(sentiment)
            .resample('D')
            .sum()
        )
    sentiment_counts['total'] = sentiment_counts[['POS', 'NEG', 'NEU']].sum(axis=1)
    
    sentiment_props = pd.DataFrame()
    sentiment_props['POS'] = sentiment_counts['POS'] / sentiment_counts['total']
    sentiment_props['NEG'] = sentiment_counts['NEG'] / sentiment_counts['total']
    sentiment_props['NEU'] = sentiment_counts['NEU'] / sentiment_counts['total']
    sentiment_props = sentiment_props.interpolate()
    
    # Apply exponential smoothing
    sentiment_props['pos_smooth'] = sentiment_props['POS'].ewm(alpha=exp_alpha, adjust=False).mean()
    sentiment_props['neu_smooth'] = sentiment_props['NEU'].ewm(alpha=exp_alpha, adjust=False).mean()
    sentiment_props['neg_smooth'] = sentiment_props['NEG'].ewm(alpha=exp_alpha, adjust=False).mean()
    
    # Load crypto data - use returns instead of prices
    crypto_ticker = "BTC-USD"
    daily_prices, daily_returns = load_crypto(crypto_ticker)
    
    # Plot sentiment over time
    plt.figure(figsize=(7, 6))
    plt.plot(sentiment_counts.index, sentiment_props['pos_smooth'], 'g-o', markersize=5, markevery=5, label='Positive')
    plt.plot(sentiment_counts.index, sentiment_props['neg_smooth'], 'r-s', markersize=5, markevery=5, label='Negative')
    plt.plot(sentiment_counts.index, sentiment_props['neu_smooth'], 'b-^', markersize=5, markevery=5, label='Neutral')
    plt.title(f"{topic_label} Sentiment\n{num_tweets:,} tweets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'topic_{topic_id}_sentiment.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Prepare sentiment metrics for correlation
    sentiment_props['pos_neg_ratio'] = sentiment_props['POS'] / (sentiment_props['NEG'] + 1e-10)
    sentiment_props['inv_negative'] = 1 / (sentiment_props['NEG'] + 1e-10)
    sentiment_props.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Align dates for correlation analysis
    start_date = max(sentiment_props.index.min(), daily_returns.index.min())
    end_date = min(sentiment_props.index.max(), daily_returns.index.max())
    
    sentiment_aligned = sentiment_props.loc[start_date:end_date]
    returns_aligned = daily_returns.loc[start_date:end_date]
    
    # Define metrics to analyze
    sentiment_metrics = {
        'pos_neg_ratio': 'Positive/Negative Ratio',
        'pos_smooth': 'Positive Sentiment',
        'NEU': 'Neutral Sentiment',
        'neg_smooth': 'Negative Sentiment'
    }
    
    # Calculate correlations for different time lags
    max_delay = 30
    all_correlation_results = {}
    
    for metric_name, metric_label in sentiment_metrics.items():
        correlation_results = []
        
        for delay in range(max_delay + 1):
            # Shift sentiment data and align with crypto returns
            shifted_sentiment = sentiment_aligned[metric_name].shift(delay)
            
            # Make sure returns_aligned is a Series, not a DataFrame
            if isinstance(returns_aligned, pd.DataFrame):
                returns_series = returns_aligned.iloc[:, 0]  # Extract the first column as a Series
            else:
                returns_series = returns_aligned  # Already a Series

            valid_data = pd.DataFrame({
                'sentiment': shifted_sentiment,
                'returns': returns_series
            }).dropna()
            
            if (len(valid_data) > 5 and
                valid_data['sentiment'].nunique() > 1 and
                valid_data['returns'].nunique() > 1):
                try:
                    correlation = valid_data['sentiment'].corr(valid_data['returns'])
                    correlation_results.append((delay, correlation))
                except Exception as e:
                    print(f"Error calculating correlation: {e}")
                    correlation_results.append((delay, float('nan')))
            else:
                correlation_results.append((delay, float('nan')))
        
        all_correlation_results[metric_name] = pd.DataFrame(correlation_results, columns=['Delay', 'Correlation'])
    
    # Plot correlation results
    plt.figure(figsize=(7, 6))
    markers = {'pos_neg_ratio': 'o', 'pos_smooth': 's', 'NEU': '^', 'neg_smooth': 'D'}
    for metric_name, metric_label in sentiment_metrics.items():
        corr_df = all_correlation_results[metric_name]
        plt.plot(corr_df['Delay'], corr_df['Correlation'], '-', marker=markers[metric_name], 
                 markersize=5, markevery=2, label=metric_label)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Correlation of {crypto_ticker} Daily Returns\nwith {topic_label} Sentiment")
    plt.xlabel("Time Delay (Days)")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'topic_{topic_id}_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot sentiment vs returns
    plt.figure(figsize=(7, 6))
    fig, ax1 = plt.subplots(figsize=(7, 6))
    
    # Plot sentiment on left axis
    ax1.plot(sentiment_counts.index, sentiment_props['pos_smooth'], 'g-o', markersize=5, markevery=5, label='Positive')
    ax1.plot(sentiment_counts.index, sentiment_props['neg_smooth'], 'r-s', markersize=5, markevery=5, label='Negative')
    ax1.set_ylabel('Sentiment Proportion')
    ax1.tick_params(axis='y')
    
    # Plot returns on right axis
    ax2 = ax1.twinx()
    ax2.plot(daily_returns.index, daily_returns*100, 'k--D', markersize=5, markevery=5, label='Daily Returns')
    ax2.set_ylabel('Daily Returns (%)')
    ax2.tick_params(axis='y')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f"{topic_label} Sentiment vs {crypto_ticker} Returns")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'topic_{topic_id}_sentiment_vs_returns.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Visualize the keyword weights for crypto topic (ID 3)
    visualize_topic_keywords(topic_id=3)
    
    # Analyze correlation between sentiment and crypto returns
    correlate_sentiment_with_crypto(topic_id=3, exp_alpha=0.1)
