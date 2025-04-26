import matplotlib.pyplot as plt
import pandas as pd
import glob
from datetime import datetime
import json
import yfinance as yf
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# Set visual style
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = 'Times New Roman'

# Helper functions to reduce repetition
def load_tweet_data(start_date="2024-05-01", end_date="2024-11-20", file_path="./results-data/merged_low_filter.csv"):
    """Load and prepare tweet data with date filtering"""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.sort_values('date')
    df = df[(df['date'] >= datetime.fromisoformat(start_date)) & 
            (df['date'] <= datetime.fromisoformat(end_date))]
    return df

def load_topic_data(topic_path='./HPC-output-dir/version_5/bertopic_model/topics.json'):
    """Load topic data from JSON file"""
    with open(topic_path, 'r') as f:
        return json.load(f)

def load_crypto(ticker='BTC-USD', start_date="2024-05-01", end_date="2024-11-20"):
    """Load cryptocurrency price data and calculate returns"""
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    daily = data.resample('D').last().ffill()
    returns = daily.pct_change().dropna()
    return daily, returns

def calculate_sentiment_metrics(df, exp_alpha=0.1):
    """Calculate sentiment counts, proportions and apply smoothing"""
    sentiment_counts = pd.DataFrame()
    
    for sentiment in ['POS', 'NEG', 'NEU']:
        sentiment_counts[sentiment] = (
            df['sentiment'].eq(sentiment)
            .resample('D')
            .sum()
        )
    sentiment_counts['total'] = sentiment_counts[['POS', 'NEG', 'NEU']].sum(axis=1)
    
    # Calculate proportions
    sentiment_props = pd.DataFrame()
    sentiment_props['POS'] = sentiment_counts['POS'] / sentiment_counts['total']
    sentiment_props['NEG'] = sentiment_counts['NEG'] / sentiment_counts['total']
    sentiment_props['NEU'] = sentiment_counts['NEU'] / sentiment_counts['total']
    sentiment_props = sentiment_props.interpolate()
    
    # Apply exponential smoothing
    for col in ['POS', 'NEG', 'NEU']:
        sentiment_counts[f'{col.lower()}_smooth'] = sentiment_counts[col].ewm(alpha=exp_alpha, adjust=False).mean()
        sentiment_props[f'{col.lower()}_smooth'] = sentiment_props[col].ewm(alpha=exp_alpha, adjust=False).mean()
    
    # Calculate derived metrics
    sentiment_props['pos_neg_ratio'] = sentiment_props['POS'] / (sentiment_props['NEG'] + 1e-10)
    sentiment_props['inv_negative'] = 1 / (sentiment_props['NEG'] + 1e-10)
    sentiment_props.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return sentiment_counts, sentiment_props

def calculate_correlation(sentiment_data, price_data, metrics, max_delay=30):
    """Calculate correlation between sentiment metrics and price data for different time delays"""
    # Align dates
    start_date = max(sentiment_data.index.min(), price_data.index.min())
    end_date = min(sentiment_data.index.max(), price_data.index.max())
    
    sentiment_aligned = sentiment_data.loc[start_date:end_date]
    price_aligned = price_data.loc[start_date:end_date]
    
    # Ensure price_aligned is a Series
    if isinstance(price_aligned, pd.DataFrame):
        price_series = price_aligned.iloc[:, 0]
    else:
        price_series = price_aligned
    
    # Calculate correlations
    all_correlation_results = {}
    
    for metric_name, metric_label in metrics.items():
        correlation_results = []
        
        for delay in range(max_delay + 1):
            shifted_sentiment = sentiment_aligned[metric_name].shift(delay)
            
            valid_data = pd.DataFrame({
                'sentiment': shifted_sentiment,
                'price': price_series
            }).dropna()
            
            if (len(valid_data) > 5 and 
                valid_data['sentiment'].nunique() > 1 and 
                valid_data['price'].nunique() > 1):
                try:
                    correlation = valid_data['sentiment'].corr(valid_data['price'])
                    correlation_results.append((delay, correlation))
                except Exception as e:
                    print(f"Error calculating correlation for {metric_name}: {e}")
                    correlation_results.append((delay, float('nan')))
            else:
                correlation_results.append((delay, float('nan')))
        
        all_correlation_results[metric_name] = pd.DataFrame(correlation_results, columns=['Delay', 'Correlation'])
    
    return all_correlation_results

def plot_sentiment_over_time(sentiment_props, title="Sentiment Trends"):
    """Plot sentiment proportions over time"""
    plt.figure(figsize=(7, 6))
    plt.plot(sentiment_props.index, sentiment_props['pos_smooth'], 'g-o', markersize=5, markevery=5, label='Positive')
    plt.plot(sentiment_props.index, sentiment_props['neg_smooth'], 'r-s', markersize=5, markevery=5, label='Negative')
    plt.plot(sentiment_props.index, sentiment_props['neu_smooth'], 'b-^', markersize=5, markevery=5, label='Neutral')
    
    # Add election date line
    election_date = pd.Timestamp('2024-11-05')
    if election_date >= sentiment_props.index.min() and election_date <= sentiment_props.index.max():
        plt.axvline(x=election_date, color='black', linestyle='--', linewidth=2, label='Election Date')
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt

def plot_correlation(correlations, ticker="BTC-USD", topic_label="Topic"):
    """Plot correlation coefficients for different metrics"""
    plt.figure(figsize=(7, 6))
    markers = {'pos_neg_ratio': 'o', 'pos_smooth': 's', 'NEU': '^', 'neg_smooth': 'D'}
    
    for metric_name, metric_label in correlations.items():
        corr_df = correlations[metric_name]
        marker = markers.get(metric_name, 'o')
        plt.plot(corr_df['Delay'], corr_df['Correlation'], '-', 
                 marker=marker, markersize=5, markevery=2, label=metric_label)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Correlation of {ticker} Price with\n{topic_label} Sentiment")
    plt.xlabel("Time Delay (Days)")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_sentiment_vs_price(sentiment_props, price_data, ticker="BTC-USD", topic_label="Topic"):
    """Create dual-axis plot with sentiment and price data"""
    fig, ax1 = plt.subplots(figsize=(7, 6))
    
    # Plot sentiment on left axis
    ax1.plot(sentiment_props.index, sentiment_props['pos_smooth'], 'g-o', markersize=5, markevery=5, label='Positive')
    ax1.plot(sentiment_props.index, sentiment_props['neg_smooth'], 'r-s', markersize=5, markevery=5, label='Negative')
    ax1.set_ylabel('Sentiment Proportion')
    ax1.tick_params(axis='y')
    
    # Plot price on right axis
    ax2 = ax1.twinx()
    ax2.plot(price_data.index, price_data, 'k--D', markersize=5, markevery=5, label=f'{ticker}')
    ax2.set_ylabel('Price/Returns')
    ax2.tick_params(axis='y')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f"{topic_label} Sentiment vs {ticker}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# Main functions that use the helpers
def plot_all_sentiment(exp_alpha=0.25):
    """Plot sentiment for all tweets"""
    dfs = [pd.read_csv(f) for f in glob.glob("./sentiment_results/*.csv")]
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.sort_values('date')
    df = df[(df['date'] >= datetime(2024, 5, 1)) & (df['date'] <= datetime(2024, 11, 20))]
    df.set_index('date', inplace=True)
    
    # Process all tweets
    sentiment_counts, sentiment_props = calculate_sentiment_metrics(df, exp_alpha)
    plot = plot_sentiment_over_time(sentiment_props, "Sentiment as a Proportion of Overall Tweets")
    plot.show()

def plot_topics(topic_id=3, exp_alpha=0.3):
    """Plot sentiment analysis for a specific topic"""
    # Load data
    df = load_tweet_data()
    topics_data = load_topic_data()
    
    # Get topic information
    topic_label = topics_data.get("topic_labels", {}).get(str(topic_id), f"Topic {topic_id}")
    
    # Filter for selected topic and calculate sentiment
    sub_df = df[df['topic'] == topic_id].copy()
    sub_df.set_index('date', inplace=True)
    sentiment_counts, sentiment_props = calculate_sentiment_metrics(sub_df, exp_alpha)
    
    # Plot basic sentiment trends
    plot = plot_sentiment_over_time(sentiment_props, f"{topic_label} : {sub_df.shape[0]} Tweets - Sentiment Graph")
    plot.show()
    
    # Load crypto data for comparison
    crypto_ticker = "BTC-USD"
    crypto_data = yf.download(crypto_ticker, start="2024-05-01", end="2024-11-20")
    crypto_data['Close'] = crypto_data['Close'] / crypto_data['Close'].max()  # Normalize
    
    # Define metrics for correlation analysis
    sentiment_metrics = {
        'pos_neg_ratio': 'Positive/Negative Ratio',
        'pos_smooth': 'Positive Sentiment',
        'NEU': 'Neutral Sentiment',
        'inv_negative': 'Inverse Negative'
    }
    
    # Calculate correlation
    correlations = calculate_correlation(sentiment_props, crypto_data['Close'], sentiment_metrics)
    
    # Plot correlation results
    corr_plot = plot_correlation(correlations, crypto_ticker, topic_label)
    corr_plot.show()
    
    # Plot combined sentiment and price
    price_plot = plot_sentiment_vs_price(sentiment_props, crypto_data['Close'], crypto_ticker, topic_label)
    price_plot.show()

def visualize_topic_keywords(topic_id=3):
    """Visualize word importance weights for a specific topic"""
    # Load topics data
    topics_data = load_topic_data()
    
    # Get topic info
    topic_sizes = topics_data.get("topic_sizes", {})
    num_tweets = topic_sizes.get(str(topic_id), 0)
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
        weights = list(range(len(words), 0, -1))
    
    # Create DataFrame and plot
    df = pd.DataFrame({'word': words, 'weight': weights}).head(15).sort_values('weight')
    
    plt.figure(figsize=(7, 6))
    bars = plt.barh(df['word'], df['weight'], color='steelblue')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', va='center')
    
    plt.title(f'Keyword Weights for the {topic_label}\n{num_tweets:,} tweets')
    plt.xlabel('Weight')
    plt.ylabel('Keywords')
    plt.tight_layout()
    plt.yticks(rotation=45)
    plt.xlim(0, df['weight'].max() * 1.2)
    plt.savefig(f'topic_{topic_id}_keywords.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlate_sentiment_with_crypto(topic_id=3, exp_alpha=0.1):
    """Analyze correlation between sentiment for a topic and crypto returns"""
    # Load data
    df = load_tweet_data()
    topics_data = load_topic_data()
    
    # Get topic information
    topic_label = topics_data.get("topic_labels", {}).get(str(topic_id), f"Topic {topic_id}")
    topic_sizes = topics_data.get("topic_sizes", {})
    num_tweets = topic_sizes.get(str(topic_id), 0)
    
    # Filter and prepare data
    sub_df = df[df['topic'] == topic_id].copy()
    sub_df.set_index('date', inplace=True)
    sentiment_counts, sentiment_props = calculate_sentiment_metrics(sub_df, exp_alpha)
    
    # Plot sentiment
    sentiment_plot = plot_sentiment_over_time(sentiment_props, f"{topic_label} Sentiment\n{num_tweets:,} tweets")
    sentiment_plot.savefig(f'topic_{topic_id}_sentiment.png', dpi=300, bbox_inches='tight')
    sentiment_plot.show()
    
    # Load crypto data - using returns
    crypto_ticker = "BTC-USD"
    daily_prices, daily_returns = load_crypto(crypto_ticker)
    
    # Define metrics to analyze
    sentiment_metrics = {
        'pos_neg_ratio': 'Positive/Negative Ratio',
        'pos_smooth': 'Positive Sentiment',
        'NEU': 'Neutral Sentiment',
        'neg_smooth': 'Negative Sentiment'
    }
    
    # Calculate correlations with returns
    correlations = calculate_correlation(sentiment_props, daily_returns, sentiment_metrics)
    
    # Plot correlation results
    corr_plot = plot_correlation(correlations, f"{crypto_ticker} Daily Returns", topic_label)
    corr_plot.savefig(f'topic_{topic_id}_correlation.png', dpi=300, bbox_inches='tight')
    corr_plot.show()
    
    # Plot combined sentiment and returns
    combined_plot = plot_sentiment_vs_price(sentiment_props, daily_returns*100, f"{crypto_ticker} Returns (%)", topic_label)
    combined_plot.savefig(f'topic_{topic_id}_sentiment_vs_returns.png', dpi=300, bbox_inches='tight')
    combined_plot.show()

if __name__ == "__main__":
    # Uncomment functions you want to run
    # plot_all_sentiment()
    # plot_topics(3, 0.1)
    visualize_topic_keywords(topic_id=3)
    correlate_sentiment_with_crypto(topic_id=3, exp_alpha=0.1)
