import matplotlib.pyplot as plt
import pandas as pd
import glob
from datetime import datetime


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

        for event, dt in events.items():
            plt.axvline(x=dt, color='black', linestyle='--', linewidth=2, label=event)

        plt.plot(sentiment_counts.index, sentiment_props['pos_smooth'], 'g-', label='Positive')
        plt.plot(sentiment_counts.index, sentiment_props['neg_smooth'], 'r-', label='Negative')
        plt.plot(sentiment_counts.index, sentiment_props['neu_smooth'], 'b-', label='Neutral')
        plt.title('Sentiment as a Proportion of Overall Tweets')
        plt.legend()

        plt.show()


def plot_topics():
    df = pd.read_csv("BERTTopic_sentiment_irony.csv")

    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.sort_values('date')
    df = df[df['date'] >= datetime(2024, 5, 1)]
    df = df[df['date'] <= datetime(2024, 11, 20)]

    # topic = 4, 'RFK Jr'
    topic = -1, 'biden/harris'

    sub_df = df[df['topic'] == topic[0]].copy()
    sub_df.set_index('date', inplace=True)

    sentiment_counts = pd.DataFrame()

    for sentiment in ['POS', 'NEG', 'NEU']:
        sentiment_counts[sentiment] = (
            sub_df['sentiment'].eq(sentiment)
            .resample('D')
            .sum()
        )
    sentiment_counts['total'] = sentiment_counts[['POS', 'NEG']].sum(axis=1)

    sentiment_props = pd.DataFrame()
    sentiment_props['POS'] = sentiment_counts['POS'] / sentiment_counts['total']
    sentiment_props['NEG'] = sentiment_counts['NEG'] / sentiment_counts['total']
    sentiment_props['NEU'] = sentiment_counts['NEU'] / sentiment_counts['total']

    sentiment_props = sentiment_props.interpolate()

    exp_alpha = 0.3
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

    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_counts.index, sentiment_counts['pos_smooth'], 'g-', label='Positive')
    plt.plot(sentiment_counts.index, sentiment_counts['neg_smooth'], 'r-', label='Negative')
    plt.plot(sentiment_counts.index, sentiment_counts['neu_smooth'], 'b-', label='Neutral')

    plt.title(f"{topic[1]} : {sub_df.shape[0]} Tweets")
    plt.legend()

    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_counts.index, sentiment_props['pos_smooth'], 'g-', label='Positive')
    plt.plot(sentiment_counts.index, sentiment_props['neg_smooth'], 'r-', label='Negative')
    plt.plot(sentiment_counts.index, sentiment_props['neu_smooth'], 'b-', label='Neutral')

    plt.title(f"{topic[1]} : {sub_df.shape[0]} Tweets 7 day sma")
    plt.legend()

    plt.show()

    """

    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_counts.index, sentiment_counts['POS'], 'g-', label='Positive')
    plt.plot(sentiment_counts.index, sentiment_counts['NEG'], 'r-', label='Negative')
    plt.plot(sentiment_counts.index, sentiment_counts['NEU'], 'b-', label='Neutral')

    plt.title(f"{topic[1]} : {sub_df.shape[0]} Tweets")

    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_counts.index, sentiment_counts['pos_smooth'], 'g-', label='Positive')
    plt.plot(sentiment_counts.index, sentiment_counts['neg_smooth'], 'r-', label='Negative')
    plt.plot(sentiment_counts.index, sentiment_counts['neu_smooth'], 'b-', label='Neutral')

    plt.title(f"{topic[1]} : {sub_df.shape[0]} Tweets 7 day sma")
    plt.legend()
    plt.show()
    """


plot_all_sentiment()
# plot_topics()
