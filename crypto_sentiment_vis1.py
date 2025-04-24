import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
import yfinance as yf
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, t

# set font size and style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'font.family': 'Times New Roman'
})

# Load and prepare topic sentiment data, now also returning raw tweet counts as 'volume'
def load_sentiment(topic=3):
    df = pd.read_csv("merged_low_filter.csv")
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.sort_values('date')
    df = df[(df['date'] >= datetime(2024, 5, 1)) & (df['date'] <= datetime(2024, 11, 20))]

    with open(r'HPC-output-dir\version_5\bertopic_model\topics.json', 'r') as f:
        topics = json.load(f)
    label = topics.get('topic_labels', {}).get(str(topic), 'Topic')

    sub = df[df['topic'] == topic].copy()
    sub.set_index('date', inplace=True)

    # count POS/NEG/NEU and total tweets per day
    counts = sub.groupby([pd.Grouper(freq='D'), 'sentiment']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=['POS', 'NEG', 'NEU'], fill_value=0)
    counts['total'] = counts.sum(axis=1)

    # proportions of each sentiment
    props = counts[['POS', 'NEG', 'NEU']].divide(counts['total'], axis=0).interpolate()

    # smooth the proportions
    alpha = 0.3
    for col in ['POS', 'NEG', 'NEU']:
        props[f'{col}_smooth'] = props[col].ewm(alpha=alpha).mean()

    # add raw tweet volume as a metric
    props['volume'] = counts['total']

    return props, label

# Download and prepare crypto data
def load_crypto(ticker='BTC-USD'):
    data = yf.download(ticker, start="2024-05-01", end="2024-11-20")['Close']
    daily = data.resample('D').last().ffill()
    returns = daily.pct_change().dropna()
    return daily, returns

# Correlation analysis: multi-method on returns, capturing sample size at each lag
def analyze_correlations(sentiment, returns, metrics, max_lag=30):
    results = {}
    for m in metrics:
        pear_vals, pear_pvals, count_vals = [], [], []
        spear_vals, kendall_vals = [], []
        for lag in range(max_lag + 1):
            shifted_ret = returns.shift(-lag)
            df = pd.concat([sentiment[m], shifted_ret], axis=1).dropna()
            df.columns = ['sent', 'ret']
            n = len(df)
            count_vals.append(n)
            if n > 10:
                r, p = pearsonr(df['sent'], df['ret'])
                pear_vals.append(r)
                pear_pvals.append(p)
                spear_vals.append(spearmanr(df['sent'], df['ret'])[0])
                kendall_vals.append(kendalltau(df['sent'], df['ret'])[0])
            else:
                pear_vals.append(np.nan)
                pear_pvals.append(np.nan)
                spear_vals.append(np.nan)
                kendall_vals.append(np.nan)

        results[m] = pd.DataFrame({
            'Lag':      np.arange(max_lag + 1),
            'Pearson':  pear_vals,
            'pval':     pear_pvals,
            'n':        count_vals,
            'Spearman': spear_vals,
            'Kendall':  kendall_vals
        })
    return results

# Find best metric and lag by Pearson
def find_best_metric_lag(corrs):
    best_m, best_lag, best_corr = None, None, -np.inf
    for m, df in corrs.items():
        idx = df['Pearson'].idxmax()
        corr = df.at[idx, 'Pearson']
        if corr > best_corr:
            best_corr = corr
            best_m = m
            best_lag = int(df.at[idx, 'Lag'])
    return best_m, best_lag, best_corr

# Plot the rolling correlation at the best lag
def plot_best_lag_rolling(sentiment, returns, corrs, metrics_labels, window=30):
    best_m, best_lag, best_corr = find_best_metric_lag(corrs)
    shifted_sent = sentiment[best_m].shift(best_lag)
    df = pd.concat([shifted_sent, returns], axis=1).dropna()
    df.columns = ['sent', 'ret']
    rc = df['sent'].rolling(window).corr(df['ret'])
    n = len(df)
    conf_band = 1.96 / np.sqrt(n)

    plt.figure(figsize=(8, 5))
    plt.plot(rc.index, rc,
             label=f"Rolling corr of {metrics_labels[best_m]} (lag={best_lag})")
    plt.fill_between(rc.index, -conf_band, conf_band, alpha=0.2,
                     label='95% conf band')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Rolling Correlation at Best Lag')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()

# Plot correlation vs lag with critical-r threshold lines
def plot_correlation_lag(corrs, metrics_labels,
                         selected_metric='inv_negative', alpha=0.05):
    if selected_metric not in corrs:
        raise ValueError(f"Metric '{selected_metric}' not in correlations.")
    df = corrs[selected_metric].dropna(subset=['Pearson', 'n'])

    # compute critical r at each lag
    t_crit = t.ppf(1 - alpha/2, df=(df['n'] - 2))
    r_crit = t_crit / np.sqrt((df['n'] - 2) + t_crit**2)

    plt.figure(figsize=(10, 6))
    plt.plot(df['Lag'], df['Pearson'],
             label=metrics_labels[selected_metric])
    plt.plot(df['Lag'],  r_crit, linestyle=':', 
             label=f'+r_crit (α={alpha})')
    plt.plot(df['Lag'], -r_crit, linestyle=':', 
             label=f'-r_crit (α={alpha})')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Correlation vs Lag Time for {metrics_labels[selected_metric]}')
    plt.xlabel('Lag Time (days)')
    plt.ylabel('Correlation Coefficient')
    plt.legend()
    plt.show()

# Main execution
def main():
    sentiment, label = load_sentiment(topic=3)
    price, returns = load_crypto('BTC-USD')

    metrics_labels = {
        'POS_smooth':   'Positive Sentiment',
        'NEG_smooth':   'Negative Sentiment',
        'NEU_smooth':   'Neutral Sentiment',
        'inv_negative': 'Inverse Negative',
        'volume':       'Tweet Volume'
    }

    # the inverse-negative metric
    sentiment['inv_negative'] = 1 / (sentiment['NEG'] + 1e-10)

    # analyze all metrics, now including raw tweet volume
    corrs = analyze_correlations(sentiment, returns,
                                 list(metrics_labels.keys()))

    # choose whichever you like as default here:
    plot_correlation_lag(corrs, metrics_labels,
                         selected_metric='volume', alpha=0.05)
    plot_best_lag_rolling(sentiment, returns,
                          corrs, metrics_labels, window=30)

if __name__ == '__main__':
    main()
