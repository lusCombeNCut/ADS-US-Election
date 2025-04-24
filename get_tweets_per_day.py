import glob

import pandas as pd
import numpy as np

DSET_PATH = "./usc-x-24-us-election"

fs = glob.glob(f"{DSET_PATH}/*.csv.gz")
df = pd.concat(
    [pd.read_csv(f, usecols=np.array(['id', 'date'])) for f in fs],
    ignore_index=True
)
df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
df.set_index('date', inplace=True)
df_counts = df.resample('D').size().to_frame(name='count')
df_counts.reset_index(inplace=True)
df_counts.to_csv('tweet_counts.csv')
