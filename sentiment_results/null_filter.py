import pandas as pd
import glob

fs = glob.glob('*.csv')
for f in fs:
    df = pd.read_csv(f)
    filtered_df = df.dropna(subset=['date'])
    if df.shape != filtered_df.shape:
        print(f"OVERWRITING {f}")
        filtered_df.to_csv(f)
