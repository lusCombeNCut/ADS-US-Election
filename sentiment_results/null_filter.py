import pandas as pd
import glob

import numpy as np

fs = glob.glob('*.csv')
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
filtered_df = df.dropna(subset=['date'])
df_no_dupes = filtered_df.drop_duplicates(subset=['id'])

print('dupes gone')

chunk_size = 40000

for i, chunk in enumerate(range(0, len(df), chunk_size)):
    df_no_dupes.iloc[chunk:chunk + chunk_size].to_csv(f"chunk{i}.csv")
    print(f"written chunk{i}.csv")
