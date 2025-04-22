import os
import glob
import pandas as pd

# Configuration: adjust these paths as needed
data_folder = r"orlando-bert\inference-output-low-filtering"
output_file = os.path.join(data_folder, "all_topics_combined.csv.gz")

# Gather all batch output files
glob_pattern = os.path.join(data_folder, "*_batch*.csv.gz")
files = sorted(glob.glob(glob_pattern))

if not files:
    print(f"No files found matching pattern: {glob_pattern}")
    exit(1)

print(f"Found {len(files)} files. Concatenating...")

# Read and concatenate
df_list = []
for f in files:
    print(f"Reading {f}...")
    df = pd.read_csv(f, compression='gzip')
    df_list.append(df)

combined = pd.concat(df_list, ignore_index=True)
print(f"Combined DataFrame has {combined.shape[0]} rows and {combined.shape[1]} columns.")

# Save combined CSV (gzip compressed)
combined.to_csv(output_file, index=False)
print(f"Saved combined file to: {output_file}")
