import os
import glob
import pandas as pd

# Define the folder containing the .csv.gz files
path = r"C:\Users\Orlan\Documents\Applied-Data-Science\part_1\may_july_chunk_1.csv.gz"

# Read and concatenate all CSV files
df = pd.read_csv(path, compression="gzip")

# Save the concatenated DataFrame to a new CSV file
df.to_csv("combined_data.csv", index=False)