import pandas as pd

# Load the IDs from the two datasets
df_ids = pd.read_csv("v3_results.csv")[["id"]].values.flatten().tolist()
mj1_ids = pd.read_csv("usc-x-24-us-election/part_1/may_july_chunk_1.csv.gz")[["id"]].values.flatten().tolist()

# Print the number of entries in each dataset
print(f"Number of entries in v3_results.csv: {len(df_ids)}")
print(f"Number of entries in may_july_chunk_1.csv.gz: {len(mj1_ids)}")

# Convert to sets for efficient comparison
df_ids_set = set(df_ids)
mj1_ids_set = set(mj1_ids)

# Calculate the intersection
common_ids = df_ids_set.intersection(mj1_ids_set)
print(common_ids)

# Print the number of common IDs
print(f"Number of common IDs: {len(common_ids)}")
