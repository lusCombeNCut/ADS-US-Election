import pandas as pd

df = pd.read_csv("orlando_BERTopic_results_v3.csv")[["id"]].values.flatten().tolist()
mj1 = pd.read_csv("../may_july_chunk_1.csv.gz")[["id"]].values.flatten().tolist()

ids = set(mj1)

for id in df:
    if id in ids:
        continue
        # print(f"found {id}")
    else:
        print(f"\n\n\n\n\n\n\n\n{id} NOT IN CHUNK 1\n\n\n\n\n\n\n\n")
