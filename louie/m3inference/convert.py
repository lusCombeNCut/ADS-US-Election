import pandas as pd
import json
import ast
import re
import glob

def value_or_null(val):
    return None if pd.isna(val) else val

def clean_value(val):
    # Remove datetime patterns
    return re.sub(r"datetime\.datetime\([^\)]+\)", '""', val)

def extract_name(val):
    try:
        # Clean the string to remove non-literal datetime objects
        clean_val = clean_value(val)
        # Convert string to dictionary safely
        data = ast.literal_eval(clean_val)
        # Return the rawDescription field
        return data.get('username', None)
    except Exception as e:
        return None
    

def extract_raw_description(val):
    try:
        # Clean the string to remove non-literal datetime objects
        clean_val = clean_value(val)
        # Convert string to dictionary safely
        data = ast.literal_eval(clean_val)
        # Return the rawDescription field
        return data.get('rawDescription', None)
    except Exception as e:
        return None
    
def load_dataset(main_dir, dir):
    files = glob.glob(f'{main_dir}/{dir}/*.csv.gz')
    # df_list = [pd.read_csv(f, compression = 'gzip') for f in files]
    df_list = [pd.read_csv(f, compression = 'infer') for f in files]
    df = pd.concat(df_list, ignore_index=True)
    return df
    

# csv_file = '/user/work/bn22907/usc-x-24-us-election'
# csv_file = 'aug_chunk_41.csv'

# jsonl_file = 'small_chunk.jsonl'
# jsonl_file = 'aug_chunk_41.jsonl'

jsonl_file = 'part_1.jsonl'
df =load_dataset('/user/work/bn22907/usc-x-24-us-election/', 'part_1')

with open(jsonl_file, 'w') as f:
    for _, row in df.iterrows():
        lang = value_or_null(row.get('lang'))
        if lang == 'zxx' or lang == 'und' or lang == 'qme' or lang is None:
            continue


        entry = {
            'id': None if pd.isna(row['id']) else str(row['id']),
            'name': value_or_null(extract_name(row['user'])),
            'screen_name': value_or_null(row['username']),
            'description': value_or_null(extract_raw_description(row['user'])),
            'lang': lang,
            'img_path': ""  # images not used; set as empty string
        }
        f.write(json.dumps(entry) + "\n")