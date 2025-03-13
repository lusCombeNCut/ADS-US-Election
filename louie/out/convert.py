import pandas as pd
import json
import ast
import re

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
    

# csv_file = 'small_chunk.csv'
csv_file = 'aug_chunk_41.csv'

jsonl_file = 'aug_chunk_41.jsonl'


df = pd.read_csv(csv_file)

with open(jsonl_file, 'w') as f:
    for _, row in df.iterrows():
        # Get the language; if it's 'zxx', assign a default value like 'en'
        lang = value_or_null(row.get('lang'))
        if lang == 'zxx' or lang is None:
            continue
        #     lang = "en"  # or any default value you prefer
        # if lang != 'en':
        #     continue

        entry = {
            'id': None if pd.isna(row['id']) else str(row['id']),
            'name': value_or_null(extract_name(row['user'])),
            'screen_name': value_or_null(row['username']),
            'description': value_or_null(extract_raw_description(row['user'])),
            'lang': lang,
            'img_path': ""  # images not used; set as empty string
        }
        f.write(json.dumps(entry) + "\n")