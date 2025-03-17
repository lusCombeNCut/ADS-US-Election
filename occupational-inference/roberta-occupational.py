#!/usr/bin/env python3

import os
import re
import ast
import nltk
import contractions
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# NEW: Import the zero-shot classification pipeline
from transformers import pipeline

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Expanded set of occupations
OCCUPATIONS = [
    "accountant", "accountant (cpa)", "actor", "actress", "architect", "artist",
    "attorney", "banker", "bartender", "blogger", "ceo", "chef", "coach", "cfo", 
    "cto", "comedian", "consultant", "counselor", "designer", "developer", 
    "director", "doctor", "dr",  # short form/abbreviation
    "economist", "editor", "educator", "engineer", "eng",  # short form/abbreviation
    "entrepreneur", "executive", "farmer", "filmmaker", 
    "financial advisor", "fa",  # short form
    "firefighter", "founder", "freelancer", "gamer", 
    "graphic designer", "healthcare", "hc", "influencer", 
    "instructor", "investigator", "investor", "journalist", 
    "jrnlst",  # short form
    "lawyer", "legal counsel", "manager", "mgr",  # short form
    "marketer", "mechanic", "model", "musician", "nurse", 
    "rn",  # short form for registered nurse
    "paralegal", "pastor", "performer", "pharmacist", 
    "photographer", "physician", "md",  # short form
    "pilot", "police", "politician", "podcaster", "producer", 
    "professor", "programmer", "coder",  # short form
    "psychologist", "psychiatrist", "publicist", "real estate agent",
    "realtor", "recruiter", "reporter", "researcher", "retired", 
    "singer", "songwriter", "scientist", "software engineer", "se", 
    "speaker", "strategist", "student", "surgeon", "teacher", 
    "tech support", "technician", "therapist", "trader", "translator", 
    "veterinarian", "writer", "author", "youtuber", "activist", 
    "digital marketer", "dj", "anchor", "host", "screenwriter", 
    "vfx artist", "composer", "lyricist", "mba", "phd"
]

# Expanded organization indicators
ORG_INDICATORS = [
    "news", "media", "organization", "party", "official", "foundation",
    "institute", "center", "group", "association", "society", "coalition",
    "network", "alliance", "federation", "union", "league", "corporation",
    "company", "llc", "inc", "corp", "incorporated", "ltd", "limited",
    "research", "university", "college", "school", "academy", "ministry",
    "committee", "nonprofit", "board", "charity", "council", "department",
    "cooperative", "ngo", "government"
]


# Stop words and Lemmatiser 
stop_word = set(stopwords.words('english')) - {"not", "no", "should", "must"}
spanish_words = set(stopwords.words('spanish'))
stop_word = stop_word.union({
    'joe','biden','donald','trump','don','el','en','que','para','hay','de','la','il','et',
    'na','ng','un','se','los','del','di','sur','su','una','como','por','al','ch','abd','srail','bir','er','ti'
})
stop_word = stop_word.union(spanish_words)
lemmatiser = WordNetLemmatizer()

# Load dataset
def load_dataset(main_dir, dirs):
    dfs = []
    for subdir in dirs:
        gz_file_path = os.path.join(main_dir, subdir)
        print(f"Looking for compressed file: {gz_file_path}")
        if os.path.exists(gz_file_path):
            print(f"File found: {gz_file_path}")
            try:
                df = pd.read_csv(gz_file_path, compression='gzip')
                print(f"Successfully loaded dataframe with {len(df)} rows")
                dfs.append(df)
            except Exception as e:
                print(f"Error loading file {gz_file_path}: {e}")
        else:
            print(f"File not found: {gz_file_path}")
    if len(dfs) > 0:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("No data was loaded from any file")

# Clean the text
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub("@[A-Za-z0-9_]+", "", text)
    text = re.sub('#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)
    text = re.sub("[^a-z0-9]", " ", text)
    tokens = word_tokenize(text, 'english')
    lemmed = [lemmatiser.lemmatize(w, pos=get_wordnet_pos(w)) for w in tokens]
    cleaned = [token for token in lemmed if token not in stop_word]
    return " ".join(cleaned)

# Parse JSON data
def parse_user_data(user_str):
    if not isinstance(user_str, str):
        return {}
    user_str = re.sub(r'datetime\.datetime\([^)]+\)', '"datetime_object"', user_str)
    try:
        return ast.literal_eval(user_str)
    except (SyntaxError, ValueError):
        # fallback
        try:
            match = re.search(r"'rawDescription':\s*'([^']*)'", user_str)
            if match:
                return {"rawDescription": match.group(1)}
            return {}
        except Exception:
            return {}

def clean_text_for_embedding(text):
    """Clean and normalize text for embedding"""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r'[^\x00-\x7F]+', '', text)   # remove non-ASCII
    cleaned = re.sub(r'[^\w\s@._-]', ' ', cleaned) # remove other special chars
    cleaned = re.sub(r'\s+', ' ', cleaned).strip() # normalise whitespace
    return cleaned[:200] if cleaned else "user"

# Zero-shot classification with RoBERTa
print("Loading zero-shot classification pipeline with roberta-large-mnli...")
zero_shot_classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")

def infer_occupations(description):
    """Get both single-label and multi-label classifications in one function."""
    if not description:
        return "Unknown", 0.0, ["Unknown"], [0.0]
    
    # Single call to the model with multi_label=True
    result = zero_shot_classifier(
    description,
    OCCUPATIONS,
    multi_label=True,
    hypothesis_template="This person works as a {}."
    )
    
    # Extract single-label result (top occupation)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    if top_score < 0.30:
        top_label = "Unknown"
        top_score = 0.0
    
    # Extract multi-label results
    valid_occupations = []
    valid_scores = []
    for label, score in zip(result["labels"], result["scores"]):
        if score >= 0.50:
            valid_occupations.append(label)
            valid_scores.append(score)
    if not valid_occupations:
        valid_occupations = ["Unknown"]
        valid_scores = [0.0]
    
    # Add this return statement
    return top_label, top_score, valid_occupations, valid_scores

if __name__ == "__main__":

    main_dir = '.'  # current directory
    dirs = ['data/may_july.gz']

    try:
        df = load_dataset(main_dir, dirs)
    except:
        print("Could not load from specified path, using sample data...")
        df = pd.read_csv('sample_tweets.csv')

    print("Applying filters...")
    df = df[df['retweetedTweet'] == False]
    df = df[df['quotedTweet'] == False]
    df = df[df['likeCount'] >= 5]
    df = df.dropna(subset=['text'])
    df = df.head(50) # CHANGE THIS TO ENTIRE TWEET SET
    print(f"Number of tweets to analyze: {df.shape[0]}")

    print("Cleaning tweet text...")
    df['clean_text'] = df['text'].apply(clean_text)

    print("Extracting user data...")
    df['user_data'] = df['user'].apply(parse_user_data)
    df['user_description'] = df['user_data'].apply(
        lambda x: x.get('rawDescription', '') if isinstance(x, dict) else ''
    )

    print("Cleaning user descriptions...")
    user_desc_list = df['user_description'].tolist()
    cleaned_desc_list = []
    total_desc = len(user_desc_list)

    for i, desc in enumerate(user_desc_list, 1):
        cleaned = clean_text_for_embedding(desc)
        cleaned_desc_list.append(cleaned)
        if i % 1 == 0:  # Print progress every iteration
            print(f"Cleaned {i} / {total_desc} descriptions...")

    df['clean_description'] = cleaned_desc_list
    desc_list = df['clean_description'].tolist()

    # For single-label approach
    inferred = []
    confidence_scores = []
    all_occupations = []
    all_confidence_scores = []
    total = len(desc_list)

    for i, desc in enumerate(desc_list, 1):
    # Get both single and multi-label results in one call
        occupation, confidence, occupations, confidences = infer_occupations(desc)
        
        # Store single-label results
        inferred.append(occupation)
        confidence_scores.append(confidence)
        
        # Store multi-label results
        all_occupations.append(occupations)
        all_confidence_scores.append(confidences)
        
        if i % 1 == 0:
            print(f"Processed {i} / {total} bios (combined approach)...")

    # Add results to dataframe
    df['inferred_occupation'] = inferred
    df['occupation_confidence'] = confidence_scores
    df['all_inferred_occupations'] = all_occupations
    df['all_occupation_confidences'] = all_confidence_scores    
    
    total_multi_occupations = sum(len(occs) for occs in all_occupations if "Unknown" not in occs)
    print(f"\nMulti-label approach identified {total_multi_occupations} total occupations across {len(df)} tweets")

    print("\nSample tweets with multi-label occupations:")
    sample_df_multi = df[["Unknown" not in occs for occs in df['all_inferred_occupations']]]
    if len(sample_df_multi) > 0:
        sample_multi = sample_df_multi.sample(min(5, len(sample_df_multi)))
        for _, row in sample_multi.iterrows():
            print(f"Text: {row['text'][:100]}...")
            print(f"User description: {row['user_description']}")
            print(f"Inferred occupations: {', '.join(row['all_inferred_occupations'])}")
            print(f"Confidence scores: {[round(score, 2) for score in row['all_occupation_confidences']]}")
            print("-" * 50)
    else:
        print("No tweets with multi-label occupations found.")

    # Count most frequent occupations from multi-label
    all_occs_flat = [occ for sublist in all_occupations for occ in sublist if occ != "Unknown"]
    if all_occs_flat:
        multi_occ_counts = pd.Series(all_occs_flat).value_counts()
        print("Top occupations (multi-label approach):")
        print(multi_occ_counts.head(15))

    # Display quick stats
    occ_counts = df['inferred_occupation'].value_counts()
    known_occupations = occ_counts.drop('Unknown', errors='ignore')

    print(f"\nIdentified {known_occupations.sum()} tweets with occupations out of {len(df)} total tweets\n")
    print("Top occupations (from zero-shot):")
    print(known_occupations.head(15))

    # Save results
    os.makedirs('../results', exist_ok=True)
    result_path = '../results/tweets_with_occupations_zero_shot.csv'
    df.to_csv(result_path, index=False)
    print(f"\nZero-shot results saved to: {result_path}")

    # Plot
    if len(known_occupations) > 0:
        top_n = 10
        plt.figure(figsize=(12, 6))
        known_occupations.head(top_n).plot(kind='bar')
        plt.title(f'Top {top_n} Inferred Occupations (Zero-Shot)')
        plt.xlabel('Occupation')
        plt.ylabel('Count')
        plt.tight_layout()

        plot_path = '../results/occupation_distribution_zero_shot.png'
        plt.savefig(plot_path)
        print(f"Occupation distribution chart saved to: {plot_path}")

    # Print sample
    print("\nSample tweets with inferred occupations:")
    sample_df = df[df['inferred_occupation'] != 'Unknown']
    if len(sample_df) > 0:
        sample = sample_df.sample(min(5, len(sample_df)))
        for _, row in sample.iterrows():
            print(f"Text: {row['text'][:100]}...")
            print(f"User description: {row['user_description']}")
            print(f"Inferred occupation: {row['inferred_occupation']} (Confidence: {row['occupation_confidence']:.2f})")
            print("-" * 50)
    else: 
        print("No tweets with a recognized occupation found.")

    
    print("\nCreating confidence score visualizations...")

    # 1. Box plot of confidence by occupation
    plt.figure(figsize=(14, 8))
    occupation_data = []
    labels = []

    # Get top occupations (exclude Unknown)
    top_occupations = known_occupations.head(10).index.tolist()

    for occupation in top_occupations:
        # Get confidence scores for this occupation
        scores = df[df['inferred_occupation'] == occupation]['occupation_confidence'].tolist()
        if scores:  # Check if any scores exist
            occupation_data.append(scores)
            labels.append(f"{occupation} (n={len(scores)})")

    if occupation_data:  # Check if we have data to plot
        plt.boxplot(occupation_data, labels=labels, vert=False)
        plt.title('Confidence Score Distribution by Top Occupations')
        plt.xlabel('Confidence Score')
        plt.tight_layout()
        plt.savefig('../results/occupation_confidence_boxplot.png')
        print(f"Confidence boxplot saved to: ../results/occupation_confidence_boxplot.png")

    # 2. Histogram of all confidence scores
    plt.figure(figsize=(12, 6))
    # Filter out Unknown occupations for the histogram
    valid_confidences = df[df['inferred_occupation'] != 'Unknown']['occupation_confidence']
    plt.hist(valid_confidences, bins=20, edgecolor='black')
    plt.title('Distribution of Confidence Scores Across All Detected Occupations')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.axvline(x=0.30, color='r', linestyle='--', label='Threshold (0.30)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../results/confidence_distribution_histogram.png')
    print(f"Confidence histogram saved to: ../results/confidence_distribution_histogram.png")

    # 3. Scatter plot of tweet likes vs. confidence
    plt.figure(figsize=(12, 6))
    plt.scatter(
        df[df['inferred_occupation'] != 'Unknown']['likeCount'],
        df[df['inferred_occupation'] != 'Unknown']['occupation_confidence'],
        alpha=0.5
    )
    plt.title('Relationship Between Tweet Popularity and Occupation Classification Confidence')
    plt.xlabel('Tweet Like Count')
    plt.ylabel('Occupation Confidence Score')
    plt.tight_layout()
    plt.savefig('../results/likes_vs_confidence_scatter.png')
    print(f"Likes vs. confidence scatter plot saved to: ../results/likes_vs_confidence_scatter.png")

    # 4. Heatmap of top multi-label occupation co-occurrences (if there are enough)
    if len(sample_df_multi) > 10:  # Only create if we have enough multi-labeled examples
        try:
            import numpy as np
            
            # Get top 15 occupations from multi-label approach
            top_multi_occs = pd.Series(all_occs_flat).value_counts().head(15).index.tolist()
            
            # Create co-occurrence matrix
            cooccur_matrix = np.zeros((len(top_multi_occs), len(top_multi_occs)))
            
            # Fill the matrix
            for occs in df['all_inferred_occupations']:
                # Skip unknown
                if "Unknown" in occs:
                    continue
                    
                # Check which top occupations are in this list
                present_occs = [occ for occ in occs if occ in top_multi_occs]
                
                # For each pair of occupations present, increment co-occurrence
                for i, occ1 in enumerate(present_occs):
                    idx1 = top_multi_occs.index(occ1)
                    for occ2 in present_occs:
                        idx2 = top_multi_occs.index(occ2)
                        cooccur_matrix[idx1, idx2] += 1
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            plt.imshow(cooccur_matrix, cmap='viridis')
            plt.colorbar(label='Co-occurrence count')
            plt.xticks(range(len(top_multi_occs)), top_multi_occs, rotation=90)
            plt.yticks(range(len(top_multi_occs)), top_multi_occs)
            plt.title('Co-occurrence of Top Occupations in Multi-label Classification')
            plt.tight_layout()
            plt.savefig('../results/occupation_cooccurrence_heatmap.png')
            print(f"Co-occurrence heatmap saved to: ../results/occupation_cooccurrence_heatmap.png")
        except Exception as e:
            print(f"Couldn't create co-occurrence heatmap: {e}")
