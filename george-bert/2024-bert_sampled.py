import pandas as pd
import glob
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag, word_tokenize
import contractions
from bertopic import BERTopic
import os
from transformers import AutoModel, AutoTokenizer
import torch
import html


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

TOP_ISSUES = ['economy', 'health care', 'supreme court appointment', 'foreign policy', 'violent crime',
              'immigration', 'gun policy', 'abortion', 'racial ethnic inequality', 'climate change']

SAMPLE_SIZE = 0.1


# Could probably be condensed - Try SpaCy
stop_word = list(set(stopwords.words('english')) - {"not", "no", "should", "must"} | {
                 "trump", "biden", "donald", "joe", "president", "kamala", "harris"})

lemmatiser = WordNetLemmatizer()


tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModel.from_pretrained("vinai/bertweet-base")

# Define custom embedding function for BERTopic


def bertweet_embedding(texts):
    """Generate embeddings for a list of texts using vinai/bertweet-base."""
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True,
                           truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        # CLS token embedding
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def filter_dataset(df):
    df = df[df['retweetedTweet'] == False]
    df = df[df['quotedTweet'] == False]
    df = df[df['likeCount'] >= 5]
    df = df[df['lang'] == 'en']
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= 3]
    df = df.dropna(subset=['text'])
    return df


def load_sampled_dataset(main_dir):
    parts = [
        f"{main_dir}/{p}" for p in os.listdir(main_dir) if p.startswith('p')]
    dfs = []

    for part in parts:
        files = glob.glob(f'{part}/*.csv.gz')
        df_list = [pd.read_csv(f, compression='gzip',
                               low_memory=False) for f in files]
        df = pd.concat(df_list, ignore_index=True)
        df = filter_dataset(df)
        df = df.sample(frac=SAMPLE_SIZE)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    return df


def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()  # Get first letter of POS tag
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def clean_text(text):
    text = html.unescape(text)
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub("@[A-Za-z0-9_]+", "", text)
    text = re.sub('#', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    text = re.sub("[^a-z0-9]", " ", text)
    text = word_tokenize(text, 'english')
    text = [lemmatiser.lemmatize(
        word, pos=get_wordnet_pos(word)) for word in text if len(word) >= 2]

    return " ".join(text)


def model_topics(df):
    vectoriser = CountVectorizer(stop_words=stop_word, ngram_range=(1, 3))
    topic_model = BERTopic(
        embedding_model=bertweet_embedding,
        vectorizer_model=vectoriser,
        verbose=True,
        low_memory=True,
        n_gram_range=(1, 3),
        nr_topics='auto'
    )
    topics, probs = topic_model.fit_transform(df["clean_text"])

    return topic_model, topics, probs


def save_model(save_dir, model, df):
    df.to_csv(
        f'{save_dir}/topics.csv', columns=["id", "clean_text", "topic", "prob", "date", "likeCount", "retweetCount" "viewCount"], index=False, compression='gzip')
    model.save(f"{save_dir}/bertopic_model",
               serialization='pytorch', save_ctfidf=True)


if __name__ == "__main__":
    save_dir = "/user/work/ne22902/ADS/models"
    new_version = "version_" + \
        str(max([int(folder[8:])
                 for folder in os.listdir(save_dir)], default=0) + 1)

    save_dir += f'/{new_version}'

    os.makedirs(save_dir, exist_ok=True)

    main_dir = '/user/home/ne22902/work/usc-x-24-us-election'
    df = load_sampled_dataset(main_dir)

    print(f"Number of tweets to analyse: {df.shape[0]}")

    df['clean_text'] = df['text'].apply(clean_text)

    topic_model, topics, probs = model_topics(df)
    df['topic'] = topics
    df['prob'] = probs
    save_model(save_dir, topic_model, df)
