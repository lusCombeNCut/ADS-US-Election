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
from sentence_transformers import SentenceTransformer


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

TOP_ISSUES = ['economy', 'health care', 'supreme court appointment', 'foreign policy', 'violent crime',
              'immigration', 'gun policy', 'abortion', 'racial ethnic inequality', 'climate change']


# Could probably be condensed - Try SpaCy
stop_word = list(set(stopwords.words('english')) -
                 {"not", "no", "should", "must"})

lemmatiser = WordNetLemmatizer()


def load_dataset(main_dir, dirs):
    dfs = []
    for dir in dirs:
        files = glob.glob(f'{main_dir}/{dir}/*.csv.gz')
        df_list = [pd.read_csv(f, compression='gzip') for f in files]

        df = pd.concat(df_list, ignore_index=True)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df


def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()  # Get first letter of POS tag
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def clean_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub("@[A-Za-z0-9_]+", "", text)
    text = re.sub('#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)
    text = re.sub("[^a-z0-9]", " ", text)
    text = word_tokenize(text, 'english')
    text = [lemmatiser.lemmatize(
        word, pos=get_wordnet_pos(word)) for word in text]

    return " ".join(text)


def model_topics(df):
    vectoriser = CountVectorizer(stop_words=stop_word)
    embedding_model = SentenceTransformer("vinai/bertweet-base")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectoriser,
        verbose=True,
        low_memory=True,
        n_gram_range=(1, 3)
    )
    topics, probs = topic_model.fit_transform(df["clean_text"])

    return topic_model, topics, probs


def save_model(save_dir, model, df):
    df.to_csv(
        f'{save_dir}/topics.csv', columns=["id", "clean_text", "topic", "prob"], index=False, compression='gzip')
    model.save(f"{save_dir}/bertopic_model", serialization='pytorch')


if __name__ == "__main__":
    save_dir = "/user/work/ne22902/ADS/models"
    new_version = "version_" + \
        str(max([int(folder[8:])
                 for folder in os.listdir(save_dir)], default=0) + 1)

    save_dir += f'/{new_version}'

    os.makedirs(save_dir, exist_ok=True)

    main_dir = '/user/work/ne22902/usc-x-24-us-election'
    dirs = ['part_23', 'part_24', 'part_25', 'part_26', 'part_27', 'part_28']
    df = load_dataset(main_dir, dirs)
    df = df[df['retweetedTweet'] == False]
    df = df[df['quotedTweet'] == False]
    df = df[df['likeCount'] >= 5]
    df = df[df['lang'] == 'en']
    df = df.dropna(subset=['text'])
    print(f"Number of tweets to analyse: {df.shape[0]}")

    df['clean_text'] = df['text'].apply(clean_text)

    topic_model, topics, probs = model_topics(df)
    df['topic'] = topics
    df['prob'] = probs
    save_model(save_dir, topic_model, df)
