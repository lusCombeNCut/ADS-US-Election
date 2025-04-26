import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bertopic import BERTopic
import json

def create_wordcloud(topic_model, topic_id, label, output_dir):
    words = dict(topic_model.get_topic(topic_id))
    wc = WordCloud(background_color='white', width=800, height=600)
    wc.generate_from_frequencies(words)

    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    filename = f'topic_{topic_id}_{label.replace(" ", "_")}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def main():
    model_path = r"HPC-output-dir\version_5\bertopic_model"
    topic_model = BERTopic.load(model_path)
    
    topics_file = os.path.join(model_path, "topics.json")
    with open(topics_file, "r") as f:
        topics = json.load(f)
    topic_labels = topics.get("topic_labels", {})

    output_dir = os.path.join(model_path, "wordclouds")
    os.makedirs(output_dir, exist_ok=True)
    topic_ids = [topic_id for topic_id in topic_model.get_topic_info().Topic if topic_id != -1]

    for topic_id in topic_ids:
        label = topic_labels.get(str(topic_id), f"topic_{topic_id}")
        create_wordcloud(topic_model, topic_id, label, output_dir)
        print(f"Word cloud saved for topic {topic_id} ({label})")

if __name__ == "__main__":
    main()
