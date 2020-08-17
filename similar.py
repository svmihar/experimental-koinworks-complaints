try:
    import ktrain
except Exception as e:
    print(e)
    pass

from pprint import pprint
from joblib import load
from util import data_path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
"""
digunakan untuk membersihkan sebuah topic yang diasumsikan adalah kumpulan keluhan
"""

df = pd.read_csv(data_path/"koinworks_cleaned.csv")
df.dropna(inplace=True)
tweets = df.flair_dataset.values


def keluhan_ktrain(df):
    df = df[df["username"] != "koinworks"]
    df_keluhan = pd.read_csv("koinworks_labeled_lda.csv")
    df_keluhan = df_keluhan[df_keluhan["label"] == 1]
    breakpoint()
    texts_keluhan = df_keluhan.text.values

    texts = df.cleaned.values
    tm = ktrain.text.get_topic_model(texts, n_features=10000, n_topics=None)
    tm.build(texts, threshold=0.25)
    tm.train_recommender()

    for keluhan in texts_keluhan:
        mirip = [a for a in tm.recommend(text=keluhan, n=5)]
        breakpoint()


def compute_distance(query_id, embeddings, df, top_k):
    # query_id because encoder not yet implemented
    # df is tweets list (text)

    print(f"querying: {df[query_id]}")
    print("computing distance to other tweets")
    similar_tweets_score = cosine_similarity(embeddings[query_id], embeddings)
    similar_tweets = [(i, tweet) for i, tweet in enumerate(similar_tweets_score)]
    sorted_similar = [
        (df[i], score) for i, score in sorted(similar_tweets, key=lambda x: x[1])
    ]
    return sorted_similar[:top_k]


def keluhan_flair():
    flair_embeddings = load(data_path / "flair.pkl")
    assert len(flair_embeddings) == len(tweets)
    query = input("query here (because time constraint please input the id): \n>")
    hasil = compute_distance(
        query_id=query, embeddings=flair_embeddings, df=tweets, top_k=10
    )
    pprint(hasil)


if __name__ == "__main__":
    keluhan_flair()
