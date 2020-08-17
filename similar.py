try:
    import ktrain
except Exception as e:
    print(e)
    pass
from pprint import pprint
from joblib import load
from util import data_path
import pandas as pd
from scipy.spatial.distance import cosine
from pprint import pprint

"""
digunakan untuk membersihkan sebuah topic yang diasumsikan adalah kumpulan keluhan
"""

df = pd.read_csv(data_path / "1_koinworks_cleaned.csv")
df.dropna(inplace=True)
df = df.reset_index()


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


def compute_distance(query_id, embeddings, tweets, top_k):
    # query_id because encoder not yet implemented
    # df is tweets list (text)
    print(f"querying: {tweets.loc[query_id].flair_dataset}")
    print("computing distance to other tweets")

    similar_tweets = [
        (i, cosine(embeddings[query_id], tweet_vector))
        for i, tweet_vector in enumerate(embeddings)
        if i != query_id
    ]
    sorted_similar = [
        (i, tweets.loc[i], score)
        for i, score in sorted(similar_tweets, key=lambda x: x[1])
    ]
    return sorted_similar[:top_k]


def keluhan_flair():
    flair_embeddings = load(data_path / "flair.pkl")
    assert len(flair_embeddings) == len(df)
    query = int(input("query here (because time constraint please input the id): \n>"))
    hasil = compute_distance(
        query_id=query, embeddings=flair_embeddings, tweets=df, top_k=5
    )
    pprint(hasil)


if __name__ == "__main__":
    keluhan_flair()
