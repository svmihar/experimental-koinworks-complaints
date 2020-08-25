try:
    import os
    import ktrain
except Exception as e:
    print(e)
    pass
import numpy as np
from collections import Counter
from pprint import pprint
from util import data_path
import pandas as pd
from sklearn.cluster import DBSCAN

"""
clustering topics
"""


def get_k_word(tweets: list):
    words = [b for a in tweets for b in a.split()]
    return [a[0] for a in Counter(words).most_common(10)]


def dbscan_method(df):
    embeddings = np.array([a for a in df.pca.values])
    os.system("rm -rf ./data/topics/*")
    db = DBSCAN(eps=0.003, min_samples=3)  # 3-> 2 * n - 1
    db.fit(embeddings)
    labels = db.labels_
    count_labels = Counter(labels)
    pprint(count_labels)
    df = df[["id", "date", "username", "tweet", "cleaned"]]
    df["dbscan_tfidf"] = labels
    df.to_csv("data/4_dbscan_tfidf.csv", index=False)
    for label in set(labels):
        hasil = get_k_word(df[df["dbscan_tfidf"] == label].cleaned.values)
        print(label, hasil)
        with open(f"./data/topics/dbscan_{label}.txt", "a", encoding="utf-8") as f:
            f.writelines(" ".join(hasil))


def kmeans_method(df):
    # see untitled.ipynb on kmeans analysis
    pass


def lda_method(df):
    """ CANNOT RETRIEVE THE TOPIC ID"""
    tweets = df.cleaned.values
    tm = ktrain.text.get_topic_model(tweets, n_topics=None, n_features=10000)
    tm.print_topics(show_counts=True)
    # precompute doc matrix (isinya probability ditribution)
    tm.build(tweets, threshold=0)  # 0 karena ada range_id yang harus persistent
    tweet_docs = tm.get_docs()
    assert len(tweets) == len(tweet_docs)
    topic_selection = input("select your input here (1 2 3 ...)\n>")
    topic_selection = topic_selection.split()
    docs = tm.get_docs(topic_ids=topic_selection, rank=True)
    df["range_id"] = [x for x in range(len(df))]
    keluhan_df = pd.DataFrame(docs, columns=["text", "range_id", "score", "topic_id"])
    keluhan_df = keluhan_df[["text", "range_id", "topic_id"]]
    df = df.merge(keluhan_df, on="range_id")
    df = df[["id", "date", "username", "cleaned", "range_id", "topic_d"]]
    df.to_csv("./data/5_keluhan_lda.csv", index=False)


if __name__ == "__main__":
    df = pd.read_pickle(data_path / "3_koinworks_embeddings.pkl")
    lda_method(df)
#     dbscan_method(df)
# keluhan_flair()
