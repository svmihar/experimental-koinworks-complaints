try:
    import ktrain
except Exception as e:
    print(e)
    pass
import numpy as np
from collections import Counter
from pprint import pprint
from joblib import load
from util import data_path
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from pprint import pprint

"""
clustering topics
"""


def get_k_word(tweets: list):
    words = [b for a in tweets for b in a.split()]
    return [a[0] for a in Counter(words).most_common(10)]


def dbscan_method(df):
    embeddings = np.array([a for a in df.pca.values])
    import os

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
    texts = df.cleaned.values
    tm = ktrain.text.get_topic_model(texts, n_topics=None, n_features=10000)
    tm.print_topics()
    # precompute doc matrix (isinya probability ditribution)
    tm.build(
        texts, threshold=0.25
    )  # kenapa .25, gue juga gak tau, still need to find out
    # TODO: get the list of topic label
    import pdb

    pdb.set_trace()

    df.to_csv("data/4_lda_tfidf.csv", index=False)
    for label in set(labels):
        with open(f"./data/topics/lda_{label}.txt", "a", encoding="utf-8") as f:
            f.writelines(df[df["lda_tfidf"] == label].tweet.values)


if __name__ == "__main__":
    df = pd.read_pickle(data_path / "3_koinworks_embeddings.pkl")
#     lda_method(df)
#     dbscan_method(df)
# keluhan_flair()
