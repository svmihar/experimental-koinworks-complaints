from numpy import w
from collections import Counter
from numpy.core.numeric import ComplexWarning
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from wordcloud import WordCloud
from top2vec import Top2Vec

pca = PCA(n_components=2)
pca1 = PCA(n_components=2)


def df_wrapper(search_result: list):
    result = [(tweet, score) for tweet, score, _ in zip(*search_result)]
    df = pd.DataFrame(result, columns=["tweets", "score"])
    return df.sort_values(by="score", ascending=False)


@st.cache
def load_vectors():
    model = Top2Vec.load("./models/top2vec.model")
    topic_vectors = model.topic_vectors
    tweet_vectors = model.model.docvecs.vectors_docs
    pca_tweet_vec = pca.fit_transform(tweet_vectors)
    pca_topic_vec = pca.fit_transform(topic_vectors)
    return pca_tweet_vec, pca_topic_vec, model


def complaint_words():
    haram = "ya ga bisa nya ada sudah dm dana aplikasi \
    min halo cek gak kasih tolong terima update kalo mohon utk".split()
    df = pd.read_csv("./data/labeled_complaint.csv")
    complaints = df[df["complaint"] == 1].cleaned.values
    words = [a for b in complaints for a in b.split() if a not in haram]
    return words


def wc(words):
    res = Counter(words).most_common(10)
    df = pd.DataFrame(res, columns=["words", "frequency"])
    return df


class _cluster:
    def __init__(self) -> None:
        self.df = pd.read_pickle("./data/4_hasil_cluster.pkl")
        self.kmeans = self.df.kmeans.values
        self.dbscan = self.df.dbscan.values
        self.x = [a[0] for a in self.df.umap_2d.values]
        self.y = [a[1] for a in self.df.umap_2d.values]

    def plot_df(self):
        records = {
            "kmeans_label":self.kmeans,
            "dbscan_label":self.dbscan,
            "x": self.x,
            "y": self.y,
        }
        df = pd.DataFrame(records)
        return df
    @classmethod
    def get_k_words(self, tweets):
        words = [b for a in tweets for b in a.split()]
        return ' '.join([a[0] for a in Counter(words).most_common(10)])

    def top_words(self):
        k_words = [self.df[self.df['kmeans'] == i].cleaned for i in set(self.kmeans)]
        d_words = [self.df[self.df['dbscan'] == i].cleaned for i in set(self.dbscan)]
        k_top = [self.get_k_words(a) for a in k_words]
        d_top = [self.get_k_words(a) for a in d_words]
        return k_top, d_top



def main():
    st.title("clustering using the top2vec")
    st.subheader("top words on complaint")
    st.write(wc(complaint_words()))
    st.subheader("tweet trends")
    st.write("this dataset based on tweets that has keyword 'koinworks'")

    st.subheader("visualization of the dataset")
    st.markdown("#### doc2vec")
    vectors, topic_vectors, model = load_vectors()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[a[0] for a in vectors], y=[a[1] for a in vectors], mode="markers")
    )
    fig.add_trace(
        go.Scatter(
            x=[a[0] for a in topic_vectors],
            y=[a[1] for a in topic_vectors],
            mode="markers",
        )
    )
    st.plotly_chart(fig)
    st.markdown("#### kmeans")
    C = _cluster()
    plot_df = C.plot_df()
    k_top, d_top = C.top_words()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["x"],
            y=plot_df["y"],
            marker_color=plot_df["kmeans_label"],
            mode="markers",
        )
    )
    st.plotly_chart(fig)
    st.write(k_top)
    st.markdown("#### dbscan")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["x"],
            y=plot_df["y"],
            marker_color=plot_df["dbscan_label"],
            mode="markers",
        )
    )
    st.plotly_chart(fig)
    st.write(d_top)

    st.subheader("search tweets")
    query = st.text_input("keyword")
    result = ""
    if query is not "":
        try:
            result = model.search_documents_by_keywords(query.split(), 50)
        except ValueError as e:
            st.write("no tweets detected, maybe try another keyword")
            # print('word is not in vocab')
    s = df_wrapper(result)
    st.dataframe(s, width=1000)
    st.subheader("similar tweets by distance")


if __name__ == "__main__":
    main()
