import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from top2vec import Top2Vec

pca = PCA(n_components=2)
pca1 = PCA(n_components=2)
tsne = TSNE(n_components=2, init='random', perplexity=5)

@st.cache
def load_vectors():
    model = Top2Vec.load("./top2vec.model")
    topic_vectors = model.topic_vectors
    tweet_vectors = model.model.docvecs.vectors_docs
    pca_tweet_vec = pca.fit_transform(tweet_vectors)
    pca_topic_vec = pca.fit_transform(topic_vectors)
    # TODO: TSNE
    # TODO: reduce dimension with tsne
    # tsne_tweet_vec = 0
    return pca_tweet_vec, pca_topic_vec


def main():
    st.title("clustering using the top2vec")
    st.write('this dataset based on tweets that has keyword \'koinworks\'')
    st.subheader("visualization of the dataset")
    vectors, topic_vectors = load_vectors()
    # visualize in plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[a[0] for a in vectors], y=[a[1] for a in vectors], mode='markers'))
    fig.add_trace(go.Scatter(x=[a[0] for a in topic_vectors], y=[a[1] for a in topic_vectors], mode='markers'))
    st.plotly_chart(fig)
    st.subheader("search tweets")
    st.subheader("similar tweets by distance")


if __name__ == "__main__":
    main()
