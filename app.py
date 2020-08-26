from os import write
from numpy.core.defchararray import array
import streamlit as st
import numpy as np
import pandas as pd
from streamlit.proto.DataFrame_pb2 import DataFrame
import plotly.express as px
import ktrain
from cleaning import preprocess
from cluster_topic import get_k_word


def main():
    select = st.sidebar.selectbox("Menu", list(MENU.keys()))
    MENU[select]()


@st.cache
def load_model(model_path="./models/"):
    return ktrain.load_predictor(model_path)


def search():
    st.write("otw")


def print_result(predict_proba):
    if predict_proba == "":
        return predict_proba
    h = np.argmax(predict_proba)
    probs = f"{predict_proba[h]:.0%}"
    if h:
        return f"NOT COMPLAINT \n probs: {probs}"
    else:
        return f"COMPLAINT \n probs: {probs}"


def classifier():
    hasil = ""
    st.title("simple tweet classifier")
    with st.spinner("loading model..."):
        model = load_model()
    check_tweet = st.text_input("Tweet: ")
    cleaned = preprocess(check_tweet)
    if check_tweet != "":
        hasil = model.predict(cleaned, return_proba=True)
        # st.components.v1.html(model.explain(cleaned)) # displays the type lol.
    st.write(print_result(hasil))


def split_array(df, array_column):
    X = df[array_column].values
    x, y = [a[0] for a in X], [a[1] for a in X]
    label_x, label_y = f"{array_column}_x", f"{array_column}_y"
    df[label_x] = x
    df[label_y] = y
    return df


def write_top_words(labels: list, method: str, df: pd.DataFrame) -> list:
    words = [df[df[method] == i].cleaned for i in labels]
    top_words = [get_k_word(a) for a in words]
    for i, topic in enumerate(top_words):
        st.write(i, ' '.join(topic))


def eda():
    st.title("clustering")
    df = pd.read_pickle("./data/4_hasil_cluster.pkl")
    df_ = df[["pca", "flair_pca", "kmeans", "dbscan", "name"]]
    plot_df = df_.pipe(split_array, "pca")
    del df_
    st.header("kmeans")
    plotlychart = px.scatter(plot_df, x="pca_x", y="pca_y", color="kmeans")
    st.plotly_chart(plotlychart)
    kmeans_label = set(df["kmeans"].values)
    word = write_top_words(kmeans_label, "kmeans", df)

    st.header("dbscan")
    plotlychart1 = px.scatter(plot_df, x="pca_x", y="pca_y", color="dbscan")
    st.plotly_chart(plotlychart1)
    dbscan_labels = set(df["dbscan"].values)
    word = write_top_words(dbscan_labels, "dbscan", df)


def generate():
    st.write("otw")
    # TODO: bikin markov modelnya


MENU = {
    "classifier": classifier,
    "explore data": eda,
    "generate": generate,
    "search": search,
}
if __name__ == "__main__":
    main()
