import streamlit as st
from top2vec import Top2Vec

@st.cache
def load_vectors():
    model = Top2Vec.load('./top2vec.model')
    return model

def main():
    st.title("clustering using the top2vec")
    st.subheader('visualization of the dataset')
    model =load_vectors()
    # get the vectors + topic labels
    # visualize in plotly
    st.subheader('search tweets')
    st.subheader('similar tweets by distance')


if __name__ == "__main__":
    main()
