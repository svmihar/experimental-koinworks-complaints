import streamlit as st
import numpy as np
import ktrain
from cleaning import preprocess



def main():
    select = st.sidebar.selectbox("Menu", list(MENU.keys()))
    MENU[select]()

@st.cache
def load_model(model_path="./models/"):
    return ktrain.load_predictor(model_path)

def search():
    st.write('otw')

def print_result(predict_proba):
    if predict_proba == '':
        return predict_proba
    h = np.argmax(predict_proba)
    probs = f'{predict_proba[h]:.0%}'
    if h:
        return f"NOT COMPLAINT \n probs: {probs}"
    else:
        return f"COMPLAINT \n probs: {probs}"


def classifier():
    hasil = ''
    st.title('simple tweet classifier')
    with st.spinner("loading model..."):
        model = load_model()
    check_tweet = st.text_input('Tweet: ')
    cleaned = preprocess(check_tweet)
    if check_tweet !='':
        hasil = model.predict(cleaned, return_proba=True)
        # st.components.v1.html(model.explain(cleaned)) # displays the type lol.
    st.write(print_result(hasil))


def eda():
    st.write("otw")


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
