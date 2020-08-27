from joblib import load
from cleaning import preprocess

def query(df, w):
    model = load('./data/tfidf_vectorizer.pkl')
    w_cleaned = preprocess(w)
    model.fit_transform(w)
