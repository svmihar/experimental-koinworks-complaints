from texthero.representation import pca, kmeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from util import data_path
import joblib
import pandas as pd


df = pd.read_csv(data_path / "koinworks_cleaned.csv")
df = df[["date", "username", "cleaned", "tweet", "name"]]
df["date"] = pd.to_datetime(df["date"])
print(f"before drop duplicate: {len(df)}")
df = df.drop_duplicates(subset=["cleaned"])
print(f"after drop duplicate: {len(df)}")
df.dropna(inplace=True)

# TFIDF embeddings
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.cleaned.values)
df["tfidf_dense"] = [list(a) for a in X.toarray()]
df["pca"] = df["tfidf_dense"].pipe(pca)
df.to_parquet(data_path / "koinworks_fix.pkl", index=False)
df["tfidf"] = X
joblib.dump(X, data_path / "tfidf.pkl")
