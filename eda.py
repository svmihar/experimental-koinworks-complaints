from texthero.representation import pca
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from util import data_path
import pandas as pd


df = pd.read_pickle(data_path / "1_koinworks_cleaned.pkl")
df = df[["id", "date", "username", "cleaned", "tweet", "flair_dataset", "name"]]
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
df["tfidf"] = X
df.to_pickle(data_path / "2_koinworks_fix.pkl")
tweets = df.flair_dataset.values
x, y = train_test_split(tweets)
y_test, y_val = train_test_split(y)
with open(data_path / "flair_format/train/train.txt", "w") as f:
    for t in tweets:
        f.writelines(f"{t}\n")
with open(data_path / "flair_format/test.txt", "w") as f:
    for t in y_test:
        f.writelines(f"{t}\n")
with open(data_path / "flair_format/valid.txt", "w") as f:
    for t in y_val:
        f.writelines(f"{t}\n")
