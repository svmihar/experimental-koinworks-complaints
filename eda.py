from texthero.representation import pca, kmeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd

df = pd.read_csv("koinworks_cleaned.csv")
df = df[["date", "username", "cleaned", "tweet", "name"]]
df["date"] = pd.to_datetime(df["date"])
print(f"before drop duplicate: {len(df)}")
df = df.drop_duplicates(subset=["cleaned"])
print(f"after drop duplicate: {len(df)}")
df.dropna(inplace=True)

vectorizer  = TfidfVectorizer()
X = vectorizer.fit_transform(df.cleaned.values)
df["tfidf_dense"] = [list(a) for a in X.toarray()]
df["pca"] = df["tfidf_dense"].pipe(pca)
df.to_parquet("koinworks_fix.pkl", index=False)
df['tfidf']=X
joblib.dump(X,'tfidf.pkl')
