from flair.data import Sentence
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
from util import data_path
import numpy as np
from flair.embeddings import (
    DocumentPoolEmbeddings,
)  # using pool because tweets are short
from flair.embeddings.token import (
    FlairEmbeddings,
    WordEmbeddings,
)

pca = PCA(n_components=2, svd_solver="full")

# TODO: a good logging will go a long way, yeah shut up
print("loading model")
model = FlairEmbeddings("./models/best-lm.pt")
fasttext_id = WordEmbeddings("id-crawl")
document_model = DocumentPoolEmbeddings([model, fasttext_id])
# load cleaned tweets
df = pd.read_pickle(data_path / "2_koinworks_fix.pkl")
tweets = df["flair_dataset"].values

# text -> flair embeddings
embeddings = []
for tweet in tqdm(tweets):
    s = Sentence(tweet)
    document_model.embed(s)
    embeddings.append(s.embedding.detach().cpu().numpy().reshape(1, -1))


# flair embeddings -> 2d by pca
print("now reducing dimensions")
embeddings_ = np.array([a[0] for a in embeddings])
flair_pca = pca.fit_transform(embeddings_)
df["flair_pca"] = [a for a in flair_pca]

print(f"pca variance, retained: {pca.explained_variance_ratio_.cumsum()}")

# insert to 3_koinworks_embeddings
df["flair_embedding"] = embeddings
df.to_pickle(data_path / "3_koinworks_embeddings.pkl")
