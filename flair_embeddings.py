from flair.data import Sentence
from tqdm import tqdm
import pandas as pd
from util import data_path
from flair.embeddings import DocumentPoolEmbeddings # using pool because tweets are short
from flair.embeddings.token import (
    FlairEmbeddings,
    WordEmbeddings, 
)  

# TODO: a good logging will go a long way, yeah shut up
print("loading model")
model = FlairEmbeddings("./models/best-lm.pt")
fasttext_id = WordEmbeddings('id-crawl')
document_model = DocumentPoolEmbeddings([model, fasttext_id])
# load cleaned tweets
df = pd.read_pickle(data_path / "2_koinworks_fix.pkl")
tweets = df["flair_dataset"].values
del df

# text -> flair embeddings
embeddings = []
for tweet in tqdm(tweets):
    s = Sentence(tweet)
    document_model.embed(s)
    embeddings.append(s.embedding.detach().cpu().numpy().reshape(1, -1))

# insert to 3_koinworks_embeddings
df['flair_embedding']=embeddings
df.to_pickle('3_koinworks_embeddings.pkl')