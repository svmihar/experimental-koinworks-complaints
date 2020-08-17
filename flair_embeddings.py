from flair.data import Sentence
from tqdm import tqdm
import pandas as pd
from util import data_path
from flair.embeddings import DocumentPoolEmbeddings
from flair.embeddings.token import FlairEmbeddings # using pool because tweets are short
import pdb
from joblib import dump

# TODO: a good logging will go a long way, yeah shut up
print('loading model')
model = FlairEmbeddings('./models/best-lm.pt')
document_model = DocumentPoolEmbeddings([model])

# load cleaned tweets
df = pd.read_csv(data_path/'koinworks_cleaned.csv')
df.dropna(inplace=True)
tweets = df['flair_dataset'].values
del df

# text -> flair embeddings
embeddings = []
for tweet in tqdm(tweets):
    s = Sentence(tweet)
    document_model.embed(s)
    embeddings.append(s.embedding.detach().cpu().numpy().reshape(1,-1))

dump(embeddings, data_path/'flair.pkl')
