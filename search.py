from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_parquet('koinworks_fix.pkl')
tweets = list(df['cleaned'])
tfidf_vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            smooth_idf=True,
            ngram_range=(1, 1),
            norm="l1",  # median, data dikit, tapi panjang
        )
X = tfidf_vectorizer.fit_transform(tweets)
X = X.todense()

def search(query, X=X, top_k=10):
    # search by id
    print(df.loc[query].tweet)
    query_cleaned =df.loc[query].cleaned
    query_vec = tfidf_vectorizer.transform([query_cleaned])
    hasil = linear_kernel(query_vec, X)
    most_similar = [{'id': i, 'score': v, 'tweet': df.loc[i].tweet} for i, v in enumerate(hasil[0])]
    hasil=sorted(most_similar, key=lambda x: x['score'], reverse=True)
    return hasil[:top_k]
