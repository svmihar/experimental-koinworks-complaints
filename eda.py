from texthero.representation import tfidf, pca, kmeans
import pandas as pd

df = pd.read_csv('koinworks_cleaned.csv')
df=  df[['date','username', 'cleaned', 'tweet', 'name']]
df['date'] = pd.to_datetime(df['date'])
print(f'before drop duplicate: {len(df)}')
df = df.drop_duplicates(subset=['cleaned'])
print(f'after drop duplicate: {len(df)}')
df.dropna(inplace=True)
df['tfidf'] = df['cleaned'].pipe(tfidf)
df['pca'] = df['tfidf'].pipe(pca)
df.to_parquet('koinworks_fix.pkl', index=False)

