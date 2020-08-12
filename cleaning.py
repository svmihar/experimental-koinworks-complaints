import texthero.preprocessing as h
import pandas as pd

stopwords = {a.replace('\n', '') for a in (open('stopwords.txt').readlines())}


def custom_pipe(tweets): 
    s = h.remove_stopwords(tweets, stopwords)
    s = h.clean(s)
    return s

def is_referral(tweet): 
    p = False
    t = tweet.split()
    keywords = ['referral', 'kode', 'referal', 'code']
    for word in keywords: 
        if word in t: 
            p=True
        return p



df = pd.read_csv('koinworks_raw.csv')
t = df['tweet']
cleaned_t = custom_pipe(t)
df['cleaned'] = cleaned_t
df['is_ref'] = df['cleaned'].apply(is_referral)
df = df[df['is_ref']==False]
df = df[df['username']!='danielchayau']
df.dropna(inplace=True)
df.reset_index(inplace=True)

df.to_csv('koinworks_cleaned.csv', index=False)


