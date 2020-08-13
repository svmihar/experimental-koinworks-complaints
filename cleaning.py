import texthero.preprocessing as h
import pandas as pd

stopwords = {a.replace("\n", "") for a in (open("stopwords.txt").readlines())}
custom_pipeline = (
        h.lowercase,
        h.remove_digits, 
        h.remove_punctuation, 
        h.remove_diacritics, 
        lambda x: h.remove_stopwords(x, stopwords=stopwords), 
        h.remove_whitespace
        )


def custom_pipe(tweets):
    return 


def is_referral(tweet):
    p = False
    t = tweet.split()
    keywords = ["referral", "kode", "referal", "code", 'click to watch']
    for word in keywords:
        if word in t:
            p = True
        return p


df = pd.read_csv("koinworks_raw.csv")
df["cleaned"] = h.clean(df['tweet'], pipeline=custom_pipeline)
df["is_ref"] = df["cleaned"].apply(is_referral)
df = df[df["is_ref"] == False]
df = df[df["username"] != "danielchayau"]
df.dropna(inplace=True)
df.reset_index(inplace=True)

df.to_csv("koinworks_cleaned.csv", index=False)


# testing kalo stopwords nya udah di remove
for tweet in df['cleaned'].values: 
    if 'kalian' in tweet: 
        print('belum ke remove')
        break
