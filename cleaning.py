import texthero.preprocessing as h
from util import data_path
import pandas as pd

stopwords = {a.replace("\n", "") for a in (open("stopwords.txt").readlines())}
custom_pipeline = (
    h.lowercase,
    h.remove_digits,
    h.remove_punctuation,
    h.remove_diacritics,
    lambda x: h.remove_stopwords(x, stopwords=stopwords),
    h.remove_whitespace,
)


def custom_pipe(tweets):
    return


def is_referral(tweet):
    p = False
    t = tweet.split()
    keywords = ["referral", "kode", "referal", "code", "click to watch"]
    for word in keywords:
        if word in t:
            p = True
        return p


df = pd.read_csv(data_path / "koinworks_raw.csv")
df["cleaned"] = h.clean(df["tweet"], pipeline=custom_pipeline)
df["flair_dataset"] = h.clean(df["tweet"], pipeline=custom_pipeline[:4])
df["flair_dataset"] = h.remove_whitespace(df["flair_dataset"])
df["is_ref"] = df["cleaned"].apply(is_referral)
df = df[df["is_ref"] == False]
df = df[df["username"] != "danielchayau"]  # spam / bot account
df = df[df["username"] != "koinworks"]  # own koinworks
df = df.dropna()
df = df.reset_index(inplace=True)

print("now saving the flair format dataset")
from sklearn.model_selection import train_test_split

tweets = df.flair_dataset.values
x, y = train_test_split(tweets)
y_test, y_val = train_test_split(y)
del y
with open(data_path / "flair_format/train/train.txt", "w") as f:
    for t in x:
        f.writelines(f"{t}\n")
with open(data_path / "flair_format/test.txt", "w") as f:
    for t in y_test:
        f.writelines(f"{t}\n")

with open(data_path / "flair_format/valid.txt", "w") as f:
    for t in y_val:
        f.writelines(f"{t}\n")

df.to_csv(data_path / "koinworks_cleaned.csv", index=False)


# testing kalo stopwords nya udah di remove
for tweet in df["cleaned"].values:
    if "kalian" in tweet:
        print("belum ke remove")
        break
