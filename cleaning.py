import os
import string
import re
from util import data_path
import pandas as pd


STOPWORDS = {a.replace("\n", "") for a in (open("stopwords.txt").readlines())}
UNALLOWED = list(string.digits) + list(string.punctuation)


def is_referral(tweet: str):
    p = False
    t = tweet.split()
    keywords = [
        "referral",
        "kode",
        "referal",
        "code",
        "click to watch",
        "youtube",
        "download",
        "gratis",
    ]
    for word in keywords:
        if word in t:
            return True
    return p


def preprocess(tweet: str) -> str:
    # lowercase
    tweet_ = tweet.lower()
    # remove link
    tweet_ = re.sub(
        r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""",
        "",
        tweet_,
    )
    # remove digits
    for d in UNALLOWED:
        tweet_ = tweet_.replace(d, "")
    # remove multiple white space + split
    tweet_split = [t for t in tweet_.split() if t]
    return " ".join([t for t in tweet_split if t not in STOPWORDS])


if __name__ == "__main__":
    df = pd.read_csv(data_path / "koinworks.csv")
    data_folder = os.listdir()
    df["cleaned"] = df["tweet"].apply(preprocess)
    df["flair_dataset"] = df["cleaned"]
    df["is_ref"] = df["cleaned"].apply(is_referral)
    print(f"sebelum kena keyword block: {len(df)}")

    df = df[df["is_ref"] == False]
    print(f"setelah kena keyword block: {len(df)}")
    df = df[df["username"] != "danielchayau"]  # spam / bot account
    df = df[df["username"] != "koinworks"]  # own koinworks
    df = df.dropna(subset=["tweet", "username"])
    df = df.reset_index(drop=True)

    print("now saving the flair format dataset")
    print(f"total rows: {len(df)}")

    df.to_pickle(data_path / "1_koinworks_cleaned.pkl")

    # testing kalo STOPWORDS nya udah di remove
    for tweet in df["cleaned"].values:
        if "kalian" in tweet:
            print("belum ke remove")
            break
