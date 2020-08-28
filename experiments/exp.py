import string
import re


def custom_preprocessing(x):
    x = x.lower()
    x = re.sub(
        r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""",
        "",
        x,
    )
    for p in list(string.punctuation)+list(string.digits):
        x = x.replace(p, "")
        x = x.replace(r'\xa0','')
    x = [a.encode('ascii', 'ignore').decode('ascii') for a in x.split() if a]
    return ' '.join([a for a in x if a])


if __name__ == "__main__":
    from top2vec import Top2Vec
    import pandas as pd

    df_docs = pd.read_pickle("../data/2_koinworks_fix.pkl")
    docs = df_docs["tweet"].apply(custom_preprocessing).values
    breakpoint()
    model = Top2Vec(docs, speed="deep-learn", workers=4)
    model.save("./top2vec.model")

    topic_sizes, topic_nums = model.get_topic_sizes()
    print(f'vocab learned: {len(model.model.wv.vocab.keys())}')
    print(topic_sizes)
    print(topic_nums)
