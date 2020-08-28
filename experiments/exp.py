from top2vec import Top2Vec
import pandas as pd

df_docs = pd.read_pickle("../data/2_koinworks_fix.pkl")
docs = df_docs["tweet"].values

model = Top2Vec(docs, speed="deep-learn", workers=4)
model.save("./top2vec.model")

topic_sizes, topic_nums = model.get_topic_sizes()
print(topic_sizes)
print(topic_nums)
