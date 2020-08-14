import ktrain
import pandas as pd

"""
digunakan untuk membersihkan sebuah topic yang diasumsikan adalah kumpulan keluhan
"""

df = pd.read_csv("koinworks_cleaned.csv")
df.dropna(inplace=True)
df = df[df["username"] != "koinworks"]
df_keluhan = pd.read_csv("koinworks_labeled_lda.csv")
df_keluhan = df_keluhan[df_keluhan["label"] == 1]
breakpoint()
texts_keluhan = df_keluhan.text.values

texts = df.cleaned.values
tm = ktrain.text.get_topic_model(texts, n_features=10000, n_topics=None)
tm.build(texts, threshold=0.25)
tm.train_recommender()

for keluhan in texts_keluhan:
    mirip = [a for a in tm.recommend(text=keluhan, n=5)]
    breakpoint()
