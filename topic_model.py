import pandas as pd
import ktrain

df = pd.read_csv('koinworks_cleaned.csv')
df.dropna(inplace=True)
df = df[df['username']!='koinworks']
texts = df.cleaned.values
tm = ktrain.text.get_topic_model(texts, n_topics=None, n_features=10000)
tm.print_topics()
# precompute doc matrix (isinya probability ditribution)
tm.build(texts, threshold=.25) # kenapa .25, gue juga gak tau, still need to find out

# bisa diliat hasilnya disini
# by eyeballing it, bisa disimpulkan kalo topic di bawah punya kemungkinan keluhan paling tinggi 
keluhan_topics = input('masukkan id_topic, yang punya kemungkinan dia adalah keluhan\n>')
keluhan_topics = [int(a) for a in keluhan_topics.split()]

docs = tm.get_docs(topic_ids=keluhan_topics, rank=True)
# save docs as csv, karena kemungkinan itu adalah tweet ngeluh cukup tinggi
print('save docs as csv, karena kemungkinan itu adalah tweet ngeluh cukup tinggi')
print(type(docs)) 
df_keluhan=pd.DataFrame(docs, columns=['text', 'id', 'score', 'topic_id'])
df_keluhan.to_csv('koinworks_keluhan_lda.csv', index=False)
breakpoint()

texts =tm.filter(texts)
