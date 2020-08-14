from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

import pandas as pd
import ktrain



embedding_size = 128
print("Loading data...")
df = pd.read_parquet("koinworks_gabung_label.pkl")
df = df[['cleaned', 'label']]
df['label'] = df['label'].apply(lambda x: 'keluhan' if x< 2 else 'not_keluhan')
df.columns= ['text', 'label']
df= pd.concat([df, df.label.astype('str').str.get_dummies()], axis=1, sort=False)
df = df[['text','keluhan', 'not_keluhan']]
# Embedding
max_features = 20000
maxlen = max([len(a.split()) for a in df.text.values])


(x_train, y_train), (x_test, y_test), preproc = ktrain.text.texts_from_df(df, 'text', ['keluhan', 'not_keluhan'],random_state=42, max_features=max_features, maxlen=maxlen, ngram_range=3)
print(len(x_train), "train sequences")
print(len(x_test), "test sequences")


# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 35


print("Build model...")

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
# model.add(Conv1D(filters, kernel_size, padding="valid", activation="relu", strides=1))
# model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

learner = ktrain.get_learner(
    model, train_data=(x_train, y_train), val_data=(x_test, y_test)
)

print("build model finish now finding learning rate")
learner.lr_find()
learner.lr_plot()
print("fitting shit")
learner.autofit(0.005, 20, reduce_on_plateau=3)
p = ktrain.get_predictor(learner.model, preproc)
p.save('./models/cnn-lstm/')
