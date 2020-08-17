from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from util import data_path

flair_datapath = data_path / "flair_format"
# instantiate an existing LM, such as one from the FlairEmbeddings
language_model = FlairEmbeddings("id-forward").lm

# are you fine-tuning a forward or backward LM?
is_forward_lm = language_model.is_forward_lm

# get the dictionary from the existing language model
dictionary: Dictionary = language_model.dictionary

# get your corpus, process forward and at the character level
corpus = TextCorpus(flair_datapath, dictionary, is_forward_lm, character_level=True)

# use the model trainer to fine-tune this model on your corpus
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train(
    "models/",
    sequence_length=108,  # max(len(tweets))
    mini_batch_size=100,
    learning_rate=20,
    patience=10,
    checkpoint=True,
)
