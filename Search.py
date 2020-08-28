from top2vec import Top2Vec
from dataclasses import dataclass


@dataclass
class model:
    model_path: str

    def __post_init__(self):
        self.model = Top2Vec.load(self.model_path)
        self.vectors = None
        self.topic_labels = None

    def search_keyword(self, query):
        self.model.search('')

    def search_doc(self, doc_id):
        pass
