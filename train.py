import ktrain
from ktrain import text
from pathlib import Path
import pandas as pd

dataset = Path("./data/labeled_complaints.csv")
models_path = Path("./models")
if not dataset.is_file():
    raise FileNotFoundError()
if not models_path.is_dir():
    print('model path is not found, creating a new one')
    models_path.mkdir(exist_ok=True, parents=True)


def _dataset(dataset=dataset):
    df = pd.read_csv(dataset)
    if "cleaned" not in df.columns:
        raise ValueError()
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(
        df,
        "cleaned",  # name of column containing review text
        label_columns=["complaint", "not_complaint"],
        maxlen=75,
        max_features=100000,
        preprocess_mode="standard",
        val_pct=0.1,
        ngram_range=1,
    )
    return x_train, y_train, x_test, y_test, preproc


def train_svm(x_train, y_train, x_test, y_test, preproc, bs=5):
    model = text.text_classifier("nbsvm", (x_train, y_train), preproc=preproc)
    learner = ktrain.get_learner(
        model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=bs
    )
    learner.lr_find(suggest=True)
    grad_lr = learner.lr_estimate()
    learner.autofit(min(grad_lr), 10)
    learner.view_top_losses(n=10, preproc=preproc)
    learner.validate(class_names = preproc.get_classes())
    predictor = ktrain.get_predictor(learner.model, preproc)
    predictor.save(str(models_path))

def train_gru(x_train, y_train, x_test, y_test, preproc, bs=5):
    model = text.text_classifier("bigru", (x_train, y_train), preproc=preproc)
    learner = ktrain.get_learner(
        model, train_data=(x_train, y_train), val_data=(x_test, y_test)
    )
    learner.lr_find(suggest=True)
    grad_lr = learner.lr_estimate()
    learner.autofit(min(grad_lr), 10)
    predictor = ktrain.get_predictor(learner.model, preproc)
    predictor.save(str(models_path))
    learner.validate(class_names = preproc.get_classes())

def validate(model_path):
    model = None
    pass

if __name__ == "__main__":
    x = _dataset()
    train_svm(*x, bs=2)
    # train_gru(*x)
