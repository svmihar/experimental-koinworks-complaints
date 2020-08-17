from pathlib import Path

data_path = Path("./data")
flair_datapath = data_path / "flair_format"
train_flair_datapath = flair_datapath / "train"
if not train_flair_datapath.is_dir():
    train_flair_datapath.mkdir(parents=True, exist_ok=True)
