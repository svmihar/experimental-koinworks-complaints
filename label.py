import pandas as pd
from rich.console import Console
from rich.markdown import Markdown
from fire import Fire

console = Console()


def label(csv="koinworks_keluhan_lda.csv"):
    df = pd.read_csv(csv)
    assert "text" in df.columns
    labels = []
    console.clear()
    for item in df.iterrows():
        item = item[1]
        text = f"""## {item.text}
        topic id is {item.topic_id}
        id is {item.id}
        - 1 is keluhan 
        - 2 is not keluhan"""
        markdown = Markdown(text)
        console.print(markdown)
        label = console.input("Your [b]input[/b] here\n")
        labels.append(int(label))
        console.clear()
    df["label"] = labels
    df.to_csv("koinworks_labeled_lda.csv")


if __name__ == "__main__":
    Fire(label)
