import pandas as pd
from rich.console import Console
from rich.markdown import Markdown
from fire import Fire

console = Console()


def label(csv="5_keluhan_label_lda.csv"):
    df = pd.read_csv("./data/" + csv)
    assert "text" in df.columns
    labels = []
    console.clear()
    for i, item in enumerate(df.iterrows()):
        item = item[1]
        text = f"""## {item.text}
        topic id is {item.topic_id}
        id is {item.range_id}
        - 1 is keluhan
        - 2 is not keluhan"""
        markdown = Markdown(text)
        console.print(markdown)
        label = 1
        labels.append(int(label))
        console.clear()
        console.print(f"[italic red]{i}/{len(df)}[/italic red]", justify="right")
    df["label"] = labels
    df.to_csv("6_hasil_label_lda.csv", index=False)


if __name__ == "__main__":
    Fire(label)
