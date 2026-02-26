import pandas as pd

from tokenizer.MultiLabelTokenizer import MultiLabelTokenizer


def tokenize_dataset():
    dataset = pd.read_json("dataset/videos.json")

    tokenizer = MultiLabelTokenizer()
    dataset["labels"] = tokenizer.fit_transform(dataset["genres"].tolist()).tolist()
    print(tokenizer.label_to_index)
    dataset.to_json("dataset/p2_dataset.json", index=False)
