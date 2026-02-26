import torch


class MultiLabelTokenizer:
    def __init__(self, label_to_index=None):
        self.label_to_index = label_to_index

    def fit(self, labels: list[list[str]]):
        unique_labels = set(label for sublist in labels for label in sublist)
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    def transform(self, labels):
        one_hot = torch.zeros(len(self.label_to_index), dtype=torch.bool)
        for label in labels:
            if label in self.label_to_index:
                index = self.label_to_index[label]
                one_hot[index] = 1.0
        return one_hot

    def fit_transform(self, labels):
        self.fit(labels)
        return torch.stack([self.transform(label_set) for label_set in labels])
