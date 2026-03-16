import pandas as pd
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from model.ModelWrapper import ModelWrapper
from model.model import PretrainedGenreTransformer, _compute_f1
from model.train import GenreDataset
from config.config import NUM_LABELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compile_model():
    model = PretrainedGenreTransformer(NUM_LABELS).to(DEVICE)
    checkpoint = torch.load("model/final_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    val_df = pd.read_json("dataset/p4_dataset_val.json")

    audio_dir = "dataset/audio_chunks"
    val_dataset = GenreDataset(val_df, audio_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1)

            y_pred.append(predictions.detach())
            y_true.append(labels.detach())

    y_pred = torch.cat(y_pred).cpu().tolist()
    y_true = torch.cat(y_true).cpu().tolist()

    res = _compute_f1(y_true, y_pred)

    print("Evaluation results on validation set:")
    print(f"F1 Micro: {res['f1_micro']:.4f}")
    print(f"F1 Macro: {res['f1_macro']:.4f}")
    print(f"F1 Weighted: {res['f1_weighted']:.4f}")
    print("F1 per class:")
    for label, f1 in res["f1_per_class"].items():
        print(f"  {label}: {f1:.4f}")

    model_wrapper = ModelWrapper(model)
    model_wrapper.serialize("model/compiled_model.tar.gz")

    model_wrapper = ModelWrapper.deserialize("model/compiled_model.tar.gz")

    session = model_wrapper.session

    all_predictions = []
    all_labels = []
    for input_values, labels in tqdm(val_dataset, desc="Deserialized Model Inference"):
        input_values_np = input_values.unsqueeze(0).cpu().numpy()

        outputs = session.run(
            None,
            {
                "input_values": input_values_np,
            },
        )
        logits = torch.from_numpy(outputs[0])

        predictions = torch.argmax(logits, dim=1)

        all_predictions.append(predictions.squeeze(0).cpu())
        all_labels.append(labels.int())

    all_predictions = torch.stack(all_predictions).cpu().tolist()
    all_labels = torch.stack(all_labels).cpu().tolist()
    assert len(all_predictions) == len(val_dataset)

    f1_scores = _compute_f1(all_labels, all_predictions)
    print("Evaluation results on validation set (after deserialization):")
    print(f"F1 Micro: {f1_scores['f1_micro']:.4f}")
    print(f"F1 Macro: {f1_scores['f1_macro']:.4f}")
    print(f"F1 Weighted: {f1_scores['f1_weighted']:.4f}")
    print("F1 per class:")
    for label, f1 in f1_scores["f1_per_class"].items():
        print(f"  {label}: {f1:.4f}")

    print("Performance Degradation Check:")
    print(f"F1 Micro Degradation: {res['f1_micro'] - f1_scores['f1_micro']:.4f}")
    print(f"F1 Macro Degradation: {res['f1_macro'] - f1_scores['f1_macro']:.4f}")
    print(
        f"F1 Weighted Degradation: {res['f1_weighted'] - f1_scores['f1_weighted']:.4f}"
    )
    print("F1 per class Degradation:")
    for label in res["f1_per_class"]:
        degradation = res["f1_per_class"][label] - f1_scores["f1_per_class"].get(
            label, 0
        )
        print(f"  {label}: {degradation:.4f}")
