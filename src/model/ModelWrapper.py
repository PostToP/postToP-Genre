import tarfile
import tempfile
from pathlib import Path
import json
import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoFeatureExtractor
from config.config import TRANSFORMER_MODEL_NAME, VERSION, SAMPLE_RATE


class ModelWrapper:
    def __init__(self, model, session=None):
        self.model_name = TRANSFORMER_MODEL_NAME
        self.model = model
        self.session = session
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            TRANSFORMER_MODEL_NAME
        )

    def serialize(self, location):
        import torch
        from onnxruntime.quantization import (
            quantize_dynamic,
            QuantType,
            quant_pre_process,
        )
        from onnxruntime.transformers import optimizer

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"

            device = next(self.model.parameters()).device
            self.model = self.model.cpu()

            dummy_input = torch.zeros(1, 1024, 128, dtype=torch.float)

            torch.onnx.export(
                self.model,
                (dummy_input,),
                model_path.as_posix(),
                input_names=["input_values"],
                output_names=["logits"],
                opset_version=18,
            )

            opt_model = optimizer.optimize_model(
                model_path.as_posix(),
                model_type="vit",
                num_heads=12,
                hidden_size=768,
            )
            opt_model.save_model_to_file(model_path.as_posix())

            final_model_path = Path(tmpdir) / "model_final.onnx"
            quant_pre_process(
                input_model_path=model_path.as_posix(),
                output_model_path=final_model_path.as_posix(),
                skip_optimization=False,
            )
            quantize_dynamic(
                model_input=final_model_path.as_posix(),
                model_output=final_model_path.as_posix(),
                weight_type=QuantType.QInt8,
            )

            self.model = self.model.to(device)

            config = {
                "model_name": self.model_name,
                "version": VERSION,
            }
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            with tarfile.open(location, "w:gz") as tar:
                tar.add(final_model_path, arcname="model.onnx")
                tar.add(config_path, arcname="config.json")

    @staticmethod
    def deserialize(location):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(location, "r:gz") as tar:
                tar.extractall(tmpdir)
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            model_name = config["model_name"]
            version = config.get("version", "unknown")
            model_path = Path(tmpdir) / "model.onnx"
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                model_path.as_posix(),
                so,
                providers=["CPUExecutionProvider"],
            )
            mw = ModelWrapper(model=None, session=session)
            mw.model_name = model_name
            mw.version = version
            return mw

    def warmup(self):
        if self.session is not None:
            dummy_input_values = np.zeros((1, 1024, 128), dtype=np.float32)
            self.session.run(
                None,
                {
                    "input_values": dummy_input_values,
                },
            )

    def preprocess_audio(self, waveform):
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
        )
        return inputs["input_values"]

    def predict(self, waveform):
        input_values = self.preprocess_audio(waveform)
        outputs = self.session.run(
            None,
            {
                "input_values": input_values,
            },
        )
        logits = outputs[0]
        return logits
