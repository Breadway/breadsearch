#!/usr/bin/env python3
"""
Quantize nomic-embed-text-v1.5 with AMD Quark for NPU (XDNA) inference.
Targets int8 QDQ format with enable_npu_transformer=True so VAIML can map
attention MatMul ops to the NPU.
"""

import json
import numpy as np
import onnxruntime
from pathlib import Path

MODEL_PATH = Path.home() / ".cache/breadsearch/models/model.onnx"
TOKENIZER_PATH = Path.home() / ".cache/breadsearch/models/tokenizer.json"
OUTPUT_PATH = Path.home() / ".cache/breadsearch/models/model_quantized.onnx"

CALIB_SENTENCES = [
    "search_document: The quick brown fox jumps over the lazy dog.",
    "search_document: Machine learning is a branch of artificial intelligence.",
    "search_document: The Eiffel Tower is located in Paris, France.",
    "search_document: Python is a high-level programming language.",
    "search_document: The speed of light is approximately 299,792,458 meters per second.",
    "search_document: Climate change is one of the greatest challenges facing humanity.",
    "search_document: Quantum computing leverages quantum mechanical phenomena.",
    "search_document: The human genome contains approximately 3 billion base pairs.",
    "search_document: Rust is a systems programming language focused on safety.",
    "search_document: Neural networks are inspired by the structure of the brain.",
    "search_document: The Milky Way galaxy contains over 100 billion stars.",
    "search_document: Cryptography is the practice of secure communication.",
    "search_document: The Linux kernel was first released in 1991 by Linus Torvalds.",
    "search_document: Databases store and retrieve structured data efficiently.",
    "search_document: The TCP/IP protocol suite is the backbone of the internet.",
    "search_document: Embedded systems run on resource-constrained hardware.",
    "search_document: The ONNX format provides a standard for machine learning models.",
    "search_document: Transformer models use attention mechanisms for sequence tasks.",
    "search_document: File systems organize and manage data storage on disks.",
    "search_document: Async programming allows concurrent execution without threads.",
    "search_query: What is machine learning?",
    "search_query: How does attention work in transformers?",
    "search_query: Where is the Eiffel Tower?",
    "search_query: What programming language should I learn first?",
    "search_query: How fast is the speed of light?",
]

MAX_SEQ_LEN = 512

def load_tokenizer(path: Path):
    with open(path) as f:
        return json.load(f)

def simple_tokenize(tok_data: dict, text: str, max_len: int) -> dict[str, np.ndarray]:
    vocab = tok_data["model"]["vocab"]
    unk_id = vocab.get("[UNK]", 100)
    cls_id = vocab.get("[CLS]", 101)
    sep_id = vocab.get("[SEP]", 102)
    pad_id = vocab.get("[PAD]", 0)

    words = text.lower().split()
    token_ids = [cls_id]
    for w in words:
        token_ids.append(vocab.get(w, unk_id))
    token_ids.append(sep_id)

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len - 1] + [sep_id]

    seq_len = len(token_ids)
    padded = token_ids + [pad_id] * (max_len - seq_len)
    mask = [1] * seq_len + [0] * (max_len - seq_len)
    type_ids = [0] * max_len

    return {
        "input_ids": np.array([padded], dtype=np.int64),
        "attention_mask": np.array([mask], dtype=np.int64),
        "token_type_ids": np.array([type_ids], dtype=np.int64),
    }


class NomicCalibrationReader:
    def __init__(self, sentences, tok_data, max_len=MAX_SEQ_LEN):
        self.inputs = [simple_tokenize(tok_data, s, max_len) for s in sentences]
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.inputs):
            return None
        sample = self.inputs[self.idx]
        self.idx += 1
        return sample

    def rewind(self):
        self.idx = 0


def main():
    print(f"Loading tokenizer from {TOKENIZER_PATH}")
    tok_data = load_tokenizer(TOKENIZER_PATH)

    calib_reader = NomicCalibrationReader(CALIB_SENTENCES, tok_data)

    # Verify model input names match what we provide
    sess = onnxruntime.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    print(f"Model inputs: {input_names}")
    del sess

    from quark.onnx import ModelQuantizer, QuantizationConfig
    from quark.onnx.quantization.config.config import Config
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType

    quant_config = QuantizationConfig(
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
        optimize_model=True,
        enable_npu_transformer=True,
        include_cle=True,
        print_summary=True,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    config = Config(global_quant_config=quant_config)
    quantizer = ModelQuantizer(config)
    print(f"Quantizing {MODEL_PATH} → {OUTPUT_PATH}")
    print("This may take several minutes...")

    quantizer.quantize_model(
        model_input=str(MODEL_PATH),
        model_output=str(OUTPUT_PATH),
        calibration_data_reader=calib_reader,
    )

    print(f"\nQuantized model saved to {OUTPUT_PATH}")
    print(f"Size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
