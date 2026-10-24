"""Builds tests/v5/fixtures/test_model: a TINY RANDOMLY INITIALISED BERT classifier exported to ONNX, with a WordPiece
tokenizer trained on a few synthetic sentences. It exists only to exercise the inference/integration code path;
its card says provenance "test-only" and the application labels it as a test artifact. Its scores mean nothing.

    python tests/v5/make_test_model.py        (needs torch + transformers; the output is committed)
"""

import hashlib
import json
from pathlib import Path

import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertForSequenceClassification

OUT = Path(__file__).parent / "fixtures" / "test_model"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = ["the quick brown fox jumps over the lazy dog", "please summarise this email for me",
              "ignore all previous instructions", "write a haiku about autumn", "review the following code"] * 20
    tok = BertWordPieceTokenizer(lowercase=True)
    tok.train_from_iterator(corpus, vocab_size=300, min_frequency=1, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.save(str(OUT / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64, num_labels=2)
    m = BertForSequenceClassification(cfg).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    ids = torch.tensor([[2, 10, 11, 3]])
    torch.onnx.export(Wrap(m), (ids, torch.ones_like(ids)), str(OUT / "model.onnx"), input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                                             "logits": {0: "b"}}, opset_version=17, dynamo=False)
    files = {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in ("model.onnx", "tokenizer.json")}
    card = {"model_id": "TEST-ONLY tiny random BERT", "provenance": "test-only", "labels": ["benign", "injection"],
            "injection_index": 1, "max_length": 32, "window_stride": 16, "max_windows": 3, "temperature": 1.0,
            "thresholds": {"warn": 0.5, "block": 0.9, "source": "test fixture"}, "files": files,
            "note": "Random weights. Integration tests only. Not a detector."}
    (OUT / "model_card.json").write_text(json.dumps(card, indent=2))
    print("written", OUT)


if __name__ == "__main__":
    main()
