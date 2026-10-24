"""Memory / latency PROXY: a RANDOM-WEIGHT model with the EXACT architecture of the official checkpoint
(values read from its onnx/config.json), exported to ONNX and run with the app's ONNX Runtime path.

Memory and CPU time depend on architecture and input length, not on weight values, so this is a valid
footprint estimate. It says NOTHING about detection quality. Replace with the real checkpoint when available.

Usage (needs torch + transformers + onnx, dev only):
  python eval/footprint_proxy.py --sets eval/sets --out /tmp/proxy
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="eval/sets")
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
out = Path(a.out) / "onnx"; out.mkdir(parents=True, exist_ok=True)

if not (out / "model.onnx").exists():
    import torch
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
    cfg = DebertaV2Config(  # values from protectai/deberta-v3-base-prompt-injection-v2/onnx/config.json
        vocab_size=128100, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=512, relative_attention=True, position_buckets=256, norm_rel_ebd="layer_norm",
        share_att_key=True, pos_att_type=["p2c", "c2p"], position_biased_input=False, max_relative_positions=-1,
        hidden_act="gelu", layer_norm_eps=1e-7, type_vocab_size=0, pooler_hidden_size=768, pooler_hidden_act="gelu",
        pooler_dropout=0, id2label={0: "SAFE", 1: "INJECTION"}, label2id={"SAFE": 0, "INJECTION": 1})
    m = DebertaV2ForSequenceClassification(cfg).eval()
    print("params:", sum(p.numel() for p in m.parameters()))
    ids = torch.ones(1, 8, dtype=torch.long)
    torch.onnx.export(m, (ids, torch.ones_like(ids)), str(out / "model.onnx"), input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}},
                      opset_version=17, dynamo=False)
    (out / "config.json").write_text(cfg.to_json_string())
    # tokenizer: reuse the stub's (token ids only matter for length; 512-token inputs are forced below)
    stub = Path(__file__).resolve().parents[1] / "tests/fixtures/stub_model/onnx/tokenizer.json"
    (out / "tokenizer.json").write_bytes(stub.read_bytes())
print("model.onnx bytes:", (out / "model.onnx").stat().st_size)

# Measure in a FRESH process that imports only the app's runtime stack (onnxruntime, tokenizers, numpy).
probe = r'''
import sys, time, json, psutil, numpy as np
sys.path.insert(0, sys.argv[1])
p = psutil.Process(); base = p.memory_info().rss
from ppa_guard.engine2 import DebertaDetector
t0 = time.perf_counter(); d = DebertaDetector(sys.argv[2], threads=int(sys.argv[3]), model_id="PROXY-random"); load = time.perf_counter() - t0
after_load = p.memory_info().rss
lat = {}
for n in (16, 64, 128, 256, 512):
    ids = np.ones((1, n), dtype=np.int64); feeds = {"input_ids": ids, "attention_mask": ids}
    d.session.run(None, feeds)  # warm-up
    ts = []
    for _ in range(5):
        t = time.perf_counter(); d.session.run(None, feeds); ts.append((time.perf_counter() - t) * 1000)
    lat[n] = round(sorted(ts)[2], 1)
print(json.dumps({"rss_mb_python_baseline": round(base / 2**20), "rss_mb_after_load": round(after_load / 2**20),
                  "rss_mb_peak_after_512": round(p.memory_info().rss / 2**20), "load_s": round(load, 1),
                  "median_latency_ms_by_tokens": lat}))
'''
src = str(Path(__file__).resolve().parents[1] / "src")
r = subprocess.run([sys.executable, "-c", probe, src, str(out.parent), str(a.threads)], capture_output=True, text=True)
print(r.stdout or r.stderr[-2000:])
