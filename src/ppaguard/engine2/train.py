"""Engine 2 training pipeline: data -> tokenizer -> fine-tuning -> ONNX export -> calibration -> thresholds -> test.

    python -m ppaguard.engine2.train --config configs/engine2/<config>.json

Modes
  pretrained     fine-tune an existing pretrained checkpoint (a local folder or a Hugging Face id such as
                 microsoft/deberta-v3-xsmall). Its own tokenizer is used unchanged.
  from_scratch   no pretrained weights are available: train a WordPiece tokenizer on the training split only and a
                 small BERT-style encoder from random initialisation. The resulting model card says so.

Data discipline
  * The training file is validated, de-duplicated and checked for conflicts; it must not share any normalised text
    with the frozen development or test sets (the run stops if it does).
  * Early stopping uses an internal, group-aware split of the TRAINING data.
  * The frozen development set is used for temperature calibration and threshold selection.
  * The frozen test set is evaluated exactly once, at the end, with everything already fixed. Nothing reads it earlier.

Needs the training extras: pip install -r requirements-train.txt   (torch, transformers, onnx; CPU is enough).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M
from . import tokenization as T


@dataclass
class TrainConfig:
    run_name: str
    train_file: str
    dev_file: str
    output_dir: str
    test_file: str | None = None
    group_flags: str | None = None
    mode: str = "pretrained"                  # pretrained | from_scratch
    base_model: str | None = None
    scratch: dict = field(default_factory=lambda: {"vocab_size": 16000, "hidden_size": 256, "num_hidden_layers": 4,
                                                   "num_attention_heads": 4, "intermediate_size": 1024,
                                                   "lowercase": True, "min_frequency": 2})
    max_length: int = 256
    batch_size: int = 32
    epochs: float = 2.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    class_weighting: str = "balanced"         # balanced | none
    internal_val_fraction: float = 0.05
    eval_every: int = 400
    patience: int = 5
    seed: int = 42
    threads: int = 2
    max_train_minutes: float | None = None
    max_train_rows: int | None = None
    block_min_precision: float = 0.98
    warn_min_recall: float = 0.95
    max_warn_fpr: float = 0.10
    max_block_fpr: float = 0.01
    truncate_layers: int | None = None         # pretrained mode: keep only the first N encoder layers
    freeze_embeddings: bool = False            # pretrained mode: keep the embedding table fixed (faster on CPU)
    quantize: bool = True
    max_quantization_ap_drop: float = 0.005

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(d) - set(cls.__dataclass_fields__) - {"_comment"}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        d.pop("_comment", None)
        cfg = cls(**d)
        if cfg.mode not in ("pretrained", "from_scratch"):
            raise ValueError("mode must be pretrained or from_scratch")
        if cfg.mode == "pretrained" and not cfg.base_model:
            raise ValueError("pretrained mode needs base_model")
        return cfg


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, event: str, **kw):
        rec = {"t": round(time.time(), 3), "event": event, **kw}
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()
        print(f"[{event}] " + " ".join(f"{k}={v}" for k, v in kw.items() if not isinstance(v, (dict, list))), flush=True)


def _seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _env() -> dict:
    import onnxruntime
    import tokenizers
    import torch
    import transformers
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "onnxruntime": onnxruntime.__version__, "numpy": np.__version__, "cpu_threads": torch.get_num_threads()}


# ---------------------------------------------------------------------------------------------------- data
def prepare_data(cfg: TrainConfig, log: Log) -> dict:
    train, rep = D.load_records(cfg.train_file, group_flags=cfg.group_flags)
    log("data.loaded", file=cfg.train_file, report=rep.as_dict())
    train, dd = D.dedup_and_conflicts(train)
    log("data.dedup", **dd)
    dev, dev_rep = D.load_records(cfg.dev_file)
    refs = {"dev": dev}
    test_rep = None
    if cfg.test_file:
        test, test_rep = D.load_records(cfg.test_file)
        refs["test"] = test
    overlaps = D.assert_no_leakage(train, refs)          # raises on any overlap
    log("data.leakage_check", overlaps=overlaps)
    if cfg.max_train_rows and len(train) > cfg.max_train_rows:
        rng = random.Random(cfg.seed)
        by = {0: [r for r in train if r.label == 0], 1: [r for r in train if r.label == 1]}
        keep = []
        for lab, rows in by.items():
            k = round(cfg.max_train_rows * len(rows) / len(train))
            keep += rng.sample(rows, k)
        train = keep
        log("data.subsampled", rows=len(train), reason="max_train_rows (CPU budget); class ratio preserved")
    tr, iv = D.grouped_split(train, cfg.internal_val_fraction, cfg.seed)
    info = {"train_file_sha256": rep.file_sha256, "dev_file_sha256": dev_rep.file_sha256,
            "test_file_sha256": test_rep.file_sha256 if test_rep else None,
            "train": D.class_balance(tr), "internal_val": D.class_balance(iv), "dev": D.class_balance(dev),
            "train_manifest": D.manifest(tr), "internal_val_manifest": D.manifest(iv), "dedup": dd, "leakage_overlaps": overlaps}
    log("data.split", **{k: v for k, v in info.items() if k in ("train", "internal_val", "dev")})
    return {"train": tr, "internal_val": iv, "dev": dev, "test": refs.get("test"), "info": info}


# ---------------------------------------------------------------------------------------------------- tokenizer + model
def build_tokenizer(cfg: TrainConfig, train_texts: list[str], out: Path, log: Log) -> Path:
    path = out / "tokenizer.json"
    if cfg.mode == "pretrained":
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(cfg.base_model)
        if not getattr(hf_tok, "is_fast", False):
            raise T.TokenizerMismatch("the base model has no fast tokenizer (tokenizer.json) for ONNX inference")
        hf_tok.backend_tokenizer.save(str(path))
        log("tokenizer.from_base_model", base=cfg.base_model, sha256=T.fingerprint(path))
    elif cfg.scratch.get("tokenizer_file"):
        # WordPiece training in `tokenizers` is not bit-reproducible (ties at the vocabulary cut-off depend on hash
        # order), so an exact re-run reuses the tokenizer.json saved by an earlier run.
        shutil.copy(cfg.scratch["tokenizer_file"], path)
        log("tokenizer.reused", source=cfg.scratch["tokenizer_file"], sha256=T.fingerprint(path))
    else:
        from tokenizers import BertWordPieceTokenizer
        s = cfg.scratch
        tok = BertWordPieceTokenizer(lowercase=s.get("lowercase", True), strip_accents=None)
        tok.train_from_iterator(train_texts, vocab_size=s["vocab_size"], min_frequency=s.get("min_frequency", 2),
                                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        tok.save(str(path))
        log("tokenizer.trained_on_train_split", vocab_size=tok.get_vocab_size(), sha256=T.fingerprint(path))
    return path


def build_model(cfg: TrainConfig, tok, log: Log):
    from transformers import AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification
    labels = {0: "benign", 1: "injection"}
    if cfg.mode == "pretrained":
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model, num_labels=2, id2label=labels, label2id={v: k for k, v in labels.items()},
            ignore_mismatched_sizes=True)
        if cfg.truncate_layers:
            enc = model.base_model.encoder
            if not 0 < cfg.truncate_layers <= len(enc.layer):
                raise ValueError(f"truncate_layers must be 1..{len(enc.layer)}")
            enc.layer = enc.layer[: cfg.truncate_layers]
            model.config.num_hidden_layers = cfg.truncate_layers
            log("model.truncated", kept_layers=cfg.truncate_layers,
                reason="smaller/faster model; artifact must fit the free-hosting limits")
        if cfg.freeze_embeddings:
            for prm in model.base_model.embeddings.parameters():
                prm.requires_grad = False
            log("model.embeddings_frozen")
    else:
        s = cfg.scratch
        conf = BertConfig(vocab_size=tok.get_vocab_size(), hidden_size=s["hidden_size"], num_hidden_layers=s["num_hidden_layers"],
                          num_attention_heads=s["num_attention_heads"], intermediate_size=s["intermediate_size"],
                          max_position_embeddings=max(512, cfg.max_length), num_labels=2, id2label=labels,
                          label2id={v: k for k, v in labels.items()}, pad_token_id=tok.token_to_id("[PAD]"))
        model = BertForSequenceClassification(conf)
    n = sum(p.numel() for p in model.parameters())
    log("model.built", mode=cfg.mode, base=cfg.base_model, parameters=n, architecture=type(model).__name__)
    return model


def _pad_id(tok) -> int:
    for t in ("[PAD]", "<pad>"):
        i = tok.token_to_id(t)
        if i is not None:
            return i
    return 0


def encode(tok, texts: list[str], max_length: int) -> list[list[int]]:
    tok.enable_truncation(max_length)
    try:
        return [e.ids for e in tok.encode_batch(texts)]
    finally:
        tok.no_truncation()


def batches(lengths: list[int], batch_size: int, rng: random.Random, shuffle=True):
    idx = list(range(len(lengths)))
    if shuffle:
        rng.shuffle(idx)
    chunk = batch_size * 50
    out = []
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i:i + chunk], key=lambda j: lengths[j])     # length bucketing: less padding, faster on CPU
        out += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(out)
    return out


def collate(ids: list[list[int]], rows: list[int], pad: int):
    import torch
    w = max(len(ids[r]) for r in rows)
    x = torch.full((len(rows), w), pad, dtype=torch.long)
    m = torch.zeros((len(rows), w), dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, :len(ids[r])] = torch.tensor(ids[r])
        m[i, :len(ids[r])] = 1
    return x, m


def torch_logits(model, ids, pad, batch_size=64) -> np.ndarray:
    import torch
    model.eval()
    out = np.zeros((len(ids), 2))
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    with torch.no_grad():
        for k in range(0, len(order), batch_size):
            rows = order[k:k + batch_size]
            x, m = collate(ids, rows, pad)
            out[rows] = model(input_ids=x, attention_mask=m).logits.float().numpy()
    model.train()
    return out


# ---------------------------------------------------------------------------------------------------- training loop
def fit(cfg: TrainConfig, model, tok, splits: dict, out: Path, log: Log) -> dict:
    import torch
    from transformers import get_linear_schedule_with_warmup
    pad = _pad_id(tok)
    tr, iv = splits["train"], splits["internal_val"]
    x_tr, y_tr = encode(tok, [r.text for r in tr], cfg.max_length), np.array([r.label for r in tr])
    x_iv, y_iv = encode(tok, [r.text for r in iv], cfg.max_length), np.array([r.label for r in iv])
    if cfg.class_weighting == "balanced":
        counts = np.bincount(y_tr, minlength=2)
        w = torch.tensor(len(y_tr) / (2 * counts), dtype=torch.float)
    else:
        w = torch.ones(2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    rng = random.Random(cfg.seed)
    steps_per_epoch = math.ceil(len(x_tr) / cfg.batch_size)
    total = max(1, int(steps_per_epoch * cfg.epochs))
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [{"params": [p for n, p in trainable if not any(k in n for k in no_decay)], "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in trainable if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
    sched = get_linear_schedule_with_warmup(opt, int(total * cfg.warmup_ratio), total)
    log("train.start", steps=total, steps_per_epoch=steps_per_epoch, class_weights=w.tolist(), train_rows=len(x_tr),
        internal_val_rows=len(x_iv))
    best = {"ap": -1.0, "step": 0}
    best_dir = out / "checkpoint_best"
    t0, step, bad_evals, stopped = time.time(), 0, 0, "completed"
    model.train()
    running = []
    while step < total:
        for rows in batches([len(s) for s in x_tr], cfg.batch_size, rng):
            if step >= total:
                break
            x, m = collate(x_tr, rows, pad)
            y = torch.tensor(y_tr[rows])
            loss = loss_fn(model(input_ids=x, attention_mask=m).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            running.append(float(loss.detach()))
            if step % cfg.eval_every == 0 or step == total:
                p = M.softmax(torch_logits(model, x_iv, pad))[:, 1]
                ap, auc = M.average_precision(y_iv, p), M.roc_auc(y_iv, p)
                f1 = M.report_at(y_iv, p, 0.5)["injection"]["f1"]
                log("train.eval", step=step, epoch=round(step / steps_per_epoch, 3), train_loss=round(float(np.mean(running)), 4),
                    internal_val_ap=round(ap, 4), internal_val_auc=round(auc, 4), internal_val_f1_at_0_5=round(f1, 4),
                    lr=sched.get_last_lr()[0], minutes=round((time.time() - t0) / 60, 2))
                running = []
                if ap > best["ap"] + 1e-4:
                    best = {"ap": ap, "auc": auc, "f1_at_0_5": f1, "step": step}
                    bad_evals = 0
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    model.save_pretrained(best_dir)
                else:
                    bad_evals += 1
                    if bad_evals >= cfg.patience:
                        stopped = f"early stopping after {cfg.patience} evaluations without improvement"
                        step = total
                        break
            if cfg.max_train_minutes and (time.time() - t0) / 60 > cfg.max_train_minutes:
                stopped = f"time budget of {cfg.max_train_minutes} minutes reached"
                step = total
                break
    log("train.end", stopped=stopped, best_step=best["step"], best_internal_val_ap=round(best["ap"], 4),
        minutes=round((time.time() - t0) / 60, 2))
    return {"best": best, "stopped": stopped, "steps_run": step, "minutes": round((time.time() - t0) / 60, 2),
            "class_weights": w.tolist(), "best_dir": str(best_dir)}


# ---------------------------------------------------------------------------------------------------- export
def export_onnx(model, tok, out_path: Path, max_length: int) -> None:
    import torch

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    model.eval()
    # Trace with a PADDED batch: if every mask value is 1 during tracing, some architectures (seen with DeBERTa-v2 and
    # DistilBERT under transformers 5.x) skip the masking branch and the exported graph ignores attention_mask.
    rows = encode(tok, ["export example text", "a somewhat longer second example sentence used for tracing"], max_length)
    width = max(len(r) for r in rows)
    pad = _pad_id(tok)
    ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows])
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    wrapped = Wrap(model).eval()   # a fresh Module starts in train mode; export restores that mode afterwards
    torch.onnx.export(wrapped, (ids, mask), str(out_path), input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                                             "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    model.eval()


def onnx_logits(path: Path, ids: list[list[int]], pad: int, batch_size=64) -> np.ndarray:
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = np.zeros((len(ids), 2))
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    for k in range(0, len(order), batch_size):
        rows = order[k:k + batch_size]
        w = max(len(ids[r]) for r in rows)
        x = np.full((len(rows), w), pad, dtype=np.int64)
        m = np.zeros_like(x)
        for i, r in enumerate(rows):
            x[i, :len(ids[r])] = ids[r]
            m[i, :len(ids[r])] = 1
        out[rows] = s.run(None, {"input_ids": x, "attention_mask": m})[0]
    return out


# ---------------------------------------------------------------------------------------------------- whole run
def run(cfg: TrainConfig) -> Path:
    import torch
    out = Path(cfg.output_dir) / cfg.run_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; choose a new run_name (runs are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    log = Log(out / "training_log.jsonl")
    torch.set_num_threads(cfg.threads)
    _seed(cfg.seed)
    env = _env()
    log("run.start", run=cfg.run_name, env=env)

    splits = prepare_data(cfg, log)
    tok_path = build_tokenizer(cfg, [r.text for r in splits["train"]], out, log)
    tok = T.load(tok_path)
    model = build_model(cfg, tok, log)
    emb = model.get_input_embeddings().weight.shape[0]
    specials = ("[CLS]", "[SEP]", "[PAD]") if cfg.mode == "from_scratch" else \
        tuple(t for t in ("<s>", "</s>", "<pad>") if tok.token_to_id(t) is not None) or ("[CLS]", "[SEP]")
    tok_report = T.verify(tok, emb, specials)
    tok_report["train_token_lengths"] = T.length_stats(tok, [r.text for r in splits["train"]])
    log("tokenizer.verified", report=tok_report)

    fit_info = fit(cfg, model, tok, splits, out, log)

    # best checkpoint -> ONNX (fp32), parity with PyTorch
    from transformers import AutoModelForSequenceClassification
    best = AutoModelForSequenceClassification.from_pretrained(fit_info["best_dir"]).eval()
    pad = _pad_id(tok)
    art = out / "artifact"
    art.mkdir()
    fp32 = out / "model_fp32.onnx"
    export_onnx(best, tok, fp32, cfg.max_length)
    iv_ids = encode(tok, [r.text for r in splits["internal_val"]], cfg.max_length)
    y_iv = np.array([r.label for r in splits["internal_val"]])
    sample = iv_ids[:256]
    diff = float(np.abs(torch_logits(best, sample, pad) - onnx_logits(fp32, sample, pad)).max())
    log("export.onnx_fp32", max_abs_logit_diff_vs_torch=diff, bytes=fp32.stat().st_size)
    if diff > 1e-3:
        raise RuntimeError(f"ONNX export parity failed: max |logit diff| {diff}")

    shipped, quant = fp32, None
    ap32 = M.average_precision(y_iv, M.softmax(onnx_logits(fp32, iv_ids, pad))[:, 1])
    if cfg.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q8 = out / "model_int8.onnx"
        quantize_dynamic(str(fp32), str(q8), weight_type=QuantType.QInt8)
        p8 = M.softmax(onnx_logits(q8, iv_ids, pad))[:, 1]
        p32 = M.softmax(onnx_logits(fp32, iv_ids, pad))[:, 1]
        ap8 = M.average_precision(y_iv, p8)
        quant = {"ap_fp32": ap32, "ap_int8": ap8, "max_prob_diff": float(np.abs(p8 - p32).max()),
                 "decision_agreement_at_0_5": float(((p8 > 0.5) == (p32 > 0.5)).mean()),
                 "bytes_fp32": fp32.stat().st_size, "bytes_int8": q8.stat().st_size}
        if ap32 - ap8 <= cfg.max_quantization_ap_drop:
            shipped = q8
            quant["shipped"] = "int8"
        else:
            quant["shipped"] = "fp32"
        log("export.quantization", **quant)
    shutil.copy(shipped, art / "model.onnx")
    shutil.copy(tok_path, art / "tokenizer.json")

    # calibration + thresholds on the frozen DEV set, using the exact shipped artifact and the runtime code path
    from .runtime import Detector
    provenance = "fine-tuned-ppaguard" if cfg.mode == "pretrained" else "trained-from-scratch-ppaguard"
    card = {"model_id": f"ppaguard/{cfg.run_name}", "provenance": provenance, "labels": ["benign", "injection"],
            "injection_index": 1, "max_length": cfg.max_length, "window_stride": cfg.max_length // 2, "max_windows": 8,
            "temperature": 1.0, "thresholds": {"warn": 0.5, "block": 0.9, "source": "provisional"}, "files": {}}
    (art / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    det = Detector(art, threads=cfg.threads, verify_hashes=False)

    def score(records):
        logits = np.array([_agg_logits(det, r.text) for r in records])
        return logits, np.array([r.label for r in records])

    dev_logits, y_dev = score(splits["dev"])
    temp = M.fit_temperature(y_dev, dev_logits)
    p_dev_raw, p_dev = M.softmax(dev_logits)[:, 1], M.softmax(dev_logits, temp)[:, 1]
    thr = M.select_thresholds(y_dev, p_dev, cfg.block_min_precision, cfg.warn_min_recall, cfg.max_warn_fpr, cfg.max_block_fpr)
    dev_eval = {"temperature": temp, "nll_before": M.nll(y_dev, dev_logits, 1.0), "nll_after": M.nll(y_dev, dev_logits, temp),
                "ece_before": M.ece(y_dev, p_dev_raw), "ece_after": M.ece(y_dev, p_dev),
                "selected_thresholds": thr, "metrics": M.summary(y_dev, p_dev, {"warn": thr["warn"], "block": thr["block"], "argmax_0_5": 0.5}),
                "by_subset": _by_subset(splits["dev"], p_dev, thr)}
    (out / "eval_dev.json").write_text(json.dumps(dev_eval, indent=2), encoding="utf-8")
    log("calibration.dev", temperature=round(temp, 4), warn=thr["warn"], block=thr["block"], notes=thr["notes"])

    card.update(temperature=temp, thresholds={"warn": thr["warn"], "block": thr["block"],
                                              "source": f"selected on frozen dev set ({len(y_dev)} rows)"})

    test_eval = None
    if splits["test"] is not None:   # evaluated once, nothing below changes the model, temperature or thresholds
        test_logits, y_test = score(splits["test"])
        p_test = M.softmax(test_logits, temp)[:, 1]
        test_eval = {"metrics": M.summary(y_test, p_test, {"warn": thr["warn"], "block": thr["block"], "argmax_0_5": 0.5}),
                     "by_subset": _by_subset(splits["test"], p_test, thr),
                     "note": "Frozen test set, evaluated once after all choices were fixed on train/dev."}
        (out / "eval_test.json").write_text(json.dumps(test_eval, indent=2), encoding="utf-8")
        log("evaluation.test", auc=test_eval["metrics"]["roc_auc"], ap=test_eval["metrics"]["average_precision"])

    files = {"model.onnx": D.sha256_file(art / "model.onnx"), "tokenizer.json": D.sha256_file(art / "tokenizer.json")}
    card.update(files=files, base_model=cfg.base_model if cfg.mode == "pretrained" else None,
                architecture=type(best).__name__, parameters=sum(p.numel() for p in best.parameters()),
                quantization=quant, tokenizer=tok_report, training={**fit_info, "config": asdict(cfg)},
                data=splits["info"], environment=env,
                metrics={"dev": _headline(dev_eval["metrics"]), "test": _headline(test_eval["metrics"]) if test_eval else None},
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (art / "model_card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    Detector(art)            # final load with hash verification
    log("run.end", artifact=str(art), shipped=quant["shipped"] if quant else "fp32")
    return art


def _agg_logits(det, text: str) -> np.ndarray:
    """Window logits with the largest injection margin (equals the max-probability window at any temperature)."""
    wins, _, _ = det._windows(text)
    lg = det._logits(wins)
    return lg[int(np.argmax(lg[:, 1] - lg[:, 0]))]


def _by_subset(records, p, thr) -> dict:
    out = {}
    for name in sorted({r.source for r in records}):
        idx = [i for i, r in enumerate(records) if r.source == name]
        y = np.array([records[i].label for i in idx])
        pp = p[idx]
        out[name] = {"n": len(idx), "injection_rows": int(y.sum()),
                     "flagged_at_warn": float((pp > thr["warn"]).mean()), "blocked": float((pp > thr["block"]).mean()),
                     "accuracy_at_warn": float(((pp > thr["warn"]).astype(int) == y).mean())}
    return out


def _headline(s: dict) -> dict:
    w, b = s["at_threshold"]["warn"], s["at_threshold"]["block"]
    return {"n": s["n"], "roc_auc": s["roc_auc"], "average_precision": s["average_precision"], "ece_10": s["ece_10"],
            "warn": {"threshold": w["threshold"], "injection_recall": w["injection"]["recall"],
                     "injection_precision": w["injection"]["precision"], "fpr": w["false_positive_rate"]},
            "block": {"threshold": b["threshold"], "injection_recall": b["injection"]["recall"],
                      "injection_precision": b["injection"]["precision"], "fpr": b["false_positive_rate"]}}


def recalibrate(run_dir: str | Path, max_warn_fpr: float, max_block_fpr: float, reason: str) -> Path:
    """Re-select temperature and thresholds for an existing run's artifact with the current threshold rule, without
    retraining. Writes artifact_recalibrated/ and eval_*_recalibrated.json next to the originals (never overwrites).
    The frozen test set is re-scored with the new thresholds; its threshold-free metrics (ROC-AUC, AP) cannot change."""
    from .runtime import Detector
    run_dir = Path(run_dir)
    cfg = TrainConfig(**{k: v for k, v in json.loads((run_dir / "config.json").read_text()).items()
                         if k in TrainConfig.__dataclass_fields__})
    cfg.max_warn_fpr, cfg.max_block_fpr = max_warn_fpr, max_block_fpr
    src, dst = run_dir / "artifact", run_dir / "artifact_recalibrated"
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    det = Detector(src)                          # verifies the original artifact's hashes
    dev, _ = D.load_records(cfg.dev_file)
    test = D.load_records(cfg.test_file)[0] if cfg.test_file else None
    lg = np.array([_agg_logits(det, r.text) for r in dev])
    y = np.array([r.label for r in dev])
    temp = M.fit_temperature(y, lg)
    p = M.softmax(lg, temp)[:, 1]
    thr = M.select_thresholds(y, p, cfg.block_min_precision, cfg.warn_min_recall, max_warn_fpr, max_block_fpr)
    dev_eval = {"temperature": temp, "selected_thresholds": thr, "reason": reason,
                "metrics": M.summary(y, p, {"warn": thr["warn"], "block": thr["block"], "argmax_0_5": 0.5}),
                "by_subset": _by_subset(dev, p, thr)}
    (run_dir / "eval_dev_recalibrated.json").write_text(json.dumps(dev_eval, indent=2), encoding="utf-8")
    test_eval = None
    if test is not None:
        lt = np.array([_agg_logits(det, r.text) for r in test])
        yt = np.array([r.label for r in test])
        pt = M.softmax(lt, temp)[:, 1]
        test_eval = {"metrics": M.summary(yt, pt, {"warn": thr["warn"], "block": thr["block"], "argmax_0_5": 0.5}),
                     "by_subset": _by_subset(test, pt, thr), "reason": reason}
        (run_dir / "eval_test_recalibrated.json").write_text(json.dumps(test_eval, indent=2), encoding="utf-8")
    shutil.copytree(src, dst)
    card = json.loads((dst / "model_card.json").read_text())
    card.setdefault("revisions", []).append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reason": reason,
                                             "previous_thresholds": card["thresholds"], "previous_temperature": card["temperature"]})
    card.update(temperature=temp, thresholds={"warn": thr["warn"], "block": thr["block"],
                                              "source": f"selected on frozen dev set ({len(y)} rows), FPR-bounded rule"},
                metrics={"dev": _headline(dev_eval["metrics"]), "test": _headline(test_eval["metrics"]) if test_eval else None},
                threshold_notes=thr["notes"])
    (dst / "model_card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    Detector(dst)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--recalibrate", metavar="RUN_DIR", help="re-select thresholds for an existing run (no retraining)")
    ap.add_argument("--max-warn-fpr", type=float, default=0.10)
    ap.add_argument("--max-block-fpr", type=float, default=0.01)
    ap.add_argument("--reason", default="threshold rule revised")
    a = ap.parse_args(argv)
    if a.recalibrate:
        print(f"artifact: {recalibrate(a.recalibrate, a.max_warn_fpr, a.max_block_fpr, a.reason)}")
    elif a.config:
        print(f"artifact: {run(TrainConfig.from_json(a.config))}")
    else:
        ap.error("--config or --recalibrate is required")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
