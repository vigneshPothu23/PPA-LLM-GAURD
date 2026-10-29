"""Where Engine 2 gets its model from.

Resolution (first match wins):
  1. PPAGUARD_MODEL_DIR     a folder with model_card.json, model.onnx and tokenizer.json (e.g. a training artifact)
  2. PPAGUARD_MODEL=protectai  (the default when nothing is set)
                            download protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0) at a pinned revision
                            from Hugging Face (free, no account) and describe it with a model card
  3. PPAGUARD_MODEL=<name>  a bundled artifact in artifacts/engine2/<name>/ produced by this repository's pipeline
                            (roberta-6l-experimental, scratch-baseline)
There is no silent fallback: if the chosen model is unavailable the workflow blocks every request (fail closed).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .runtime import ModelUnavailable, _sha256

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts" / "engine2"
DEFAULT_MODEL = "protectai"      # EXPERIMENTAL default, not validated - see docs/v5/EVALUATION.md

PROTECTAI = {
    "repo": "protectai/deberta-v3-base-prompt-injection-v2",
    "revision": "89b085cd330414d3e7d9dd787870f315957e1e9f",
    "files": ("onnx/model.onnx", "onnx/tokenizer.json", "onnx/config.json"),
    "license": "Apache-2.0",
    # provider guidance: 0.5 is the model's argmax; 0.92 is the default threshold in ProtectAI's LLM Guard
    "thresholds": {"warn": 0.5, "block": 0.92, "source": "provider defaults (not calibrated by PPA Guard)"},
}

MINILM = {
    "repo": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "license": "Apache-2.0",
    # the ONNX export used when the context guard was fitted and selected (scripts/context_guard/paths.py)
    "files": {"onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
              "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"},
}


def prepare_protectai(dest: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Downloads the pinned ONNX export and writes a model card next to it. Network access to huggingface.co needed.

    If the model card already exists, updates it with any new evaluation.json thresholds.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ModelUnavailable("huggingface_hub is not installed") from e
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    card_path = dest / "model_card.json"

    # If card already cached, load it; otherwise create a new one
    if card_path.is_file():
        card = json.loads(card_path.read_text(encoding="utf-8"))
    else:
        try:
            paths = {f: Path(hf_hub_download(PROTECTAI["repo"], f, revision=PROTECTAI["revision"], cache_dir=cache_dir))
                     for f in PROTECTAI["files"]}
        except Exception as e:
            raise ModelUnavailable(f"could not download {PROTECTAI['repo']}: {type(e).__name__}: {e}") from e
        cfg = json.loads(paths["onnx/config.json"].read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in cfg["id2label"].items()}
        labels = [id2label[i].lower() for i in sorted(id2label)]
        if "injection" not in labels:
            raise ModelUnavailable(f"unexpected labels {labels}")
        model, tok = str(paths["onnx/model.onnx"]), str(paths["onnx/tokenizer.json"])
        card = {"model_id": f"{PROTECTAI['repo']}@{PROTECTAI['revision'][:12]}", "provenance": "pretrained-external",
                "labels": labels, "injection_index": labels.index("injection"), "max_length": 512, "window_stride": 256,
                "max_windows": 8, "temperature": 1.0, "thresholds": PROTECTAI["thresholds"],
                "model_file": model, "tokenizer_file": tok,
                "files": {model: _sha256(Path(model)), tok: _sha256(Path(tok))},
                "source": {"repo": PROTECTAI["repo"], "revision": PROTECTAI["revision"], "license": PROTECTAI["license"],
                           "note": "file hashes recorded at first download (trust on first use) from the pinned revision"}}

    # Always check and merge evaluation.json, even if card was cached
    ev_path = ARTIFACTS / "protectai-v2" / "evaluation.json"
    if ev_path.is_file():   # thresholds and measured results from this repository's frozen-set evaluation
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        card.update(thresholds=ev["thresholds"], status=ev["status"], limitations=ev["limitations"], metrics=ev["metrics"],
                    threshold_notes=ev["threshold_notes"], provider_thresholds=ev["provider_thresholds"],
                    evaluation_source=ev["generated_from"])

    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return dest


def prepare_minilm(dest: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Places the pinned MiniLM-L6 ONNX export (context guard embeddings) in `dest` and verifies its SHA-256 hashes."""
    dest = Path(dest)
    missing = [f for f in MINILM["files"] if not (dest / f).is_file()]
    if missing:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover
            raise ModelUnavailable("huggingface_hub is not installed") from e
        try:
            for f in missing:
                src = Path(hf_hub_download(MINILM["repo"], f, revision=MINILM["revision"], cache_dir=cache_dir))
                (dest / f).parent.mkdir(parents=True, exist_ok=True)
                (dest / f).write_bytes(src.read_bytes())
        except Exception as e:
            raise ModelUnavailable(f"could not download {MINILM['repo']}: {type(e).__name__}: {e}") from e
    for f, expected in MINILM["files"].items():
        if _sha256(dest / f) != expected:
            raise ModelUnavailable(f"MiniLM {f} checksum mismatch")
    return dest


def bundled(name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", name) or ".." in name:
        raise ModelUnavailable(f"invalid model name {name!r}")
    d = ARTIFACTS / name
    if not (d / "model_card.json").is_file():
        raise ModelUnavailable(f"no bundled Engine 2 artifact named {name!r} in artifacts/engine2/")
    return d


def resolve(environ=None) -> Path:
    env = os.environ if environ is None else environ
    if env.get("PPAGUARD_MODEL_DIR"):
        return Path(env["PPAGUARD_MODEL_DIR"])
    choice = (env.get("PPAGUARD_MODEL") or DEFAULT_MODEL).strip()
    if choice.lower() == "protectai":
        return prepare_protectai(Path(env.get("PPAGUARD_CACHE", Path.home() / ".cache" / "ppaguard")) / "protectai-v2")
    return bundled(choice)
