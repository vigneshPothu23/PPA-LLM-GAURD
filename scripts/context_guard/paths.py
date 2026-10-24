"""Default locations for the context-guard scripts (all overridable on the command line).

  sources  : eval/context_sets/_sources            (or $PPAGUARD_CTX_SOURCES)  - see download_sources.py
  detector : $PPAGUARD_CACHE/protectai-v2          (default ~/.cache/ppaguard) - the app's own ProtectAI folder,
             prepared through ppaguard.engine2.registry exactly as the application does
  minilm   : $PPAGUARD_CACHE/minilm                - sentence-transformers/all-MiniLM-L6-v2 @1110a243 (Apache-2.0)
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("PPAGUARD_CACHE", Path.home() / ".cache" / "ppaguard"))
SOURCES = Path(os.environ.get("PPAGUARD_CTX_SOURCES", ROOT / "eval" / "context_sets" / "_sources"))
MINILM_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_URL = f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{MINILM_REV}/"


def detector_dir(path: str | None = None) -> str:
    if path:
        return path
    d = CACHE / "protectai-v2"
    if not (d / "model_card.json").is_file():
        sys.path.insert(0, str(ROOT / "src"))
        from ppaguard.engine2.registry import prepare_protectai
        prepare_protectai(d)
    return str(d)


def minilm_dir(path: str | None = None) -> str:
    if path:
        return path
    d = CACHE / "minilm"
    for f in ("onnx/model.onnx", "tokenizer.json"):
        target = d / f
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(MINILM_URL + f, headers={"User-Agent": "ppaguard-context-guard"})
            with urllib.request.urlopen(req, timeout=120) as r, open(target.with_suffix(".part"), "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            target.with_suffix(".part").replace(target)
    return str(d)
