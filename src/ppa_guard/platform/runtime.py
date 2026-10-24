"""Loads Engine 2 and reports honestly what was loaded. Never substitutes another model or a canned prediction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..engine2 import HF_REPO, HF_REVISION, DebertaDetector, download_model


@dataclass(frozen=True)
class ModelStatus:
    loaded: bool
    source: str            # "pinned-download" | "local-directory" | "none"
    model_id: str | None
    verified: str          # plain-language provenance statement shown in the UI
    error: str | None = None


def load_detector(model_dir: str | None = None, threads: int | None = None):
    """Returns (detector or None, ModelStatus). On any failure the detector is None and the gateway blocks."""
    threads = threads or int(os.environ.get("PPA_THREADS", "2"))
    expected = os.environ.get("PPA_MODEL_SHA256") or None
    try:
        if model_dir:
            det = DebertaDetector(Path(model_dir), revision=HF_REVISION, threads=threads, expected_sha256=expected,
                                  model_id=os.environ.get("PPA_MODEL_ID") or f"local:{Path(model_dir).name}")
            how = ("checksum matched PPA_MODEL_SHA256" if expected
                   else "local directory - provenance not verified by the app")
            return det, ModelStatus(True, "local-directory", det.model_id, how)
        path = download_model(revision=HF_REVISION)
        det = DebertaDetector(path, revision=HF_REVISION, threads=threads, expected_sha256=expected)
        return det, ModelStatus(True, "pinned-download", det.model_id,
                                f"downloaded from {HF_REPO} at pinned revision {HF_REVISION[:12]}"
                                + (", checksum matched" if expected else ""))
    except Exception as exc:  # reported in the UI; every scan then BLOCKs (fail closed)
        return None, ModelStatus(False, "none", None, "not loaded", f"{type(exc).__name__}: {exc}")
