"""Loads the Engine 2 detector for the application and reports honestly what was loaded."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..engine2 import Detector, ModelUnavailable
from ..engine2.context import ContextSignal
from ..engine2.context.embed import Embedder
from ..engine2.registry import ROOT, prepare_minilm, resolve

CONTEXT_ARTIFACT = ROOT / "artifacts" / "context_guard" / "guard_full.json"


@dataclass(frozen=True)
class ModelStatus:
    loaded: bool
    model_id: str | None
    provenance: str | None
    detail: dict
    error: str | None = None


def load_detector(environ=None):
    """Returns (Detector or None, ModelStatus). Never substitutes another model; on failure every scan BLOCKs.

    The Engine 2 detector and its context signal (round-1 context guard + MiniLM-L6 embeddings) are one unit: if either
    cannot be loaded, nothing is loaded and the workflow fails closed. PPAGUARD_CONTEXT=off disables the context signal
    explicitly (reported in the status; used only to reproduce the earlier whole-document configuration).
    """
    env = os.environ if environ is None else environ
    try:
        det = Detector(resolve(env), threads=int(env.get("PPAGUARD_THREADS", "2")))
        det.context_signal = None
        if env.get("PPAGUARD_CONTEXT", "on").lower() != "off":
            det.context_signal = load_context(det, env)
        status = det.status() | {"context": det.context_signal.model_id if det.context_signal else "disabled (PPAGUARD_CONTEXT=off)"}
        return det, ModelStatus(True, det.model_id, det.card.provenance, status)
    except (ModelUnavailable, OSError, ValueError) as e:
        return None, ModelStatus(False, None, None, {}, f"{type(e).__name__}: {e}")


def load_context(detector, environ=None) -> ContextSignal:
    env = os.environ if environ is None else environ
    if not CONTEXT_ARTIFACT.is_file():
        raise ModelUnavailable(f"context guard artifact missing: {CONTEXT_ARTIFACT}")
    minilm = prepare_minilm(Path(env.get("PPAGUARD_CACHE", Path.home() / ".cache" / "ppaguard")) / "minilm")
    try:
        embedder = Embedder(minilm, threads=int(env.get("PPAGUARD_THREADS", "2")))
    except Exception as e:
        raise ModelUnavailable(f"MiniLM embedder could not be loaded: {type(e).__name__}: {e}") from e
    return ContextSignal(CONTEXT_ARTIFACT, detector, embedder)
