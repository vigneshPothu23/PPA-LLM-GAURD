"""Loads the Engine 2 detector for the application and reports honestly what was loaded."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..engine2 import Detector, ModelUnavailable
from ..engine2.context import ContextSignal
from ..engine2.registry import resolve, prepare_minilm


@dataclass(frozen=True)
class ModelStatus:
    loaded: bool
    model_id: str | None
    provenance: str | None
    detail: dict
    error: str | None = None


def load_detector(environ=None):
    """Returns (Detector or None, ModelStatus). Never substitutes another model; on failure every scan BLOCKs.

    Also attempts to load the MiniLM sentence transformer and context guard for context-based protection.
    If context guard is unavailable, the detector still loads and context scoring is skipped.
    """
    env = os.environ if environ is None else environ
    try:
        det = Detector(resolve(env), threads=int(env.get("PPAGUARD_THREADS", "2")))

        # Try to load MiniLM and context guard (optional; skip if unavailable)
        try:
            minilm_dir = prepare_minilm(
                Path(env.get("PPAGUARD_CACHE", Path.home() / ".cache" / "ppaguard")) / "minilm"
            )
            # Load the embedder from the MiniLM model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                # sentence_transformers not available; context guard will be None
                det.context_signal = None
            else:
                embedder = SentenceTransformer(str(minilm_dir), device="cpu")

                # Create the context signal
                context_artifact = Path(__file__).resolve().parent.parent / "artifacts" / "context_guard" / "guard_full.json"
                if context_artifact.is_file():
                    try:
                        context_signal = ContextSignal(context_artifact, det, embedder)
                        det.context_signal = context_signal
                    except Exception:
                        # Context guard loading failed; skip it
                        det.context_signal = None
        except (ModelUnavailable, OSError):
            # MiniLM not available; context guard will be None
            det.context_signal = None

        return det, ModelStatus(True, det.model_id, det.card.provenance, det.status())
    except (ModelUnavailable, OSError, ValueError) as e:
        return None, ModelStatus(False, None, None, {}, f"{type(e).__name__}: {e}")
