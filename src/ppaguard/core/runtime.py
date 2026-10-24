"""Loads the Engine 2 detector for the application and reports honestly what was loaded."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..engine2 import Detector, ModelUnavailable
from ..engine2.registry import resolve


@dataclass(frozen=True)
class ModelStatus:
    loaded: bool
    model_id: str | None
    provenance: str | None
    detail: dict
    error: str | None = None


def load_detector(environ=None):
    """Returns (Detector or None, ModelStatus). Never substitutes another model; on failure every scan BLOCKs."""
    env = os.environ if environ is None else environ
    try:
        det = Detector(resolve(env), threads=int(env.get("PPAGUARD_THREADS", "2")))
        return det, ModelStatus(True, det.model_id, det.card.provenance, det.status())
    except (ModelUnavailable, OSError, ValueError) as e:
        return None, ModelStatus(False, None, None, {}, f"{type(e).__name__}: {e}")
