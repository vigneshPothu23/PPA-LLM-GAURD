"""Engine 2: transformer prompt-injection detector (independent implementation). See docs/v5/ENGINE2.md.

Inference (runtime.py) needs only numpy, onnxruntime and tokenizers. Training (train.py) additionally needs torch and
transformers (requirements-train.txt) and is never imported by the application.
"""

from .runtime import PROVENANCE, Detection, Detector, InferenceError, ModelCard, ModelUnavailable

__all__ = ["Detection", "Detector", "InferenceError", "ModelCard", "ModelUnavailable", "PROVENANCE"]
