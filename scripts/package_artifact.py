"""Copy a training run's artifact into artifacts/engine2/<name>/ (the folder the app ships with) and record the
pretrained base model's provenance in the card. Refuses to overwrite an existing folder.

    python scripts/package_artifact.py models/runs/<run>/artifact default
"""

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(src: str, name: str) -> None:
    src_p, dst = Path(src), ROOT / "artifacts" / "engine2" / name
    if dst.exists():
        sys.exit(f"{dst} exists; remove it deliberately first")
    card = json.loads((src_p / "model_card.json").read_text())
    base = card.get("base_model")
    if base and (ROOT / base / "PROVENANCE.json").is_file():
        card["base_provenance"] = json.loads((ROOT / base / "PROVENANCE.json").read_text())
    card["packaged"] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "from": str(src_p)}
    dst.mkdir(parents=True)
    for f in ("model.onnx", "tokenizer.json"):
        shutil.copy(src_p / f, dst / f)
    (dst / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    print("packaged", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
