"""Download the external sources for the context-guard datasets at pinned revisions (about 520 MB).

    python scripts/context_guard/download_sources.py [--dest eval/context_sets/_sources]

Layout written under --dest (default eval/context_sets/_sources, git-ignored):
  llmail/   labelled_unique_submissions_phase1.json, emails_for_fp_tests.json   (unseen_email)
  llmail2/  labelled_unique_submissions_phase2.json                             (unseen_email_2)
  enron/    chunk_0..9.bin  byte ranges [i*98e6, i*98e6+3e6]                    (unseen_email)
  enron2/   chunk_0..9.bin  byte ranges [i*98e6+49e6, i*98e6+51e6]              (unseen_email_2)
  enron3/   chunk_0..9.bin  byte ranges [i*98e6+24.5e6, +2e6]                  (round-2 test sets)
  enron4/   chunk_0..9.bin  byte ranges [i*98e6+73.5e6, +2e6]                  (round-2 development set)
  enron5/   chunk_0..9.bin  byte ranges [i*98e6+36.75e6, +2e6]                 (final held-out set)

Sources (licences as published on Hugging Face):
  microsoft/llmail-inject-challenge @1063bdf01ec8762b812d5e06ee768a06faa5a6f7 (MIT)
  LLM-PBE/enron-email               @a4761f17a981d79bd286bfff7dd0d53645aa5027 (Apache-2.0; Enron corpus)
Only public files are fetched; no token is needed. Existing complete files are skipped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLMAIL = "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/1063bdf01ec8762b812d5e06ee768a06faa5a6f7/data/"
ENRON = ("https://huggingface.co/datasets/LLM-PBE/enron-email/resolve/"
         "a4761f17a981d79bd286bfff7dd0d53645aa5027/enron_email_all.jsonl")


def fetch(url: str, dest: Path, byte_range: tuple[int, int] | None = None) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest} (exists)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ppaguard-context-guard"})
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
        if byte_range and r.status != 206:
            raise SystemExit(f"server ignored the byte range for {url} (HTTP {r.status})")
        shutil.copyfileobj(r, fh, 1 << 20)
    tmp.replace(dest)
    print(f"got  {dest} ({dest.stat().st_size:,} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "eval/context_sets/_sources"))
    d = Path(ap.parse_args().dest)
    fetch(LLMAIL + "emails_for_fp_tests.json", d / "llmail/emails_for_fp_tests.json")
    fetch(LLMAIL + "labelled_unique_submissions_phase1.json", d / "llmail/labelled_unique_submissions_phase1.json")
    fetch(LLMAIL + "labelled_unique_submissions_phase2.json", d / "llmail2/labelled_unique_submissions_phase2.json")
    for i in range(10):
        s = i * 98_000_000
        fetch(ENRON, d / f"enron/chunk_{i}.bin", (s, s + 3_000_000))
        fetch(ENRON, d / f"enron2/chunk_{i}.bin", (s + 49_000_000, s + 51_000_000))
        fetch(ENRON, d / f"enron3/chunk_{i}.bin", (s + 24_500_000, s + 26_500_000))
        fetch(ENRON, d / f"enron4/chunk_{i}.bin", (s + 73_500_000, s + 75_500_000))
        fetch(ENRON, d / f"enron5/chunk_{i}.bin", (s + 36_750_000, s + 38_750_000))
    print("done:", d)


if __name__ == "__main__":
    sys.exit(main())
