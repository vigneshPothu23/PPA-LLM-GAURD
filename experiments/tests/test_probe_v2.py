"""Checks for option_c_probe_v2 strategies (run on the random STUB model: code paths only)."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("probe", ROOT / "experiments/scripts/option_c_probe_v2.py")
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
pytest.importorskip("onnxruntime")
from ppa_guard.engine2 import DebertaDetector  # noqa: E402

STUB = ROOT / "tests/fixtures/stub_model"


def reference_llm_guard_chunks(input_length, chunk_length=256, overlap_length=25):
    # verbatim logic of llm_guard.util.split_text_to_word_chunks (0.3.16)
    if input_length < chunk_length:
        return [(0, input_length)]
    return [(i, min([i + chunk_length, input_length])) for i in range(0, input_length - overlap_length, chunk_length - overlap_length)]


@pytest.mark.parametrize("n", [0, 1, 255, 256, 257, 486, 487, 1000, 4096])
def test_chunks_match_llm_guard(n):
    assert probe.llm_guard_chunks(n) == reference_llm_guard_chunks(n)


@pytest.mark.parametrize("n", [256, 300, 999, 5000])
def test_chunks_cover_whole_text(n):
    spans = probe.llm_guard_chunks(n)
    covered = set()
    for a, b in spans:
        covered.update(range(a, b))
    assert covered == set(range(n))


@pytest.fixture(scope="module")
def st():
    return probe.Strategies(DebertaDetector(STUB, threads=1, model_id="STUB"))


def test_short_input_identical_for_full_and_head_tail(st):
    t = "Summarize this email please. What is the capital of Brazil?"
    assert st.score(t, "FULL")[:2] == st.score(t, "HEAD_TAIL")[:2]


def test_long_input_head_tail_fits_window_and_keeps_both_ends(st):
    t = "alpha " * 400 + "MIDDLE " + "omega " * 400
    s, r, pieces, n = st.score(t, "HEAD_TAIL")
    assert n > st.window and pieces == 1
    offs = st._content_offsets(t)
    head = t[: offs[probe.HEAD - 1][1]]; tail = t[offs[n - probe.TAIL][0]:]
    assert head.startswith("alpha") and tail.rstrip().endswith("omega") and "MIDDLE" not in head + tail
    assert len(st._content_offsets(head + " " + tail)) <= st.window + 2


def test_chunks_preserve_formatting_and_count(st):
    t = "| a | b |\n|---|---|\n" * 40
    s, r, pieces, n = st.score(t, "CHUNKS")
    assert pieces == len(probe.llm_guard_chunks(len(t))) > 1


def test_official_rounded_threshold_semantics():
    assert not (round(0.9249, 2) > 0.92) and (0.9249 > 0.92)   # the v1 probe would have flagged this
    assert round(0.9251, 2) > 0.92


def test_statistics_helpers():
    assert probe.mcnemar_exact(0, 0) == 1.0
    assert probe.mcnemar_exact(10, 0) < 0.01
    lo, hi = probe.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
