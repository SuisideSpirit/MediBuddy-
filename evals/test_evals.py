"""pytest entry point for the eval suite: every case in run_evals.CASES, real graph, real LLM.
For the written report (what's checked, pass criteria, replies) run:  python evals/run_evals.py
"""
import pytest

from run_evals import CASES

pytestmark = pytest.mark.live


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_eval_case(case):
    (ok, detail), _ = case["fn"]()
    assert ok, f"{case['checking']}\nPass looks like: {case['passes_if']}\nGot: {detail}"
