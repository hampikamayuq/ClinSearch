import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.main import _build_pico_query, _dedupe, _paper_flags, _rank_by_evidence


def test_pico_query_uses_structured_fields():
    query = _build_pico_query(
        "diabetes",
        patient="adults with type 2 diabetes",
        intervention="semaglutide",
        comparator="placebo",
        outcome="cardiovascular mortality",
    )
    assert query == "adults with type 2 diabetes AND semaglutide AND placebo AND cardiovascular mortality"


def test_dedupe_prefers_populated_record():
    papers = [
        {"title": "Aspirin for prevention", "year": "2020", "citations": 1},
        {"title": "Aspirin for prevention", "year": "2020", "citations": 10, "abstract": "data", "doi": "10.1/x"},
    ]
    out = _dedupe(papers)
    assert len(out) == 1
    assert out[0]["doi"] == "10.1/x"


def test_rank_best_evidence_first_when_relevant():
    papers = [
        {"title": "Aspirin cohort study", "study_type": "Cohort", "year": "2025", "citations": 100},
        {"title": "Aspirin guideline", "study_type": "Guideline", "year": "2020", "citations": 1},
        {"title": "Unrelated guideline", "study_type": "Guideline", "year": "2026", "citations": 9999},
    ]
    ranked = _rank_by_evidence(papers, "aspirin")
    assert ranked[0]["title"] == "Aspirin guideline"
    assert ranked[-1]["title"] == "Unrelated guideline"


def test_evidence_flags_detect_risk_signals():
    flags = _paper_flags({
        "title": "Retracted preprint about treatment",
        "journal": "medRxiv",
        "study_type": "Cohort",
    })
    types = {f["type"] for f in flags}
    assert {"retracted", "preprint", "low_evidence"} <= types


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
