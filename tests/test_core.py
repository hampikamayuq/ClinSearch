import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import backend.main as main
from backend.main import _build_pico_query, _dedupe, _paper_flags, _rank_by_evidence, app


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


def test_health_reports_monitoring_fields():
    client = TestClient(app)
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "latency_ms" in data
    assert "database" in data
    assert "gemini" in data["ai"]


def test_session_and_quota_endpoints():
    client = TestClient(app)
    created = client.post("/api/session", json={"action": "create", "data": {"topic": "test"}}).json()
    assert created["id"]
    got = client.post("/api/session", json={"action": "get", "session_id": created["id"]}).json()
    assert got["id"] == created["id"]
    quota = client.get("/api/quota").json()
    assert quota["unlimited"] is True


def test_ai_tool_returns_provider_error_without_keys():
    client = TestClient(app)
    res = client.post("/api/ai-tool", json={"tool": "clinical_bottom_line", "papers": [], "query": "test"})
    assert res.status_code in {429, 502, 503}


def test_search_endpoint_ranks_and_returns_pico_query():
    client = TestClient(app)
    fake_pubmed = AsyncMock(return_value=[
        {"id": "1", "title": "Drug cohort adults", "study_type": "Cohort", "year": "2024", "source": "PubMed"},
        {"id": "2", "title": "Drug guideline adults", "study_type": "Guideline", "year": "2020", "source": "PubMed"},
    ])
    with patch.object(main, "_pubmed_search", fake_pubmed):
        data = client.get(
            "/api/search?q=drug&sources=pubmed&pico=true&patient=adults&intervention=drug&outcome=mortality"
        ).json()
    assert data["pico_query"] == "adults AND drug AND mortality"
    assert data["results"][0]["study_type"] == "Guideline"


def test_ai_tool_success_path_is_cacheable_with_provider():
    client = TestClient(app)
    with patch.object(main, "GROQ_API_KEY", "test"), patch.object(
        main, "call_groq", AsyncMock(return_value="## Clinical Bottom Line\nCited [[1] A, 2024](https://x)")
    ):
        payload = {
            "tool": "clinical_bottom_line",
            "query": "drug",
            "papers": [{"id": "p1", "title": "Drug trial", "year": "2024", "url": "https://x"}],
        }
        res = client.post("/api/ai-tool", json=payload)
    assert res.status_code == 200
    assert res.json()["tool"] == "clinical_bottom_line"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
