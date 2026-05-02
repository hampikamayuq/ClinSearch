import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import backend.main as main
from backend.main import _build_pico_query, _cache_get, _cache_set, _dedupe, _extract_effect_data, _grade_signal, _grade_summary, _paper_flags, _rank_by_evidence, _tool_cache, _validate_ai_citations, app


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


def test_effect_extraction_detects_ratios_and_nnt_signal():
    data = _extract_effect_data({
        "title": "Drug trial",
        "abstract": "The hazard ratio was 0.72 (95% CI 0.60-0.86). Mortality was 10% vs 15%."
    })
    assert data["measures"][0]["measure"] == "Hazard Ratio"
    assert data["measures"][0]["value"] == 0.72
    assert data["nnt"]["value"] == 20
    assert "NNT/NNH" in data["summary"]


def test_grade_signal_uses_design_flags_and_effects():
    high = _grade_signal({"study_type": "Meta-Analysis", "effect_data": {"summary": "RR 0.8"}})
    low = _grade_signal({"study_type": "Cohort", "evidence_flags": [{"type": "low_evidence", "level": "warning"}], "effect_data": {"summary": "NR"}})
    summary = _grade_summary([{"grade_signal": high}, {"grade_signal": low}])
    assert high["certainty"] == "high"
    assert low["certainty"] == "low"
    assert summary["overall"] == "high"


def test_ai_citation_validator_blocks_missing_or_invalid_sources():
    papers = [{"title": "Trial A", "url": "https://example.org/a"}, {"title": "Trial B"}]
    assert _validate_ai_citations("Cited [[1] A, 2024](https://example.org/a)", papers)["ok"] is True
    missing = _validate_ai_citations("No citations here", papers)
    invalid = _validate_ai_citations("Bad [[3] C, 2024](https://z)", papers)
    bad_url = _validate_ai_citations("Bad URL [[1] A, 2024](https://wrong.example)", papers)
    assert missing["missing_citations"] is True
    assert invalid["invalid_citations"][0]["index"] == 3
    assert bad_url["invalid_urls"][0]["index"] == 1


def test_tool_cache_persists_through_memory_clear():
    key = "unit-test-cache-key"
    payload = {"ok": True, "value": 1}
    _cache_set(_tool_cache, key, payload)
    _tool_cache.clear()
    assert _cache_get(_tool_cache, key, 900) == payload


def test_health_reports_monitoring_fields():
    client = TestClient(app)
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "latency_ms" in data
    assert "database" in data
    assert "gemini" in data["ai"]


def test_metrics_reports_cache_and_database_counts():
    client = TestClient(app)
    data = client.get("/metrics").json()
    assert data["status"] == "ok"
    assert "cache" in data
    assert "database" in data
    assert "providers" in data
    assert "endpoints" in data
    assert "persistent_tool_entries" in data["cache"]


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


def test_frontend_keeps_core_workflow_controls():
    html = (ROOT / "frontend" / "index.html").read_text()
    required = [
        "showJournalClubByKey",
        "showPatientSummaryByKey",
        "showPICOByKey",
        "showEvidenceTable",
        "setMode('stats')",
        "saveCurrentSearch",
        "loadProviderStatus",
        "showSystemDashboard",
        "mt-system",
        "effectSummaryHtml",
        "gradeHtml",
    ]
    for marker in required:
        assert marker in html


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
