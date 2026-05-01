"""
ClinSearch v3 — Backend API
AI: Gemini 2.0 Flash (free) → Groq (free fallback) → User key (Claude/OpenAI)
Auth: Google OAuth
Research: PubMed, Semantic Scholar, OpenAlex, ClinicalTrials, Unpaywall
"""

import sqlite3
import logging
import re
logger = logging.getLogger("clinsearch")
import time
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, List
import httpx, os, json, asyncio, time
import xml.etree.ElementTree as ET
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import jwt, secrets
from datetime import datetime, timedelta

app = FastAPI(title="ClinSearch API v3", version="3.0.0")

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("FRONTEND_URL", "http://localhost:3000").split(",")]
ALLOWED_ORIGINS += ["http://localhost:3000", "http://localhost:5500", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Environment ───────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
JWT_SECRET           = os.environ.get("JWT_SECRET", secrets.token_hex(32))
FRONTEND_URL         = os.environ.get("FRONTEND_URL", "https://clinsearch.vercel.app")

# AI keys (server-side free tier)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")    # Free tier
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")      # Free fallback

# Research keys
PUBMED_EMAIL   = os.environ.get("PUBMED_EMAIL", "api@clinsearch.app")
PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY", "")
S2_API_KEY     = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

TIMEOUT = 20

FREE_DAILY_LIMIT = None
_DB_PATH = "/tmp/clinsearch.db"

def _init_db():
    conn = sqlite3.connect(_DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS quotas (
            user_id TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0,
            date TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            topic TEXT,
            papers TEXT DEFAULT '[]',
            notes TEXT DEFAULT '[]',
            messages TEXT DEFAULT '[]',
            created TEXT,
            updated TEXT
        );
        CREATE TABLE IF NOT EXISTS workspace (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            paper_json TEXT,
            notes TEXT DEFAULT '',
            folder TEXT DEFAULT 'default',
            saved_at TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            query TEXT,
            sources TEXT DEFAULT 'pubmed',
            last_check TEXT,
            created TEXT
        );
    """)
    conn.commit()
    conn.close()

_init_db()

def _migrate_db():
    conn = sqlite3.connect(_DB_PATH)
    for col, default in [("messages", "'[]'"), ("updated", "''")]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT DEFAULT {default}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.close()

_migrate_db()


# ── Models ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: List[dict]
    session_token: Optional[str] = None     # for quota tracking (Google login)
    # Note: user API keys are handled directly in the browser, never sent here

class SearchRequest(BaseModel):
    query: str
    sources: List[str] = ["pubmed", "semantic_scholar", "openalex"]
    max_per_source: int = 5
    year_from: Optional[int] = None
    open_access: bool = False
    reviews_only: bool = False


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "ai": {
            "gemini": bool(GEMINI_API_KEY),
            "groq":   bool(GROQ_API_KEY),
        }
    }


# ── Google OAuth ──────────────────────────────────────────────────────────────
@app.get("/auth/google")
async def google_login():
    """Redirect to Google OAuth consent screen."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(400, "Google OAuth not configured")
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  f"{os.environ.get('BACKEND_URL','')}/auth/google/callback",
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "online",
    }
    from urllib.parse import urlencode
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(url)


@app.get("/auth/google/callback")
async def google_callback(code: str):
    """Exchange code for tokens, return JWT session."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  f"{os.environ.get('BACKEND_URL','')}/auth/google/callback",
                "grant_type":    "authorization_code",
            }
        )
        tokens = r.json()

        r2 = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {tokens.get('access_token','')}"}
        )
        user = r2.json()

    payload = {
        "sub":   user.get("sub", ""),
        "email": user.get("email", ""),
        "name":  user.get("name", ""),
        "pic":   user.get("picture", ""),
        "exp":   datetime.utcnow() + timedelta(days=30),
    }
    session_token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    return RedirectResponse(
        f"{FRONTEND_URL}/auth/callback?token={session_token}&name={user.get('name','')}&email={user.get('email','')}"
    )


@app.post("/auth/verify")
async def verify_token(request: Request):
    """Verify a Google ID token (for direct Google Sign-In)."""
    body = await request.json()
    credential = body.get("credential", "")
    try:
        idinfo = id_token.verify_oauth2_token(credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        payload = {
            "sub":   idinfo["sub"],
            "email": idinfo["email"],
            "name":  idinfo.get("name", ""),
            "pic":   idinfo.get("picture", ""),
            "exp":   datetime.utcnow() + timedelta(days=30),
        }
        session_token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
        return {"token": session_token, "user": {"name": idinfo.get("name"), "email": idinfo.get("email"), "picture": idinfo.get("picture")}}
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")


def get_user(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        return None


def check_quota(user_id: str) -> bool:
    if FREE_DAILY_LIMIT is None:
        return True
    today = datetime.utcnow().date().isoformat()
    conn = sqlite3.connect(_DB_PATH)
    row = conn.execute("SELECT count, date FROM quotas WHERE user_id=?", (user_id,)).fetchone()
    if not row or row[1] != today:
        conn.execute("INSERT OR REPLACE INTO quotas VALUES (?,1,?)", (user_id, today))
        conn.commit(); conn.close()
        return True
    if row[0] >= FREE_DAILY_LIMIT:
        conn.close()
        return False
    conn.execute("UPDATE quotas SET count=count+1 WHERE user_id=?", (user_id,))
    conn.commit(); conn.close()
    return True


def ai_provider_error_response(provider: str, exc: Exception):
    status = getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc)
    if status == 429 or "429" in text or "Too Many Requests" in text:
        return JSONResponse(
            status_code=429,
            content={
                "error": "provider_rate_limit",
                "message": "The free AI provider is rate-limited right now. Try again in a few minutes or use your own API key.",
                "provider": provider,
            },
        )
    return JSONResponse(
        status_code=502,
        content={
            "error": "provider_error",
            "message": f"{provider} error: {text}",
            "provider": provider,
        },
    )


# ── AI Chat ───────────────────────────────────────────────────────────────────
_ip_requests: dict = {}

@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    # Basic IP rate limit: 60 req/min per IP
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _ip_requests.get(client_ip, [])
    window = [t for t in window if now - t < 60]
    if len(window) >= 60:
        raise HTTPException(429, "Too many requests. Please wait a minute.")
    window.append(now)
    _ip_requests[client_ip] = window

    user_id = "anonymous"
    if req.session_token:
        user = get_user(req.session_token)
        if user:
            user_id = user.get("sub", "anonymous")

    if not check_quota(user_id):
        return JSONResponse(
            status_code=429,
            content={
                "error": "daily_limit",
                "message": f"You've reached the {FREE_DAILY_LIMIT} free queries/day limit.",
                "options": ["Add your Claude or OpenAI key for unlimited use", "Come back tomorrow"]
            }
        )

    # Agentic: inject real papers into context before calling AI
    messages = await _inject_search_context(req.messages)

    if GEMINI_API_KEY:
        try:
            response = await call_gemini(messages, GEMINI_API_KEY)
            return {"response": response, "provider": "gemini-flash", "quota_used": True}
        except Exception as e:
            if "quota" not in str(e).lower() and "rate" not in str(e).lower():
                raise HTTPException(500, f"Gemini error: {e}")

    if GROQ_API_KEY:
        try:
            response = await call_groq(messages, GROQ_API_KEY)
            return {"response": response, "provider": "groq-llama", "quota_used": True}
        except Exception as e:
            return ai_provider_error_response("groq", e)

    raise HTTPException(503, "No AI provider available. Please add your API key.")


_SEARCH_KEYWORDS = re.compile(
    r'\b(evidence|papers?|studi(es|y)|trial|systematic|meta.?analysis|treatment|efficacy|'
    r'versus|compar|review|biolog|drug|therap|intervention|outcome|rct|placebo|'
    r'guidelines?|recommend|adverse|safety|dose|mechanism)\b', re.I
)

async def _inject_search_context(messages: list) -> list:
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not _SEARCH_KEYWORDS.search(last_user):
        return messages
    try:
        results = await asyncio.gather(
            _pubmed_search(last_user, 6),
            _s2_search(last_user, 5),
            return_exceptions=True
        )
        papers = []
        for r in results:
            if isinstance(r, list):
                papers.extend(r)
        if not papers:
            return messages
        context = "\n\n[REAL PAPERS RETRIEVED — you MUST cite these using [1],[2] etc. and extract their actual data: study design, N, effect sizes, CI, p-values, NNT. Do NOT invent additional papers.]\n"
        for i, p in enumerate(papers[:9], 1):
            authors = ", ".join((p.get("authors") or [])[:3])
            abstract = (p.get('abstract') or '')[:700]
            context += (f"\n[{i}] \"{p['title']}\"\n"
                       f"    Authors: {authors} | Year: {p.get('year','')} | Journal: {p.get('journal','')} | Citations: {p.get('citations','')}\n"
                       f"    Abstract: {abstract}\n"
                       f"    URL: {p.get('url','')}\n")
        enriched = []
        for m in messages:
            if m["role"] == "system":
                enriched.append({**m, "content": m["content"] + context})
            else:
                enriched.append(m)
        return enriched
    except Exception as e:
        logger.warning(f"Agentic search inject error: {e}")
        return messages


async def call_gemini(messages: list, key: str) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    history = [m for m in messages if m["role"] != "system"]

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
            json={
                "system_instruction": {"parts": [{"text": system}]} if system else None,
                "contents": [
                    {"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]}
                    for m in history
                ],
                "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.3}
            }
        )
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def call_groq(messages: list, key: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [m for m in messages if m["role"] != "system"] if not any(m["role"]=="system" for m in messages) else messages,
                "max_tokens": 2048,
                "temperature": 0.3,
            }
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ── Research tools (REST endpoints) ──────────────────────────────────────────
@app.get("/api/search")
async def search_all(
    q: str,
    n: int = 5,
    offset: int = 0,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    open_access: bool = False,
    reviews_only: bool = False,
    study_type: Optional[str] = None,
    sources: Optional[str] = None,
):
    """Parallel search across selected sources with dedup and study-type classification."""
    requested = set((sources or "pubmed,s2,openalex").split(","))
    pubmed_q = q + (" AND (Review[pt] OR Meta-Analysis[pt])" if reviews_only else "")

    coros = []
    if "pubmed" in requested:
        coros.append(_pubmed_search(pubmed_q, n, year_from, open_access))
    if "s2" in requested:
        coros.append(_s2_search(q, n, year_from, open_access))
    if "openalex" in requested:
        coros.append(_openalex_search(q, n, year_from, open_access,
                                      "review" if reviews_only else None))
    if "europepmc" in requested or "cochrane" in requested:
        coros.append(_europepmc_search(q, n, offset, year_from, year_to,
                                       open_access, "cochrane" in requested or reviews_only))
    if "who" in requested:
        coros.append(_who_iris_search(q, n))

    if not coros:
        coros = [
            _pubmed_search(pubmed_q, n, year_from, open_access),
            _s2_search(q, n, year_from, open_access),
            _openalex_search(q, n, year_from, open_access, None),
        ]

    results = await asyncio.gather(*coros, return_exceptions=True)
    combined = []
    for r in results:
        if isinstance(r, list):
            combined.extend(r)

    for p in combined:
        if not p.get("study_type"):
            p["study_type"] = _classify_study_type(
                p.get("title", ""), p.get("abstract", "")
            )

    deduped = _dedupe(combined)

    if study_type and study_type.lower() not in ("all", ""):
        deduped = [p for p in deduped
                   if (p.get("study_type") or "").lower() == study_type.lower()]

    return {"results": deduped, "total": len(deduped), "query": q,
            "sources_used": list(requested)}


@app.get("/api/pubmed")
async def pubmed(q: str, n: int = 10, year_from: Optional[int] = None, free_only: bool = False):
    return {"results": await _pubmed_search(q, n, year_from, free_only), "source": "PubMed"}


@app.get("/api/europepmc")
async def europepmc_endpoint(q: str, n: int = 10, offset: int = 0,
                              year_from: Optional[int] = None, year_to: Optional[int] = None,
                              oa_only: bool = False, systematic_only: bool = False):
    results = await _europepmc_search(q, n, offset, year_from, year_to, oa_only, systematic_only)
    return {"results": results, "total": len(results), "source": "Europe PMC"}


@app.get("/api/mesh-suggest")
async def mesh_suggest(q: str):
    """MeSH term autocomplete via NLM API."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                "https://id.nlm.nih.gov/mesh/suggest",
                params={"label": q, "format": "json"}
            )
            if r.status_code == 200:
                hits = r.json().get("hits", [])
                terms = [h.get("label", "") for h in hits[:10] if h.get("label")]
                return {"suggestions": terms}
    except Exception as e:
        logger.warning(f"MeSH suggest error: {e}")
    return {"suggestions": []}


@app.get("/api/check-journal")
async def check_journal(journal: str):
    """Check journal quality via DOAJ API."""
    if not journal or len(journal) < 3:
        return {"indexed": None, "doaj": None, "score": "unknown"}
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                "https://doaj.org/api/search/journals",
                params={"q": journal, "pageSize": 1}
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("total", 0) > 0:
                    return {"indexed": True, "doaj": True, "score": "trusted"}
    except Exception as e:
        logger.warning(f"DOAJ check error: {e}")
    return {"indexed": False, "doaj": False, "score": "unknown"}


async def _who_iris_search(q: str, n: int = 10) -> list:
    """Search WHO IRIS knowledge base (guidelines, technical reports, policy briefs)."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(
                "https://iris.who.int/rest/search",
                params={"query": q, "expand": "metadata", "limit": min(n, 20), "offset": 0}
            )
            if r.status_code != 200:
                return []
            items = r.json()
        results = []
        for item in (items if isinstance(items, list) else []):
            meta = {}
            for m in (item.get("metadata") or []):
                k = m.get("key", "")
                v = m.get("value", "")
                if k and v and k not in meta:
                    meta[k] = v
            title = meta.get("dc.title", "")
            if not title:
                continue
            abstract = (meta.get("dc.description.abstract", "")
                        or meta.get("dc.description", ""))
            year = (meta.get("dc.date.issued", "") or "")[:4]
            doi = meta.get("dc.identifier.doi", "")
            handle = meta.get("dc.identifier.uri", "")
            if not handle:
                h = item.get("handle", "")
                handle = f"https://iris.who.int/handle/{h}" if h else ""
            author = (meta.get("dc.creator", "")
                      or meta.get("dc.contributor.author", ""))
            results.append({
                "id": doi or handle,
                "title": title,
                "authors": [author] if author else ["World Health Organization"],
                "year": year,
                "journal": "WHO Publications",
                "abstract": (abstract or "")[:500],
                "doi": doi,
                "url": handle,
                "source": "WHO IRIS",
                "study_type": _classify_study_type(title, abstract or ""),
                "isOpenAccess": True,
            })
        return results
    except Exception as e:
        logger.warning(f"WHO IRIS search error: {e}")
        return []


@app.get("/api/who-iris")
async def who_iris_endpoint(q: str, n: int = 10):
    results = await _who_iris_search(q, n)
    return {"results": results, "total": len(results), "source": "WHO IRIS"}


@app.get("/api/trials")
async def trials(q: str, n: int = 10, status: Optional[str] = None, phase: Optional[str] = None):
    params = {"query.term": q, "pageSize": min(n,25), "format": "json",
              "fields": "NCTId,BriefTitle,OverallStatus,Phase,StartDate,CompletionDate,EnrollmentCount,BriefSummary,Condition,InterventionName,LeadSponsorName"}
    if status: params["filter.overallStatus"] = status
    if phase:  params["filter.phase"] = phase
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get("https://clinicaltrials.gov/api/v2/studies", params=params)
        r.raise_for_status()
        studies = r.json().get("studies", [])
    results = []
    for s in studies:
        proto = s.get("protocolSection", {})
        id_m  = proto.get("identificationModule", {})
        st_m  = proto.get("statusModule", {})
        ds_m  = proto.get("descriptionModule", {})
        de_m  = proto.get("designModule", {})
        sp_m  = proto.get("sponsorCollaboratorsModule", {})
        nct   = id_m.get("nctId", "")
        results.append({
            "id": nct, "title": id_m.get("briefTitle",""), "status": st_m.get("overallStatus",""),
            "phase": de_m.get("phases",[]), "sponsor": (sp_m.get("leadSponsor") or {}).get("name",""),
            "enrollment": de_m.get("enrollmentInfo",{}).get("count",""),
            "start": st_m.get("startDateStruct",{}).get("date",""),
            "end": st_m.get("completionDateStruct",{}).get("date",""),
            "summary": (ds_m.get("briefSummary","") or "")[:400],
            "url": f"https://clinicaltrials.gov/study/{nct}",
        })
    return {"results": results, "total": len(results)}


@app.get("/api/journal")
async def journal_impact(name: str):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get("https://api.openalex.org/sources",
            params={"search": name, "per-page": 3, "mailto": PUBMED_EMAIL,
                    "select": "display_name,issn_l,cited_by_count,works_count,summary_stats,is_oa,host_organization_name"})
        r.raise_for_status()
        results = r.json().get("results", [])
    out = []
    for j in results[:3]:
        stats = j.get("summary_stats") or {}
        if2yr = stats.get("2yr_mean_citedness", 0) or 0
        q = "Q1" if if2yr >= 8 else "Q1/Q2" if if2yr >= 4 else "Q2/Q3" if if2yr >= 2.5 else "Q3/Q4"
        out.append({"name": j.get("display_name",""), "publisher": j.get("host_organization_name",""),
                    "if2yr": round(if2yr,2), "h_index": stats.get("h_index",0),
                    "quartile": q, "is_oa": j.get("is_oa",False)})
    return {"results": out}


@app.get("/api/fulltext")
async def fulltext(doi: str):
    doi = doi.strip().lstrip("https://doi.org/")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": PUBMED_EMAIL})
        if r.status_code == 404:
            return {"found": False, "doi": doi}
        r.raise_for_status()
        data = r.json()
    best = data.get("best_oa_location") or {}
    return {
        "found": data.get("is_oa", False), "doi": doi,
        "title": data.get("title",""), "journal": data.get("journal_name",""),
        "oa_status": data.get("oa_status",""),
        "best_url": best.get("url",""),
        "best_version": best.get("version",""),
        "all_locations": [{"url": l.get("url",""), "version": l.get("version",""), "host": l.get("host_type","")}
                          for l in (data.get("oa_locations") or [])[:5] if l.get("url")]
    }


@app.get("/api/quota")
async def get_quota(token: Optional[str] = None):
    if FREE_DAILY_LIMIT is None:
        return {"used": 0, "limit": None, "remaining": None, "unlimited": True}
    user_id = "anonymous"
    if token:
        user = get_user(token)
        if user: user_id = user.get("sub", "anonymous")
    today = datetime.utcnow().date().isoformat()
    conn = sqlite3.connect(_DB_PATH)
    row = conn.execute("SELECT count, date FROM quotas WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    used = row[0] if row and row[1] == today else 0
    return {"used": used, "limit": FREE_DAILY_LIMIT, "remaining": max(0, FREE_DAILY_LIMIT - used)}


# ── Internal search helpers ───────────────────────────────────────────────────
async def _pubmed_search(q, n=5, year_from=None, free_only=False):
    query = q
    if year_from: query += f" AND {year_from}:3000[dp]"
    if free_only: query += " AND free full text[sb]"
    base = {"tool":"clinsearch","email":PUBMED_EMAIL}
    if PUBMED_API_KEY: base["api_key"] = PUBMED_API_KEY
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={**base,"db":"pubmed","term":query,"retmax":n,"retmode":"json","sort":"relevance"})
            ids = r.json().get("esearchresult",{}).get("idlist",[])
            if not ids: return []
            r2 = await client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={**base,"db":"pubmed","id":",".join(ids),"retmode":"xml","rettype":"abstract"})
        root = ET.fromstring(r2.text)
        results = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID","")
            title = article.findtext(".//ArticleTitle","").strip()
            year = article.findtext(".//PubDate/Year","") or (article.findtext(".//PubDate/MedlineDate","") or "")[:4]
            journal = article.findtext(".//Journal/Title","")
            authors = []
            for a in article.findall(".//Author")[:3]:
                last = a.findtext("LastName",""); first = a.findtext("ForeName","")
                if last: authors.append(f"{last} {first[0]}." if first else last)
            abstract = " ".join(
                (f"[{p.get('Label')}] " if p.get('Label') else "") + (p.text or "")
                for p in article.findall(".//AbstractText"))[:500]
            doi = ""
            for eid in article.findall(".//ArticleId"):
                if eid.get("IdType")=="doi": doi = eid.text or ""
            pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
            badge = next((pt for pt in pub_types if pt in (
                "Randomized Controlled Trial","Systematic Review","Meta-Analysis","Clinical Trial","Review"
            )), "")
            results.append({"id":pmid,"title":title,"authors":authors,"year":year,"journal":journal,
                           "abstract":abstract,"doi":doi,"pmid":pmid,"badge":badge,
                           "url":f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/","source":"PubMed"})
        return results
    except Exception as e:
        logger.warning(f"PubMed search error: {e}")
        return []


async def _s2_search(q, n=5, year_from=None, oa_only=False):
    headers = {"Accept":"application/json"}
    if S2_API_KEY: headers["x-api-key"] = S2_API_KEY
    params = {"query":q,"limit":n,"fields":"title,year,abstract,authors,citationCount,openAccessPdf,externalIds,journal"}
    if year_from: params["year"] = f"{year_from}-"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get("https://api.semanticscholar.org/graph/v1/paper/search",
                params=params, headers=headers)
            papers = r.json().get("data",[])
        if oa_only: papers = [p for p in papers if p.get("openAccessPdf")]
        return [{"id":(p.get("externalIds") or {}).get("DOI","") or p.get("paperId",""),
                 "title":p.get("title",""),"authors":[a.get("name","") for a in (p.get("authors") or [])[:3]],
                 "year":str(p.get("year","")),"journal":(p.get("journal") or {}).get("name",""),
                 "abstract":(p.get("abstract") or "")[:500],"doi":(p.get("externalIds") or {}).get("DOI",""),
                 "citations":p.get("citationCount",0),"pdf_url":(p.get("openAccessPdf") or {}).get("url",""),
                 "source":"Semantic Scholar"} for p in papers]
    except Exception as e:
        logger.warning(f"S2 search error: {e}")
        return []


async def _openalex_search(q, n=5, year_from=None, oa_only=False, study_type=None):
    params = {"search":q,"per-page":n,"select":"title,publication_year,authorships,primary_location,cited_by_count,doi,open_access,abstract_inverted_index","sort":"cited_by_count:desc","mailto":PUBMED_EMAIL}
    filters = []
    if year_from:   filters.append(f"publication_year:>{year_from-1}")
    if study_type:  filters.append(f"type:{study_type}")
    if oa_only:     filters.append("open_access.is_oa:true")
    if filters:     params["filter"] = ",".join(filters)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get("https://api.openalex.org/works", params=params)
            works = r.json().get("results",[])
        def reconstruct(inv):
            if not inv: return ""
            words = {}
            for w,pos in inv.items():
                for p in pos: words[p] = w
            return " ".join(words[k] for k in sorted(words))[:500]
        return [{"id":(w.get("doi","") or "").replace("https://doi.org/",""),
                 "title":w.get("title",""),"authors":[(a.get("author") or {}).get("display_name","") for a in (w.get("authorships") or [])[:3]],
                 "year":str(w.get("publication_year","")),"journal":((w.get("primary_location") or {}).get("source") or {}).get("display_name",""),
                 "abstract":reconstruct(w.get("abstract_inverted_index")),"doi":(w.get("doi","") or "").replace("https://doi.org/",""),
                 "citations":w.get("cited_by_count",0),"pdf_url":(w.get("open_access") or {}).get("oa_url",""),
                 "source":"OpenAlex"} for w in works]
    except Exception as e:
        logger.warning(f"OpenAlex search error: {e}")
        return []


def _classify_study_type(title: str, abstract: str) -> str:
    """Quick regex-based study type classifier for cards."""
    text = (f"{title} {abstract}").lower()
    if re.search(r'systematic.{0,10}review|meta.?analysis|cochrane.{0,10}review', text):
        return "SR/MA"
    if re.search(r'randomized controlled trial|randomised controlled trial|\brct\b|randomized.{0,20}trial|randomised.{0,20}trial', text):
        return "RCT"
    if re.search(r'phase (i|ii|iii|iv)\b|double.blind|placebo.controlled', text):
        return "RCT"
    if re.search(r'\bcohort\b|case.?control|observational|cross.?sectional', text):
        return "Observational"
    if re.search(r'case report|case series|case presentation', text):
        return "Case Report"
    if re.search(r'guideline|recommendation|consensus statement|practice parameter', text):
        return "Guideline"
    return ""


def _dedupe(papers: list) -> list:
    """Deduplicate papers by DOI, PMID, or title prefix (keep highest-cited)."""
    sorted_p = sorted(papers, key=lambda x: int(x.get("citations") or 0), reverse=True)
    seen_doi, seen_pmid, seen_title = set(), set(), set()
    out = []
    for p in sorted_p:
        doi = (p.get("doi") or "").lower().strip()
        pmid = str(p.get("pmid") or "").strip()
        title_key = re.sub(r"\W+", "", (p.get("title") or "").lower())[:40]
        if doi and doi in seen_doi:
            continue
        if pmid and pmid in seen_pmid:
            continue
        if title_key and len(title_key) > 10 and title_key in seen_title:
            continue
        if doi:
            seen_doi.add(doi)
        if pmid:
            seen_pmid.add(pmid)
        if title_key and len(title_key) > 10:
            seen_title.add(title_key)
        out.append(p)
    return out


async def _europepmc_search(q: str, n: int = 10, offset: int = 0,
                             year_from: int = None, year_to: int = None,
                             oa_only: bool = False, systematic_only: bool = False) -> list:
    """Search Europe PMC — 42M+ articles including Cochrane Systematic Reviews."""
    query = q
    if year_from or year_to:
        y1 = year_from or 1900
        y2 = year_to or 2099
        query += f" AND FIRST_PDATE:[{y1}-01-01 TO {y2}-12-31]"
    if oa_only:
        query += " AND OPEN_ACCESS:Y"
    if systematic_only:
        query += " AND (REVIEW_TYPE:SYSTEMATIC_REVIEW OR SRC:CBA)"
    params = {
        "query": query,
        "resultType": "core",
        "pageSize": min(n, 25),
        "format": "json",
        "sort": "CITED desc",
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params=params
            )
            r.raise_for_status()
            data = r.json()
        results = []
        for p in data.get("resultList", {}).get("result", []):
            doi = p.get("doi", "") or ""
            pmid = str(p.get("pmid", "") or "")
            authors = []
            author_list = (p.get("authorList") or {}).get("author", [])
            if isinstance(author_list, list):
                for a in author_list[:3]:
                    if isinstance(a, dict):
                        ln = a.get("lastName", "") or ""
                        ini = a.get("initials", "") or ""
                        name = (ln + " " + ini).strip()
                        if name:
                            authors.append(name)
            article_id = p.get("id", "") or ""
            source_db = p.get("source", "MED")
            ep_url = (f"https://europepmc.org/article/{source_db}/{article_id}"
                      if article_id else "")
            study_type = _classify_study_type(
                p.get("title", "") or "",
                p.get("abstractText", "") or ""
            )
            results.append({
                "id": doi or pmid or article_id,
                "title": p.get("title", "") or "",
                "authors": authors,
                "year": str(p.get("pubYear", "") or ""),
                "journal": p.get("journalTitle", "") or "",
                "abstract": (p.get("abstractText", "") or "")[:500],
                "doi": doi,
                "pmid": pmid,
                "citations": p.get("citedByCount", 0) or 0,
                "isOpenAccess": (p.get("isOpenAccess") or "N") == "Y",
                "source": "Europe PMC",
                "study_type": study_type,
                "url": ep_url,
            })
        return results
    except Exception as e:
        logger.warning(f"Europe PMC search error: {e}")
        return []


# ───────────────────────────────────────
# RESEARCH SESSION — context accumulation
# ───────────────────────────────────────

import uuid

class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    action: str  # 'create' | 'get' | 'save_paper' | 'remove_paper' | 'add_note' | 'clear'
    data: Optional[dict] = None

@app.post("/api/session")
async def session_manager(req: SessionRequest):
    """Persistent research session — workspace for papers and notes."""
    if req.action == "create":
        sid = str(uuid.uuid4())[:8]
        topic = req.data.get("topic","") if req.data else ""
        created = datetime.utcnow().isoformat()
        conn = sqlite3.connect(_DB_PATH)
        conn.execute(
            "INSERT INTO sessions (id, topic, papers, notes, messages, created, updated) VALUES (?,?,?,?,?,?,?)",
            (sid, topic, "[]", "[]", "[]", created, created)
        )
        conn.commit(); conn.close()
        return {"id": sid, "topic": topic, "papers": [], "notes": [], "created": created}

    if req.action == "get":
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute(
            "SELECT id, topic, papers, notes, messages, created, updated FROM sessions WHERE id=?",
            (req.session_id,)
        ).fetchone()
        conn.close()
        if not row: return {"error": "Session not found"}
        return {
            "id": row[0],
            "topic": row[1],
            "papers": json.loads(row[2] or "[]"),
            "notes": json.loads(row[3] or "[]"),
            "messages": json.loads(row[4] or "[]"),
            "created": row[5],
            "updated": row[6],
        }

    if req.action == "save_paper":
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute("SELECT papers FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if not row: conn.close(); return {"error": "Session not found"}
        papers = json.loads(row[0])
        paper = req.data or {}
        if paper.get("id") not in [p.get("id") for p in papers]:
            papers.append(paper)
            conn.execute(
                "UPDATE sessions SET papers=?, updated=? WHERE id=?",
                (json.dumps(papers), datetime.utcnow().isoformat(), req.session_id)
            )
            conn.commit()
        conn.close()
        return {"saved": True, "total": len(papers)}

    if req.action == "remove_paper":
        pid = (req.data or {}).get("id")
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute("SELECT papers FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if row:
            papers = [p for p in json.loads(row[0]) if p.get("id") != pid]
            conn.execute(
                "UPDATE sessions SET papers=?, updated=? WHERE id=?",
                (json.dumps(papers), datetime.utcnow().isoformat(), req.session_id)
            )
            conn.commit()
        conn.close()
        return {"removed": True}

    if req.action == "add_note":
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute("SELECT notes FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if row:
            notes = json.loads(row[0])
            notes.append({"text": (req.data or {}).get("text",""), "timestamp": datetime.utcnow().isoformat()})
            conn.execute(
                "UPDATE sessions SET notes=?, updated=? WHERE id=?",
                (json.dumps(notes), datetime.utcnow().isoformat(), req.session_id)
            )
            conn.commit()
        conn.close()
        return {"saved": True}

    if req.action == "clear":
        conn = sqlite3.connect(_DB_PATH)
        conn.execute(
            "UPDATE sessions SET papers='[]', notes='[]', updated=? WHERE id=?",
            (datetime.utcnow().isoformat(), req.session_id)
        )
        conn.commit(); conn.close()
        return {"cleared": True}

    if req.action == "list":
        conn = sqlite3.connect(_DB_PATH)
        rows = conn.execute(
            "SELECT id, topic, created FROM sessions ORDER BY created DESC LIMIT 30"
        ).fetchall()
        conn.close()
        return {"sessions": [{"id": r[0], "topic": r[1] or "Untitled", "created": r[2]} for r in rows]}

    if req.action == "save_messages":
        msgs = (req.data or {}).get("messages", [])
        topic = (req.data or {}).get("topic", "")
        conn = sqlite3.connect(_DB_PATH)
        if topic:
            conn.execute("UPDATE sessions SET messages=?, topic=?, updated=? WHERE id=?",
                        (json.dumps(msgs), topic, datetime.utcnow().isoformat(), req.session_id))
        else:
            conn.execute("UPDATE sessions SET messages=?, updated=? WHERE id=?",
                        (json.dumps(msgs), datetime.utcnow().isoformat(), req.session_id))
        conn.commit(); conn.close()
        return {"saved": True}

    if req.action == "get_messages":
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute("SELECT messages, topic FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        conn.close()
        if not row: return {"messages": [], "topic": ""}
        return {"messages": json.loads(row[0] or "[]"), "topic": row[1] or ""}

    return {"error": "Unknown action"}


# ───────────────────────────────────────
# PERSISTENT WORKSPACE ENDPOINTS
# ───────────────────────────────────────

@app.get("/api/workspace")
async def get_workspace(session_token: Optional[str] = None):
    if not session_token:
        raise HTTPException(401, "session_token required")
    user = get_user(session_token)
    if not user:
        raise HTTPException(401, "Invalid token")
    user_id = user.get("sub", "")
    conn = sqlite3.connect(_DB_PATH)
    rows = conn.execute(
        "SELECT id, paper_json, notes, folder, saved_at FROM workspace WHERE user_id=? ORDER BY saved_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    items = []
    for r in rows:
        try:
            paper = json.loads(r[1] or "{}")
        except Exception:
            paper = {}
        items.append({"id": r[0], "paper": paper, "notes": r[2] or "", "folder": r[3] or "default", "saved_at": r[4] or ""})
    return {"items": items, "total": len(items)}


class WorkspaceSaveRequest(BaseModel):
    session_token: str
    paper: dict
    notes: Optional[str] = ""
    folder: Optional[str] = "default"


@app.post("/api/workspace")
async def save_workspace(req: WorkspaceSaveRequest):
    user = get_user(req.session_token)
    if not user:
        raise HTTPException(401, "Invalid token")
    user_id = user.get("sub", "")
    item_id = str(uuid.uuid4())[:12]
    saved_at = datetime.utcnow().isoformat()
    paper_json = json.dumps(req.paper)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        "INSERT INTO workspace VALUES (?,?,?,?,?,?)",
        (item_id, user_id, paper_json, req.notes or "", req.folder or "default", saved_at)
    )
    conn.commit()
    conn.close()
    return {"saved": True, "id": item_id}


@app.delete("/api/workspace/{item_id}")
async def delete_workspace(item_id: str, session_token: Optional[str] = None):
    if not session_token:
        raise HTTPException(401, "session_token required")
    user = get_user(session_token)
    if not user:
        raise HTTPException(401, "Invalid token")
    user_id = user.get("sub", "")
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("DELETE FROM workspace WHERE id=? AND user_id=?", (item_id, user_id))
    conn.commit()
    conn.close()
    return {"deleted": True}


# ───────────────────────────────────────
# ALERTS ENDPOINTS
# ───────────────────────────────────────

@app.get("/api/alerts")
async def get_alerts(session_token: Optional[str] = None):
    if not session_token:
        raise HTTPException(401, "session_token required")
    user = get_user(session_token)
    if not user:
        raise HTTPException(401, "Invalid token")
    user_id = user.get("sub", "")
    conn = sqlite3.connect(_DB_PATH)
    rows = conn.execute(
        "SELECT id, query, sources, last_check, created FROM alerts WHERE user_id=? ORDER BY created DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return {"alerts": [{"id": r[0], "query": r[1], "sources": r[2], "last_check": r[3], "created": r[4]} for r in rows]}


class AlertCreateRequest(BaseModel):
    session_token: str
    query: str
    sources: Optional[str] = "pubmed"


@app.post("/api/alerts")
async def create_alert(req: AlertCreateRequest):
    user = get_user(req.session_token)
    if not user:
        raise HTTPException(401, "Invalid token")
    user_id = user.get("sub", "")
    alert_id = str(uuid.uuid4())[:12]
    created = datetime.utcnow().isoformat()
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        "INSERT INTO alerts VALUES (?,?,?,?,?,?)",
        (alert_id, user_id, req.query, req.sources or "pubmed", created, created)
    )
    conn.commit()
    conn.close()
    return {"created": True, "id": alert_id}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: str, session_token: Optional[str] = None):
    if not session_token:
        raise HTTPException(401, "session_token required")
    user = get_user(session_token)
    if not user:
        raise HTTPException(401, "Invalid token")
    user_id = user.get("sub", "")
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("DELETE FROM alerts WHERE id=? AND user_id=?", (alert_id, user_id))
    conn.commit()
    conn.close()
    return {"deleted": True}


# ───────────────────────────────────────
# USER SESSIONS ENDPOINT
# ───────────────────────────────────────

@app.get("/api/user-sessions")
async def get_user_sessions(session_token: Optional[str] = None):
    if not session_token:
        raise HTTPException(401, "session_token required")
    user = get_user(session_token)
    if not user:
        raise HTTPException(401, "Invalid token")
    user_email = user.get("email", user.get("sub", ""))
    conn = sqlite3.connect(_DB_PATH)
    rows = conn.execute(
        "SELECT id, topic, created FROM sessions ORDER BY created DESC LIMIT 30"
    ).fetchall()
    conn.close()
    return {"sessions": [{"id": r[0], "topic": r[1] or "Untitled", "created": r[2] or ""} for r in rows]}


# ───────────────────────────────────────
# PREPRINTS FEED
# ───────────────────────────────────────

@app.get("/api/preprints")
async def preprints(query: str = ""):
    from datetime import timedelta as td
    end = datetime.utcnow()
    start = end - td(days=30)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    results = []
    query_words = [w.lower() for w in query.split() if len(w) > 2] if query else []

    async def fetch_server(server: str):
        url = f"https://api.medrxiv.org/details/{server}/{start_str}/{end_str}/json"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return []
                data = r.json()
                papers = data.get("collection", [])
                out = []
                for p in papers:
                    title = (p.get("title") or "").lower()
                    abstract = (p.get("abstract") or "").lower()
                    if query_words and not any(w in title or w in abstract for w in query_words):
                        continue
                    doi = p.get("doi", "")
                    out.append({
                        "title": p.get("title", ""),
                        "doi": doi,
                        "authors": p.get("authors", ""),
                        "date": p.get("date", ""),
                        "abstract": (p.get("abstract") or "")[:500],
                        "server": server,
                        "url": f"https://doi.org/{doi}" if doi else f"https://www.{server}.org",
                    })
                return out[:20]
        except Exception as e:
            logger.warning(f"{server} preprints error: {e}")
            return []

    med_results, bio_results = await asyncio.gather(
        fetch_server("medrxiv"),
        fetch_server("biorxiv"),
        return_exceptions=True
    )
    if isinstance(med_results, list):
        results.extend(med_results)
    if isinstance(bio_results, list):
        results.extend(bio_results)
    return {"results": results, "total": len(results), "query": query}


# ───────────────────────────────────────
# CITATION GRAPH
# ───────────────────────────────────────

@app.get("/api/citations")
async def citations(pmid: Optional[str] = None, doi: Optional[str] = None):
    if not pmid and not doi:
        raise HTTPException(400, "pmid or doi required")
    headers = {"Accept": "application/json"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    paper_id = f"PMID:{pmid}" if pmid else doi
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    paper_data = {}
    references = []
    cited_by = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(base_url, params={"fields": "title,year,authors,citationCount,abstract"}, headers=headers)
            if r.status_code == 200:
                paper_data = r.json()

            r2 = await client.get(
                base_url + "/references",
                params={"fields": "title,year,authors,citationCount", "limit": 15},
                headers=headers
            )
            if r2.status_code == 200:
                for item in r2.json().get("data", []):
                    cited = item.get("citedPaper", {})
                    authors_list = cited.get("authors", [])
                    references.append({
                        "paperId": cited.get("paperId", ""),
                        "title": cited.get("title", ""),
                        "year": cited.get("year"),
                        "authors": [a.get("name", "") for a in authors_list[:3]],
                        "citationCount": cited.get("citationCount", 0),
                    })

            r3 = await client.get(
                base_url + "/citations",
                params={"fields": "title,year,authors,citationCount", "limit": 10},
                headers=headers
            )
            if r3.status_code == 200:
                for item in r3.json().get("data", []):
                    citing = item.get("citingPaper", {})
                    authors_list = citing.get("authors", [])
                    cited_by.append({
                        "paperId": citing.get("paperId", ""),
                        "title": citing.get("title", ""),
                        "year": citing.get("year"),
                        "authors": [a.get("name", "") for a in authors_list[:3]],
                        "citationCount": citing.get("citationCount", 0),
                    })
    except Exception as e:
        logger.warning(f"Citation graph error: {e}")

    return {"paper": paper_data, "references": references, "cited_by": cited_by}


# ───────────────────────────────────────
# PRE-REGISTRATION CHECK
# ───────────────────────────────────────

@app.get("/api/check-registration")
async def check_registration(title: Optional[str] = None, pmid: Optional[str] = None):
    query_term = title or pmid or ""
    nct_match = re.search(r'NCT\d+', query_term, re.I)
    nct_id = nct_match.group(0).upper() if nct_match else ""
    registered = False
    status = ""
    primary_completion_date = ""
    try:
        search_q = nct_id if nct_id else (title or pmid or "")
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params={"query.term": search_q, "pageSize": 3, "format": "json",
                        "fields": "NCTId,BriefTitle,OverallStatus,PrimaryCompletionDate"}
            )
            if r.status_code == 200:
                studies = r.json().get("studies", [])
                if studies:
                    registered = True
                    first = studies[0].get("protocolSection", {})
                    id_m = first.get("identificationModule", {})
                    st_m = first.get("statusModule", {})
                    nct_id = nct_id or id_m.get("nctId", "")
                    status = st_m.get("overallStatus", "")
                    pcd = st_m.get("primaryCompletionDateStruct", {})
                    primary_completion_date = pcd.get("date", "") if isinstance(pcd, dict) else ""
    except Exception as e:
        logger.warning(f"Registration check error: {e}")
    return {
        "registered": registered,
        "nct_id": nct_id,
        "status": status,
        "primary_completion_date": primary_completion_date,
    }


# ───────────────────────────────────────
# PDF ANALYZER
# ───────────────────────────────────────

class PDFAnalyzeRequest(BaseModel):
    url: str
    session_token: Optional[str] = None


@app.post("/api/pdf-analyze")
async def pdf_analyze(req: PDFAnalyzeRequest):
    text = ""
    source_url = req.url
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(req.url, headers={"User-Agent": "Mozilla/5.0 ClinSearch/3.0"})
            if r.status_code == 200:
                ct = r.headers.get("content-type", "")
                if "html" in ct:
                    html = re.sub(r'<script[^>]*>.*?</script>', '', r.text, flags=re.DOTALL)
                    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
                    html = re.sub(r'<[^>]+>', ' ', html)
                    text = re.sub(r'\s+', ' ', html).strip()[:4000]
                elif "pdf" in ct:
                    doi_match = re.search(r'10\.\d{4,}/\S+', req.url)
                    if doi_match:
                        doi = doi_match.group(0).rstrip('/')
                        r2 = await client.get(
                            f"https://api.unpaywall.org/v2/{doi}",
                            params={"email": PUBMED_EMAIL}
                        )
                        if r2.status_code == 200:
                            best = (r2.json().get("best_oa_location") or {})
                            oa_url = best.get("url_for_landing_page") or best.get("url", "")
                            if oa_url and not oa_url.endswith(".pdf"):
                                r3 = await client.get(oa_url, headers={"User-Agent": "Mozilla/5.0"})
                                if r3.status_code == 200 and "html" in r3.headers.get("content-type", ""):
                                    html = re.sub(r'<script[^>]*>.*?</script>', '', r3.text, flags=re.DOTALL)
                                    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
                                    html = re.sub(r'<[^>]+>', ' ', html)
                                    text = re.sub(r'\s+', ' ', html).strip()[:4000]
                                    source_url = oa_url
    except Exception as e:
        logger.warning(f"PDF fetch error: {e}")

    if not text:
        return {"analysis": "Could not retrieve readable text from the URL.", "source_url": source_url}

    analysis_prompt = "Analyze this paper: extract key findings, methods, limitations, and clinical implications\n\n" + text[:4000]
    messages = [{"role": "user", "content": analysis_prompt}]
    try:
        if GEMINI_API_KEY:
            analysis = await call_gemini(messages, GEMINI_API_KEY)
        elif GROQ_API_KEY:
            analysis = await call_groq(messages, GROQ_API_KEY)
        else:
            return {"analysis": "No AI provider configured.", "source_url": source_url}
    except Exception as e:
        raise HTTPException(500, str(e))

    return {"analysis": analysis, "source_url": source_url}


# ───────────────────────────────────────
# SMART QUERY BUILDER
# ───────────────────────────────────────

class QueryBuildRequest(BaseModel):
    question: str
    session_token: Optional[str] = None

@app.post("/api/build-query")
async def build_query(req: QueryBuildRequest):
    """Convert natural language clinical question to optimised PubMed query."""
    prompt = f"""Convert this clinical question to an optimised PubMed search query.
Return JSON only with these fields:
- "pubmed_simple": basic PubMed query
- "pubmed_advanced": full MeSH query with Boolean operators
- "mesh_terms": list of relevant MeSH terms
- "suggested_filters": list of filter suggestions
- "search_strategy": brief explanation

Question: {req.question}"""

    messages = [{"role": "user", "content": prompt}]
    try:
        if GEMINI_API_KEY:
            raw = await call_gemini(messages, GEMINI_API_KEY)
        elif GROQ_API_KEY:
            raw = await call_groq(messages, GROQ_API_KEY)
        else:
            return {"error": "No AI provider available"}

        import json as _json
        match = re.search(r'\{[\s\S]+\}', raw)
        if match:
            return _json.loads(match.group())
        return {"pubmed_simple": req.question, "error": "Could not parse"}
    except Exception as e:
        return {"error": str(e)}


# ───────────────────────────────────────
# WIZARD — guided research flow
# ───────────────────────────────────────

class WizardRequest(BaseModel):
    intent: str  # 'quick_review' | 'systematic' | 'single_paper' | 'trials' | 'gap_analysis'
    query: str
    session_token: Optional[str] = None

@app.post("/api/wizard")
async def wizard(req: WizardRequest):
    """Return recommended tool sequence for the research intent."""
    flows = {
        "quick_review": {
            "label": "Quick Evidence Review",
            "steps": ["search_all_sources","rank_evidence","check_retraction","summarize"],
            "description": "Best evidence on a topic in 2 minutes",
            "sources": ["pubmed","semantic_scholar","openalex"],
        },
        "systematic": {
            "label": "Systematic Review Support",
            "steps": ["build_query","search_all_sources","search_trials","rank_evidence","extract_pico","grade","export_ris","export_csv"],
            "description": "Full systematic review workflow with PRISMA support",
            "sources": ["pubmed","semantic_scholar","openalex","europe_pmc","cochrane"],
        },
        "single_paper": {
            "label": "Paper Deep Dive",
            "steps": ["get_details","read_fulltext","extract_pico","assess_rob","find_related","citation_network"],
            "description": "Complete analysis of a single article",
            "sources": ["pubmed","pmc","unpaywall"],
        },
        "trials": {
            "label": "Clinical Trials Search",
            "steps": ["search_trials","search_preprints","check_results"],
            "description": "Ongoing and completed clinical trials",
            "sources": ["clinicaltrials","pubmed","biorxiv"],
        },
        "gap_analysis": {
            "label": "Research Gap Analysis",
            "steps": ["search_all_sources","rank_evidence","find_gaps","suggest_designs"],
            "description": "Identify unexplored areas and research opportunities",
            "sources": ["pubmed","semantic_scholar","openalex","arxiv"],
        },
    }
    flow = flows.get(req.intent, flows["quick_review"])
    return {"intent": req.intent, "query": req.query, "flow": flow}


# ───────────────────────────────────────
# EXPORT WORKSPACE
# ───────────────────────────────────────

class ExportRequest(BaseModel):
    papers: List[dict]
    format: str = "ris"  # 'ris' | 'csv' | 'vancouver' | 'apa' | 'summary'
    session_token: Optional[str] = None

@app.post("/api/export")
async def export_papers(req: ExportRequest):
    """Export a list of papers in various formats."""
    papers = req.papers

    if req.format == "ris":
        lines = []
        for p in papers:
            doi = p.get("doi","")
            lines.append("TY  - JOUR")
            for a in (p.get("authors") or [])[:6]:
                lines.append(f"AU  - {a}")
            lines.append(f"TI  - {p.get('title','')}")
            lines.append(f"JO  - {p.get('journal','')}")
            lines.append(f"PY  - {p.get('year','')}")
            if doi: lines.append(f"DO  - {doi}")
            if doi: lines.append(f"UR  - https://doi.org/{doi}")
            lines.append("ER  - ")
            lines.append("")
        return {"content": "\n".join(lines), "filename": "references.ris", "mimetype": "application/x-research-info-systems"}

    if req.format == "csv":
        import csv, io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["title","authors","year","journal","doi","url","citations","source"])
        w.writeheader()
        for p in papers:
            w.writerow({
                "title": p.get("title",""), "year": p.get("year",""),
                "journal": p.get("journal",""), "doi": p.get("doi",""),
                "authors": "; ".join(p.get("authors") or []),
                "url": p.get("url",""), "citations": p.get("citations",""),
                "source": p.get("source",""),
            })
        return {"content": buf.getvalue(), "filename": "references.csv", "mimetype": "text/csv"}

    if req.format in ("vancouver","apa"):
        refs = []
        for i, p in enumerate(papers, 1):
            authors = p.get("authors") or []
            author_str = ", ".join(authors[:6]) + (" et al" if len(authors) > 6 else "")
            doi = p.get("doi","")
            if req.format == "vancouver":
                ref = f"{i}. {author_str}. {p.get('title','')}. {p.get('journal','')}. {p.get('year','')}."
                if doi: ref += f" doi:{doi}"
            else:
                ref = f"{author_str} ({p.get('year','')}). {p.get('title','')}. {p.get('journal','')}."
                if doi: ref += f" https://doi.org/{doi}"
            refs.append(ref)
        return {"content": "\n\n".join(refs), "filename": "bibliography.txt", "mimetype": "text/plain"}

    if req.format == "bibtex":
        lines = []
        for i, p in enumerate(papers):
            authors_list = p.get("authors") or []
            first_author = authors_list[0] if authors_list else ""
            parts = first_author.split()
            last_name = parts[-1] if parts else "Author"
            key = last_name + str(p.get("year", "")) + str(i)
            lines.append(f"@article{{{key},")
            if authors_list:
                joined = " and ".join(authors_list)
                lines.append(f"  author  = {{{joined}}},")
            title_val = p.get("title", "")
            if title_val:
                lines.append(f"  title   = {{{title_val}}},")
            journal_val = p.get("journal", "")
            if journal_val:
                lines.append(f"  journal = {{{journal_val}}},")
            year_val = p.get("year", "")
            if year_val:
                lines.append(f"  year    = {{{year_val}}},")
            doi_val = p.get("doi", "")
            if doi_val:
                lines.append(f"  doi     = {{{doi_val}}},")
            url_val = p.get("url", "")
            if url_val and not doi_val:
                lines.append(f"  url     = {{{url_val}}},")
            lines.append("}")
            lines.append("")
        return {
            "content": "\n".join(lines),
            "filename": "references.bib",
            "mimetype": "application/x-bibtex",
        }

    return {"error": "Unknown format"}


# ── Fulltext text extraction ──────────────────────────────────────────────────
@app.get("/api/fulltext-text")
async def fulltext_text(pmid: Optional[str] = None, doi: Optional[str] = None):
    """Extract readable text from PMC (via PMID) or Unpaywall OA HTML."""
    text = ""

    if pmid:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Convert PMID → PMCID
                conv = await client.get(
                    "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                    params={"ids": pmid, "format": "json", "email": PUBMED_EMAIL}
                )
                pmcid = ((conv.json().get("records") or [{}])[0]).get("pmcid", "").replace("PMC", "")
                if pmcid:
                    r = await client.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                        params={"db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml",
                                "email": PUBMED_EMAIL}
                    )
                    if r.status_code == 200:
                        root = ET.fromstring(r.text)
                        paragraphs = [
                            (p.text or "").strip()
                            for p in root.findall(".//p")
                            if p.text and len(p.text.strip()) > 40
                        ]
                        text = " ".join(paragraphs)[:6000]
        except Exception as e:
            logger.warning(f"PMC fulltext error: {e}")

    if not text and doi:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"https://api.unpaywall.org/v2/{doi.lstrip('https://doi.org/')}",
                    params={"email": PUBMED_EMAIL}
                )
                if r.status_code == 200:
                    best = (r.json().get("best_oa_location") or {})
                    url = best.get("url_for_landing_page") or best.get("url", "")
                    if url and not url.endswith(".pdf"):
                        r2 = await client.get(url, follow_redirects=True, timeout=10,
                                              headers={"User-Agent": "Mozilla/5.0"})
                        if r2.status_code == 200 and "html" in r2.headers.get("content-type", ""):
                            html = re.sub(r'<script[^>]*>.*?</script>', '', r2.text, flags=re.DOTALL)
                            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
                            html = re.sub(r'<[^>]+>', ' ', html)
                            text = re.sub(r'\s+', ' ', html).strip()[:6000]
        except Exception as e:
            logger.warning(f"Fulltext HTML fetch error: {e}")

    return {"text": text, "chars": len(text), "found": bool(text)}


# ── Streaming chat (SSE) ──────────────────────────────────────────────────────
from fastapi.responses import StreamingResponse as _StreamingResponse

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE streaming endpoint for free-tier AI (Gemini / Groq)."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _ip_requests.get(client_ip, [])
    window = [t for t in window if now - t < 60]
    if len(window) >= 60:
        raise HTTPException(429, "Too many requests.")
    window.append(now)
    _ip_requests[client_ip] = window

    user_id = "anonymous"
    if req.session_token:
        user = get_user(req.session_token)
        if user:
            user_id = user.get("sub", "anonymous")

    if not check_quota(user_id):
        async def quota_err():
            yield f"data: {json.dumps({'error': 'daily_limit'})}\n\n"
        return _StreamingResponse(quota_err(), media_type="text/event-stream")

    messages = await _inject_search_context(req.messages)

    async def generate():
        if GEMINI_API_KEY:
            try:
                async for chunk in _stream_gemini(messages, GEMINI_API_KEY):
                    yield f"data: {json.dumps({'text': chunk, 'provider': 'gemini-flash'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            except Exception as e:
                logger.warning(f"Gemini stream error: {e}")

        if GROQ_API_KEY:
            try:
                async for chunk in _stream_groq(messages, GROQ_API_KEY):
                    yield f"data: {json.dumps({'text': chunk, 'provider': 'groq-llama'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            except Exception as e:
                logger.warning(f"Groq stream error: {e}")

        yield f"data: {json.dumps({'error': 'No AI provider available'})}\n\n"
        yield "data: [DONE]\n\n"

    return _StreamingResponse(generate(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _stream_gemini(messages: list, key: str):
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    history = [m for m in messages if m["role"] != "system"]
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?key={key}&alt=sse",
            json={
                "system_instruction": {"parts": [{"text": system}]} if system else None,
                "contents": [
                    {"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]}
                    for m in history
                ],
                "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.3}
            }
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        chunk = (data.get("candidates", [{}])[0]
                                 .get("content", {}).get("parts", [{}])[0].get("text", ""))
                        if chunk:
                            yield chunk
                    except Exception:
                        pass


async def _stream_groq(messages: list, key: str):
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages,
                  "max_tokens": 2048, "temperature": 0.3, "stream": True}
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        data = json.loads(line[6:])
                        chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if chunk:
                            yield chunk
                    except Exception:
                        pass


# ── Retraction Check ──────────────────────────────────────────────────────────
@app.get("/api/retraction")
async def check_retraction(doi: str = "", pmid: str = ""):
    """Check if a paper has been retracted via CrossRef and PubMed."""
    retracted = False
    reason = ""
    source = ""

    if doi:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"https://api.crossref.org/works/{doi.lstrip('https://doi.org/')}",
                    headers={"User-Agent": f"ClinSearch/3.0 (mailto:{PUBMED_EMAIL})"}
                )
                if r.status_code == 200:
                    data = r.json().get("message", {})
                    updates = data.get("update-to", []) or []
                    for upd in updates:
                        if "retract" in upd.get("type", "").lower():
                            retracted = True
                            reason = "Retraction notice found in CrossRef"
                            source = "CrossRef"
                            break
                    if not retracted and "retract" in str(data.get("title", "")).lower():
                        retracted = True
                        reason = "Title indicates retraction"
                        source = "CrossRef"
        except Exception as e:
            logger.warning(f"CrossRef retraction check error: {e}")

    if not retracted and pmid:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={"db": "pubmed", "id": pmid, "rettype": "xml",
                            "retmode": "xml", "email": PUBMED_EMAIL}
                )
                if r.status_code == 200:
                    xml = r.text
                    if "RetractionOf" in xml or "RetractionIn" in xml or "retract" in xml.lower()[:2000]:
                        retracted = True
                        reason = "Retraction notice found in PubMed record"
                        source = "PubMed"
        except Exception as e:
            logger.warning(f"PubMed retraction check error: {e}")

    return {"retracted": retracted, "reason": reason, "source": source,
            "doi": doi, "pmid": pmid}


# ── AI-Powered Tools ──────────────────────────────────────────────────────────
class AIToolRequest(BaseModel):
    tool: str           # 'journal_club' | 'gap_analysis' | 'head_to_head' | 'patient_summary' | 'forest_plot'
    pmid: Optional[str] = None
    doi: Optional[str] = None
    papers: Optional[List[dict]] = None
    query: Optional[str] = None
    session_token: Optional[str] = None


@app.post("/api/ai-tool")
async def ai_tool(req: AIToolRequest):
    """Run a specialised AI tool: journal club, gap analysis, head-to-head, patient summary."""

    if req.tool == "journal_club":
        paper_text = ""
        if req.pmid or req.doi:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    qid = req.pmid or req.doi
                    db_param = "pubmed"
                    r = await client.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                        params={"db": db_param, "id": qid, "rettype": "abstract",
                                "retmode": "text", "email": PUBMED_EMAIL}
                    )
                    paper_text = r.text[:3000] if r.status_code == 200 else ""
            except Exception:
                pass

        paper_section = ("Paper content:\n" + paper_text) if paper_text else ("Query: " + (req.query or ""))
        prompt = f"""You are a senior clinician running a journal club.

{paper_section}

Generate a complete journal club presentation with these sections:
## 1. Background & Clinical Question
## 2. PICO Framework
- P (Population):
- I (Intervention):
- C (Comparison):
- O (Outcome):
## 3. Methods Summary
- Study design, N, randomisation, blinding, follow-up
## 4. Key Results (with exact numbers)
- Primary endpoint: [effect size, 95% CI, p-value, NNT/NNH]
- Secondary endpoints
- Subgroup analyses
## 5. Critical Appraisal
- Risk of bias (CONSORT/STROBE criteria)
- Internal validity concerns
- External validity / generalisability
## 6. GRADE Certainty: ⊕⊕⊕⊕/⊕⊕⊕◯/⊕⊕◯◯/⊕◯◯◯
## 7. Clinical Bottom Line
- Does this change practice?
- For which patients?
## 8. Discussion Questions (5 questions for the group)

Be rigorous, cite specific numbers from the paper."""

    elif req.tool == "gap_analysis":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:12], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:400]}"
        prompt = f"""You are an expert researcher analyzing the evidence landscape for: {req.query or 'the provided papers'}

Papers analyzed:{papers_ctx}

Identify the most important RESEARCH GAPS with this structure:

## 🗺 Evidence Map
Brief overview of what IS known (2-3 sentences).

## 🔍 Critical Research Gaps
For each gap, provide:
### Gap [N]: [Title]
- **What's missing**: specific description
- **Why it matters clinically**: patient impact
- **Current best evidence**: what we have instead
- **Suggested study design**: RCT/SR/cohort, estimated N, endpoints
- **Priority**: 🔴 High / 🟡 Medium / 🟢 Low

## 📊 Evidence Quality by Subgroup
Table: Population | Evidence Quality | Key Gap

## 💡 Most Fundable Research Questions
Top 3 specific, answerable research questions with rationale.

## ⚠️ Risks of Acting on Current Evidence
What could go wrong if clinicians rely on existing data."""

    elif req.tool == "head_to_head":
        papers_ctx = ""
        treatments = set()
        if req.papers:
            for i, p in enumerate(req.papers[:10], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:500]}"
                treatments.add(p.get('title', '')[:40])

        prompt = f"""You are a systematic reviewer comparing treatments for: {req.query or 'the provided topic'}

Evidence base:{papers_ctx}

Create a comprehensive HEAD-TO-HEAD comparison:

## Treatments Compared
List all interventions identified.

## Evidence Matrix
| Outcome | Treatment A | Treatment B | Treatment C | Winner |
|---------|------------|------------|------------|--------|
| Primary efficacy | [OR/RR, CI] | ... | ... | ... |
| Mortality | | | | |
| Major adverse events | | | | |
| Discontinuation rate | | | | |
| Quality of life | | | | |
| Cost-effectiveness | | | | |

## Network of Evidence
- Direct comparisons available: [list]
- Indirect comparisons only: [list]
- No data: [list]

## Subgroup Differences
Who benefits most from each treatment?

## Clinical Decision Framework
```
If [patient profile A] → prefer [Treatment X] because [evidence]
If [patient profile B] → prefer [Treatment Y] because [evidence]
If [contraindication Z] → avoid [Treatment] because [evidence]
```

## Confidence in Conclusions
GRADE rating for each comparison with justification."""

    elif req.tool == "patient_summary":
        paper = (req.papers or [{}])[0]
        title = paper.get("title", req.query or "")
        abstract = paper.get("abstract", "")[:1000]
        prompt = f"""Convert this medical research into a clear patient-friendly summary.

Paper: {title}
Abstract: {abstract}

Write a patient summary with:
## What this study is about
(2-3 simple sentences, no medical jargon)

## What the researchers did
(Simple description of how the study worked)

## What they found
(Key results in plain language — use "X out of 100 patients" instead of percentages or OR)

## What this means for you
(Practical takeaway — should patients ask their doctor about this?)

## Questions to ask your doctor
(3-5 specific questions based on these findings)

## Important limitations
(Why this might not apply to everyone, in simple terms)

Use analogies, avoid jargon, write at 8th-grade reading level. Respond in the same language as the paper title."""

    elif req.tool == "forest_plot":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:10], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:500]}"
        prompt = f"""Extract forest plot data from these papers for: {req.query or 'the primary outcome'}

Papers:{papers_ctx}

Return a JSON object with this EXACT structure (no markdown, just JSON):
{{
  "outcome": "outcome name",
  "measure": "OR|RR|HR|MD",
  "unit": "unit if continuous",
  "studies": [
    {{
      "author": "First author name",
      "year": 2023,
      "n_treatment": 150,
      "n_control": 148,
      "effect": 0.65,
      "ci_lower": 0.45,
      "ci_upper": 0.95,
      "weight": 15.2,
      "events_treatment": 12,
      "events_control": 18
    }}
  ],
  "pooled": {{
    "effect": 0.72,
    "ci_lower": 0.58,
    "ci_upper": 0.89,
    "i2": 23,
    "p_heterogeneity": 0.24,
    "p_value": 0.003
  }}
}}
If exact data is not available, estimate from the abstract. Ensure effect and CIs are numeric."""
    elif req.tool == "drug_interaction":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:8], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} — {(p.get('abstract','') or '')[:300]}"
        drugs = req.query or "drugs mentioned in papers"
        fda_ctx = ""
        if req.query:
            try:
                drug_term = req.query.split()[0]
                async with httpx.AsyncClient(timeout=8) as client:
                    r = await client.get(
                        "https://api.fda.gov/drug/event.json",
                        params={"search": f"patient.drug.medicinalproduct:{drug_term}", "limit": 3}
                    )
                    if r.status_code == 200:
                        results = r.json().get("results", [])[:2]
                        fda_ctx = f"\nOpenFDA sample data: {json.dumps(results)[:600]}"
            except Exception:
                pass

        prompt = f"""You are a clinical pharmacologist. Analyze drug interactions for: {drugs}

Papers:{papers_ctx}{fda_ctx}

## 💊 Drugs Identified
List all drugs/interventions mentioned in the papers.

## ⚠️ Key Drug-Drug Interactions
| Drug A | Drug B | Mechanism | Severity | Management |
|--------|--------|-----------|----------|-----------|

## 🔴 Absolute Contraindications
Combinations to never use and why.

## 🟡 Use With Caution
Dose adjustments or enhanced monitoring required.

## 📊 Special Populations
| Population | Key Consideration |
|-----------|------------------|
| Renal impairment (CrCl<30) | |
| Hepatic impairment | |
| Elderly (>65) | |
| Pregnancy/lactation | |

## 💡 Clinical Pearls
Key prescribing safety tips."""

    elif req.tool == "guidelines":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:5], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')})"
        topic = req.query or "the topic"
        prompt = f"""You are a medical guidelines expert. Summarize clinical practice guidelines for: {topic}

Papers context:{papers_ctx}

## 📋 Major Guidelines at a Glance
| Organization | Year | Key Recommendation | Evidence Grade |
|-------------|------|--------------------|---------------|

## 🎯 First-Line Treatment Algorithm
Step-by-step clinical pathway based on current guidelines.

## 🔄 Escalation Criteria
Specific thresholds/triggers to move to next-line therapy.

## 🔬 Evidence vs Guidelines Gaps
Where current trial evidence differs from guideline recommendations.

## 📅 Recent Updates (Last 2 Years)
New or changed recommendations with rationale.

## 🌍 Regional Differences
Where NICE / AHA / ESC / ESMO / WHO recommendations diverge and why."""

    elif req.tool == "case_simulator":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:8], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:350]}"
        case = req.query or "No case provided"
        prompt = f"""You are a senior clinician providing evidence-based clinical decision support.

CLINICAL CASE:
{case}

EVIDENCE BASE:{papers_ctx}

## 🏥 Case Analysis
- Key clinical features and significance
- Working diagnosis
- Risk stratification

## 🔍 Most Relevant Evidence
Which papers best apply to this patient and why?

## 💊 Recommended Treatment Plan
Tailored to THIS patient's profile:
- **First-line:** [drug, dose, frequency, duration]
- **If fails/contraindicated:** [alternative]
- **Monitoring:** [parameters and frequency]

## ⚠️ Red Flags & Contraindications
Specific concerns for this patient.

## 📊 Expected Outcomes
Based on trial data: response rates and NNT for this patient type.

## 🔄 Follow-up Plan
Timeline and decision points.

## ❓ Evidence Gaps
What data is lacking for this exact patient profile?"""

    elif req.tool == "dosing":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:10], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:350]}"
        drug = req.query or "drugs in these papers"
        prompt = f"""You are a clinical pharmacologist. Extract and synthesize dosing information for: {drug}

Papers:{papers_ctx}

## 💊 Dosing Summary
| Drug | Indication | Dose | Frequency | Route | Duration | Trial Source |
|------|-----------|------|-----------|-------|----------|-------------|

## 📈 Dose-Response Data
What trials showed about different doses (if dose-ranging data available).

## ⚙️ Special Population Dosing
| Population | Standard Dose | Adjusted Dose | Rationale |
|-----------|--------------|--------------|-----------|
| Renal impairment (CrCl <30) | | | |
| Hepatic impairment (Child-Pugh B/C) | | | |
| Elderly (>65 years) | | | |
| Pediatric (<18 years) | | | |
| Obesity (BMI >30) | | | |

## ⚠️ Dose Modification for Toxicity
When to reduce, hold, or discontinue.

## 💡 Practical Administration Tips
Loading doses, titration schedules, food interactions, timing pearls."""

    elif req.tool == "audio_summary":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:6], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:350]}"
        topic = req.query or "the provided research"
        prompt = f"""You are a medical podcaster. Write a 5-minute audio script (~750 words) on: {topic}

Research base:{papers_ctx}

Write in conversational spoken style — NO bullet points, flowing prose only:

[INTRO - 30 sec]
Open with a compelling clinical scenario or striking statistic. Introduce the topic.

[BACKGROUND - 60 sec]
Why this matters. What was missing before this research.

[THE EVIDENCE - 2 min]
Walk through key studies in storytelling format. Use phrases like "researchers found that...", "what's remarkable is...", "here's the twist...". Cite specific numbers.

[CLINICAL IMPLICATIONS - 90 sec]
"So what does this mean for your patients..." Practical takeaways.

[BOTTOM LINE - 30 sec]
Three numbered take-home messages.

[OUTRO]
Brief sign-off.

Write naturally as if speaking. Use pauses (...) and smooth transitions."""

    elif req.tool == "slides":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:8], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:250]}"
        topic = req.query or "the provided research"
        prompt = f"""You are a medical educator. Create a 15-slide presentation outline on: {topic}

Papers:{papers_ctx}

For EACH slide provide:
### Slide N: [Title]
**Key message:** one sentence
**Bullets:**
- point 1
- point 2
- point 3
**Visual:** chart/table/diagram description
**Speaker notes:** 2-3 sentences to say aloud

Slide structure:
- Slide 1: Title + learning objectives
- Slide 2: Epidemiology / burden of disease
- Slides 3-4: Pathophysiology
- Slides 5-9: Key evidence (one landmark study per slide)
- Slides 10-11: Evidence synthesis / meta-analysis
- Slides 12-13: Clinical implications and guidelines
- Slide 14: Controversies and limitations
- Slide 15: Take-home messages"""

    elif req.tool == "pico":
        paper = (req.papers or [{}])[0]
        title = paper.get("title", req.query or "")
        abstract = (paper.get("abstract", "") or "")[:2000]
        prompt = f"""Extract the full PICO framework from this medical paper.

Title: {title}
Abstract: {abstract}

## Population (P)
- **Inclusion criteria:** age, diagnosis, severity, prior treatments
- **Exclusion criteria:** key exclusions
- **N =** [sample size]
- **Setting:** inpatient/outpatient, country, years

## Intervention (I)
- **Treatment:** name, dose, frequency, route, duration
- **Co-interventions:** background therapy allowed

## Comparison (C)
- **Control:** placebo / active comparator / usual care
- **Dose and duration of comparator**

## Outcomes (O)
### Primary Outcome
| Measure | Timepoint | Result | 95% CI | p-value | NNT |
|---------|-----------|--------|--------|---------|-----|

### Key Secondary Outcomes
| Outcome | Result | p-value |
|---------|--------|---------|

## Study Design
- **Type:** RCT / cohort / case-control / SR / meta-analysis
- **Blinding:** double-blind / single-blind / open-label
- **Follow-up duration:**
- **Risk of bias:** low / moderate / high + key concerns

## GRADE Certainty
Level: High / Moderate / Low / Very Low
**Rationale:** why this GRADE level?"""

    elif req.tool == "rob":
        paper = (req.papers or [{}])[0]
        title = paper.get("title", req.query or "")
        abstract = (paper.get("abstract", "") or "")[:2000]
        study_type = "RCT" if any(w in (title+abstract).lower() for w in ["randomiz","randomis","rct","placebo","double-blind"]) else "observational"
        tool_name = "RoB 2.0 (Cochrane)" if study_type == "RCT" else "ROBINS-I"
        rct_domains = """### Domain 1: Randomisation process
- Judgement: Low / Some concerns / High
- Rationale:

### Domain 2: Deviations from intended interventions
- Judgement:
- Rationale:

### Domain 3: Missing outcome data
- Judgement:
- Rationale:

### Domain 4: Measurement of the outcome
- Judgement:
- Rationale:

### Domain 5: Selection of the reported result
- Judgement:
- Rationale:"""
        observational_domains = """### Domain 1: Bias due to confounding
- Judgement: Low / Moderate / Serious / Critical
- Rationale:

### Domain 2: Bias in selection of participants
- Judgement:
- Rationale:

### Domain 3: Bias in classification of interventions
- Judgement:
- Rationale:

### Domain 4: Bias due to deviations from intended interventions
- Judgement:
- Rationale:

### Domain 5: Bias due to missing data
- Judgement:
- Rationale:

### Domain 6: Bias in measurement of outcomes
- Judgement:
- Rationale:

### Domain 7: Bias in selection of the reported result
- Judgement:
- Rationale:"""
        rob_domains = rct_domains if study_type == "RCT" else observational_domains
        prompt = f"""You are a systematic reviewer performing a {tool_name} risk of bias assessment.

Paper: {title}
Abstract: {abstract}

## Risk of Bias Assessment — {tool_name}

{rob_domains}

## Overall Risk of Bias
**Judgement:** Low / Some concerns / High / Critical
**Summary:** 2-3 sentences explaining the overall assessment.

## Impact on Conclusions
How does this risk of bias affect the reliability of the results?"""

    elif req.tool == "grade_table":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:10], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}, N={p.get('citations','?')}) — {(p.get('abstract','') or '')[:350]}"
        topic = req.query or "the intervention vs comparator"
        prompt = f"""You are a systematic reviewer creating a GRADE Evidence Profile for: {topic}

Papers:{papers_ctx}

Generate a complete GRADE Evidence Profile table:

## GRADE Summary of Findings

| Outcome | Studies (N) | Study Design | Risk of Bias | Inconsistency | Indirectness | Imprecision | Effect (95% CI) | Certainty |
|---------|------------|--------------|--------------|---------------|-------------|-------------|-----------------|-----------|
| Primary efficacy | | RCT | | | | | | ⊕⊕⊕⊕ |
| Mortality | | | | | | | | |
| Serious AEs | | | | | | | | |
| QoL | | | | | | | | |
| Discontinuation | | | | | | | | |

## Footnotes
Explain each downgrade/upgrade decision with specific rationale.

## Authors' Conclusions
What can be confidently recommended based on this evidence?

## Implications for Practice
What clinicians should do NOW with this level of certainty.

## Implications for Research
What studies are needed to upgrade the certainty."""

    elif req.tool == "adverse_events":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:10], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:400]}"
        topic = req.query or "the drugs in these papers"
        prompt = f"""You are a clinical pharmacovigilance expert. Extract and synthesize safety/adverse event data for: {topic}

Papers:{papers_ctx}

## 🚨 Adverse Events Summary

### Serious Adverse Events (SAEs)
| Adverse Event | Incidence (treatment) | Incidence (control) | RR/OR | NNH | Grade |
|--------------|----------------------|---------------------|-------|-----|-------|

### Common Adverse Events (>5%)
| Adverse Event | Incidence (treatment) | Incidence (control) | Significant? |
|--------------|----------------------|---------------------|-------------|

## 🔴 Black Box Warnings / Critical Safety Signals
Any life-threatening risks identified.

## ⚠️ Discontinuation Due to AEs
Rate of discontinuation and leading reasons.

## 📊 Safety by Subgroup
Any subgroups with higher AE risk (age, comorbidities, dose).

## 🔄 Long-term Safety Data
What is known beyond the trial duration.

## 💊 Drug-Specific Safety Monitoring
Recommended laboratory tests, imaging, or clinical monitoring."""

    elif req.tool == "nma":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:10], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:400]}"
        topic = req.query or "the treatments"
        prompt = f"""You are a network meta-analysis expert. Interpret NMA findings for: {topic}

Papers:{papers_ctx}

## 🕸️ Network of Evidence

### Direct vs Indirect Comparisons Available
| Treatment A | Treatment B | Direct evidence? | Indirect via? |
|------------|------------|-----------------|---------------|

## 📊 NMA Results (Primary Outcome)
| Treatment | vs Placebo (OR, 95% CrI) | Ranking (P-score/SUCRA) |
|----------|--------------------------|-------------------------|

## 🏆 Treatment Ranking
1. Best treatment: [name] — rationale
2. Second: [name]
...

## ⚠️ Heterogeneity & Inconsistency
- Global I²
- Key sources of heterogeneity
- Inconsistency test results

## 📉 Publication Bias Assessment
Funnel plot asymmetry / small study effects.

## 🔍 Key Assumptions
Transitivity assumption validity for this network.

## 💡 Clinical Interpretation
What the ranking means for clinical practice — caution around indirect comparisons."""

    elif req.tool == "cost_effectiveness":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:8], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:350]}"
        topic = req.query or "the intervention"
        prompt = f"""You are a health economist. Perform a pharmacoeconomic analysis for: {topic}

Papers:{papers_ctx}

## 💰 Cost-Effectiveness Analysis

### Cost Data (from papers or estimates)
| Intervention | Annual cost (USD) | Cost source |
|-------------|------------------|-------------|

### Effectiveness Data
| Intervention | QALYs gained | LYs saved | Response rate |
|-------------|-------------|-----------|---------------|

## 📊 ICER Calculation
Incremental Cost-Effectiveness Ratio:
- ICER = (Cost_A - Cost_B) / (Effect_A - Effect_B) = [value] per QALY
- WTP threshold comparison: $50,000/QALY (US) · £20,000-30,000/QALY (NICE)
- Verdict: Cost-effective / Not cost-effective / Borderline

## 🏥 Budget Impact
- Target population size
- Annual budget impact estimate
- Per-patient, per-year cost

## 📈 Sensitivity Analysis
Key drivers of uncertainty in the economic model.

## 🌍 Healthcare System Perspective
How cost-effectiveness varies by country/payer system.

## 💡 Value-Based Recommendations
At what price point does this intervention become cost-effective?"""

    elif req.tool == "grant_proposal":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:8], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:300]}"
        topic = req.query or "the identified research gap"
        prompt = f"""You are a senior academic researcher. Write a competitive grant proposal abstract and specific aims for: {topic}

Evidence base:{papers_ctx}

## 📝 Grant Proposal

### Title
[Concise, compelling title under 200 characters]

### Abstract (250 words)
Background, gap, hypothesis, aims, methods, significance. Written for a non-specialist reviewer.

### Specific Aims (1 page)

**Background & Significance (3-4 sentences)**
What is known, what is the gap, why does it matter.

**Central Hypothesis**
We hypothesize that [specific, testable statement].

**Aim 1:** [Action verb + specific measurable goal]
- Rationale:
- Approach: [study design, N, primary endpoint]
- Expected outcome:

**Aim 2:** [Action verb + specific measurable goal]
- Rationale:
- Approach:
- Expected outcome:

**Aim 3 (exploratory):** [Action verb + specific measurable goal]
- Rationale:
- Approach:

**Innovation**
What is novel about this approach compared to prior work?

**Impact**
If successful, this research will... (patient outcomes, clinical practice, policy)

### Budget Justification (brief)
Personnel, equipment, consumables — rough estimates with rationale.

### Timeline
Month-by-month milestones for a 3-year grant."""

    elif req.tool == "cme_questions":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:6], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:350]}"
        topic = req.query or "the evidence in these papers"
        prompt = f"""You are a medical educator. Create 5 CME multiple-choice questions based on: {topic}

Papers:{papers_ctx}

For each question:

### Question [N]
**Stem:** [Clinical vignette or direct knowledge question — 2-4 sentences]

A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
E) [Option E]

**Correct Answer:** [Letter]

**Explanation:** (3-5 sentences) Why is this the correct answer? Why are the distractors wrong? Reference the evidence.

**Learning Objective:** What concept does this question test?

**Difficulty:** Easy / Intermediate / Advanced

---

Questions should:
- Cover different aspects of the topic (not all the same concept)
- Include at least 2 clinical vignettes
- Have plausible distractors (common misconceptions)
- Be at postgraduate medical education level"""

    elif req.tool == "stats_critic":
        paper = (req.papers or [{}])[0]
        title = paper.get("title", req.query or "")
        abstract = (paper.get("abstract", "") or "")[:2000]
        prompt = f"""You are a biostatistician and methodologist. Critically appraise the statistical methods of this paper.

Paper: {title}
Abstract: {abstract}

## 📐 Statistical Methods Critique

### Study Design Assessment
- Appropriate design for the research question? Y/N + rationale
- Sample size: adequate / underpowered / overpowered?
- Power calculation: reported / missing / flawed?

### Primary Analysis
- Statistical test used: [name]
- Appropriate for data type and distribution? Y/N
- Multiple comparisons adjustment: adequate / missing?
- Intention-to-treat vs per-protocol: which used, appropriate?

### Effect Measures
| Measure reported | More appropriate measure | Reason |
|-----------------|--------------------------|--------|

### Common Statistical Errors Checklist
- [ ] P-hacking / outcome switching
- [ ] Baseline imbalance not adjusted
- [ ] Inappropriate subgroup analyses
- [ ] Surrogate outcomes without validation
- [ ] Missing data handling inadequate
- [ ] Confidence intervals not reported
- [ ] Correlation vs causation conflated

### Clinically vs Statistically Significant
Is the effect size clinically meaningful despite statistical significance (or vice versa)?

### Overall Methodological Quality Score
[1-10] with justification.

### Recommendations for Readers
How should clinicians interpret and apply these results given the statistical limitations?"""

    elif req.tool == "sr_protocol":
        topic = req.query or "the research question"
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:5], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')})"
        prompt = f"""You are a systematic review methodologist. Write a PRISMA-compliant systematic review protocol for: {topic}

Related papers:{papers_ctx}

## Systematic Review Protocol

### Title
[Structured title: Intervention in Population — A Systematic Review and Meta-Analysis]

### Background
Rationale for the review, existing evidence, and gaps (3-4 paragraphs).

### Objectives
To assess the [effectiveness/safety/accuracy] of [intervention] in [population] compared to [comparator] on [outcomes].

### Methods

#### Eligibility Criteria (PICOS)
- **P (Population):** inclusion/exclusion criteria
- **I (Intervention):** specific interventions included
- **C (Comparators):** acceptable comparators
- **O (Outcomes):**
  - Primary: [specific outcome, timepoint]
  - Secondary: [list]
- **S (Study design):** RCTs / observational / all designs

#### Information Sources
Databases: MEDLINE/PubMed, EMBASE, Cochrane CENTRAL, ClinicalTrials.gov, WHO ICTRP
Search dates: [from] to present
Grey literature: conference abstracts, regulatory documents

#### Search Strategy (PubMed example)
```
([MeSH terms]) AND ([free text terms]) AND ([study type filters])
```

#### Data Extraction
Two independent reviewers. Disagreements resolved by consensus or third reviewer.

#### Risk of Bias Assessment
RCTs: Cochrane RoB 2.0. Observational: ROBINS-I.

#### Statistical Analysis Plan
- Pooling method: random effects (DerSimonian-Laird) if I²>50%, fixed effects otherwise
- Heterogeneity: Cochran Q, I², tau²
- Subgroup analyses: [pre-specified list]
- Sensitivity analyses: [list]
- Publication bias: funnel plot + Egger's test if N>10 studies

#### GRADE Assessment
Certainty rated for each outcome.

### Registration
PROSPERO registration planned. Protocol DOI: pending."""

    elif req.tool == "sample_size":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:6], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:350]}"
        topic = req.query or "the proposed study"
        prompt = f"""You are a biostatistician. Perform a sample size calculation for: {topic}

Evidence base (to extract effect sizes and control event rates):{papers_ctx}

## 🧮 Sample Size Calculation

### Parameters Extracted from Evidence
- Control event rate (CER): [%] — source: [paper]
- Expected effect size: OR/RR/MD = [value]  — source: [paper]
- Standard deviation (if continuous): [value]

### Primary Calculation

**For dichotomous outcome (two proportions):**
- Alpha (type I error): 0.05 (two-tailed)
- Power (1-beta): 80% / 90%
- CER: [%]
- EER: [%]
- n per group: [N]
- Total N: [2N]

**For continuous outcome (two means):**
- Effect size (Cohen's d): [value]
- n per group: [N]
- Total N: [2N]

### Adjusted Sample Size
Accounting for:
- Expected dropout (15%): adjusted N = [value]
- 2:1 randomisation ratio: n_treatment=[X], n_control=[Y]
- Stratification factors: [minimal inflation]

### Sensitivity Table
| Power | Alpha | Expected Effect | N per arm | Total N |
|-------|-------|----------------|-----------|---------|
| 80% | 0.05 | [base] | | |
| 90% | 0.05 | [base] | | |
| 80% | 0.05 | [base × 0.75] | | |

### Feasibility Assessment
- Is this sample size feasible? Recruitment rate from literature: [X/year]
- Estimated recruitment time: [months]
- Multi-centre requirement: [yes/no]

### Software/Formula Reference
G*Power formula used. Verify with: G*Power 3.1 / pwr package in R."""

    elif req.tool == "translation":
        paper = (req.papers or [{}])[0]
        title = paper.get("title", req.query or "")
        abstract = (paper.get("abstract", "") or "")[:2000]
        lang = paper.get("_lang", "Portuguese (Brazilian)")
        prompt = f"""Translate and adapt this medical research for patients. Write entirely in {lang}.

Paper: {title}
Abstract: {abstract}

Create a patient-friendly summary (no jargon, 8th-grade reading level):

## O que foi pesquisado
(2-3 simple sentences about the question studied)

## Como foi feito
(Simple description of how the study worked)

## O que encontraram
(Key results — use "X em cada 10 pacientes" style, avoid percentages and statistics)

## O que isso significa para você
(Practical implications — should patients discuss this with their doctor?)

## Perguntas para fazer ao seu médico
1.
2.
3.
4.
5.

## Limitações importantes
(Why results might not apply to everyone, in simple terms)

Write entirely in {lang}. Use simple vocabulary. No medical abbreviations."""

    elif req.tool == "contradiction":
        papers_ctx = ""
        if req.papers:
            for i, p in enumerate(req.papers[:12], 1):
                papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) — {(p.get('abstract','') or '')[:500]}"
        topic = req.query or "the provided papers"
        prompt = f"""You are a systematic reviewer specializing in identifying contradictions in the medical literature for: {topic}

Papers analyzed:{papers_ctx}

## 🔍 Contradiction Analysis

### Papers With Contradictory Findings
For each contradiction, provide:

#### Contradiction [N]: [Brief description of the conflict]
- **Paper A:** [title, year] — finding: [specific result with numbers]
- **Paper B:** [title, year] — finding: [specific result with numbers]
- **Effect size discrepancy:** [quantify the difference — e.g., OR 1.8 vs OR 0.6]
- **Clinical significance:** Is this difference clinically meaningful? [yes/no + rationale]

### Methodological Reasons for Contradictions
For each pair, analyze:
- Population differences (age, severity, comorbidities)
- Intervention differences (dose, duration, co-interventions)
- Outcome measurement differences (instruments, timepoints)
- Study design quality (RoB, confounding)
- Publication bias or selective reporting
- Statistical differences (underpowering, different effect measures)

### 🟡 Papers With Different Effect Sizes (Same Direction)
| Paper | Effect Size | 95% CI | Clinically Different? |
|-------|------------|--------|----------------------|

### 🔴 Papers With Opposite Conclusions
| Paper A | Conclusion | Paper B | Conclusion | Likely Explanation |
|---------|-----------|---------|-----------|-------------------|

### ⚖️ How to Reconcile
For each contradiction: what additional evidence would resolve it? Which finding is more trustworthy and why?

### 📋 Clinical Guidance Despite Contradictions
What can clinicians safely conclude despite the contradictory evidence?"""

    elif req.tool == "sr_screen":
        papers = req.papers or []
        inclusion_criteria = (req.query or "").split("|||")[0].strip()
        exclusion_criteria = (req.query or "").split("|||")[1].strip() if "|||" in (req.query or "") else ""
        papers_ctx = ""
        for i, p in enumerate(papers[:30], 1):
            papers_ctx += f"\n[{i}] Title: {p.get('title','')}\n    Abstract: {(p.get('abstract','') or '')[:400]}\n    Year: {p.get('year','')} | Authors: {', '.join((p.get('authors') or [])[:2])}\n"

        inc_text = inclusion_criteria if inclusion_criteria else "RCTs and systematic reviews on the topic"
        exc_text = exclusion_criteria if exclusion_criteria else "Non-human studies, case reports, editorials"
        prompt = f"""You are a systematic review screener. Screen the following papers for inclusion.

INCLUSION CRITERIA: {inc_text}
EXCLUSION CRITERIA: {exc_text}

PAPERS TO SCREEN:
{papers_ctx}

For EACH paper, return a JSON array (no markdown, just JSON) with this structure:
[
  {{
    "index": 1,
    "decision": "INCLUDE",
    "rationale": "Brief reason why this paper meets inclusion criteria"
  }},
  {{
    "index": 2,
    "decision": "EXCLUDE",
    "rationale": "Specific exclusion criterion met"
  }},
  {{
    "index": 3,
    "decision": "UNCERTAIN",
    "rationale": "Cannot determine without full text — specific uncertainty"
  }}
]

Assess ALL {len(papers)} papers. Return only the JSON array."""

    elif req.tool == "quadas2":
        paper_title = (req.papers or [{}])[0].get("title", req.query or "")
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:3], 1):
            abstract_excerpt = (p.get("abstract", "") or "")[:600]
            papers_ctx += f"\n[{i}] {p.get('title','')} ({p.get('year','')}) | {p.get('journal','')}\nAbstract: {abstract_excerpt}\n"
        prompt = f"""You are a systematic reviewer trained in QUADAS-2.

Apply QUADAS-2 to this diagnostic accuracy study:
{papers_ctx}

## QUADAS-2 Assessment

Rate each domain as LOW / HIGH / UNCLEAR risk and give 1-2 sentence rationale.

### Domain 1: Patient Selection
Signalling: consecutive/random sample? case-control design avoided? no inappropriate exclusions?
**Risk of bias:** [LOW/HIGH/UNCLEAR]
**Applicability concern:** [LOW/HIGH/UNCLEAR]
**Rationale:**

### Domain 2: Index Test
Signalling: results interpreted blinded to reference standard? threshold pre-specified?
**Risk of bias:** [LOW/HIGH/UNCLEAR]
**Applicability concern:** [LOW/HIGH/UNCLEAR]
**Rationale:**

### Domain 3: Reference Standard
Signalling: reference standard likely correct? interpreted blinded to index test?
**Risk of bias:** [LOW/HIGH/UNCLEAR]
**Applicability concern:** [LOW/HIGH/UNCLEAR]
**Rationale:**

### Domain 4: Flow and Timing
Signalling: appropriate interval? same reference standard for all? all patients in analysis?
**Risk of bias:** [LOW/HIGH/UNCLEAR]
**Rationale:**

## Summary Table
| Domain | Risk of Bias | Applicability |
|--------|-------------|---------------|
| Patient Selection | | |
| Index Test | | |
| Reference Standard | | |
| Flow and Timing | — | |

## Overall Assessment
Overall bias risk and implication for GRADE certainty of diagnostic evidence."""

    elif req.tool == "publication_bias":
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:10], 1):
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')}) | Citations: {p.get('citations',0)}\nAbstract: {(p.get('abstract','') or '')[:300]}\n\n"
        n_papers = len(req.papers or [])
        prompt = f"""You are a meta-analyst specializing in publication bias assessment.

Analyze these {n_papers} papers for publication bias:

{papers_ctx}

## Publication Bias Analysis

### 1. Funnel Plot Data (estimated)
Generate estimated funnel plot coordinates as JSON:
{{"funnel_data": [{{"study": "Author Year", "effect": 0.0, "se": 0.0, "weight": 1.0}}]}}

### 2. Asymmetry Assessment
- Direction of funnel asymmetry (if detectable)
- Egger's test interpretation (likely result based on study characteristics)
- Most likely explanation: small-study effects, reporting bias, heterogeneity

### 3. Trim-and-Fill Estimate
- Estimated missing studies (low/moderate/high number)
- Expected direction of adjusted effect

### 4. Evidence of Selective Reporting
- Overrepresentation of statistically significant findings?
- Missing negative/null trials?
- Time-lag bias indicators?

### 5. GRADE Impact
Should publication bias downgrade evidence certainty? (Undetected / Suspected / Strongly suspected)

### 6. Recommended Actions
Specific steps to address publication bias in this evidence base."""

    elif req.tool == "evidence_score":
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:10], 1):
            st = p.get("study_type", "")
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')}) | Design: {st or 'unknown'} | Journal: {p.get('journal','')}\nAbstract: {(p.get('abstract','') or '')[:400]}\n\n"
        prompt = f"""You are an evidence-based medicine expert. Rate each paper using Oxford Levels of Evidence and GRADE.

PAPERS:
{papers_ctx}

## Evidence Quality Scorecard

For EACH paper produce ONE table row:
| # | First Author Year | Study Design | Oxford LoE | GRADE Certainty | Methodological Quality (1-10) | Clinical Relevance | Key Limitation |
|---|------------------|--------------|-----------|-----------------|-------------------------------|-------------------|----------------|

Oxford LoE: 1a (SR of RCTs) > 1b (RCT) > 2a (SR cohorts) > 2b (cohort) > 3 (case-control) > 4 (case series) > 5 (expert opinion)
GRADE: High | Moderate | Low | Very Low
Clinical Relevance: High (changes practice) | Medium (informative) | Low (exploratory)

## Overall Body of Evidence
- Dominant evidence level: ___
- Overall GRADE certainty: ___
- Key gaps in evidence quality
- Recommended action: Sufficient for clinical decisions / More RCTs needed / SR recommended"""

    elif req.tool == "absolute_risk":
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:8], 1):
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')})\nAbstract: {(p.get('abstract','') or '')[:600]}\n\n"
        prompt = f"""You are a biostatistician. Extract absolute risk data from these papers.

PAPERS:
{papers_ctx}

## Absolute Risk Analysis

For each paper, extract or estimate:
| Paper | CER (Control Event Rate) | EER (Experimental Event Rate) | ARR (Absolute Risk Reduction) | RRR (Relative Risk Reduction) | NNT / NNH | 95% CI | Outcome Measured | Time Frame |
|-------|-------------------------|------------------------------|------------------------------|------------------------------|-----------|--------|-----------------|-----------|

If data not directly reported, estimate from abstract context. Mark estimated values with *.

## Clinical Interpretation
- Best NNT for primary outcome: ___
- Clinical significance threshold: ___
- Comparison across studies: consistent / inconsistent?

## Pooled Estimate (if applicable)
- Weighted average ARR: ___
- Overall NNT: ___
- Heterogeneity concern: yes / no

## Patient Counseling Translation
"For every ___ patients treated, ___ benefit / ___ are harmed" """

    elif req.tool == "subgroup_analysis":
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:8], 1):
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')})\nAbstract: {(p.get('abstract','') or '')[:500]}\n\n"
        topic = req.query or ""
        prompt = f"""You are a clinical epidemiologist specializing in heterogeneity of treatment effects.

Analyze subgroup effects for: {topic}

PAPERS:
{papers_ctx}

## Subgroup Analysis Report

### 1. Reported Subgroups
For each paper, list every subgroup reported (or "none reported"):
| Paper | Subgroups Analyzed | Direction of Effect | P-interaction | Conclusion |
|-------|--------------------|--------------------|--------------| -----------|

### 2. Effect Modifiers Identified
For each significant subgroup finding:
- **Subgroup:** (e.g., age >65, female sex, CKD stage 3+)
- **Effect in subgroup:** (effect size + CI)
- **Effect in main population:** (for comparison)
- **P for interaction:** (if reported)
- **Credibility:** Is this a pre-specified subgroup? (credible / exploratory / data-dredging risk)

### 3. High-Priority Subgroups to Check
Based on pathophysiology, which subgroups should be prioritized:
- Sex-based differences
- Age (pediatric / elderly)
- Ethnic/genetic differences
- Comorbidities (renal/hepatic impairment, diabetes, etc.)
- Disease severity

### 4. Clinical Guidance
Does the evidence justify differential treatment by any subgroup? Specify which."""

    elif req.tool == "systematic_comparison":
        papers = req.papers or []
        papers_ctx = ""
        for i, p in enumerate(papers[:6], 1):
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')})\nAbstract: {(p.get('abstract','') or '')[:500]}\n\n"
        n_papers = len(papers[:6])
        prompt = f"""You are a systematic reviewer. Create a structured side-by-side comparison of {n_papers} studies.

STUDIES TO COMPARE:
{papers_ctx}

## Structured Comparison Table

| Domain | {" | ".join([f"Study [{i+1}]" for i in range(n_papers)])} |
|--------|{"-------|"*n_papers}
| **Population** | | |
| **Intervention** | | |
| **Comparator** | | |
| **Primary outcome** | | |
| **Effect size (95% CI)** | | |
| **p-value** | | |
| **Sample size** | | |
| **Follow-up** | | |
| **Setting** | | |
| **Risk of bias** | | |
| **Study design** | | |
| **Key limitation** | | |
| **Overall quality** | | |

Fill in ALL cells based on abstract data. Use "NR" (not reported) where unavailable.

## Key Differences Between Studies
1. Population differences that may explain result variation
2. Methodological differences (dose, duration, outcome definition)
3. Which study is most applicable to real-world practice and why

## Synthesis
Consistent direction? What does the totality suggest?"""

    elif req.tool == "replication_check":
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:8], 1):
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')}) | {p.get('journal','')}\nAbstract: {(p.get('abstract','') or '')[:400]}\n\n"
        topic = req.query or ""
        prompt = f"""You are a scientific integrity and reproducibility expert.

Assess replication and reproducibility for: {topic}

PAPERS:
{papers_ctx}

## Replication Assessment

### 1. Replication Status per Study
| Paper | Finding Independently Replicated? | Number of Replication Attempts | Consistency |
|-------|----------------------------------|-------------------------------|-------------|

### 2. Reproducibility Indicators
For each paper (check if any signals present in abstract):
| Paper | Pre-registration Mentioned | Sample Size Powered | Multiple Comparison Concern | Result Too Perfect | Reproducibility Score (1-10) |
|-------|--------------------------|--------------------|-----------------------------|---------------------|------------------------------|

### 3. Statistical Red Flags
- p-values just below 0.05 across multiple papers (p-hacking risk)
- Effect sizes larger than typical for the field
- Overfitting in prediction models
- Surrogate endpoints without clinical validation

### 4. Open Science Practices (from abstracts)
- Any mention of pre-registration: yes/no/unclear
- Data sharing statements: yes/no/unclear
- Code availability: yes/no/unclear

### 5. Overall Replication Confidence
**Score:** HIGH / MODERATE / LOW / VERY LOW
**Key concern:** ___
**Recommended verification:** ___"""

    elif req.tool == "open_science":
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:8], 1):
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')}) | {p.get('journal','')}\nAbstract: {(p.get('abstract','') or '')[:400]}\n\n"
        prompt = f"""You are an open science advocate. Evaluate transparency and reproducibility practices.

PAPERS:
{papers_ctx}

## Open Science Scorecard

### 1. Transparency Table
| Paper | Trial/Study Registered? | Open Data? | Open Code? | Pre-print Available? | COI Declared? | Funder Disclosed? | Open Access? |
|-------|-----------------------|-----------|-----------|---------------------|--------------|------------------|-------------|

Rate each: YES | NO | UNCLEAR | NR (not reported)

### 2. Journal-Level Practices
- Do these journals require data sharing? (Yes/No/Some)
- TOP (Transparency and Openness Promotion) factor considerations

### 3. Best Practices Present
List any exemplary open science practices mentioned

### 4. Missing Practices
Critical transparency gaps in this evidence base

### 5. Overall Open Science Score
| Domain | Score (0-3) |
|--------|-------------|
| Pre-registration | |
| Data availability | |
| Code availability | |
| Reporting quality | |
| COI transparency | |
| **TOTAL** | **/15** |

**Interpretation:** 0-5 = Poor · 6-10 = Moderate · 11-15 = Excellent"""

    elif req.tool == "pretest_prob":
        scenario = req.query or "clinical scenario"
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:6], 1):
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')})\nAbstract: {(p.get('abstract','') or '')[:500]}\n\n"
        prompt = f"""You are a clinical epidemiologist applying Bayesian reasoning.

Clinical scenario: {scenario}

Diagnostic test evidence from papers:
{papers_ctx}

## Pre/Post-Test Probability Analysis

### 1. Pre-Test Probability (Prevalence)
Based on the clinical scenario (symptoms, risk factors, setting):
- Estimated pre-test probability: ___ %
- Rationale: (prevalence data, clinical gestalt)

### 2. Diagnostic Test Accuracy (from papers)
For each relevant test found:
| Test | Sensitivity | Specificity | LR+ | LR- | Source |
|------|-------------|-------------|-----|-----|--------|

Formula: LR+ = Sens/(1-Spec) · LR- = (1-Sens)/Spec

### 3. Post-Test Probability (Fagan Nomogram)
| Scenario | Pre-test P | LR Used | Pre-test Odds | Post-test Odds | Post-test P |
|---------|-----------|---------|--------------|----------------|------------|
| Positive test | _% | LR+ | | | |
| Negative test | _% | LR- | | | |

Formula: Post-test P = (Pre-test odds × LR) / (1 + Pre-test odds × LR)

### 4. Clinical Decision Thresholds
- **Test threshold** (below which no testing needed): ___ %
- **Treatment threshold** (above which treat without testing): ___ %
- **Current pre-test P vs thresholds:** Action = Test / Treat empirically / Observe

### 5. Recommendations
Which test, in which patients, changes management?"""

    elif req.tool == "mechanism":
        topic = req.query or ""
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:6], 1):
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')})\nAbstract: {(p.get('abstract','') or '')[:500]}\n\n"
        prompt = f"""You are a pharmacologist and molecular biologist. Explain the mechanism of action for: {topic}

Evidence from papers:
{papers_ctx}

## Mechanism of Action Report

### 1. Primary Mechanism
- Molecular target(s): receptor, enzyme, ion channel, transporter
- Binding characteristics: affinity, selectivity, reversibility
- Downstream signaling pathway(s)

### 2. Secondary Mechanisms
Any pleiotropic effects, off-target effects, or downstream consequences

### 3. Pharmacokinetics (ADME)
| Parameter | Value | Clinical Relevance |
|-----------|-------|-------------------|
| Absorption | | |
| Distribution (Vd) | | |
| Metabolism (CYP) | | |
| Elimination (t½) | | |
| Renal adjustment | | |
| Hepatic adjustment | | |

### 4. Pharmacodynamic Effects
- Onset of action
- Duration of effect
- Dose-response relationship (linear / bell-shaped / ceiling)
- Tolerance / tachyphylaxis

### 5. Clinically Important Drug Interactions
| Interacting Drug | Mechanism | Effect | Management |
|-----------------|-----------|--------|-----------|

### 6. Special Populations
Pediatrics, elderly, pregnancy (category), lactation

### 7. Emerging Evidence
Novel mechanisms or targets identified in the included papers"""

    elif req.tool == "coi_detector":
        papers_ctx = ""
        for i, p in enumerate((req.papers or [])[:8], 1):
            papers_ctx += f"[{i}] {p.get('title','')} ({p.get('year','')}) | {p.get('journal','')}\nAbstract: {(p.get('abstract','') or '')[:400]}\n\n"
        topic = req.query or ""
        prompt = f"""You are an evidence-based medicine expert analyzing conflicts of interest.

Analyze these papers on '{topic}' for funding and COI signals:

{papers_ctx}

## Conflict of Interest Analysis

### 1. Funding Source Detection
For each paper, identify funding signals from abstract/affiliations:
| Paper | Likely Funding | Type (Industry/Public/Mixed/NR) |
|-------|---------------|----------------------------------|

### 2. Industry Influence Indicators
| Paper | Industry Link | Spin Risk | Notes |
|-------|-------------|-----------|-------|

### 3. COI Red Flags
- Industry-funded trials with exclusively positive results
- Statistical significance patterns (p-hacking signals)
- Outcome reporting inconsistencies
- Author-industry financial links (if mentioned)

### 4. Aggregate Assessment
- Proportion with industry funding signals: X/Y papers
- Overall COI risk: LOW / MODERATE / HIGH
- Impact on reliability of this evidence base

### 5. Recommendations
What additional COI information is needed and where to find it."""

    else:
        raise HTTPException(400, "Unknown tool")

    messages = [{"role": "user", "content": prompt}]
    raw = None
    try:
        if GEMINI_API_KEY:
            try:
                raw = await call_gemini(messages, GEMINI_API_KEY)
            except Exception:
                raw = None
        if not raw and GROQ_API_KEY:
            try:
                raw = await call_groq(messages, GROQ_API_KEY)
            except Exception as e:
                return ai_provider_error_response("groq", e)
        if not raw:
            raise HTTPException(503, "No AI provider available")

        if req.tool == "forest_plot":
            try:
                match = re.search(r'\{[\s\S]+\}', raw)
                data = json.loads(match.group()) if match else {}
                return {"tool": req.tool, "data": data, "raw": raw}
            except Exception:
                return {"tool": req.tool, "data": {}, "raw": raw}

        if req.tool == "sr_screen":
            try:
                match = re.search(r'\[[\s\S]+\]', raw)
                decisions = json.loads(match.group()) if match else []
                return {"tool": req.tool, "decisions": decisions, "raw": raw}
            except Exception:
                return {"tool": req.tool, "decisions": [], "raw": raw}

        return {"tool": req.tool, "result": raw}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
