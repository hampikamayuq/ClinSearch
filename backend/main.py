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

FREE_DAILY_LIMIT = 20
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
            created TEXT
        );
    """)
    conn.commit()
    conn.close()

_init_db()

def _migrate_db():
    conn = sqlite3.connect(_DB_PATH)
    for col, default in [("messages", "[]"), ("updated", "''")]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT DEFAULT '{default}'")
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
            raise HTTPException(500, f"Groq error: {e}")

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
    year_from: Optional[int] = None,
    open_access: bool = False,
    reviews_only: bool = False
):
    """Parallel search across all sources."""
    pubmed_q = q + (" AND (Review[pt] OR Meta-Analysis[pt])" if reviews_only else "")
    tasks = [
        _pubmed_search(pubmed_q, n, year_from, open_access),
        _s2_search(q, n, year_from, open_access),
        _openalex_search(q, n, year_from, open_access, "review" if reviews_only else None),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    combined = []
    for r in results:
        if isinstance(r, list):
            combined.extend(r)
    return {"results": combined, "total": len(combined), "query": q}


@app.get("/api/pubmed")
async def pubmed(q: str, n: int = 10, year_from: Optional[int] = None, free_only: bool = False):
    return {"results": await _pubmed_search(q, n, year_from, free_only), "source": "PubMed"}


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
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)", (sid, topic, "[]", "[]", "[]", created))
        conn.commit(); conn.close()
        return {"id": sid, "topic": topic, "papers": [], "notes": [], "created": created}

    if req.action == "get":
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        conn.close()
        if not row: return {"error": "Session not found"}
        return {"id":row[0],"topic":row[1],"papers":json.loads(row[2]),"notes":json.loads(row[3]),"created":row[4]}

    if req.action == "save_paper":
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute("SELECT papers FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if not row: conn.close(); return {"error": "Session not found"}
        papers = json.loads(row[0])
        paper = req.data or {}
        if paper.get("id") not in [p.get("id") for p in papers]:
            papers.append(paper)
            conn.execute("UPDATE sessions SET papers=? WHERE id=?", (json.dumps(papers), req.session_id))
            conn.commit()
        conn.close()
        return {"saved": True, "total": len(papers)}

    if req.action == "remove_paper":
        pid = (req.data or {}).get("id")
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute("SELECT papers FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if row:
            papers = [p for p in json.loads(row[0]) if p.get("id") != pid]
            conn.execute("UPDATE sessions SET papers=? WHERE id=?", (json.dumps(papers), req.session_id))
            conn.commit()
        conn.close()
        return {"removed": True}

    if req.action == "add_note":
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute("SELECT notes FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if row:
            notes = json.loads(row[0])
            notes.append({"text": (req.data or {}).get("text",""), "timestamp": datetime.utcnow().isoformat()})
            conn.execute("UPDATE sessions SET notes=? WHERE id=?", (json.dumps(notes), req.session_id))
            conn.commit()
        conn.close()
        return {"saved": True}

    if req.action == "clear":
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("UPDATE sessions SET papers='[]', notes='[]' WHERE id=?", (req.session_id,))
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
            conn.execute("UPDATE sessions SET messages=?, topic=? WHERE id=?",
                        (json.dumps(msgs), topic, req.session_id))
        else:
            conn.execute("UPDATE sessions SET messages=? WHERE id=?",
                        (json.dumps(msgs), req.session_id))
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
            raw = await call_groq(messages, GROQ_API_KEY)
        if not raw:
            raise HTTPException(503, "No AI provider available")

        if req.tool == "forest_plot":
            try:
                match = re.search(r'\{[\s\S]+\}', raw)
                data = json.loads(match.group()) if match else {}
                return {"tool": req.tool, "data": data, "raw": raw}
            except Exception:
                return {"tool": req.tool, "data": {}, "raw": raw}

        return {"tool": req.tool, "result": raw}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
