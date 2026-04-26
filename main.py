"""
ClinSearch v2 — Backend API
AI: Gemini 2.0 Flash (free) → Groq (free fallback) → User key (Claude/OpenAI)
Auth: Google OAuth
Research: PubMed, Semantic Scholar, OpenAlex, ClinicalTrials, Unpaywall
"""

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

app = FastAPI(title="ClinSearch API v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# Simple in-memory quota tracker (use Redis in production)
_quotas: dict = {}  # user_id -> {count, date}
FREE_DAILY_LIMIT = 20  # queries per day


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
        "version": "2.0.0",
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
        # Exchange code for tokens
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
        id_token_str = tokens.get("id_token", "")

        # Get user info
        r2 = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {tokens.get('access_token','')}"}
        )
        user = r2.json()

    # Issue JWT session token
    payload = {
        "sub":   user.get("sub", ""),
        "email": user.get("email", ""),
        "name":  user.get("name", ""),
        "pic":   user.get("picture", ""),
        "exp":   datetime.utcnow() + timedelta(days=30),
    }
    session_token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    # Redirect to frontend with token
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
    rec = _quotas.get(user_id, {"count": 0, "date": today})
    if rec["date"] != today:
        rec = {"count": 0, "date": today}
    if rec["count"] >= FREE_DAILY_LIMIT:
        return False
    rec["count"] += 1
    _quotas[user_id] = rec
    return True


# ── AI Chat ───────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Route AI request:
    1. If user provided their own key → use it (no quota)
    2. Else try Gemini free tier
    3. Fallback to Groq free tier
    4. If all fail → ask user for key
    """
    # Check session & quota for free tier
    user_id = "anonymous"
    if req.session_token:
        user = get_user(req.session_token)
        if user:
            user_id = user.get("sub", "anonymous")

    # Free tier only — user keys are called directly from browser
    # Check quota
    if not check_quota(user_id):
        return JSONResponse(
            status_code=429,
            content={
                "error": "daily_limit",
                "message": f"You've reached the {FREE_DAILY_LIMIT} free queries/day limit.",
                "options": ["Add your Claude or OpenAI key for unlimited use", "Come back tomorrow"]
            }
        )

    # Try Gemini (free)
    if GEMINI_API_KEY:
        try:
            response = await call_gemini(req.messages, GEMINI_API_KEY)
            return {"response": response, "provider": "gemini-flash", "quota_used": True}
        except Exception as e:
            if "quota" not in str(e).lower() and "rate" not in str(e).lower():
                raise HTTPException(500, f"Gemini error: {e}")

    # Fallback: Groq (free)
    if GROQ_API_KEY:
        try:
            response = await call_groq(req.messages, GROQ_API_KEY)
            return {"response": response, "provider": "groq-llama", "quota_used": True}
        except Exception as e:
            raise HTTPException(500, f"Groq error: {e}")

    raise HTTPException(503, "No AI provider available. Please add your API key.")


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


# User AI calls happen directly in the browser — no server involvement


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
    rec = _quotas.get(user_id, {"count": 0, "date": today})
    if rec["date"] != today: rec = {"count": 0, "date": today}
    return {"used": rec["count"], "limit": FREE_DAILY_LIMIT, "remaining": max(0, FREE_DAILY_LIMIT - rec["count"])}


# ── Internal search helpers ──────────────────────────────────────────────────
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
    except Exception:
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
    except Exception:
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
    except Exception:
        return []
