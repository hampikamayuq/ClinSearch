# ClinSearch v2 — Setup Guide

## Architecture

```
User → Google Sign-In (1 click)
           ↓
Frontend (Vercel, free)
           ↓
Backend API (Render, free)
     ├── Gemini 2.0 Flash (FREE — default AI)
     ├── Groq Llama 3.3 (FREE — fallback)
     ├── User's Claude key (optional, unlimited)
     ├── User's OpenAI key (optional, unlimited)
     └── Research tools (all free)
```

## Step 1 — Get free API keys

### Gemini (primary AI — free)
1. Go to https://aistudio.google.com/apikey
2. Create API key
3. Free tier: 15 req/min, 1M tokens/day

### Groq (fallback AI — free)
1. Go to https://console.groq.com/keys
2. Create API key
3. Free tier: 14,400 req/day with Llama 3.3 70B

### Google OAuth (for user login — free)
1. Go to https://console.cloud.google.com
2. New project → APIs & Services → Credentials
3. Create OAuth 2.0 Client ID (Web application)
4. Add authorized redirect URIs:
   - https://clinsearch-api.onrender.com/auth/google/callback
5. Copy Client ID and Client Secret

## Step 2 — Deploy Backend (Render)

1. Create GitHub repo `clinsearch-api` with `backend/` contents
2. Render → New Web Service → connect repo
3. Add environment variables:
   - GOOGLE_CLIENT_ID = (from step 1)
   - GOOGLE_CLIENT_SECRET = (from step 1)
   - GEMINI_API_KEY = (from step 1)
   - GROQ_API_KEY = (from step 1)
   - FRONTEND_URL = https://your-app.vercel.app
   - BACKEND_URL = https://clinsearch-api.onrender.com
   - PUBMED_API_KEY = (optional, from ncbi.nlm.nih.gov/account)
   - SEMANTIC_SCHOLAR_API_KEY = (optional)
4. Deploy → copy your backend URL

## Step 3 — Deploy Frontend (Vercel)

1. Edit `frontend/index.html`:
   - Line `const API = '...'` → your Render backend URL
   - Line `data-client_id="GOOGLE_CLIENT_ID_PLACEHOLDER"` → your Google Client ID
2. Create GitHub repo `clinsearch` with `frontend/` contents
3. Vercel → New Project → connect repo → Deploy
4. Copy your Vercel URL (e.g. https://clinsearch.vercel.app)
5. Update FRONTEND_URL in Render environment variables

## Free tier limits

| Resource | Limit | Notes |
|---|---|---|
| Gemini 2.0 Flash | 15 req/min, 1M tokens/day | Primary AI |
| Groq Llama 3.3 | 14,400 req/day | Fallback AI |
| Free queries/user | 20/day | Increase in main.py |
| Render (backend) | Always free | Sleeps after 15min |
| Vercel (frontend) | Always free | No sleep |

## Estimated cost at scale

| Users | Queries/day | Gemini cost | Groq cost |
|---|---|---|---|
| 100 | 1,000 | $0 (free tier) | $0 (free tier) |
| 500 | 5,000 | ~$2-5/mo | $0 |
| 2,000+ | 20,000+ | ~$15-30/mo | $0 (fallback) |

Users with their own API keys don't consume your quota at all.

## Files

```
backend/
  main.py           — FastAPI app (AI routing, OAuth, research tools)
  requirements.txt  — Dependencies
  render.yaml       — Render deployment config

frontend/
  index.html        — Complete single-file web app
  vercel.json       — Vercel routing config
```
