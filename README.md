# 🔬 ClinSearch

**Free AI-powered medical literature search for clinicians and investigators.**

No account required. No subscription. Search 200M+ papers from PubMed, Semantic Scholar, OpenAlex, ClinicalTrials.gov and more — with AI synthesis powered by Gemini, Groq, Claude, or OpenAI.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

---

## Features

- **AI-powered synthesis** — ask clinical questions in natural language, get evidence summaries
- **10+ databases** — PubMed, Semantic Scholar, OpenAlex, Europe PMC, ClinicalTrials.gov, arXiv, CORE, bioRxiv/medRxiv
- **Free AI by default** — Gemini 2.0 Flash + Groq Llama 3.3 (no cost to user)
- **BYOK** — users can connect their own Claude or OpenAI key for unlimited use
- **Google Sign-In** — 1-click login, 20 free queries/day
- **Key security** — user API keys stay in the browser, never sent to servers
- **$0/month** to operate for moderate usage

---

## Architecture

```
User → Google Sign-In (1 click, optional)
              ↓
    Frontend  (Vercel, free)
              ↓
    Backend   (Render, free)
         ├── Gemini 2.0 Flash  ← free, server-side
         ├── Groq Llama 3.3    ← free fallback, server-side
         └── Research tools    ← all free APIs
              ↓ (if user has own key)
    Browser → Claude API       ← direct, key never leaves browser
    Browser → OpenAI API       ← direct, key never leaves browser
```

---

## Quick Deploy

### Prerequisites (all free)

| Service | Purpose | Free tier |
|---|---|---|
| [Google AI Studio](https://aistudio.google.com/apikey) | Gemini API key | 1M tokens/day |
| [Groq Console](https://console.groq.com/keys) | Groq API key | 14,400 req/day |
| [Google Cloud Console](https://console.cloud.google.com) | OAuth Client ID | Free |
| [Render.com](https://render.com) | Backend hosting | Free |
| [Vercel.com](https://vercel.com) | Frontend hosting | Free |

### 1. Backend (Render)

1. Fork this repository
2. Render → **New → Web Service** → connect your fork
3. Add environment variables:

```
GOOGLE_CLIENT_ID      = (from Google Cloud Console)
GOOGLE_CLIENT_SECRET  = (from Google Cloud Console)
GEMINI_API_KEY        = (from Google AI Studio)
GROQ_API_KEY          = (from Groq Console)
FRONTEND_URL          = https://your-app.vercel.app
BACKEND_URL           = https://your-api.onrender.com
PUBMED_API_KEY        = (optional)
PUBMED_EMAIL          = your@email.com
```

4. Deploy → copy your backend URL

### 2. Frontend (Vercel)

1. Edit `frontend/index.html` — replace:
   - `const API = 'https://clinsearch-api.onrender.com'` → your Render URL
   - `data-client_id="GOOGLE_CLIENT_ID_PLACEHOLDER"` → your Google Client ID
2. Vercel → **New Project** → connect repo → Deploy

### 3. Google OAuth

1. Google Cloud Console → Credentials → Create OAuth 2.0 Client ID
2. Authorized redirect URIs: `https://your-api.onrender.com/auth/google/callback`
3. Copy Client ID and Secret → add to Render env vars

---

## Cost at scale

| Users/month | Queries/day | Total cost |
|---|---|---|
| 200 | 2,000 | **$0** (free tiers) |
| 1,000 | 10,000 | **~$8/mo** (Gemini) |
| 5,000+ | 50,000+ | **~$40/mo** |

Users with their own Claude/OpenAI key consume zero server quota.

---

## Security

| What | Where | Who sees it |
|---|---|---|
| User's Claude/OpenAI key | Browser localStorage only | Only the user |
| Gemini API key | Render env vars | Backend only |
| Groq API key | Render env vars | Backend only |
| Session JWT | Browser localStorage | User + backend |

The user's API key **never reaches the server**.

---

## Project structure

```
backend/
  main.py           FastAPI — AI routing, OAuth, research tools
  requirements.txt
  render.yaml

frontend/
  index.html        Complete single-file app
  vercel.json

SETUP.md            Detailed step-by-step guide
```

---

## Roadmap

- [ ] PICO extraction (Pro)
- [ ] Risk of Bias — RoB 2 / ROBINS-I (Pro)
- [ ] GRADE assessment (Pro)
- [ ] Stripe subscriptions
- [ ] Export — RIS, CSV, Vancouver
- [ ] Mobile PWA

---

## License

MIT
