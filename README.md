# 🔬 ClinSearch v3

**The most complete free AI research assistant for medical professionals.**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## What's new in v3

- **Research session** — AI maintains context across an entire investigation
- **Workspace** — save, annotate, and export papers in one click
- **Guided wizard** — choose Quick Review / Systematic / Paper Analysis / Gap Analysis
- **Visual evidence** — GRADE, RoB, PICO, statistics rendered as interactive visuals
- **44 tools exposed** — tools panel with one-click injection into chat
- **AI synthesis of saved papers** — generate evidence summary from your workspace
- **Export from workspace** — RIS, CSV, Vancouver with one click
- **Smart query builder** — natural language → optimised PubMed query
- **Mode tabs** — Chat / Search / Trials / Paper / Stats / Export

## Architecture

```
Browser → Google Sign-In → 20 free queries/day (Gemini + Groq)
Browser → Own API key   → Unlimited (Claude/OpenAI, key stays in browser)
Backend → Research APIs → All free (PubMed, S2, OpenAlex, ClinicalTrials…)
```

## Deploy

Backend → Render.com (free)
Frontend → Vercel (free)
AI → Gemini 2.0 Flash + Groq Llama 3.3 (free)

## License

MIT
