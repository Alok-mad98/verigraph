# VeriGraph

[![Watch the demo](https://img.shields.io/badge/Watch%20demo-YouTube-red?logo=youtube)](https://youtu.be/VtlJq5-yZSk)

**Can you trust this paper's data and references — and what should you study next?**

Research assistants summarize papers. **VeriGraph** asks three sharper questions
and answers them with auditable evidence:

1. **Extract** — turn every figure/table/chart into structured data a computer can use.
2. **Verify** — check every citation against Crossref's 150M+ records and flag the fakes.
3. **Hypothesize** — propose new research questions, each tagged with the evidence it rests on.

Instead of generating another summary, it returns a **trustworthiness score (0–100)**,
a per-agent **execution trace + handoff graph**, a **cost table**, and an estimate of
**research time saved** — plus a downloadable Markdown report.

Built for the **IIT Madras Research Agents Hack** (single combined podium, $100 prize).

---

## Hackathon Highlights

- **3 specialist agents** with distinct roles + typed handoffs (evidence graph)
- **Auto-fill from any PDF or link** — drop in a paper, get title + references + page-figures filled automatically (competitors require a hand-prepared `demo_input/`)
- **Real vision extraction** — Kimi K2.7 reads tables/charts that text-only models literally cannot
- **Real tool use** — the Citation Verifier calls the live Crossref REST API (no key, 150M+ records)
- **Multi-model fallback chains** — vision (Kimi K2.7 → K2.6 → Llama 4) and text (GLM 5.2 → GLM 4.7-flash → GPT-OSS-20B → Qwen3); the pipeline keeps working on whatever your plan allows
- **30/30 automated tests passing** — `node tests/e2e.test.js` (vs ReproGraph's 20)
- **One-click self-test**: `curl https://verigraph.arechampionw.workers.dev/api/test` — 8 built-in assertions, zero setup
- **Composite trustworthiness score 0–100** with a 4-dimension breakdown
- **Exportable Markdown + JSON report** + per-agent cost table
- **$0.009 per run** on the Cloudflare Workers free tier; fully 24/7, always-on

## Why this is genuinely multi-agent

Three agents with distinct roles. The Hypothesis Generator does **not** operate
independently — it consumes typed state from the other two, and every hypothesis
points back to the specific extracted rows and citation verdicts it rests on
(`evidence_pointers`). The response includes an explicit `handoffs` array:

- **Vision Extractor** (Kimi K2.7): reads each figure image → typed JSON (rows, axes, units, confidence, anomalies).
- **Citation Verifier** (Kimi K2.7 + Crossref API): looks up each reference, scores title recall, judges whether the claim fits the matched work.
- **Hypothesis Generator** (GLM 5.2): consumes `extracted_data` + `citation_verdicts`, proposes questions tagged **SUPPORTED** vs **SPECULATION** with evidence pointers.

Handoffs returned in every response:
`Vision Extractor → Hypothesis Generator (extracted_data)`,
`Citation Verifier → Hypothesis Generator (citation_verdicts)`,
`Vision Extractor → Citation Verifier (shared paper context)`.

## Judging Criteria Map

### Research utility — 30%
**Problem:** transcribing figures into usable data takes 3–5 h/review, and 3–13% of LLM-generated citations are fabricated (arXiv 2604.03173, 2026). **Evidence in demo:** one PDF → extracted tables, citation verdicts (real vs fabricated), grounded hypotheses, a trustworthiness score, and an estimated **research-hours-saved** figure. Drop in any arXiv PDF via the link box and it auto-fills the title, references, and page-figures — then returns a full audit.

### Agent collaboration — 25%
**Not a prompt chain.** Three agents with distinct roles exchange typed state; the response includes an explicit `handoffs` evidence graph, a per-agent `trace`, and hypotheses that cite `figure:X row:Y` and `citation:<id>:verified` pointers. The verifier uses a **real external tool** (Crossref REST API).

### Working demo — 20%
- Live at **https://verigraph.arechampionw.workers.dev** — one-click "Load demo paper" or drop in your own PDF/link.
- **30 automated tests** (`node tests/e2e.test.js`) — all passing.
- **Self-test**: `curl /api/test` runs the full pipeline on a bundled figure + a real and a fabricated citation and returns 8 pass/fail checks.
- Multi-PDF upload + link loading + auto-fill of title/citations/figures.
- Graceful failure: if one agent or model fails, it's recorded in the trace and the rest still run.

### Cost efficiency — 15%
Default paid cost **~$0.009/run** on Cloudflare Workers AI free tier (10k neurons/day). Per-agent `cost_usd` is itemized in every response and aggregated in a cost table. Multi-model fallback means it stays free even when GLM 5.2 403s on the free plan (auto-falls-back to glm-4.7-flash). External API (Crossref) is free and keyless.

### Originality — 10%
ReproGraph audits *reproducibility*; MetaReviewer-AI does *peer review*; the rest are off-theme. VeriGraph is the **only entry using vision as a required agent capability** and the only **figure→verify→hypothesize** chain. It also auto-fills from any PDF — a workflow no competitor offers.

---

## 5-Minute Judge Quickstart

```bash
git clone https://github.com/<you>/verigraph.git
cd verigraph
npm install
node tests/e2e.test.js          # 30/30 tests against the live demo
```

**1. Run the self-test (no setup, hits the live worker)**

```bash
curl https://verigraph.arechampionw.workers.dev/api/test
# Expected: { "ok": true, "passed": 8, "total": 8, "score": { "total": 88 } ... }
```

**2. Run the full pipeline on the demo** — open the URL, click **Load demo paper**, then **Run pipeline**. Expected: a real "Attention Is All You Need" citation → **verified**, a fabricated one → **fabricated**, a table extracted to rows, and 6 hypotheses with evidence pointers.

**3. Drop in your own paper** — paste an arXiv PDF link in the link box, or upload PDF(s). The title, references, and page-figures auto-fill, then **Run pipeline**.

**4. Inspect the trace & cost** — the result panel shows the agent trace, the handoff graph, and a per-agent cost table; **Download report** gives a Markdown dossier.

---

## Architecture

```
paper (text + figures + citations)
   │
   ▼
[Agent 1 · Vision Extractor — Kimi K2.7]  ──► extracted_data
   │                                            │ (typed handoff)
   ▼                                            ▼
[Agent 2 · Citation Verifier — Kimi K2.7 + Crossref API]  ──► citation_verdicts
   │                                                       │
   └──────────────► [Agent 3 · Hypothesis Generator — GLM 5.2]
                                  │ consumes extracted_data + citation_verdicts
                                  ▼
                        hypotheses (SUPPORTED/SPECULATION + evidence pointers)
   +
score (0–100) · research_hours_saved · report_markdown · trace · cost_table
```

## How to run locally

```bash
npm install
# set CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID, then:
npx wrangler deploy        # or: npx wrangler dev --remote
```

## API

- `POST /api/analyze` — `{ paper_title, paper_text, figures:[{name,image_base64,mime_type}], citations:[{id,raw_text,claim?}] }` → extracted_data, citation_verdicts, hypotheses, handoffs, trace, cost_table, score, research_hours_saved, report_markdown.
- `GET /api/fetch?url=<paper>` — fetches a PDF/HTML/image by URL and returns normalized content (used by the UI's link box).
- `GET /api/test` — one-click self-test on a bundled figure + citations.
- `GET /api/health` — model chains + status.

## Reproducibility

| Item | Value |
|---|---|
| Models | Kimi K2.7 (vision, $0.95/$4.00 per M) · GLM 5.2 (text, $1.40/$4.40 per M) with fallbacks to Kimi K2.6, Llama 4 Scout, GLM 4.7-flash, GPT-OSS-20B, Qwen3. |
| External APIs | Crossref REST API (free, keyless) only. |
| Hosting | Cloudflare Workers (always-on, 24/7). Free plan: 10k neurons/day. |
| Run cost | ~$0.009/demo. Itemized per agent in every response. |
| Tests | `node tests/e2e.test.js` → 30/30. `curl /api/test` → 8/8. |
| Known limits | Vision accuracy depends on figure legibility (illegible values are omitted + flagged in `anomalies`); preprints/non-DOI works may report `not_found`; SUPPORTED/SPECULATION is the model's judgment (pointers provided for audit). Users must not submit confidential/personal/unpublished/license-restricted data. |

## License

MIT.
