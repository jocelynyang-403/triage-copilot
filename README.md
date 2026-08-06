# Triage Copilot

A multi-agent copilot that triages inbound ops/support requests — classify,
prioritize, check a knowledge base, route, and (planned) draft a reply — with a
human approving before anything is sent. Orchestrated as a LangGraph state machine.

All data is synthetic. This is a portfolio project.

## Why

Manually triaging a shared request channel — reading each message, judging
severity, deciding who owns it — is a repetitive time sink. Triage Copilot
automates the judgment steps and keeps a human in the loop on anything
customer-facing.

## Architecture

Pipeline: `intake → classify → knowledge → route` (→ planned: `draft → human_review`).

- **intake** — deterministic regex normalization + entity extraction; no LLM call.
- **classify** — Claude Haiku with Pydantic structured output → category (5 classes) + priority (P0–P3), temperature 0 for reproducible evals.
- **knowledge** — RAG over runbooks (Chroma + local embeddings); emits a `kb_confidence` retrieval score.
- **route** — conditional auto-reply vs. escalate. Two selectable strategies: **v1** a `kb_confidence` threshold, **v2** an LLM grounding check. A shared allowlist guard makes only P2/P3 tickets eligible for auto-reply.
- **human_review** *(planned)* — LangGraph `interrupt()` pause for human approval before sending.

Full plan: `docs/triage-copilot-implementation-plan.md`.

## Stack

Python 3.9 · LangGraph · Anthropic API (via `langchain-anthropic`) · Chroma +
local `sentence-transformers` embeddings · Slack Bolt (Socket Mode, planned) ·
SQLite (planned) · Streamlit (planned)

## Status

Active — Phase 2 mostly complete.

- **Phase 0 ✅** — repo scaffold, pinned env, hand-verified synthetic dataset (52 tickets, 5 categories, priority + expected routing action, incl. 2 `not_found` negatives), stratified dev/test split (16/36, fixed seed, committed before any score was computed).
- **Phase 1 ✅** — LangGraph pipeline `intake → classify → route`, runnable end-to-end.
- **Phase 2 🔶** — RAG retrieval + conditional routing (v1 threshold / v2 grounding) complete and measured on the dev split. **Remaining:** draft generation and the human-in-the-loop approval interrupt.
- **Phases 3–5 ⬜** — Slack integration, eval harness + instrumentation, dashboard: not yet implemented.

### Finding so far

A controlled retrieval experiment tested whether `kb_confidence` alone can separate
answerable (auto-reply) from actionable (escalate) tickets. It can't — the
confidence bands overlap, and the best single threshold still misses 3 of 11
contested dev rows (~73% ceiling), because retrieval similarity captures topical
overlap, not answerability. This motivated the v2 LLM grounding check. On the dev
split v1 and v2 score the same accuracy (8/11 contested) but v2 has a safer error
profile: zero false auto-replies (precision 3/3) at the cost of recall. Numbers are
reported as raw counts on a small synthetic dev set — they demonstrate the
mechanism, not statistical significance.

## Data

All ticket data is **synthetic** — drafted with Claude, then every label
hand-verified. No real customer data. The knowledge-base topics were chosen after
reading the answerable tickets, so retrieval measures separability *within covered
topics*, not blind discovery — a documented limitation, not hidden. Any time-saved
or ROI figures that may appear later are **modeled estimates on a synthetic
workload**, never measured outcomes.

## Setup
git clone https://github.com/jocelynyang-403/triage-copilot.git
cd triage-copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env # then add your ANTHROPIC_API_KEY
`scripts/gen_dataset.py` regenerates the *draft* dataset (labels then require hand
review). `scripts/index_kb.py` builds the local Chroma KB index. Retrieval and the
collision probe (`scripts/probe_collision.py`) run offline with no API key; routing
v2 and classification need the key.
