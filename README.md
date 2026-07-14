Update README.md in the triage-copilot repo to reflect that Phase 0 is complete.
Keep it concise and recruiter-readable. Rewrite the file to contain these sections
in this order. Do not add badges, do not invent metrics, do not claim anything is
working that isn't.

# Triage Copilot

One-line description: A multi-agent copilot that triages inbound ops/support
requests — classifies, prioritizes, checks a knowledge base, routes, and drafts a
reply — with a human approving before anything is sent.

## Why
2-3 sentences: manual triage of a shared request channel is a time sink. This
system automates the judgment steps and keeps a human in the loop on the output.

## Architecture (planned)
A short bullet list of the agent nodes and their roles: intake (normalize),
classify (category + priority), knowledge (RAG over runbooks), route (conditional
on retrieval confidence), human_review (approval interrupt). Note that these are
orchestrated as a LangGraph state machine.

## Stack
Python 3.9 · LangGraph · Anthropic API (via langchain-anthropic) · Chroma +
local sentence-transformers embeddings · Slack Bolt (Socket Mode) · SQLite ·
Streamlit

## Status
Phase 0 complete — repo scaffold, pinned environment, and a hand-verified
synthetic eval dataset (52 labeled tickets across 5 categories with priority and
expected routing action, including 2 not_found negatives).
Phases 1-5 (agent graph, RAG, Slack, evals, dashboard) are not yet implemented.
See docs/triage-copilot-implementation-plan.md for the full plan.

## Data
State plainly that all ticket data is SYNTHETIC — generated with Claude, then
every label hand-verified. No real customer data. Any time-saved or ROI figures
that appear later are modeled estimates on a synthetic workload, not measured
outcomes.

## Setup
Brief: clone, python -m venv .venv, source .venv/bin/activate,
pip install -r requirements.txt, cp .env.example .env and add an
ANTHROPIC_API_KEY. Note that scripts/gen_dataset.py regenerates the draft
dataset (labels then require hand review).

Show me the diff, then STOP. Do not commit, do not push.