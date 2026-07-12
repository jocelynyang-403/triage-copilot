# triage-copilot

A multi-agent ops-ticket triage copilot that classifies, enriches, and routes incoming support tickets with human-in-the-loop review.

## Stack

- **LangGraph** — multi-agent orchestration
- **Anthropic API** (via `langchain-anthropic`) — classification & generation
- **Chroma** (via `langchain-chroma` + `langchain-huggingface` / `sentence-transformers`) — knowledge base retrieval
- **Slack** (`slack-bolt`) — human-in-the-loop review & notifications
- **Streamlit** — metrics & run-review dashboard
- **pydantic**, **python-dotenv** — config & validation

## Status

Phase 0 — scaffold. Uses synthetic data; not production.
