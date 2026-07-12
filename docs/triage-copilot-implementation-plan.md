# Triage Copilot — Implementation Plan

**What it is:** A multi-agent copilot for an internal ops/support team. It watches a Slack request channel, and for each incoming message a LangGraph agent pipeline **classifies → prioritizes → checks a knowledge base → decides routing → drafts a reply** — then pauses for a human to approve before anything is sent. Everything is instrumented (time-saved, cost, adoption) and covered by an eval harness.

**Stack:** Python 3.11 · LangGraph (orchestration) · Anthropic API via `langchain-anthropic` (`ChatAnthropic`) · local `sentence-transformers` embeddings + Chroma (RAG) · Slack Bolt (Socket Mode) · SQLite (run log) · Streamlit (dashboard).

**Repo layout (target):**
```
triage-copilot/
├── src/
│   ├── graph.py            # LangGraph assembly + compile
│   ├── state.py            # TriageState TypedDict
│   ├── nodes/              # one file per agent node
│   │   ├── intake.py
│   │   ├── classify.py
│   │   ├── knowledge.py
│   │   ├── route.py
│   │   └── human_review.py
│   ├── prompts/            # versioned prompts (v1.py, v2.py) for eval comparison
│   ├── kb/                 # 10–20 seed runbook markdown files
│   ├── slack_app.py        # Bolt Socket Mode handler
│   ├── instrument.py       # SQLite run logger
│   └── config.py
├── evals/
│   ├── golden.jsonl        # 40–60 labeled synthetic tickets
│   ├── run_evals.py
│   └── results/            # per-prompt-version metrics
├── dashboard.py            # Streamlit
├── scripts/gen_dataset.py  # synthetic ticket generator
└── README.md
```

> **Core deliverable = Phases 0–4** (≈3 focused days). Phase 5 is polish/stretch. Ship the README the moment Phase 4 is green — a graded, instrumented, eval-covered agent beats a half-polished one with no numbers.

---

## Phase 0 — Foundation & Synthetic Dataset

**Goal:** Runnable repo skeleton, dependencies pinned, secrets loaded, and a labeled synthetic ticket dataset that every later phase depends on.

**Prerequisites:** Anthropic API key. (Slack tokens deferred to Phase 3 — do not block here on Slack setup.)

### Task 0.1 — Repo + env `feature/p0-scaffold`

- `python -m venv .venv && source .venv/bin/activate`
- `pip install langgraph langchain-anthropic langchain-chroma langchain-huggingface sentence-transformers chromadb slack-bolt streamlit python-dotenv`
- `pip freeze > requirements.txt`
- `.env` with `ANTHROPIC_API_KEY`; load via `python-dotenv` in `config.py`. **Add `.env` to `.gitignore` before the first commit** — do not leak the key into git history.

### Task 0.2 — Synthetic dataset generator `feature/p0-dataset`

**File: `scripts/gen_dataset.py`** (new)

Generate 40–60 realistic tickets across the 5 categories with ground-truth labels. Use Claude to draft varied phrasings, then **you hand-verify every label** — an eval set is only as trustworthy as its labels.

```python
# Categories and target priority the ROUTER should reach.
# expected_action drives the routing-accuracy metric in Phase 4.
CATEGORIES = ["billing", "bug", "access_request", "sales_lead", "other"]

# Emit JSONL: one object per line.
# {"id","raw_text","category","priority","expected_action","expected_destination"}
# expected_action ∈ {"auto_reply","escalate"}
```

> **Label the *expected* behavior, not just the category.** A "P0 production outage" bug should have `expected_action: "escalate"` even though it's a bug; a "how do I reset my password" access_request should be `auto_reply` if the KB covers it. If you only label category, your Phase 4 routing metric measures nothing.

### Definition of Done — Phase 0

- [ ] `pip install -r requirements.txt` reproduces the env from clean
- [ ] `.env` loads; a one-line `ChatAnthropic(...).invoke("ping")` returns text
- [ ] `evals/golden.jsonl` has ≥40 rows, every label human-verified, roughly balanced across categories
- [ ] `.env` and `.venv/` are gitignored

---

## Phase 1 — Core Agent Graph (Intake → Classify → Route)

**Goal:** A linear LangGraph pipeline that takes raw ticket text and produces `{category, priority, destination, action, draft_reply}` end-to-end, runnable from a script (no Slack, no RAG yet).

**Prerequisites:** Phase 0 complete.

### Task 1.1 — Graph state `feature/p1-state`

**File: `src/state.py`** (new)

```python
from typing import TypedDict, Optional, Annotated
from operator import add

class TriageState(TypedDict):
    # inputs
    raw_text: str
    requester: str
    channel: str
    # intake
    normalized: Optional[str]
    entities: dict
    # classification
    category: Optional[str]      # billing|bug|access_request|sales_lead|other
    priority: Optional[str]      # P0|P1|P2|P3
    # knowledge (Phase 2)
    kb_chunks: list
    kb_confidence: float
    # routing
    destination: Optional[str]
    action: Optional[str]        # auto_reply|escalate
    draft_reply: Optional[str]
    # human-in-the-loop (Phase 2)
    approved: Optional[bool]
    # instrumentation — accumulates across nodes, so it needs a reducer
    trace: Annotated[list, add]
```

> **`trace` needs the `add` reducer, the scalar fields must not.** Any state key written by more than one node needs an `Annotated[..., reducer]` or LangGraph raises `InvalidUpdateError` on concurrent writes. Single-writer fields (each node owns its output) stay plain — adding a reducer to them silently duplicates data instead.

### Task 1.2 — Intake node `feature/p1-intake`

**File: `src/nodes/intake.py`** — normalize the raw Slack text (strip mentions/emoji, pull obvious entities like order IDs, error codes). Cheap deterministic cleanup first; only call the LLM if you actually need entity extraction. Return `{"normalized": ..., "entities": {...}, "trace": ["intake done"]}`.

### Task 1.3 — Classifier node `feature/p1-classify`

**File: `src/nodes/classify.py`** (new)

Use **structured output** so the classifier can never return unparseable text — this is what makes it eval-able.

```python
from typing import Literal
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from src.prompts.v1 import CLASSIFY_SYSTEM

class Classification(BaseModel):
    category: Literal["billing","bug","access_request","sales_lead","other"]
    priority: Literal["P0","P1","P2","P3"]
    reasoning: str = Field(description="one sentence")

# Haiku is the right call here: classification is cheap + latency-sensitive.
# Reserve Sonnet for draft *generation* in Phase 2. Verify current model IDs at build time.
_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0).with_structured_output(Classification)

def classify_node(state):
    r = _llm.invoke([("system", CLASSIFY_SYSTEM),
                     ("human", state.get("normalized") or state["raw_text"])])
    return {"category": r.category, "priority": r.priority,
            "trace": [f"classify: {r.category}/{r.priority}"]}
```

### Task 1.4 — Router node (stub) + assembly `feature/p1-graph`

**File: `src/nodes/route.py`** — for Phase 1, a rule-based first pass: map (category, priority) → destination team, set `action = "escalate"`, and leave `draft_reply = None`. RAG-driven auto-reply arrives in Phase 2.

**File: `src/graph.py`** (new)

```python
from langgraph.graph import StateGraph, START, END
from src.state import TriageState
# ... import nodes

def build_graph():
    b = StateGraph(TriageState)
    b.add_node("intake", intake_node)
    b.add_node("classify", classify_node)
    b.add_node("route", route_node)
    b.add_edge(START, "intake")
    b.add_edge("intake", "classify")
    b.add_edge("classify", "route")
    b.add_edge("route", END)
    return b.compile()   # no checkpointer yet — added in Phase 2 for interrupts
```

### Definition of Done — Phase 1

- [ ] `python -c "from src.graph import build_graph; print(build_graph().invoke({...}))"` returns a fully-populated state
- [ ] Classifier returns a valid `Classification` for all 5 categories (spot-check 5 golden rows)
- [ ] Router maps every (category, priority) pair to a non-null destination — no `None` leaks through

### Watch-outs — Phase 1

- **Prompt lives in a versioned file, not inline.** Put `CLASSIFY_SYSTEM` in `src/prompts/v1.py` from day one. Phase 4's whole value is comparing `v1` vs `v2` metrics; if the prompt is buried in the node you can't diff it cleanly.
- **`temperature=0` for the classifier.** Non-determinism here makes your eval numbers noisy and non-reproducible. Save temperature for generation, not classification.

---

## Phase 2 — Knowledge Agent (RAG) + Conditional Routing + Human-in-the-Loop

**Goal:** Add a RAG knowledge agent, make routing *conditional* on retrieval confidence, draft auto-replies with Claude, and pause the graph for human approval before sending.

**Prerequisites:** Phase 1 graph runs. 10–20 seed runbook markdown files in `src/kb/`.

### Task 2.1 — Build the KB index `feature/p2-kb-index`

> **API correction — Anthropic has no embeddings endpoint.** Do not reach for `ChatAnthropic` to embed. Use a **local `sentence-transformers` model (free, no API key)** or Voyage AI (Anthropic's recommended embeddings partner, paid). For a side project, local wins: zero cost, offline, fast enough.

**File: `src/nodes/knowledge.py`** (new)

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_db = Chroma(persist_directory="./kb_store", embedding_function=_emb)

def knowledge_node(state):
    query = state.get("normalized") or state["raw_text"]
    hits = _db.similarity_search_with_relevance_scores(query, k=3)
    top = hits[0][1] if hits else 0.0
    return {"kb_chunks": [d.page_content for d, _ in hits],
            "kb_confidence": top,
            "trace": [f"kb: top_score={top:.2f}"]}
```

A one-off `scripts/index_kb.py` chunks the `src/kb/*.md` files and writes them into Chroma once.

### Task 2.2 — Conditional routing + draft generation `feature/p2-conditional-route`

Rewrite the router: if `kb_confidence >= THRESHOLD` **and** priority isn't P0 → draft an auto-reply grounded in `kb_chunks`; otherwise escalate to a human team with no draft.

```python
from langgraph.graph import END
from typing import Literal

THRESHOLD = 0.5   # tune against golden set in Phase 4; store-dependent, see watch-out

def route_decision(state) -> Literal["draft_reply", "escalate"]:
    if state["priority"] == "P0":            # never auto-reply a P0
        return "escalate"
    return "draft_reply" if state["kb_confidence"] >= THRESHOLD else "escalate"

# wiring:
# b.add_conditional_edges("knowledge", route_decision,
#     {"draft_reply": "draft", "escalate": "human_review"})
# b.add_edge("draft", "human_review")   # even auto-replies get approved
```

Draft node uses **Sonnet** (quality matters for customer-facing text) with a strict grounding prompt: answer *only* from `kb_chunks`, and if the chunks don't cover it, say so — same RAG-guardrail discipline as Paw Sync's dog-expert endpoint.

### Task 2.3 — Human-in-the-loop interrupt `feature/p2-hitl`

**File: `src/nodes/human_review.py`** (new) — pause the graph and surface the draft for approval.

```python
from langgraph.types import interrupt

def human_review_node(state):
    decision = interrupt({          # graph halts here; state is checkpointed
        "category": state["category"], "priority": state["priority"],
        "destination": state["destination"], "action": state["action"],
        "draft_reply": state.get("draft_reply"),
    })
    return {"approved": decision["approved"],
            "trace": [f"human: approved={decision['approved']}"]}
```

Recompile **with a checkpointer** — `interrupt()` requires one to persist the paused state:

```python
from langgraph.checkpoint.memory import MemorySaver
# graph = b.compile(checkpointer=MemorySaver())   # dev
# resume later with a thread_id-scoped config:
#   config = {"configurable": {"thread_id": ticket_id}}
#   graph.invoke({...}, config)                    # runs, then pauses at interrupt
#   graph.invoke(Command(resume={"approved": True}), config)   # continues
```

> **Verify the interrupt/resume API surface at build time.** LangGraph's `interrupt()` + `Command(resume=...)` pattern and the state-snapshot shape (`graph.get_state(config).tasks[...].interrupts`) have moved between minor versions. Pin your LangGraph version in `requirements.txt` and confirm the exact call against the installed version's docs before wiring Slack.

### Definition of Done — Phase 2

- [ ] A ticket the KB covers (e.g. "how do I reset MFA") → `action = auto_reply` with a grounded draft
- [ ] A ticket the KB doesn't cover → `action = escalate`, `draft_reply = None`
- [ ] Any P0 always escalates regardless of KB confidence
- [ ] `graph.invoke({...}, config)` pauses at `human_review`; a follow-up `Command(resume=...)` completes the run and sets `approved`

### Watch-outs — Phase 2

- **Relevance-score semantics are store-specific.** `similarity_search_with_relevance_scores` returns a normalized 0–1 score, but the mapping from raw cosine distance depends on the vector store. Don't hardcode `0.5` on faith — eyeball the scores your golden queries actually produce (Phase 4) and set the threshold from data.
- **Checkpointer choice = durability tradeoff.** `MemorySaver` loses paused tickets on process restart. The Slack app is long-running so it's fine for the demo, but if you want approvals to survive a crash, swap in `SqliteSaver`. State it in the README either way.
- **Ground the draft or you've built a hallucination machine.** The draft prompt must forbid using anything outside `kb_chunks`. This is the exact `faithfulness` property your Phase 4 LLM-judge scores — build it grounded from the start.

---

## Phase 3 — Slack Integration

**Goal:** Read real messages from a Slack channel, run the graph, post the draft with Approve/Reject buttons, and resume the graph on button click. No public URL — Socket Mode.

**Prerequisites:** Phase 2 graph pauses/resumes correctly. A Slack workspace you can create an app in.

### Task 3.1 — Slack app config (no code)

In the Slack app dashboard: enable **Socket Mode**, subscribe to `message.channels`, enable **Interactivity**, add bot scopes (`chat:write`, `channels:history`). Grab `SLACK_BOT_TOKEN` (`xoxb-`) and `SLACK_APP_TOKEN` (`xapp-`) into `.env`.

### Task 3.2 — Bolt handler `feature/p3-slack`

**File: `src/slack_app.py`** (new)

```python
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langgraph.types import Command
from src.graph import build_graph

app = App(token=os.environ["SLACK_BOT_TOKEN"])
graph = build_graph()   # compiled WITH checkpointer

@app.event("message")
def on_message(event, say):
    # ignore the bot's own posts, edits, joins, threads — or you'll loop on yourself
    if event.get("bot_id") or event.get("subtype"):
        return
    ticket_id = event["ts"]                       # stable per-message id → thread_id
    cfg = {"configurable": {"thread_id": ticket_id}}
    graph.invoke({"raw_text": event["text"], "requester": event["user"],
                  "channel": event["channel"], "entities": {}}, cfg)
    payload = graph.get_state(cfg).tasks[0].interrupts[0].value   # verify shape vs version
    say(blocks=approval_blocks(payload, ticket_id), text="New ticket triaged")

@app.action("approve_ticket")
def approve(ack, body):
    ack()
    ticket_id = body["actions"][0]["value"]
    graph.invoke(Command(resume={"approved": True}),
                 {"configurable": {"thread_id": ticket_id}})

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
```

> **Guard against the self-reply loop first.** The bot posts into the same channel it listens to. Without the `bot_id`/`subtype` guard, every approval card the bot posts re-triggers `on_message` → infinite tickets. This is the #1 way this kind of app melts down — write the guard before you test.

### Definition of Done — Phase 3

- [ ] Posting a ticket in the channel yields an approval card within a few seconds
- [ ] "Approve" resumes the graph and the run completes (check the trace / log)
- [ ] The bot never triages its own messages (no loop)
- [ ] Killing and restarting the app with `MemorySaver` documented as losing in-flight approvals (or `SqliteSaver` swapped in)

---

## Phase 4 — Eval Harness & Instrumentation

**Goal:** The differentiators. A reproducible eval harness that scores the pipeline across prompt versions, plus per-run instrumentation capturing time-saved, adoption, and cost.

**Prerequisites:** Phases 1–2 (graph produces gradeable outputs). Slack not required — evals run the graph directly.

### Task 4.1 — Eval runner `feature/p4-evals`

**File: `evals/run_evals.py`** (new)

```python
import json, argparse
from src.graph import build_graph

def run(prompt_version: str):
    graph = build_graph()                       # graph reads prompts/{version}
    rows = [json.loads(l) for l in open("evals/golden.jsonl")]
    cat = pri = act = 0
    for r in rows:
        cfg = {"configurable": {"thread_id": f"eval-{r['id']}"}}
        out = graph.invoke({"raw_text": r["raw_text"], "requester": "eval",
                            "channel": "eval", "entities": {}}, cfg)
        cat += out["category"]   == r["category"]
        pri += out["priority"]   == r["priority"]
        act += out["action"]     == r["expected_action"]
    n = len(rows)
    return {"version": prompt_version, "n": n,
            "category_acc": cat/n, "priority_acc": pri/n, "routing_acc": act/n}
```

Add an **LLM-as-judge** faithfulness check on drafted replies: given the draft + `kb_chunks`, score 0/1 whether every claim is supported.

> **Judge with a different (stronger) model than the generator.** If Sonnet drafts *and* Sonnet judges, you get self-preference bias — inflated faithfulness scores. Use the stronger tier as judge (or at minimum a different model), and hand-audit ~10 judgments to confirm the judge tracks your own read.

Write results to `evals/results/{version}.json`, and print a comparison table across versions — that table *is* your "iterative loop" evidence for the interview.

### Task 4.2 — Instrumentation `feature/p4-instrument`

**File: `src/instrument.py`** (new) — SQLite `runs` table logged on every *real* (Slack) run:

```python
# columns: ts, ticket_id, category, priority, action, auto_resolved(bool),
#          minutes_saved, input_tokens, output_tokens, cost_usd
```

- `minutes_saved`: a static per-category lookup (billing=5, access_request=8, ...). **This is modeled, not measured — say so in the README.**
- Token counts: pull from `response.usage_metadata` on the `ChatAnthropic` calls; convert to `cost_usd` with current per-token pricing (verify at build time).
- `auto_resolved`: `action == "auto_reply" and approved`.

### Definition of Done — Phase 4

- [ ] `python evals/run_evals.py --prompt-version v1` prints category/priority/routing accuracy over the full golden set
- [ ] A `v2` prompt produces a *different* score, and the comparison table shows the delta
- [ ] LLM-judge faithfulness scores the drafts; ~10 judgments hand-audited
- [ ] Every Slack run writes a `runs` row with tokens + cost + minutes_saved
- [ ] Thresholds (KB confidence) tuned against golden data, not guessed

### Watch-outs — Phase 4

- **Eval determinism.** Keep `temperature=0` everywhere the eval touches, and pin model IDs — otherwise your v1-vs-v2 delta is noise, not signal. Note the exact model IDs in the results file.
- **Small-N honesty.** 40–60 tickets is enough to show the loop, not to claim statistical significance. Report raw counts (e.g. "47/52") alongside percentages, and don't over-claim in the README.

---

## Phase 5 — Dashboard & Polish (stretch)

**Goal:** Surface the instrumentation to a non-technical audience, add a second integration, and gate quality in CI. This is where "student demo" becomes "internal product."

**Prerequisites:** Phase 4 logging live.

### Task 5.1 — Streamlit dashboard `feature/p5-dashboard`

**File: `dashboard.py`** — read the SQLite `runs` table and render: ticket volume over time, category distribution, an approval-queue count, and a headline **"~X minutes saved this week (modeled)"**. `streamlit run dashboard.py`.

### Task 5.2 — Notion escalation sink `feature/p5-notion` *(second integration → bonus points)*

On `escalate`, create a row in a Notion database via `notion-client` (`NOTION_TOKEN`, `NOTION_DB_ID`). This gives you the "ties together Slack + Notion" story the JD explicitly calls out.

### Task 5.3 — CI eval regression gate `feature/p5-ci`

**File: `.github/workflows/evals.yml`** — run `evals/run_evals.py` on push; fail if category accuracy drops below a floor. Shows you treat prompts like code. (Store `ANTHROPIC_API_KEY` as a repo secret.)

### Task 5.4 — README `feature/p5-readme`

The README is what a recruiter actually reads. Include: a **mermaid architecture diagram** of the agent graph, a demo GIF (Slack ticket → approval card → resolved), the **v1-vs-v2 metrics table**, and a short "How I'd productionize this" section (swap Chroma→pgvector, MemorySaver→Postgres checkpointer, Socket Mode→Events API behind a real endpoint, Redis-backed rate limiting).

### Definition of Done — Phase 5

- [ ] `streamlit run dashboard.py` shows live volume/category/time-saved from real runs
- [ ] Escalated tickets appear as Notion rows
- [ ] CI runs evals on push and can fail the build
- [ ] README has architecture diagram, demo GIF, metrics table, productionization notes

---

## Integrity checklist (before it goes public)

- Synthetic data is labeled **synthetic** in the README; ROI is framed as **"modeled on a synthetic workload,"** never "saved real users X hours."
- Eval numbers report raw counts, not just percentages; model IDs + prompt versions are recorded.
- No secrets in git history (`.env`, tokens, Notion IDs all gitignored).
- Completion-tense claims only for what actually runs; anything stubbed is marked as such.

---

## Appendix — Scale AI JD coverage map

| JD line | Covered by |
|---|---|
| Multi-step agentic workflows (LangChain/LangGraph) | Phases 1–2 (LangGraph state machine + conditional edges) |
| API-connected automations tying together Slack/Notion | Phases 3, 5.2 |
| Human-in-the-loop prompts | Phase 2.3 interrupt + Phase 3 approval buttons |
| Lightweight dashboards surfacing AI outputs to business teams | Phase 5.1 |
| Instrument work / time-saved / adoption metrics | Phase 4.2 |
| Tag projects to value categories (ROI framework) | Phase 4.2 category-keyed minutes_saved |
| LLM APIs — built something real (Anthropic) | Whole project, `ChatAnthropic` throughout |
| RAG pipeline + prompt engineering + LLM evals (bonus) | Phase 2 (RAG), prompts/ versioning, Phase 4 (evals) |
| Multi-agent architecture (bonus) | Phase 1–2 (intake/classify/knowledge/route/review nodes) |
| BizOps/ops automation context (bonus) | The scenario itself |
| Active GitHub side project (bonus) | This repo |
