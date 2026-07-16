"""Shared graph state for the triage copilot."""
from operator import add
from typing import Optional
from typing_extensions import TypedDict, Annotated


class TriageState(TypedDict, total=False):
    # --- inputs ---
    raw_text: str
    requester: str
    channel: str
    # --- intake ---
    normalized: Optional[str]
    entities: dict
    # --- classification ---
    category: Optional[str]       # billing|bug|access_request|sales_lead|other
    priority: Optional[str]       # P0|P1|P2|P3
    # --- knowledge (Phase 2) ---
    kb_chunks: list
    kb_confidence: float
    # --- routing ---
    destination: Optional[str]
    action: Optional[str]         # auto_reply|escalate
    draft_reply: Optional[str]
    # --- human-in-the-loop (Phase 2) ---
    approved: Optional[bool]
    # --- instrumentation ---
    # `trace` is the ONLY multi-writer key, so it is the only one with a reducer.
    # Adding a reducer to a single-writer field would silently duplicate data.
    trace: Annotated[list, add]
