"""Central configuration: env loading, API keys, model IDs, and categories."""

import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
    )

CLASSIFY_MODEL = "claude-haiku-4-5"   # supports temperature (verified 2026-07-12)
GENERATE_MODEL = "claude-sonnet-5"    # temperature DEPRECATED — do NOT pass temperature=
DATASET_MODEL  = "claude-sonnet-5"    # temperature DEPRECATED — do NOT pass temperature=

CATEGORIES = ["billing", "bug", "access_request", "sales_lead", "other"]

DESTINATIONS = ["Sales", "Billing Ops", "Engineering", "IT / Access", "Support", "Human Review"]
PRIORITIES = ["P0", "P1", "P2", "P3"]
