"""Central configuration: env loading, API keys, model IDs, and categories."""

import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
    )

CLASSIFY_MODEL = "claude-haiku-4-5"   # verify current ID at build time
GENERATE_MODEL = "claude-sonnet-5"    # verify current ID at build time
DATASET_MODEL = "claude-sonnet-5"     # verify current ID at build time

CATEGORIES = ["billing", "bug", "access_request", "sales_lead", "other"]
