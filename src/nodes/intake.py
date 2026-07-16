"""Intake node: normalize and ingest incoming ops tickets.

Deterministic only — no LLM call in Phase 1. This keeps intake free, fast, and
fully reproducible across eval runs.
"""
import re

# Slack channel refs: <#C0123|general> -> keep the label ("general").
_CHANNEL_RE = re.compile(r"<#C[^|>]*\|([^>]+)>")
# User mentions: <@U0123> -> drop entirely (no useful label).
_MENTION_RE = re.compile(r"<@[UW][^>]*>")
# Links with a label: <https://x.com|click here> -> keep the label.
_LINK_LABELED_RE = re.compile(r"<(https?://[^|>]+)\|([^>]+)>")
# Bare links: <https://x.com> -> keep the URL itself.
_LINK_BARE_RE = re.compile(r"<(https?://[^>]+)>")
# Emoji shortcodes: :tada: -> drop.
_EMOJI_RE = re.compile(r":[a-z0-9_+-]+:")

_ID_RE = re.compile(r"#([A-Za-z0-9][A-Za-z0-9\-]*)")
_CODE_RE = re.compile(r"\b([A-Z]{2,}-\d+)\b")

_WS_RE = re.compile(r"\s+")


def _dedupe(items):
    """Deduplicate while preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _normalize(text):
    text = _CHANNEL_RE.sub(r"\1", text)
    text = _MENTION_RE.sub("", text)
    text = _LINK_LABELED_RE.sub(r"\2", text)
    text = _LINK_BARE_RE.sub(r"\1", text)
    text = _EMOJI_RE.sub("", text)
    # Decode the handful of HTML entities Slack emits. Do this after artifact
    # stripping so a decoded '<' cannot resurrect a fake mention/link.
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Do NOT lowercase: ALL-CAPS is a priority signal and codes are uppercase.
    text = _WS_RE.sub(" ", text).strip()
    return text


def intake_node(state):
    raw = state["raw_text"]
    normalized = _normalize(raw)

    ids = _dedupe(_ID_RE.findall(normalized))
    codes = _dedupe(_CODE_RE.findall(normalized))

    return {
        "normalized": normalized,
        "entities": {"ids": ids, "codes": codes},
        "trace": ["intake: {} ids, {} codes".format(len(ids), len(codes))],
    }
