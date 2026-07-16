"""Versioned prompt loading. Phase 4 compares metrics across versions."""
import importlib
import re

_VERSION_RE = re.compile(r"^v[0-9]+$")


def load_prompts(version="v1"):
    if not _VERSION_RE.match(version):
        raise ValueError("Invalid prompt version: {!r} (expected e.g. 'v1')".format(version))
    try:
        return importlib.import_module("src.prompts." + version)
    except ImportError as e:
        raise ImportError("Prompt version {!r} not found: {}".format(version, e))
