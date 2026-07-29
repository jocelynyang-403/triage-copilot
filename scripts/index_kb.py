"""Index the KB runbooks into a persistent Chroma store for Phase 2 retrieval.

Local embeddings (sentence-transformers/all-MiniLM-L6-v2): no Anthropic API key and
no per-call token cost. The first run downloads the ~90MB MiniLM model; after that it
is fully local. Rebuilds the store from scratch every run so re-indexing can never
append duplicate chunks.

Run from the repo root:  python -m scripts.index_kb
"""
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "kb"))
STORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kb_store"))
COLLECTION = "triage_kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Split each runbook on H2 (## ) sections. Every chunk is prefixed with the H1
# title so an embedded section is never orphaned from its document topic.
_H2_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def _split_markdown(text, source):
    title = ""
    for ln in text.splitlines():
        if ln.startswith("# "):
            title = ln[2:].strip()
            break

    docs = []
    matches = list(_H2_RE.finditer(text))
    if not matches:
        body = text.strip()
        if body:
            docs.append(Document(page_content=body,
                                 metadata={"source": source, "section": title or source}))
        return docs

    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        prefix = "{} — {}".format(title, header) if title else header
        content = "{}\n{}".format(prefix, section)
        docs.append(Document(page_content=content,
                             metadata={"source": source, "section": header}))
    return docs


def build():
    paths = sorted(glob.glob(os.path.join(KB_DIR, "*.md")))
    if not paths:
        raise SystemExit("No KB markdown files found in {}".format(KB_DIR))

    all_docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        all_docs.extend(_split_markdown(text, os.path.basename(p)))

    print("KB files: {} | chunks: {}".format(len(paths), len(all_docs)))

    # Rebuild from scratch: never append into an existing collection.
    shutil.rmtree(STORE_DIR, ignore_errors=True)

    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(
        documents=all_docs,
        embedding=emb,
        persist_directory=STORE_DIR,
        collection_name=COLLECTION,
        # Cosine so relevance = 1 - cosine_distance lands in [0, 1] and is interpretable.
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("Indexed into {}".format(STORE_DIR))

    # Self-check: sample queries, so indexing is verified not assumed.
    db = Chroma(persist_directory=STORE_DIR, collection_name=COLLECTION, embedding_function=emb)
    for q in ["how do I get access to a shared drive", "do you offer nonprofit pricing"]:
        hits = db.similarity_search_with_relevance_scores(q, k=1)
        if hits:
            doc, score = hits[0]
            print("  probe {!r} -> {:.3f} [{}]".format(q, score, doc.metadata.get("source")))
        else:
            print("  probe {!r} -> NO HITS".format(q))


if __name__ == "__main__":
    build()
