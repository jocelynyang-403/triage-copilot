"""Knowledge node: retrieve KB context and expose a retrieval-confidence signal.

Read-only over the Chroma store built by scripts/index_kb.py. Local embeddings, so
no API key and no token cost. Returns kb_confidence = the top relevance score in
[0, 1]. Phase 2 routing consumes it; the Phase 4 story is whether this single scalar
can separate answerable (auto_reply) from actionable (escalate) tickets.

Deliberately does NOT import src.config: this node needs no API key.
"""
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_STORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "kb_store"))
_COLLECTION = "triage_kb"
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_knowledge_node(k=3):
    """Load embedder + store once, return the node closure (mirrors make_classify_node)."""
    if not os.path.isdir(_STORE_DIR):
        raise RuntimeError(
            "KB store not found at {}. Run `python -m scripts.index_kb` first.".format(_STORE_DIR)
        )
    emb = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    db = Chroma(persist_directory=_STORE_DIR, collection_name=_COLLECTION, embedding_function=emb)

    def knowledge_node(state):
        query = state.get("normalized") or state["raw_text"]
        hits = db.similarity_search_with_relevance_scores(query, k=k)
        top = float(hits[0][1]) if hits else 0.0
        top_source = hits[0][0].metadata.get("source") if hits else None
        return {
            "kb_chunks": [doc.page_content for doc, _ in hits],
            "kb_confidence": top,
            "trace": ["kb: top={:.3f} src={}".format(top, top_source)],
        }

    return knowledge_node
