"""
rag/retrieval.py
────────────────
RAG retrieval layer for the OSP orbital intelligence system.

Builds a FAISS vector index over the maritime knowledge base and exposes
a simple retrieve(query) → list[KnowledgeChunk] interface.

Two embedding backends are supported:
  1. sentence-transformers (local, no API key needed) — default
  2. Google Gemini text-embedding-004 (cloud, higher quality)

The retrieved chunks are injected into the LLM system prompt to ground the
analyst's reasoning in verifiable domain knowledge rather than hallucinated
maritime facts.

Architecture:
  knowledge_base.py   →  embed all chunks  →  FAISS index (persisted to disk)
                                                    ↓
  llm_analyst.py ← retrieve(query) ← anomaly description string
                         ↓
  [retrieved chunks injected into system prompt]  →  LLM response

Usage:
    from rag.retrieval import OrbitalRAG
    rag = OrbitalRAG()
    chunks = rag.retrieve("vessel loitering Indian Ocean EEZ", k=3)
    context = rag.format_context(chunks)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Index persistence path
_DEFAULT_INDEX_DIR = Path(__file__).parent / "vector_store"


# ── Embedding backends ─────────────────────────────────────────────────────────

def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    """
    Local embedding using sentence-transformers.
    Model: all-MiniLM-L6-v2 (~22MB, runs on CPU in <100ms for small batches)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def _embed_gemini(texts: list[str], api_key: Optional[str] = None) -> list[list[float]]:
    """
    Cloud embedding using Google Gemini text-embedding-004.
    Higher dimensional (768-d) and domain-aware.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed.")

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY required for Gemini embedding backend.")
    genai.configure(api_key=key)

    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )
        embeddings.append(result["embedding"])
    return embeddings


# ── OrbitalRAG ─────────────────────────────────────────────────────────────────

class OrbitalRAG:
    """
    Retrieval-Augmented Generation layer for the OSP orbital intelligence system.

    Embeds the maritime knowledge base into a FAISS vector index.
    At analysis time, retrieves the most relevant knowledge chunks
    for a given anomaly query and formats them for LLM prompt injection.
    """

    def __init__(
        self,
        embedding_backend: str = "sentence_transformers",   # or "gemini"
        index_dir: Path = _DEFAULT_INDEX_DIR,
        api_key: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        self.backend   = embedding_backend
        self.index_dir = Path(index_dir)
        self.api_key   = api_key
        self._index    = None     # FAISS index (lazy loaded)
        self._chunks   = []       # parallel list of KnowledgeChunk objects
        self._dim      = None     # embedding dimension

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Load or build the index
        if force_rebuild or not self._index_exists():
            log.info("RAG: Building vector index from knowledge base ...")
            self._build_index()
        else:
            log.info("RAG: Loading persisted vector index ...")
            self._load_index()

    # ── Index management ───────────────────────────────────────────────────────

    def _index_exists(self) -> bool:
        return (self.index_dir / "faiss.index").exists()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "gemini":
            return _embed_gemini(texts, self.api_key)
        return _embed_sentence_transformers(texts)

    def _build_index(self) -> None:
        """Embed all knowledge chunks and store in a FAISS flat-L2 index."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

        from rag.knowledge_base import get_all_chunks
        chunks = get_all_chunks()
        texts  = [f"{c.title}. {c.content}" for c in chunks]

        log.info(f"RAG: Embedding {len(texts)} knowledge chunks ({self.backend}) ...")
        embeddings = self._embed(texts)

        vectors = np.array(embeddings, dtype=np.float32)
        self._dim    = vectors.shape[1]
        self._chunks = chunks

        # Flat inner-product index (cosine sim on normalised vectors = dot product)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vectors)

        # Persist
        faiss.write_index(self._index, str(self.index_dir / "faiss.index"))
        meta = {
            "dim":      self._dim,
            "backend":  self.backend,
            "n_chunks": len(chunks),
            "chunk_ids": [c.id for c in chunks],
        }
        (self.index_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        log.info(f"RAG: Index built and saved → {self.index_dir}")

    def _load_index(self) -> None:
        """Load persisted FAISS index and metadata."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed.")

        from rag.knowledge_base import get_all_chunks

        self._index  = faiss.read_index(str(self.index_dir / "faiss.index"))
        meta         = json.loads((self.index_dir / "meta.json").read_text())
        self._dim    = meta["dim"]
        self._chunks = get_all_chunks()

        log.info(
            f"RAG: Loaded index — {self._index.ntotal} vectors "
            f"(dim={self._dim}, backend={meta['backend']})"
        )

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.0,
    ) -> list:   # list[KnowledgeChunk]
        """
        Retrieve the k most relevant knowledge chunks for a query string.

        Args:
            query:           Natural language description of the current scene/anomaly
            k:               Number of chunks to return
            score_threshold: Minimum cosine similarity score (0.0–1.0)

        Returns:
            List of KnowledgeChunk objects, ordered by relevance
        """
        if self._index is None or self._index.ntotal == 0:
            log.warning("RAG: Index is empty — skipping retrieval.")
            return []

        try:
            import faiss
            import numpy as np
        except ImportError:
            log.error("faiss-cpu not installed — RAG retrieval disabled.")
            return []

        # Embed the query
        q_vecs = self._embed([query])
        q_arr  = np.array(q_vecs, dtype=np.float32)

        # Faiss inner-product search (cosine sim since vectors are normalised)
        k_safe = min(k, self._index.ntotal)
        scores, indices = self._index.search(q_arr, k_safe)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score < score_threshold:
                continue
            results.append(self._chunks[idx])

        log.info(
            f"RAG: Retrieved {len(results)} chunk(s) for query "
            f"'{query[:60]}...' (top score={scores[0][0]:.3f})"
        )
        return results

    def retrieve_for_payload(
        self,
        payload: dict,
        k: int = 4,
    ) -> list:
        """
        Build a retrieval query from an OSP payload and return relevant chunks.
        Constructs a rich query string from anomaly types, location, and conditions.
        """
        anomalies = payload.get("anomalies", [])
        footprint = payload.get("tile_footprint", {})
        cloud     = payload.get("cloud_cover", 0.0)

        # Build a descriptive query from the payload
        type_list = list({a.get("type", "unknown") for a in anomalies})
        lat_c = ((footprint.get("lat_min", 0) + footprint.get("lat_max", 0)) / 2)
        lon_c = ((footprint.get("lon_min", 0) + footprint.get("lon_max", 0)) / 2)

        query_parts = []
        if type_list:
            query_parts.append(f"{', '.join(type_list)} detection")
        query_parts.append(f"Indian Ocean maritime zone lat {lat_c:.2f} lon {lon_c:.2f}")
        if cloud > 0.3:
            query_parts.append(f"cloud cover {cloud:.0%} degraded sensing")
        if len(anomalies) >= 3:
            query_parts.append("vessel cluster multiple detections")
        # Low confidence anomalies
        low_conf = [a for a in anomalies if a.get("conf", 1.0) < 0.55]
        if low_conf:
            query_parts.append("low confidence detection uncertain identification")

        query = ". ".join(query_parts)
        return self.retrieve(query, k=k)

    # ── Formatting ─────────────────────────────────────────────────────────────

    def format_context(
        self,
        chunks: list,
        header: str = "RETRIEVED MARITIME KNOWLEDGE CONTEXT",
    ) -> str:
        """
        Format retrieved chunks as a compact, LLM-injectable context block.
        Designed to fit within a 1000-token budget.
        """
        if not chunks:
            return ""

        lines = [f"\n--- {header} ---"]
        for chunk in chunks:
            lines.append(
                f"\n[{chunk.id}] {chunk.title}\n{chunk.content}"
            )
        lines.append("--- END CONTEXT ---\n")
        return "\n".join(lines)


# ── Module-level singleton ─────────────────────────────────────────────────────

_rag_instance: Optional[OrbitalRAG] = None


def get_rag(
    backend: str = "sentence_transformers",
    api_key: Optional[str] = None,
) -> OrbitalRAG:
    """
    Return the module-level RAG singleton, initialising it on first call.
    Safe to call repeatedly — only builds the index once per process.
    """
    global _rag_instance
    if _rag_instance is None:
        try:
            _rag_instance = OrbitalRAG(
                embedding_backend=backend,
                api_key=api_key,
            )
        except Exception as e:
            log.error(f"RAG initialisation failed: {e}. RAG will be disabled.")
            return None
    return _rag_instance


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rag = OrbitalRAG(force_rebuild=True)

    test_queries = [
        "ship detected near Indian Ocean with low confidence",
        "cloud cover degrading SWIR band visibility",
        "vessel AIS disabled suspicious behaviour",
        "OVV trigger policy priority",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        chunks = rag.retrieve(q, k=2)
        for c in chunks:
            print(f"  → [{c.id}] {c.title}")
