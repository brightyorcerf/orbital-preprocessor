"""
rag/retrieval.py
────────────────
RAG retrieval layer for the OSP orbital intelligence system.

Embeds the maritime knowledge base and exposes a simple
retrieve(query) → list[KnowledgeChunk] interface.

Two embedding backends are supported:
  1. sentence-transformers (local, no API key needed) — default
  2. Google Gemini text-embedding-004 (cloud, higher quality)

The retrieved chunks are injected into the LLM system prompt to ground the
analyst's reasoning in verifiable domain knowledge rather than hallucinated
maritime facts.

Architecture:
  knowledge_base.py   →  embed all chunks  →  in-memory (N, D) matrix
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

import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ── Embedding backends ─────────────────────────────────────────────────────────

_ST_MODEL_NAME = "all-MiniLM-L6-v2"
_st_model_cache = None


def _get_st_model():
    """
    Lazily load and cache the sentence-transformers model.

    This used to be constructed inside the embed function, so every single
    retrieval re-loaded ~22MB of weights from disk — roughly 1-2s of pure
    overhead per query, in a system whose entire thesis is latency discipline.
    Loading once per process drops steady-state query embedding to ~10ms.
    """
    global _st_model_cache
    if _st_model_cache is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        log.info(f"RAG: loading embedding model {_ST_MODEL_NAME} (once per process) ...")
        _st_model_cache = SentenceTransformer(_ST_MODEL_NAME)
    return _st_model_cache


def _embed_sentence_transformers(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Local embedding using sentence-transformers.
    Model: all-MiniLM-L6-v2 (~22MB, ~10ms per small batch on CPU once cached).

    `is_query` is accepted for interface symmetry with the Gemini backend;
    MiniLM is a symmetric model and needs no query/document distinction.
    """
    model = _get_st_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def _embed_gemini(
    texts: list[str],
    api_key: Optional[str] = None,
    is_query: bool = False,
) -> list[list[float]]:
    """
    Cloud embedding using Google Gemini text-embedding-004 (768-d).

    Two correctness details this backend previously got wrong:

    1. task_type. text-embedding-004 is an *asymmetric* model: it projects
       documents and queries into deliberately different regions of the space.
       Embedding a query with `retrieval_document` (the old behaviour) means
       querying with a vector the index was never built to be searched by,
       which measurably degrades ranking. We now switch on `is_query`.

    2. Normalisation. Retrieval ranks by inner product, which only equals
       cosine similarity when vectors are unit-length. Gemini does not return
       normalised vectors, so raw dot products let vector *magnitude* dominate
       ranking. We normalise explicitly.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed.")

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY required for Gemini embedding backend.")
    genai.configure(api_key=key)

    task = "retrieval_query" if is_query else "retrieval_document"

    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type=task,
        )
        vec = np.asarray(result["embedding"], dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-12   # unit-length → IP == cosine
        embeddings.append(vec.tolist())
    return embeddings


# ── OrbitalRAG ─────────────────────────────────────────────────────────────────

class OrbitalRAG:
    """
    Retrieval-Augmented Generation layer for the OSP orbital intelligence system.

    Embeds the maritime knowledge base once per process and ranks chunks by
    cosine similarity at analysis time.

    There is no vector database here on purpose. The knowledge base is a
    fourteen-chunk hand-written corpus, so the search is a dot product against
    a (14, 384) matrix — microseconds, and exactly what a flat inner-product
    index computes anyway. The FAISS build this replaces also carried an
    on-disk index, a metadata sidecar and a corpus fingerprint whose only job
    was to notice when the persisted vectors had gone stale against the corpus
    they were built from. Embedding at construction cannot go stale.
    """

    def __init__(
        self,
        embedding_backend: str = "sentence_transformers",   # or "gemini"
        api_key: Optional[str] = None,
    ):
        self.backend  = embedding_backend
        self.api_key  = api_key
        self._chunks  = []          # list[KnowledgeChunk], row i == vectors[i]
        self._vectors = None        # (N, D) float32, unit-length rows
        self._build()

    # ── Index management ───────────────────────────────────────────────────────

    def _embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if self.backend == "gemini":
            return _embed_gemini(texts, self.api_key, is_query=is_query)
        return _embed_sentence_transformers(texts, is_query=is_query)

    def _build(self) -> None:
        """Embed every knowledge chunk into an in-memory matrix."""
        from rag.knowledge_base import get_all_chunks

        chunks = get_all_chunks()
        texts  = [f"{c.title}. {c.content}" for c in chunks]

        log.info(f"RAG: Embedding {len(texts)} knowledge chunks ({self.backend}) ...")
        vectors = np.asarray(self._embed(texts), dtype=np.float32)

        self._chunks  = chunks
        self._vectors = vectors
        log.info(f"RAG: {vectors.shape[0]} vectors (dim={vectors.shape[1]}) ready")

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
        if self._vectors is None or len(self._chunks) == 0:
            log.warning("RAG: Index is empty — skipping retrieval.")
            return []

        # is_query=True selects the asymmetric query projection on backends
        # that distinguish queries from documents.
        q = np.asarray(self._embed([query], is_query=True)[0], dtype=np.float32)

        # Both sides are unit-length, so the inner product is cosine similarity.
        scores = self._vectors @ q
        order  = np.argsort(-scores)[:min(k, len(self._chunks))]

        results = [self._chunks[i] for i in order if scores[i] >= score_threshold]

        log.info(
            f"RAG: Retrieved {len(results)} chunk(s) for query "
            f"'{query[:60]}...' (top score={scores[order[0]]:.3f})"
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
    Safe to call repeatedly — only embeds the corpus once per process.
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
