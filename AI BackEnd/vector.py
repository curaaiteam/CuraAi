# vector.py — Pinecone Memory Manager for CuraAi
# =====================================================
import logging
import uuid
import re
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CuraAI.Vector")


def sanitize(text: str) -> str:
    return re.sub(r"[^a-z0-9\-:]", "-", text.lower())


class PineconeMemoryManager:
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_base: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_base
        self.dimension = self.embedder.get_sentence_embedding_dimension()

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.index_name)
        logger.info(f"Pinecone index `{self.index_name}` ready")

    # -------------------------------------------------
    def embed_text(self, text: str) -> List[float]:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    # -------------------------------------------------
    def _namespace(self, user_id: str, kind: str) -> str:
        return sanitize(f"user:{user_id}:{kind}")

    # -------------------------------------------------
    def store_memory(
        self,
        user_id: str,
        text: str,
        importance: float,
        memory_type: str,
        session_id: Optional[str] = None,
    ):
        if not text.strip():
            return

        namespace = self._namespace(
            user_id,
            "session" if memory_type == "session" else "longterm",
        )

        vector = self.embed_text(text)
        vector_id = str(uuid.uuid4())

        metadata = {
            "text": text,
            "importance": float(min(max(importance, 0.0), 1.0)),
            "type": memory_type,
            "session_id": session_id,
        }

        self.index.upsert(
            vectors=[{"id": vector_id, "values": vector, "metadata": metadata}],
            namespace=namespace,
        )

    # -------------------------------------------------
    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 4,
    ) -> str:
        query_vector = self.embed_text(query)

        texts = []

        for kind in ("session", "longterm"):
            namespace = self._namespace(user_id, kind)
            try:
                res = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace,
                )
                texts.extend(
                    m.metadata.get("text", "")
                    for m in sorted(
                        res.matches,
                        key=lambda x: x.metadata.get("importance", 0),
                        reverse=True,
                    )
                )
            except Exception:
                continue

        return " | ".join(texts)

    # -------------------------------------------------
    def store_persona_update(self, user_id: str, persona_summary: str):
        self.store_memory(
            user_id=user_id,
            text=persona_summary,
            importance=0.9,
            memory_type="persona",
        )
