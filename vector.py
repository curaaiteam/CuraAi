# =====================================================
# Pinecone Memory Manager for Cura AI
# =====================================================

import os
import logging
import hashlib
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

logger = logging.getLogger("CuraAI.VectorDB")

class PineconeMemoryManager:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Missing Pinecone API key")

        pinecone.init(api_key=api_key)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("✅ Pinecone initialized for vector memory.")

    def _index_name(self, session_id: str) -> str:
        """Generate unique index name for each session."""
        hashed = hashlib.sha1(session_id.encode()).hexdigest()[:24]
        return f"cura-session-{hashed}"

    def store_conversation(self, session_id: str, user_message: str, ai_response: str):
        """Embed and store messages."""
        index_name = self._index_name(session_id)

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=384, metric="cosine")

        index = pinecone.Index(index_name)
        embed_user = self.embedding_model.embed_query(user_message)
        embed_ai = self.embedding_model.embed_query(ai_response)

        index.upsert([
            (f"user_{hashlib.md5(user_message.encode()).hexdigest()}", embed_user, {"text": user_message, "role": "user"}),
            (f"ai_{hashlib.md5(ai_response.encode()).hexdigest()}", embed_ai, {"text": ai_response, "role": "ai"}),
        ])
        logger.info(f"💾 Stored new message pair in {index_name}")

    def get_context(self, session_id: str) -> str:
        """Fetch top 5 most relevant messages for context."""
        index_name = self._index_name(session_id)
        if index_name not in pinecone.list_indexes():
            return ""

        index = pinecone.Index(index_name)
        results = index.query(vector=[0.0]*384, top_k=5, include_metadata=True)

        if not results or not results.get("matches"):
            return ""

        context = [m["metadata"]["text"] for m in results["matches"] if "metadata" in m]
        return " | ".join(context[:5])
