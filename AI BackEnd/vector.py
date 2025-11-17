# =====================================================
# Pinecone Memory Manager for Cura AI (New API)
# =====================================================

import os
import logging
import hashlib
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec

logger = logging.getLogger("CuraAI.VectorDB")

class PineconeMemoryManager:
    def __init__(self, api_key: str, environment: str = "us-east-1"):
        if not api_key:
            raise ValueError("Missing Pinecone API key")

        # ✅ Create a Pinecone client instance
        self.pc = PineconeClient(api_key=api_key)

        # Save embeddings model
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.environment = environment
        logger.info("✅ Pinecone client initialized for vector memory.")

    def _index_name(self, session_id: str) -> str:
        """Generate unique index name for each user/session."""
        hashed = hashlib.sha1(session_id.encode()).hexdigest()[:24]
        return f"cura-session-{hashed}"

    def _get_or_create_index(self, index_name: str, dimension: int = 384):
        """Check if index exists, create if not."""
        existing_indexes = self.pc.list_indexes()  # returns a list of strings
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.environment)
            )
            logger.info(f"✅ Created new Pinecone index: {index_name}")
        return self.pc.index(index_name)

    def store_conversation(self, session_id: str, user_message: str, ai_response: str):
        """Embed and store conversation pairs for context learning."""
        index_name = self._index_name(session_id)
        index = self._get_or_create_index(index_name)

        embed_user = self.embedding_model.embed_query(user_message)
        embed_ai = self.embedding_model.embed_query(ai_response)

        index.upsert([
            (f"user_{hashlib.md5(user_message.encode()).hexdigest()}", embed_user, {"type": "user_msg"}),
            (f"ai_{hashlib.md5(ai_response.encode()).hexdigest()}", embed_ai, {"type": "ai_reply"})
        ])

        logger.info(f"💾 Stored new chat pair in {index_name}")

    def get_context(self, session_id: str) -> str:
        """Fetch session summary from past conversations."""
        index_name = self._index_name(session_id)
        existing_indexes = self.pc.list_indexes()  # fix applied here
        if index_name not in existing_indexes:
            return ""

        index = self.pc.index(index_name)
        results = index.query(vector=[0.0]*384, top_k=5, include_metadata=True)

        if not results or "matches" not in results:
            return ""

        context = []
        for match in results["matches"]:
            meta = match.get("metadata", {})
            text_type = meta.get("type", "")
            if text_type:
                context.append(text_type)
        return " | ".join(context)
