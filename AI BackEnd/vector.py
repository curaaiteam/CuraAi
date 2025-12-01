# vector.py
# =====================================================
# PineconeMemoryManager — robust & defensive for new 'pinecone' client
# =====================================================

import logging
import re
import time
from typing import List
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger("CuraAI.VectorDB")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_DIM = 384


class PineconeMemoryManager:
    def __init__(self, api_key: str, environment: str):
        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set")
        self.environment = environment
        self.pc = Pinecone(api_key=api_key)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"🔧 Pinecone client initialized. Environment: {environment}")
        logger.info(f"🔤 Embedding model loaded: {EMBEDDING_MODEL}")

    def _index_name(self, session_id: str) -> str:
        safe = re.sub(r'[^a-zA-Z0-9-]', '-', session_id.lower())
        return f"curaai-{safe}"

    def _list_index_names(self) -> List[str]:
        try:
            out = self.pc.list_indexes()
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return [i.get("name") or i.get("index_name") for i in out]
            if isinstance(out, list):
                return out
            return []
        except Exception:
            return []

    def _create_index(self, index_name: str, dimension: int = DEFAULT_DIM):
        spec = ServerlessSpec(cloud="aws", region=self.environment)
        try:
            self.pc.create_index(name=index_name, dimension=dimension, metric="cosine", spec=spec)
        except TypeError:
            # fallback positional style
            self.pc.create_index(index_name, dimension, "cosine", spec)
        except Exception as e:
            # allow already exists to pass
            if "ALREADY_EXISTS" in str(e):
                logger.info(f"Index {index_name} already exists, skipping creation.")
            else:
                raise

    def _get_or_create_index(self, index_name: str, dimension: int = DEFAULT_DIM):
        existing = self._list_index_names()
        if index_name in existing:
            logger.debug(f"Using existing index: {index_name}")
            try:
                return self.pc.Index(index_name)
            except AttributeError:
                return self.pc.index(index_name)

        # create new index if missing
        logger.info(f"Creating Pinecone index: {index_name}")
        try:
            self._create_index(index_name, dimension=dimension)
            time.sleep(1)
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                logger.info(f"Index {index_name} already exists, using existing.")
            else:
                logger.error(f"Failed to create index {index_name}: {e}")
                raise

        try:
            return self.pc.Index(index_name)
        except AttributeError:
            return self.pc.index(index_name)

    # -------------------------------------------------
    # Public: retrieve short context
    # -------------------------------------------------
    def get_context(self, session_id: str, top_k: int = 5) -> str:
        index_name = self._index_name(session_id)
        try:
            index = self._get_or_create_index(index_name)
            query_vec = self.embedding_model.encode("summary of past conversation").tolist()
            results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
            matches = getattr(results, "matches", None) or results.get("matches", []) if isinstance(results, dict) else []
            texts = []
            for m in matches:
                meta = getattr(m, "metadata", None) or m.get("metadata", {})
                if meta and "text" in meta:
                    texts.append(meta["text"])
            return " | ".join(texts)
        except Exception as e:
            logger.error(f"Error retrieving context for {session_id}: {e}")
            return ""

    # -------------------------------------------------
    # Public: store conversation pair
    # -------------------------------------------------
    def store_conversation(self, session_id: str, user_message: str, assistant_message: str):
        index_name = self._index_name(session_id)
        try:
            index = self._get_or_create_index(index_name)
            ts = str(int(time.time() * 1000))
            q_vec = self.embedding_model.encode(user_message).tolist()
            r_vec = self.embedding_model.encode(assistant_message).tolist()
            vectors = [
                {
                    "id": f"q-{ts}",
                    "values": q_vec,
                    "metadata": {"text": user_message, "role": "user", "timestamp": ts}
                },
                {
                    "id": f"r-{ts}",
                    "values": r_vec,
                    "metadata": {"text": assistant_message, "role": "assistant", "timestamp": ts}
                }
            ]
            index.upsert(vectors=vectors)
            logger.info(f"Stored conversation vectors for session {session_id}")
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                logger.info(f"Vector already exists for session {session_id}, skipping.")
            else:
                logger.error(f"Error storing conversation for {session_id}: {e}")
