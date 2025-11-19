# =====================================================
# VectorDB — Pinecone Memory Manager
# =====================================================

import logging
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as VectorStore
from sentence_transformers import SentenceTransformer

# Setup logger for this module
logger = logging.getLogger("CuraAI.VectorDB")

class PineconeMemoryManager:
    def __init__(self, api_key: str, environment: str):
        """Initialize Pinecone client and embedding model."""
        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set.")
        self.environment = environment
        self.pc = Pinecone(api_key=api_key)
        # Using the same embedding model as specified in main.py
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("✅ Pinecone client initialized for vector memory.")

    def _index_name(self, session_id: str) -> str:
        """Generate a unique index name for a given session."""
        # Sanitize session_id to be a valid Pinecone index name
        safe_session_id = re.sub(r'[^a-zA-Z0-9-]', '-', session_id.lower())
        return f"curaai-{safe_session_id}"

    def _get_or_create_index(self, index_name: str, dimension: int = 384):
        """Check if index exists, create if not, and return the vector store."""
        try:
            # The new Pinecone API returns a list of index names
            existing_indexes = self.pc.list_indexes().names()
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            # Handle the case where list_indexes() might return a list directly
            existing_indexes = self.pc.list_indexes()

        if index_name not in existing_indexes:
            logger.info(f"Index '{index_name}' not found. Creating new index...")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.environment)
            )
            logger.info(f"✅ Created new Pinecone index: {index_name}")
        
        # Connect to the index
        index = self.pc.Index(index_name)

        # Return a LangChain Pinecone vector store
        return VectorStore(
            embedding_function=self.embedding_model.embed_query,
            index=index,
            text_key="text" # Specify the key for the text content
        )

    def get_context(self, session_id: str) -> str:
        """Fetch session summary from past conversations."""
        index_name = self._index_name(session_id)
        try:
            existing_indexes = self.pc.list_indexes().names()
        except Exception:
            existing_indexes = self.pc.list_indexes()

        if index_name not in existing_indexes:
            return ""

        try:
            vector_store = self._get_or_create_index(index_name)
            # Create a dummy embedding for the query to find similar past interactions
            dummy_embedding = self.embedding_model.encode("summary of past conversation", convert_to_tensor=False).tolist()
            
            results = vector_store.similarity_search_by_vector(
                embedding=dummy_embedding,
                k=5
            )
            
            if not results:
                return ""

            context = []
            for res in results:
                # The page_content is where the text is stored in LangChain's Pinecone integration
                context.append(res.page_content)
            
            return " | ".join(context)
        except Exception as e:
            logger.error(f"Error getting context for session {session_id}: {e}")
            return ""

    def store_conversation(self, session_id: str, query: str, response: str):
        """Store a conversation turn (query and response) in Pinecone."""
        index_name = self._index_name(session_id)
        vector_store = self._get_or_create_index(index_name)
        
        # Store the user query
        vector_store.add_texts(
            texts=[query],
            metadatas=[{"type": "user_query"}]
        )
        
        # Store the AI response
        vector_store.add_texts(
            texts=[response],
            metadatas=[{"type": "ai_response"}]
        )
        
        logger.info(f"✅ Stored conversation for session {session_id}")