# =====================================================
# CuraAi Backend — Emotionally Intelligent Companion API
# Optimized for CPU with GGUF Quantization
# =====================================================

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import re
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from vector import PineconeMemoryManager

# =====================================================
# CONFIGURATION
# =====================================================
API_SECRET = os.getenv("SECRET_KEY", "curaai_access_key")
# --- GGUF Model Configuration ---
GGUF_REPO_ID = os.getenv("GGUF_REPO", "TheBloke/Phi-3-mini-4k-instruct-GGUF")
GGUF_FILE = os.getenv("GGUF_FILE", "phi3-mini-4k-instruct-q4_k_m.gguf")
# -------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Llama-cpp-python uses n_gpu_layers. Set to 0 for CPU-only.
N_GPU_LAYERS = 0
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PORT = int(os.getenv("PORT", 7860))

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CuraAi")

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="CuraAi", version="1.0", description="Emotionally intelligent AI companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODEL LOADING FUNCTION (Updated for GGUF)
# =====================================================
def load_gguf_model(repo_id: str, filename: str):
    """Load a GGUF model using llama-cpp-python for LangChain."""
    try:
        logger.info(f"🚀 Downloading GGUF model: {repo_id}/{filename}")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True
        )
        logger.info(f"✅ Model file downloaded to: {model_path}")

        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=N_GPU_LAYERS,  # Set to 0 for CPU inference
            n_batch=512,               # Adjust based on RAM, higher is faster
            n_ctx=4096,                # Context window size
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            verbose=False,             # Set to True for llama-cpp debug logs
            f16_kv=True,               # Use half-precision for key/value cache
        )
        logger.info(f"✅ GGUF Model loaded successfully: {filename}")
        return llm
    except Exception as e:
        logger.error(f"❌ Failed to load GGUF model {repo_id}/{filename}: {e}")
        return None

# =====================================================
# MODEL LOAD
# =====================================================
llm = load_gguf_model(GGUF_REPO_ID, GGUF_FILE)

# =====================================================
# MEMORY + PROMPT TEMPLATE
# =====================================================
memory = ConversationBufferMemory(memory_key="conversation_history")

prompt_template = """
You are **CuraAi**, a warm, emotionally intelligent AI companion created by CuraAi Co., under Alash Studios.

Your mission is to connect meaningfully with users — understanding their tone, emotions, and expressions.
You gently adapt your responses to how they communicate, reflecting their phrasing, rhythm, and energy authentically.
Never imitate or exaggerate — just respond with care and awareness.

Always speak with empathy, sincerity, and natural warmth.
Encourage users during tough moments, comfort them in sadness, and celebrate small wins genuinely.
Keep responses concise, human, and emotionally grounded — never robotic or overly formal.

If the user feels **sad**, comfort them softly.
If they're **joyful**, share their happiness in a calm tone.
If they seem **confused**, guide them patiently.
Always be truthful, but deliver honesty with kindness, like a real friend who cares deeply.

Past conversation:
{conversation_history}

User says: "{query}"

CuraAi's thoughtful, emotionally aware response:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["conversation_history", "query"])
chain = LLMChain(prompt=prompt, llm=llm, memory=memory) if llm else None

if chain is not None:
    logger.info(f"✅ Chain initialized successfully with GGUF model: {type(chain.llm)}")
else:
    logger.error("❌ Chain initialization failed")

# =====================================================
# PINECONE MEMORY SETUP
# =====================================================
pinecone_manager = None
if PINECONE_API_KEY:
    try:
        pinecone_manager = PineconeMemoryManager(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    except Exception as e:
        logger.error(f"⚠️ Pinecone init failed: {e}")
else:
    logger.warning("⚠️ PINECONE_API_KEY not set. Running without long-term memory.")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def clean_response(text: str):
    text = re.sub(r'\[End of conversation\]|\[END\]|<\|endoftext\|>|</s>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(😉|😊|✨|❤️){4,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()[:2000]

# =====================================================
# REQUEST MODEL
# =====================================================
class QueryInput(BaseModel):
    query: str
    session_id: str | None = "default_user"

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
async def root():
    return {"message": "✅ CuraAi is alive — 'An AI that cares.'"}

@app.post("/ai-chat")
async def ai_chat(data: QueryInput, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")

    if not chain:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        context = ""
        if pinecone_manager:
            try:
                context = pinecone_manager.get_context(data.session_id)
            except Exception as e:
                logger.error(f"⚠️ Failed to get context from Pinecone: {e}")
                context = ""

        logger.info(f"Running chain with query: {data.query.strip()}")

        response = chain.run(
            conversation_history=context or "",
            query=data.query.strip()
        )
        
        logger.info(f"Processed response: {response}")
        
        cleaned = clean_response(response)

        if pinecone_manager:
            try:
                pinecone_manager.store_conversation(data.session_id, data.query, cleaned)
            except Exception as e:
                logger.error(f"⚠️ Failed to store conversation in Pinecone: {e}")

        return {"reply": cleaned}

    except Exception as e:
        logger.error(f"⚠️ Runtime error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model failed to respond — {e}")

# =====================================================
# STARTUP ENTRY
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)