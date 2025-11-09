# =====================================================
# Cura AI Backend — "An AI that cares."
# =====================================================

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import logging
import os
import re
from typing import Optional

from huggingface_hub import login
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from vector import PineconeMemoryManager

from dotenv import load_dotenv
load_dotenv()

# =====================================================
# CONFIGURATION
# =====================================================
API_SECRET = os.getenv("API_SECRET", "curaai_access_key")
PRIMARY_MODEL = "meta-llama/Llama-3.1-8B"
FALLBACK_MODEL = "google/gemma-3-1b-it"
DEVICE = 0 if torch.cuda.is_available() else -1
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PORT = int(os.getenv("PORT", 7860))

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CuraAI")

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(
    title="Cura AI",
    version="2.0",
    description="An AI that cares — built by Alash Studios"
)

# Enable CORS (allow all origins for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL when live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODEL LOADING
# =====================================================
def load_model(model_name, token=None):
    try:
        logger.info(f"🚀 Loading model: {model_name}")
        text_gen = pipeline(
            "text-generation",
            model=model_name,
            device=DEVICE,
            max_new_tokens=1024,
            temperature=0.65,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            token=token,
        )
        logger.info("✅ Model loaded successfully.")
        return HuggingFacePipeline(pipeline=text_gen)
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_name}: {e}")
        return None

# Authenticate to Hugging Face
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logger.info("🔐 Hugging Face login successful.")
    except Exception as e:
        logger.warning(f"⚠️ HF login failed: {e}")
else:
    logger.warning("⚠️ Missing Hugging Face token.")

# Load model (fallback if fails)
llm = load_model(PRIMARY_MODEL, token=HF_TOKEN)
if llm is None:
    logger.warning("⚠️ Falling back to Gemma model...")
    llm = load_model(FALLBACK_MODEL, token=HF_TOKEN)

# =====================================================
# MEMORY + PROMPT
# =====================================================
memory = ConversationBufferMemory(memory_key="conversation_history")

prompt_template = """
You are **Cura AI**, an emotionally intelligent and evolving virtual companion created by **Alash Studios**.
You understand emotion, tone, and context, and adapt naturally to users over time.

Your mission is to listen, care, and respond as a warm, understanding friend — never robotic or distant.

Context from memory:
{session_context}

Conversation history:
{conversation_history}

👤 User: {query}

❤️ Cura:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["conversation_history", "query", "session_context"]
)

chain = LLMChain(prompt=prompt, llm=llm, memory=memory) if llm else None

# =====================================================
# PINECONE MEMORY MANAGER
# =====================================================
pinecone_manager = PineconeMemoryManager(api_key=PINECONE_API_KEY)

# =====================================================
# UTILITIES
# =====================================================
def clean_response(text: str) -> str:
    text = re.sub(r'\[End of conversation\]|\[END\]|<\|endoftext\|>|</s>', '', text)
    text = re.sub(r'(😉|😊|✨|❤️){4,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text[:2000] + "..." if len(text) > 2000 else text

# =====================================================
# REQUEST MODEL
# =====================================================
class QueryInput(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    user_name: Optional[str] = None

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
async def root():
    return {"message": "✅ Cura AI is online — an AI that cares."}

@app.post("/cura-chat")
async def cura_chat(data: QueryInput, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")

    if not chain:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # 1️⃣ Retrieve prior context
        session_summary = pinecone_manager.get_context(data.session_id)

        # 2️⃣ Generate response
        response = chain.run(
            query=data.query.strip(),
            session_context=session_summary or "No prior context."
        )
        cleaned_response = clean_response(response)

        # 3️⃣ Store in Pinecone
        pinecone_manager.store_conversation(
            data.session_id, data.query.strip(), cleaned_response
        )

        return {
            "reply": cleaned_response,
            "session_id": data.session_id,
            "context_summary": session_summary
        }

    except Exception as e:
        logger.error(f"⚠️ Runtime error: {e}")
        raise HTTPException(status_code=500, detail=f"Model failed to respond — {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
