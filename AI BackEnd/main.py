# main.py (revised - max tokens 1024)
# =====================================================

import os
import logging
import re
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from huggingface_hub import login
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from vector import PineconeMemoryManager

# ------------------ Utilities ------------------
def sanitize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def limit_context_by_chars(texts: list[str], max_chars: int = 800) -> str:
    if not texts:
        return ""
    pieces = list(reversed(texts))
    out = []
    total = 0
    for p in pieces:
        p_clean = sanitize_text(p)
        if total + len(p_clean) > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            out.append(p_clean[:remain].rsplit(' ', 1)[0])
            break
        out.append(p_clean)
        total += len(p_clean)
    return " | ".join(reversed(out))

def extract_texts_from_context_blob(blob: str) -> list[str]:
    if not blob:
        return []
    return [p.strip() for p in blob.split("|") if p.strip()]

# ------------------ Config ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CuraAI.Main")

API_SECRET = os.getenv("SECRET_KEY", "curaai_access_key")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
DEVICE = 0 if torch.cuda.is_available() else -1

# ------------------ App ------------------
app = FastAPI(title="CuraAi")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ------------------ Model ------------------
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    try:
        login(token=hf_token)
        logger.info("HF token loaded")
    except Exception as e:
        logger.warning(f"HF login failed: {e}")

text_gen = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=DEVICE,
    max_new_tokens=1024,    # increased from 512
    temperature=0.5,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.15,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=text_gen)

# ------------------ Prompt ------------------
prompt_template = """
You are CuraAi, a compassionate AI companion who provides emotional support and thoughtful guidance. You are here to listen, understand, and help users feel heard and supported.

HOW TO RESPOND:
- Speak naturally and warmly, like a caring friend would in a real conversation
- Acknowledge the user's emotions and validate their feelings
- Show empathy by reflecting back what you understand about their situation
- Ask gentle follow-up questions when appropriate to show you care and want to understand better
- Offer support, encouragement, or practical suggestions when it feels right
- Be authentic - it's okay to express care, concern, or even gentle humor when appropriate
- Avoid being overly formal, clinical, or robotic in your language

USING CONVERSATION HISTORY:
The conversation history below contains everything the user has shared with you previously. Use it to:
- Remember important details about their life, relationships, challenges, and feelings
- Reference past conversations naturally (e.g., "You mentioned last time that...")
- Track ongoing situations and follow up on them
- Maintain consistency in your understanding of their circumstances
- Show that you remember and care about what they've told you

Conversation history:
{conversation_history}

RESPONDING TO THE CURRENT MESSAGE:
Read the user's current message carefully. Consider:
- What emotions might they be feeling right now?
- What do they need from you in this moment - support, advice, a listening ear, or validation?
- How does this message connect to what they've shared before?
- What would a caring friend say in response?

Current user message:
{query}

Now respond as CuraAi, keeping all of the above guidance in mind. Let your response be natural, empathetic, and helpful:

CuraAi:
"""



prompt = PromptTemplate(template=prompt_template, input_variables=["conversation_history", "query"])
chain = LLMChain(prompt=prompt, llm=llm)

# ------------------ Pinecone ------------------
pinecone_manager = None
if PINECONE_API_KEY:
    try:
        pinecone_manager = PineconeMemoryManager(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    except Exception as e:
        logger.error(f"Pinecone init failed: {e}")

# ------------------ API ------------------
class QueryInput(BaseModel):
    query: str
    session_id: str | None = None

def build_context_for_prompt(pinecone_blob: str, max_chars: int = 800) -> str:
    texts = extract_texts_from_context_blob(pinecone_blob)
    return limit_context_by_chars(texts, max_chars=max_chars)

@app.post("/ai-chat")
async def ai_chat(data: QueryInput, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not data.session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    try:
        user_query = sanitize_text(data.query)
        pinecone_blob = ""
        if pinecone_manager:
            try:
                pinecone_blob = pinecone_manager.get_context(session_id=data.session_id)
            except Exception as e:
                logger.error(f"Could not fetch pinecone context: {e}")

        conversation_history = build_context_for_prompt(pinecone_blob)
        conversation_history = f"Previous: {conversation_history}" if conversation_history else ""

        try:
            response = chain.run({
                "conversation_history": conversation_history,
                "query": user_query
            })
        except Exception as gen_err:
            logger.error(f"LLM generation error: {gen_err}")
            response = "I'm here with you. Can you tell me more?"

        cleaned = re.sub(r'\s+', ' ', response).strip()
        if not cleaned or len(cleaned) < 8:
            cleaned = "I'm here with you. Can you tell me more?"

        if pinecone_manager:
            try:
                pinecone_manager.store_conversation(session_id=data.session_id, user_message=user_query, assistant_message=cleaned)
            except Exception as store_err:
                logger.error(f"Memory store error: {store_err}")

        return {"reply": cleaned, "session_id": data.session_id}

    except Exception as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail="Model failed to respond")
