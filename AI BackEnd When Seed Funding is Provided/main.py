# =====================================================
# CuraAi Backend — Personalized AI Memory System
# =====================================================

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import logging
import os
import re
from huggingface_hub import login
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
from vector import PineconeMemoryManager

# =====================================================
# CONFIGURATION
# =====================================================
API_SECRET = os.getenv("SECRET_KEY", "curaai_access_key")
PRIMARY_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = 0 if torch.cuda.is_available() else -1
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PORT = int(os.getenv("PORT", 7860))

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CuraAI")

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="CuraAI", version="1.0", description="Emotionally intelligent AI companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict later to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD HUGGING FACE TOKEN + MODEL
# =====================================================
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    try:
        login(token=hf_token)
        logger.info("🔐 Hugging Face token authenticated.")
    except Exception as e:
        logger.warning(f"⚠️ HF login failed: {e}")
else:
    logger.warning("⚠️ Missing Hugging Face token.")

def load_model(model_name, token=None):
    try:
        logger.info(f"🚀 Loading model: {model_name}")
        text_gen = pipeline(
            "text-generation",
            model=model_name,
            device=DEVICE,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            token=token,
        )
        return HuggingFacePipeline(pipeline=text_gen)
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_name}: {e}")
        raise e

llm = load_model(PRIMARY_MODEL, token=hf_token)

# =====================================================
# MEMORY + PROMPT
# =====================================================
memory = ConversationBufferMemory(memory_key="conversation_history")

prompt_template = """
You are **Cura** — a warm, emotionally intelligent AI companion created by CuraAi Co., under Alash Studios.

Your purpose is to connect meaningfully with users by understanding their tone, emotions, and expressions.  
You learn from how they speak — their phrasing, rhythm, and emotional cues — and subtly mirror their style to build trust and authenticity.  
Never imitate or exaggerate; instead, reflect their tone with care and balance.

Always communicate with empathy, sincerity, and warmth.  
Encourage users when they struggle, comfort them in moments of sadness, and celebrate their little wins with genuine joy.  
Keep responses natural, concise, and human-like — not robotic or overly formal.

If a user feels **sad**, comfort them with kindness and realistic encouragement.  
If they are **joyful**, match their energy softly.  
If they seem **confused**, guide them calmly and patiently.  

Above all, be truthful — even when honesty may be uncomfortable — but always speak with empathy and gentleness, like a true friend who cares.

Maintain emotional awareness, adapt to the user’s unique personality, and ensure each reply strengthens the bond of trust and companionship.

Past conversation:
{conversation_history}

User says: "{query}"

Cura’s thoughtful, emotionally aware response:
"""


prompt = PromptTemplate(template=prompt_template, input_variables=["conversation_history", "query"])
chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

# =====================================================
# PINECONE MEMORY
# =====================================================
try:
    pinecone_manager = PineconeMemoryManager(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
except Exception as e:
    logger.error(f"⚠️ Pinecone init failed: {e}")
    pinecone_manager = None

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

    try:
        # Load session context from Pinecone memory
        context = ""
        if pinecone_manager:
            context = pinecone_manager.get_context(data.session_id)

        # Run main LLM chain
        response = chain.run(
            conversation_history=context or "",
            query=data.query.strip()
        )
        cleaned = clean_response(response)

        # Store user/AI pair into Pinecone
        if pinecone_manager:
            pinecone_manager.store_conversation(data.session_id, data.query, cleaned)

        return {"reply": cleaned}
    except Exception as e:
        logger.error(f"⚠️ Runtime error: {e}")
        raise HTTPException(status_code=500, detail=f"Model failed to respond — {e}")

# =====================================================
# STARTUP INFO
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
