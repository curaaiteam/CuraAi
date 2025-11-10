# =====================================================
# CuraAi Backend — Emotionally Intelligent Companion API
# =====================================================

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import logging
import os
import re
from huggingface_hub import login
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from vector import PineconeMemoryManager

# =====================================================
# CONFIGURATION
# =====================================================
API_SECRET = os.getenv("SECRET_KEY", "curaai_access_key")
PRIMARY_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = 0 if torch.cuda.is_available() else -1
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
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# HUGGING FACE LOGIN + MODEL LOAD
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
    """
    Load a text2text-generation model and wrap it for LangChain.
    This wrapper ensures the pipeline output (a list of dicts) is converted to a string.
    """
    try:
        logger.info(f"🚀 Loading model: {model_name}")
        text2text_pipe = pipeline(
            task="text2text-generation",
            model=model_name,
            device=DEVICE,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            use_auth_token=token
        )

        # Wrap pipeline to return a string instead of a list
        def model_fn(prompt: str):
            output = text2text_pipe(prompt)
            return output[0]["generated_text"]

        return HuggingFacePipeline(pipeline=model_fn)

    except Exception as e:
        logger.error(f"❌ Failed to load model {model_name}: {e}")
        raise e

llm = load_model(PRIMARY_MODEL, token=hf_token)

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
If they’re **joyful**, share their happiness in a calm tone.
If they seem **confused**, guide them patiently.
Always be truthful, but deliver honesty with kindness, like a real friend who cares deeply.

Past conversation:
{conversation_history}

User says: "{query}"

CuraAi’s thoughtful, emotionally aware response:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["conversation_history", "query"])
chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

# =====================================================
# PINECONE MEMORY SETUP
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
    """Clean model output by removing unwanted markers, emojis, and long loops."""
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
# STARTUP ENTRY
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
