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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.memory import ConversationBufferMemory

# =====================================================
# CONFIGURATION
# =====================================================
API_SECRET = os.getenv("SECRET_KEY", "curaai_access_key")
PRIMARY_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-small")  # Changed to small for 2CPU/8GB
DEVICE = "cpu"  # Force CPU
MAX_LENGTH = 256  # Reduced for stability
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
# MODEL LOADING (Direct Transformers - More Reliable)
# =====================================================
model = None
tokenizer = None

def load_model_direct():
    """Load model directly with transformers for better control"""
    global model, tokenizer
    try:
        logger.info(f"🚀 Loading model: {PRIMARY_MODEL}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            PRIMARY_MODEL,
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        # Load model with minimal memory footprint
        model = AutoModelForSeq2SeqLM.from_pretrained(
            PRIMARY_MODEL,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        model.eval()  # Set to evaluation mode
        logger.info("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# Load model at startup
load_model_direct()

# =====================================================
# MEMORY MANAGEMENT
# =====================================================
# Simple in-memory conversation storage (replacing Pinecone for now)
conversation_store = {}

def get_conversation_history(session_id: str, max_turns: int = 3):
    """Get last N turns of conversation"""
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    
    history = conversation_store[session_id][-max_turns:]
    return "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in history])

def store_conversation(session_id: str, user_msg: str, ai_msg: str):
    """Store conversation turn"""
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    
    conversation_store[session_id].append({
        "user": user_msg,
        "ai": ai_msg
    })
    
    # Keep only last 10 turns to save memory
    if len(conversation_store[session_id]) > 10:
        conversation_store[session_id] = conversation_store[session_id][-10:]

# =====================================================
# PROMPT TEMPLATE (Simplified for small model)
# =====================================================
def create_prompt(query: str, history: str = ""):
    """Create a concise prompt for the model"""
    if history:
        return f"""You are CuraAi, a caring AI companion.

Previous conversation:
{history}

User: {query}

Respond with empathy and warmth:"""
    else:
        return f"""You are CuraAi, a caring AI companion.

User: {query}

Respond with empathy and warmth:"""

# =====================================================
# GENERATION FUNCTION
# =====================================================
def generate_response(prompt_text: str):
    """Generate response using the model"""
    try:
        if model is None or tokenizer is None:
            raise Exception("Model not loaded")
        
        # Tokenize input
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Generate with conservative parameters for stability
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=MAX_LENGTH,
                min_length=20,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                num_beams=1,  # Greedy for speed
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise e

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def clean_response(text: str):
    """Clean up the generated response"""
    # Remove special tokens and artifacts
    text = re.sub(r'\[End of conversation\]|\[END\]|<\|endoftext\|>|</s>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(😉|😊|✨|❤️){4,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    # Limit length
    if len(text) > 500:
        text = text[:500].rsplit('.', 1)[0] + '.'
    
    return text

# =====================================================
# REQUEST MODEL
# =====================================================
class QueryInput(BaseModel):
    query: str
    session_id: str = "default_user"

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
async def root():
    return {
        "message": "✅ CuraAi is alive — 'An AI that cares.'",
        "model_loaded": model is not None,
        "model_name": PRIMARY_MODEL
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model": PRIMARY_MODEL,
        "device": DEVICE
    }

@app.post("/ai-chat")
async def ai_chat(data: QueryInput, x_api_key: str = Header(None)):
    """Main chat endpoint"""
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get conversation history
        history = get_conversation_history(data.session_id)
        
        # Create prompt
        prompt = create_prompt(data.query.strip(), history)
        
        # Generate response
        logger.info(f"Generating response for session: {data.session_id}")
        raw_response = generate_response(prompt)
        
        # Clean response
        cleaned = clean_response(raw_response)
        
        # Store conversation
        store_conversation(data.session_id, data.query, cleaned)
        
        return {
            "reply": cleaned,
            "session_id": data.session_id
        }
        
    except Exception as e:
        logger.error(f"⚠️ Runtime error: {e}")
        raise HTTPException(status_code=500, detail=f"Model failed to respond: {str(e)}")

@app.post("/clear-history")
async def clear_history(session_id: str, x_api_key: str = Header(None)):
    """Clear conversation history for a session"""
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")
    
    if session_id in conversation_store:
        del conversation_store[session_id]
    
    return {"message": f"History cleared for session: {session_id}"}

# =====================================================
# STARTUP ENTRY
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)