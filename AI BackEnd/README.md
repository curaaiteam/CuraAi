# CuraAi — Emotionally Intelligent AI Companion

CuraAi is a production-grade conversational AI system designed to provide emotionally grounded support through advanced memory capabilities and stable persona management. The platform functions as a memory-aware companion that maintains context across sessions while adhering to strict safety and ethical boundaries.

## System Architecture

CuraAi implements a sophisticated dual-layer memory system that combines immediate conversational context with persistent emotional intelligence. The architecture leverages in-session RAM buffering for real-time dialogue continuity alongside vector-based long-term storage through Pinecone, enabling the system to retain and retrieve emotionally significant information across extended user relationships.

The platform currently operates in production using FastAPI as its deployment framework, with Grok's latest reasoning model powering the conversational intelligence. Memory operations utilize a single Pinecone index deployed in AWS us-east-1, with strict namespace isolation ensuring complete user data segregation.

## Performance Profile

The system delivers consistent performance across typical usage patterns. Initial requests following cold starts complete within approximately ten seconds, while subsequent interactions achieve response times between five and eight seconds. This latency profile reflects the comprehensive processing pipeline that includes memory embedding generation, emotion-weighted retrieval operations, model inference execution, and vector database queries.

## Core Capabilities

### Memory Management

CuraAi maintains conversational continuity through an intelligent six-turn session buffer that captures recent dialogue context. This buffer operates at the session level, providing immediate recall without persisting beyond session boundaries. Every interaction benefits from this recent context, ensuring responses remain coherent and relevant to the ongoing conversation.

The system automatically evaluates conversational content for long-term storage potential. Rather than indiscriminately capturing all dialogue, CuraAi applies sophisticated analysis to identify personally significant statements, emotionally charged content, and persistent life themes. This selective approach prevents memory pollution while ensuring genuinely important information receives appropriate retention.

### Emotional Intelligence

Each stored memory receives a quantified importance score ranging from 0.3 to 1.0, calculated based on emotional language intensity, statement repetition patterns, and identity relevance. This scoring mechanism directly influences retrieval priority, ensuring that emotionally significant memories surface more readily during contextually appropriate moments.

When session buffers reach capacity, CuraAi performs intelligent summarization that distills conversational essence into compact memory representations. These summaries capture key emotional and thematic elements while dramatically reducing storage requirements. This approach prevents context window overflow, eliminates memory fragmentation, and reduces retrieval noise.

### Persona Stability

The platform builds user understanding through deliberate, incremental updates rather than reactive adjustments. Persona modifications require substantial supporting evidence and clear patterns before implementation. This conservative approach ensures behavioral consistency, builds user trust, and enables natural relationship evolution without jarring discontinuities.

### Multimodal Processing

CuraAi extends beyond text-based interaction to support image interpretation through optical character recognition, audio transcription and analysis, and optional supplementary text input. All multimodal content undergoes interpretation and conversion to textual representation before entering the standard memory-aware processing pipeline, ensuring consistent handling regardless of input modality.

## Safety Framework

The system operates under non-negotiable identity constraints that remain immune to prompt injection attempts or user override efforts. CuraAi explicitly maintains boundaries around its role limitations, consistently rejecting therapeutic positioning, diagnostic claims, or suggestions that it can replace professional human support. The platform actively encourages users to maintain real-world connections and seek appropriate professional assistance when situations warrant such intervention.

## API Interface

The production API operates at the following base URL:

```
https://curaaiteam-curaai.hf.space
```

### Endpoints

#### Health Check

```http
GET /
```

Returns system status confirmation.

**Response:**
```json
{
  "status": "ok"
}
```

#### Text Chat

```http
POST /ai-chat
```

**Request Body:**
```json
{
  "user_id": "user_123",
  "session_id": "session_001",
  "query": "I've been feeling really overwhelmed lately"
}
```

**Response:**
```json
{
  "reply": "That sounds like a lot to carry. Do you want to talk about what's been weighing on you most?"
}
```

#### Multimodal Input

```http
POST /multimodal
```

**Content-Type:** `multipart/form-data`

**Form Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | text | yes | User identifier |
| session_id | text | yes | Session identifier |
| file | file | yes | Image or audio file |
| text | text | no | Optional supplementary text |

**Response:**
```json
{
  "reply": "I've looked through what you shared. Want me to explain it or talk about what it brought up for you?"
}
```

## Technical Foundation

The processing pipeline follows a structured flow from client request through FastAPI routing, session buffer consultation, emotional analysis, Pinecone memory retrieval, persona control application, Grok model invocation, and finally response delivery. This architecture ensures comprehensive context awareness while maintaining processing efficiency.

### Project Structure

```
curaai/
├── main.py              # API definitions, session management, model interactions
├── vector.py            # Pinecone memory operations
├── requirements.txt     # Dependency specifications
└── README.md           # Documentation
```

## Proprietary Status

This software remains strictly proprietary under private licensing terms. No permissions exist for copying, modification, redistribution, independent hosting, commercial exploitation, derivative model training, or public API exposure without explicit written authorization from CuraAi, Co. All intellectual property rights remain fully reserved.

---

**CuraAi** — A private AI system developed by CuraAi, Co, designed to provide meaningful companionship without replacing essential human connection.
