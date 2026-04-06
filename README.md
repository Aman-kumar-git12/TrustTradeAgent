# 🤖 TrustTrade Strategic AI Agent

High-performance, semantic-aware LLM service designed for business asset transactions on the TrustTrade platform.

## 🏗️ Architecture Overview

The agent operates as a **Strategic Partner** service, using a "Single Source of Truth" knowledge base and semantic vector retrieval.

1.  **Knowledge Source**: Raw platform documentation is maintained in human-editable `.txt` files within `app/data/`.
2.  **Vector Ingestion**: The `scripts/vectorize_knowledge.py` utility chunks these text files and generates 384-dimensional embeddings (using the `all-MiniLM-L6-v2` transformer), storing them in MongoDB.
3.  **Semantic Retrieval**: During conversation, the agent performs a similarity search across these embeddings to inject relevant platform context into the LLM prompt.
4.  **Interface**: A FastAPI server (port 8000) provides the chat endpoint used by the Node.js backend.

---

## 🔄 Conversational Mode Flow

The `/api/chat` endpoint supports both `conversation` mode and `agent` mode. The flow below describes the `conversation` path:

```text
Client / Frontend
    |
    v
POST /api/chat
    |
    v
FastAPI route in main.py
    |
    v
ChatService.handle(request)
    |
    +--> Normalize request into ChatRequest
    |
    +--> If mode == "agent"
    |        |
    |        +--> Route to LangGraph purchase flow
    |
    v
Conversation mode
    |
    +--> Read user role, session id, and recent history
    |
    +--> Build retrieval query
    |        current message + recent history
    |
    +--> Detect intent, active topics, and reply format
    |
    +--> Greeting check
    |        |
    |        +--> Return source="greeting"
    |
    +--> Capability question check
    |        |
    |        +--> Return source="capability"
    |
    v
KnowledgeService.search(retrieval_query)
    |
    +--> Fetch TrustTrade semantic context from vector knowledge
    |
    v
GroundingEngine.extract_grounded_items(...)
    |
    +--> No grounded items found
    |        |
    |        +--> Return source="scope-guard"
    |             LLM is not called
    |
    v
Build conversation system prompt
    |
    +--> Add role, intent, topics, TrustTrade context,
    |    and strict JSON response instructions
    |
    v
TrustTradeAgent.chat(...)
    |
    v
Groq model response
    |
    +--> Valid JSON parsed
    |        |
    |        +--> Return source="python-agent"
    |
    +--> LLM call or JSON parse fails
             |
             +--> Return source="fallback-grounding"
```

### Response Sources

These `source` values help explain how a conversational reply was produced:

- `greeting`: Fast-path greeting response.
- `capability`: Fast-path capability/help response.
- `scope-guard`: The request looked out of scope or lacked grounded TrustTrade context.
- `python-agent`: Normal conversational LLM response.
- `fallback-grounding`: The model failed, so a grounded fallback response was returned.

---

## 🛠️ Setup & Operations

### 1. Requirements
Ensure you have the required Python dependencies:
```bash
cd Agent
pip install -r requirements.txt
```

### 2. Environment Configuration
Create or update `Agent/.env` with the following:
- `GROQ_API_KEY`: Your Groq platform key.
- `MONGODB_URI`: Connection string for the knowledge store.
- `DATABASE_NAME`: (e.g., `assetdirect`)
- `KNOWLEDGE_COLLECTION_NAME`: (typically `knowledges`)

### 3. Knowledge Vectorization
Run this script whenever you update the `.txt` files in `app/data/` to refresh the semantic brain:
```bash
python3 scripts/vectorize_knowledge.py
```

### 4. Running the Service
Start the FastAPI server using the absolute-path-aware startup script:
```bash
./start_agent.sh
```

---

## 🧠 System Design Features

- **Lazy Initialization**: The server boots instantly. Heavy AI dependencies (models/database) are loaded on-demand during the first request.
- **Fault Tolerance**: If the knowledge base is offline, the agent continues to provide strategic advice based on its internal logic rather than failing.
- **Transparent Status**: Check the health and initialization state of the AI layer at:
  - `GET http://localhost:8000/health`

---

## 📁 Repository Structure

- `scripts/`: Manual utilities for data ingestion and vectorization.
- `app/config/`: Centralized settings module (pydantic-style).
- `app/core/`: Strategic LLM logic and prompt engineering.
- `app/data/`: **Single Source of Truth** for platform intelligence (Text files).
- `app/services/`: Knowledge retrieval and orchestration services.
- `app/schemas/`: Typed Request/Response models.
