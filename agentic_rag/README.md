# Agentic RAG Service

This is a standalone conversational Agentic RAG system that integrates with your remote Supabase instance (data and vector queries) and Upstash Redis (distributed conversational memory).

## Features

* **Multi-Step Reasoning Agent**: Uses LangGraph to plan, fetch context, self-correct/critique findings, and synthesize responses.
* **Distributed Memory**: Integrates **Upstash Redis** to cache user/assistant conversation history on a rolling 24-hour window.
* **Cloud Storage Integration**: Queries **Supabase** tables (`company_reports` and `report_embeddings`) for context lookup and vector comparisons.
* **Fallback Search**: Automatically falls back to news search and Wikipedia lookup if a company is missing from the database.

## Directory Structure

```text
agentic_rag/
│
├── rag_app/
│   ├── main.py        # FastAPI API endpoints
│   ├── agent.py       # LangGraph Agent state loops
│   ├── tools.py       # Supabase, Vector, and live news tools
│   └── database.py    # Supabase SDK and Upstash Redis connections
│
├── .env
├── pyproject.toml
└── README.md
```

## Setup & Running

### 1. Synchronize Dependencies
Run from this directory:
```bash
uv sync
```

### 2. Configure Environment `.env`
Ensure your `agentic_rag/.env` file contains your credentials:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-publishable-anon-key
UPSTASH_REDIS_URL=https://your-redis-instance.upstash.io
UPSTASH_REDIS_TOKEN=your-redis-token
GOOGLE_API_KEY=your-gemini-key
NEWS_API_KEY=your-newsapi-key
MEDIASTACK_API_KEY=your-mediastack-key
```

### 3. Run the Server
Start the service on port `8001` (to prevent conflicts with the main api port `8000`):
```bash
uv run uvicorn agentic_rag.rag_app.main:app --port 8001
```

Open interactive Swagger docs at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs).

---

## API Endpoints

### 1. Chat with RAG Agent
* **Endpoint**: `POST /rag/chat`
* **Request Body**:
```json
{
  "query": "Compare Zoho and Nestle's priorities",
  "session_id": "optional-custom-session-id"
}
```
*Note: Conversation history is saved in Redis using the `session_id` as the key.*
