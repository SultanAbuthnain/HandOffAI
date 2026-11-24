# AI Customer Support Widget with RAG & Human-in-the-Loop (HITL)

An open-source, self-hosted, **Plug & Play** AI customer support widget built with **FastAPI**, **WebSockets**, **SQLite + NumPy local vector search**, **Shadow DOM**, and a real-time **Human-in-the-Loop Agent Dashboard**.

Zero external database servers. Zero build steps. Clone, add your API key, run one command, and embed a single `<script>` tag on any website.

---

## Key Features

- **Zero External Dependencies:** Sessions, chat history, HITL state, documents, and L2-normalized `float32` vector embeddings all live inside a single local SQLite file (`app.db` with WAL mode).
- **Provider-Agnostic AI Layer:** Switch seamlessly between **OpenAI** (`gpt-4o-mini` + `text-embedding-3-small`) and **Google Gemini** (`gemini-2.5-flash` + `gemini-embedding-001`) via `.env`.
- **Strict Grounded RAG Pipeline:** Ingests plain text, Markdown, or `.txt`/`.md` files into ~500-character boundary-aware overlapping chunks. The AI answers **only** from retrieved context and outputs `[ESCALATE]` whenever confidence is low or information is missing.
- **Race-Condition-Proof HITL State Machine:**
  - Every session is strictly tracked in SQLite as `BOT_ACTIVE`, `WAITING_FOR_HUMAN`, or `HUMAN_ACTIVE`.
  - **Auto-Escalation:** Triggered when a visitor requests a human, retrieval confidence is below threshold, or the LLM outputs `[ESCALATE]`. Alerts `/dashboard` in real time with a visual banner and chime.
  - **Manual Takeover:** Agents can click **Take Over Chat** at any moment—immediately cancelling in-flight AI generation tasks and locking the session to `HUMAN_ACTIVE` until **Release to Bot** is clicked.
  - **Atomic Dispatch Guard:** Verifies `state == 'BOT_ACTIVE'` inside both a per-session asyncio lock and an atomic SQLite `BEGIN IMMEDIATE` transaction right before persisting and dispatching any AI reply.
- **Shadow DOM Isolation:** Pure Vanilla JS (`widget.js`) isolated inside a Shadow Root so host website CSS never breaks the widget UI.
- **Built-in Security:** `ALLOWED_ORIGINS` allowlist on CORS & WebSockets, sliding-window IP rate limiting, XSS-safe DOM rendering, and optional `AGENT_API_KEY` protection for `/dashboard` and Knowledge Base routes.

---

## 3-Step Quickstart

### Step 1: Clone & Install Dependencies

Requires **Python 3.10+**.

```bash
git clone https://github.com/your-org/ai-support-hitl.git
cd ai-support-hitl

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
```

### Step 2: Configure `.env`

Copy `.env.example` to `.env` and add your OpenAI or Google Gemini API key:

```bash
cp .env.example .env
```

Minimal `.env` example (OpenAI):

```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

Or using Google Gemini:

```env
LLM_PROVIDER=gemini
LLM_API_KEY=AIza...
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

### Step 3: Start the Server & Test

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

1. Open the **Agent Dashboard** at `http://localhost:8000/dashboard` → switch to **Knowledge Base** and upload [`examples/acme_support_faq.md`](examples/acme_support_faq.md) (or add your own FAQ).
2. Open the **Built-in Widget Demo** at `http://localhost:8000/demo` in a second tab (intentionally styled with aggressive CSS to demonstrate Shadow DOM isolation).
3. Ask a question from your Knowledge Base (the AI replies with grounded citations), then ask an unanswerable question or click **Talk to a human** to watch real-time escalation and **Take Over** in `/dashboard`.

### Run with Docker Compose

```bash
cp .env.example .env
# Edit .env and set your LLM_API_KEY
docker compose up --build -d
```

### Run Automated Tests

```bash
pytest -v
```

---

## Embed on Any Website

Add a single `<script>` tag before `</body>` on your site and whitelist your site's origin in `ALLOWED_ORIGINS`:

```html
<script
  src="https://your-support-server.com/widget.js"
  data-title="Acme Support"
  data-subtitle="AI answers + live human agents"
  data-color="#2563eb"
  data-position="right"
  data-greeting="Hi! How can we help you today?"
></script>
```

| Attribute | Default | Description |
| --- | --- | --- |
| `data-title` | `Support Assistant` | Header title displayed in the chat widget |
| `data-subtitle` | `Instant AI answers with human handoff` | Subtitle under header title |
| `data-color` | `#2563eb` | Primary brand color (hex/CSS color) |
| `data-position` | `right` | Launcher placement: `right` or `left` |
| `data-greeting` | *(built-in welcome)* | Initial greeting shown in new sessions |
| `data-visitor-name` | `Visitor` | Optional display name for logged-in users |
| `data-server` | *(derived from script src)* | Optional override for backend base URL |

---

## HITL State Machine

```text
                [ESCALATE] / low confidence / visitor clicks "Talk to a human"
  ┌────────────┐ ────────────────────────────────────────────────────────────► ┌───────────────────┐
  │ BOT_ACTIVE │                                                               │ WAITING_FOR_HUMAN │
  └────────────┘ ◄────────────────── Agent clicks "Release to Bot" ─────────── └───────────────────┘
        │  ▲                                                                             │
        │  │                                                                             │
  Agent clicks                                                                     Agent clicks
  "Take Over"                                                                      "Take Over"
  (aborts AI)                                                                      or sends reply
        │  │                                                                             │
        ▼  │                                                                             ▼
  ┌──────────────┐ ◄─────────────────────────────────────────────────────────────────────┘
  │ HUMAN_ACTIVE │   AI is strictly locked out until agent clicks "Release to Bot"
  └──────────────┘
```

---

## Security & Production Checklist

1. **Restrict `ALLOWED_ORIGINS`:** Replace `*` or localhost entries with your exact production origins (e.g. `https://example.com,https://app.example.com`). Unauthorized origins are rejected with WebSocket close code `1008` on both `/ws/chat` and `/ws/agent`.
2. **Set `AGENT_API_KEY`:** Set a strong random secret in `.env` so `/dashboard`, `/ws/agent` (via in-band first-frame auth or `X-Agent-Key`), and `/api/kb/*` require authentication.
3. **Run behind HTTPS / Reverse Proxy:** Terminate TLS with Nginx, Caddy, or Cloudflare so the widget uses `wss://` automatically. Set `TRUST_PROXY_HEADERS=true` in `.env` **only** when your reverse proxy overwrites `X-Forwarded-For` with the verified client IP.
4. **Single-Worker Process Model (`--workers 1`):** Because active WebSocket connections, per-session locks, and in-flight AI task handles are coordinated in-process alongside SQLite WAL mode, always run Uvicorn with `--workers 1` (the default in `Dockerfile`).
5. **Rate Limiting:** Tune `RATE_LIMIT_REQUESTS` (default `20`) and `RATE_LIMIT_WINDOW_SECONDS` (default `60`) per IP as needed.
6. **Switching Embedding Providers:** If you switch `LLM_PROVIDER` between `openai` and `gemini`, delete and re-ingest existing Knowledge Base documents so stored vector dimensions match the active embedding model.

---

## Project Structure

```text
ai-support-hitl/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes, security & static file serving
│   ├── database.py          # SQLite schema, atomic HITL state & cached vector BLOB CRUD
│   ├── rag_engine.py        # Chunking, OpenAI/Gemini embeddings, retrieval & LLM calls
│   ├── ws_manager.py        # WebSocket manager, HITL coordinator & IP rate limiter
│   └── requirements.txt
├── static/
│   ├── widget.js            # Self-contained Shadow DOM chat widget
│   └── dashboard.html       # Real-time Agent Dashboard & Knowledge Base manager
├── examples/
│   └── acme_support_faq.md  # Sample Knowledge Base FAQ for instant RAG testing
├── tests/
│   ├── test_database.py     # SQLite schema, atomic HITL race guard & vector cache tests
│   ├── test_rag_engine.py   # Chunking, human intent regex & grounded RAG tests
│   └── test_api_and_ws.py   # FastAPI REST, WebSocket handoff & security tests
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## API & WebSocket Reference

### REST Endpoints
- `GET /dashboard` — Agent Dashboard & Knowledge Base Manager UI
- `GET /widget.js` — Embeddable Shadow DOM widget script
- `GET /demo` — Local Shadow DOM isolation demo page
- `GET /api/health` — System health, active provider, KB stats & connection counts
- `GET /api/kb/documents` — List indexed documents & stats
- `POST /api/kb/ingest/text` — Ingest plain text or Markdown (`{title, content, source_type}`)
- `POST /api/kb/ingest/file` — Upload `.txt` or `.md` file (`multipart/form-data`)
- `POST /api/kb/search` — Test cosine similarity retrieval (`{query, top_k}`)
- `DELETE /api/kb/documents/{document_id}` — Remove a document & its chunks
- `GET /api/sessions` — List sessions (optional `?state=WAITING_FOR_HUMAN`)
- `GET /api/sessions/{session_id}` — Fetch session metadata & message history
- `POST /api/sessions/{session_id}/takeover` — Lock session to `HUMAN_ACTIVE` & cancel pending AI
- `POST /api/sessions/{session_id}/release` — Return session to `BOT_ACTIVE`
- `POST /api/sessions/{session_id}/messages` — Send an agent reply
- `DELETE /api/sessions/{session_id}` — Delete a session

### WebSocket Endpoints
- `WS /ws/chat/{session_id}` — Visitor chat (`message`, `request_human`, `ping` → receives `session_init`, `message`, `state_change`, `typing`, `error`)
- `WS /ws/agent` — Agent Dashboard (`take_over`, `release_to_bot`, `agent_message`, `typing`, `ping` → receives `dashboard_init`, `session_updated`, `message`, `state_change`, `escalation`, `presence`, `kb_updated`)

---

## License

MIT License — free for personal and commercial self-hosted deployments.
