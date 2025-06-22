# Memory Capsule

**A personal journaling assistant powered by LLMs and RAG**

---

## Current Progress

- **FastAPI Backend**  
  - Endpoints for creating entries (`/entry`), semantic lookup (`/flashback/{user_id}`), and stats (`/stats/{user_id}`)  
  - Saves each entry as a plain-text file under `data/{user_id}/entries/{entry_id}.txt`  
  - Tracks streaks and badges in `data/{user_id}/meta.json`  

- **RAG-style Flashbacks**  
  - Embeds each saved entry with `sentence-transformers/all-MiniLM-L6-v2` via HF Inference  
  - Indexes embeddings in a per-user FAISS index (`data/{user_id}/index/faiss.index`)  
  - Maps FAISS IDs to entry files via `id_map.json`  
  - Returns top-k semantically related entries for any query  

- **LLM-powered Generation**  
  - Uses Featherless-AI serverless endpoint to host `meta-llama/Meta-Llama-3-8B-Instruct`  
  - `generate_entry()` wraps a chat-completions call (`client.chat.completions.create`)  
  - AI-mode builds a “raw Q&A block” from your answers to 10 prompts and feeds it as user content  

- **Streamlit Front-End** (`streamlit_app.py`)  
  - **Mode selector**: Manual vs. AI  
  - **Manual**: free-form text area  
  - **AI**: one text box per prompt, then “Generate”  
  - **Flashback** form with error handling  
  - **Stats** metrics (total entries, streak, badges)  
  - **Day/Night theme toggle** with custom CSS  

---

## Directory Structure

```

.
├── .streamlit/
│   └── config.toml             # Base Streamlit theme
├── app/
│   ├── main.py                 # FastAPI app setup
│   ├── routers/
│   │   ├── entry.py            # /entry
│   │   ├── flashback.py        # /flashback/{user\_id}
│   │   ├── stats.py            # /stats/{user\_id}
│   │   └── tune.py             # /tune/{user\_id} (placeholder)
│   ├── storage.py              # Persistence + RAG indexing
│   └── hf\_client.py            # HF inference clients
├── data/                       # Per-user journals & indexes
├── streamlit\_app.py            # Streamlit UI
├── train\_adapter.py            # (Future) LoRA fine-tuning script
├── requirements.txt
└── .env                        # HF\_TOKEN, BASE\_MODEL, API\_URL, etc.

````

---

## Setup & Run

1. **Clone & install**  
   ```bash
   git clone <repo-url>
   cd memory-capsule
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
````

2. **Configure**
   Copy `.env.example` → `.env` and fill in:

   ```dotenv
   HF_TOKEN=hf_xxx
   BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
   API_URL=http://127.0.0.1:8000
   ```

3. **Start the backend**

   ```bash
   uvicorn app.main:app --reload
   ```

4. **Launch the front-end**

   ```bash
   streamlit run streamlit_app.py
   ```

5. **Interact**

   * Choose **Manual** or **AI** mode
   * Fill in your entry or answer the 10 prompts
   * Click **Save Entry**, then view your journal text
   * Use **Flashbacks** to retrieve past moments
   * Check your **Stats** (entries, streak, badges)
   * Toggle **Day/Night** theme in the sidebar

---

## Next Steps

1. **Fine-tuning pipeline**

   * Collect voice samples over time
   * Implement `train_adapter.py` & `/tune` to push a LoRA adapter via LangChain or PEFT
   * Use per-user adapters in generation for personalized tone

2. **Background tasks**

   * Offload embedding & training to Celery/RQ or FastAPI `BackgroundTasks`
   * Keep the API responsive even under load

3. **UI enhancements**

   * Add login/auth for multi-user support
   * Display badge icons and streak calendar
   * Expand “View Past Entries” with `st.expander`

4. **Deployment & CI**

   * Containerize with Docker
   * Add GitHub Actions for linting, testing, and auto-deploy
   * Deploy backend to a cloud provider (Heroku, AWS, etc.) and front-end to Streamlit Cloud

---

