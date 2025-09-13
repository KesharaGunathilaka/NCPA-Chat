# NCPA Chat

This repository contains a Streamlit-based question-answering app for the National Child Protection Authority (NCPA) website. The app uses embeddings and a vector store (Qdrant) plus a hosted LLM (Groq) for retrieval-augmented generation.

## Prerequisites

- Python 3.11
- Git (optional, to clone the repo)
- (Optional) Docker & Docker Compose if you want to run in a container

If you use another shell (cmd, WSL, bash), adapt the commands accordingly.

## Project layout (important files)

- `app.py` — Streamlit application entrypoint
- `rag.py` — Retrieval + LLM logic (calls Groq and Qdrant)
- `ingest.py` — scraper / ingestion script that builds embeddings and uploads to Qdrant
- `requirements.txt` — Python dependencies
- `Dockerfile`, `docker-compose.yml` — container instructions

## 1) Clone repository

```powershell
git clone https://github.com/KesharaGunathilaka/NCPA-Chat.git
cd NCPA-Chat
```

## 2) Create and activate a Python virtual environment (PowerShell)

```powershell
# create venv in folder .venv
python -m venv .venv

# activate the venv (PowerShell)
.\.venv\Scripts\Activate.ps1
```

## 3) Install Python dependencies

With the virtualenv active:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Some packages (like `sentence-transformers`) may pull heavy ML dependencies. Installation can take several minutes and may compile wheels.
- If you prefer, create a lightweight environment with `conda` and install the packages there.

## 4) Environment variables

The app and ingest pipeline read configuration from environment variables via `python-dotenv` (.env). Create a `.env` file in the project root with the following variables filled in:

```
# .env example
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=https://your-qdrant-host
QDRANT_API_KEY=your_qdrant_api_key_here
```

Important notes:
- The `rag.py` module and `ingest.py` both expect `QDRANT_URL` and `QDRANT_API_KEY` to be set when interacting with Qdrant.
- `rag.py` also expects `GROQ_API_KEY` for the Groq hosted LLM.
- Do not commit your `.env` file containing secrets. Add `.env` to `.gitignore` if not present.

## 5) Ingesting site content into Qdrant (optional)

The project contains `ingest.py` which crawls `https://childprotection.gov.lk/`, downloads PDFs, extracts text, chunks it, computes embeddings, and uploads points to Qdrant. Use this only if you control the Qdrant instance and API key.

Run the script with the same virtualenv active and `.env` present:

```powershell
python ingest.py
```

Notes:
- `ingest.py` will create folders under `data/` for raw HTML, pdfs, and chunk JSON files.
- The script uses the `SentenceTransformer('all-mpnet-base-v2')` model which will download model weights on first run and requires network access and disk space.

## 6) Running the Streamlit app locally

With the virtualenv active and `.env` created, run:

```powershell
streamlit run app.py
```

Open your browser and visit http://localhost:8501.


## Troubleshooting

- Streamlit not found: ensure your venv is activated and `pip install -r requirements.txt` completed successfully.
- Model downloads (sentence-transformers) are slow or fail: check network, increase disk, or pre-download models in a separate step.
- Connection errors to Qdrant: verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env` and that the Qdrant instance allows connections from your IP.
- Groq API errors: verify `GROQ_API_KEY` and that your account has access to the model used in `rag.py`.
