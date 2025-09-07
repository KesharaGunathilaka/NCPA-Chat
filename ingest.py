import os
import re
import json
import time
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

BASE = "https://childprotection.gov.lk/"
OUTPUT_DIR = "data/raw"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Chroma
# Persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Get or create collection
collection = client.get_or_create_collection(name="childprotection_docs")

# Embedding model (multilingual)
embed_model = SentenceTransformer("all-mpnet-base-v2")  # or "all-MiniLM-L6-v2"

visited = set()
to_crawl = [BASE]


def download_pdf(url, dest_folder="data/pdfs"):
    os.makedirs(dest_folder, exist_ok=True)
    r = requests.get(url, timeout=20)
    fname = os.path.join(dest_folder, os.path.basename(
        urlparse(url).path) or f"doc_{int(time.time())}.pdf")
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname


def extract_text_from_pdf(path):
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return "\n".join(text)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + size)
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks


while to_crawl:
    url = to_crawl.pop(0)
    if url in visited:
        continue
    visited.add(url)
    try:
        r = requests.get(url, timeout=15)
        print("Crawling:", url, r.status_code)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print("skip", url, e)
        continue

    # save raw html
    filepath = os.path.join(OUTPUT_DIR, re.sub(
        r"[^\w\-\.]", "_", url) + ".html")
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(r.text)

    # collect internal links and pdf links
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        if href.startswith(BASE):
            if href.lower().endswith(".pdf"):
                try:
                    pdf_path = download_pdf(href)
                    print("Downloaded PDF:", href)
                    text = extract_text_from_pdf(pdf_path)
                    chunks = chunk_text(text)
                    embeds = embed_model.encode(chunks).tolist()
                    # upsert into chroma with metadata containing original url and type
                    for i, chunk in enumerate(chunks):
                        collection.add(
                            documents=[chunk],
                            metadatas=[
                                {"source_url": href, "source_type": "pdf", "pdf_path": pdf_path}],
                            ids=[f"{os.path.basename(pdf_path)}_{i}"],
                            embeddings=[embeds[i]]
                        )
                except Exception as e:
                    print("pdf error", href, e)
            else:
                if href not in visited:
                    to_crawl.append(href)

    # also optionally extract main content blocks for this HTML page
    main_text = " ".join([p.get_text(separator=" ", strip=True)
                         for p in soup.find_all(["p", "h1", "h2", "h3"])])
    if len(main_text) > 50:
        chunks = chunk_text(main_text)
        embeds = embed_model.encode(chunks).tolist()
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"source_url": url, "source_type": "html"}],
                ids=[f"{urlparse(url).path.replace('/', '_')}_{i}"],
                embeddings=[embeds[i]]
            )

print("Done ingest")
