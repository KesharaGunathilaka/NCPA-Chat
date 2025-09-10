import re
from groq import Client
from sentence_transformers import SentenceTransformer
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import unicodedata
import html
from urllib.parse import urlsplit, urlunsplit, quote, unquote

load_dotenv()

# Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq = Client(api_key=GROQ_API_KEY)

# Qdrant Cloud connection
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60.0
)

COLLECTION_NAME = "childprotection_ncpa"

# Embedding model
embed_model = SentenceTransformer("all-mpnet-base-v2")


def normalize_url(href: str) -> str:
    if not href:
        return href

    # unescape HTML entities, normalize Unicode
    href = html.unescape(href)
    href = unicodedata.normalize("NFKC", href)

    # parse URL so we operate only on the path/query fragment parts
    parts = urlsplit(href)
    scheme, netloc, path, query, fragment = parts

    # operate on decoded path (so percent-encoded unicode becomes characters)
    path = unquote(path)

    # replace any dash-like punctuation (Unicode category 'Pd') with ASCII hyphen
    path = ''.join('-' if unicodedata.category(ch)
                   == 'Pd' else ch for ch in path)

    # replace any whitespace characters (including NBSP, zero-width) with normal spaces
    path = ''.join(' ' if ch.isspace() else ch for ch in path).strip()

    # re-quote the path so spaces and special chars are percent-encoded safely
    path = quote(path, safe="/%")

    # also normalize query/fragment (unquote then quote to avoid double-encoding)
    query = quote(unquote(query), safe="=&?/") if query else ""
    fragment = quote(unquote(fragment), safe="") if fragment else ""

    return urlunsplit((scheme, netloc, path, query, fragment))


# regex to find urls in arbitrary text (keeps unicode chars inside the url match)
URL_RE = re.compile(r"https?://[^\s\)\]\}\'\"<>]+")


def normalize_urls_in_text(text: str) -> str:
    def _repl(m):
        url = m.group(0)
        try:
            return normalize_url(url)
        except Exception:
            return url  # if normalization fails, keep original
    return URL_RE.sub(_repl, text)


def generate_answer(user_question, language="en"):
    # 1) Embed the user question
    query_embedding = embed_model.encode(user_question).tolist()

    # 2) Retrieve top docs from Qdrant and deduplicate by normalized URL
    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=8,  # fetch more to allow deduping, then reduce in context
        with_payload=True
    )

    context = []
    seen = set()
    source_urls = []

    # Build deduped context (stop when we reach 4 unique sources)
    for h in hits:
        payload = h.payload or {}
        raw_url = payload.get("source_url", "") or ""
        url = normalize_url(raw_url)
        if url in seen:
            continue
        seen.add(url)
        source_urls.append(url)
        snippet = payload.get("text", "")
        stype = payload.get("source_type", "unknown")
        context.append(f"---\nSource: {url}\nType: {stype}\nText: {snippet}\n")
        if len(source_urls) >= 4:
            break

    # Add official NCPA contact info as a special source
    official_info = {
        "contact": {
            "phone": "+94 11 2 778 911 – 12 – 14",
            "fax": "+94 11 2 778 915",
            "email": "ncpa@childprotection.gov.lk",
            "website": "https://childprotection.gov.lk/en/"
        },
        "address": "No. 330, Thalawathugoda Road, Madiwela, Sri Lanka",
        "helpline": {
            "number": "1929",
            "description": "24/7 Child Protection Helpline (Sinhala, Tamil, English)"
        }
    }

    if any(keyword in user_question.lower() for keyword in ["contact", "phone", "address", "helpline"]):
        context.insert(0, f"---\nSource: official\n"
                       f"Text: Official NCPA Contact Information:\n"
                       f"Phone: {official_info['contact']['phone']}\n"
                       f"Fax: {official_info['contact']['fax']}\n"
                       f"Email: {official_info['contact']['email']}\n"
                       f"Website: {official_info['contact']['website']}\n"
                       f"Address: {official_info['address']}\n"
                       f"Helpline: {official_info['helpline']['number']} ({official_info['helpline']['description']})\n")

    # 3) System prompt: explicitly instruct model not to change URLs
    system_prompt = (
        f"You are a helpful assistant for the National Child Protection Authority Sri Lanka.\n"
        f"Answer in {language}.\n"
        f"Use only the provided sources.\n"
        "If the user is asking about emergency/abuse, display the emergency hotline and do not give diagnostic advice.\n"
        "IMPORTANT: Do NOT modify, rewrite, or replace any URLs present in the CONTEXT. "
        "When you list sources, include each URL exactly as provided, on its own line, with no surrounding punctuation."
    )

    user_prompt = (
        "CONTEXT:\n" + "\n".join(context) +
        f"\n\nUser question: {user_question}\nGive a concise answer and then list sources."
    )

    # 4) Call Groq LLM
    resp = groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=512,
    )

    model_text = resp.choices[0].message.content

    # 5) Post-process model output: normalize any URLs the model printed
    sanitized_text = normalize_urls_in_text(model_text)

    # 6) Ensure the verified normalized sources are present for programmatic reuse
    #    If any source URL is missing or appears different in model output, append a verified list.
    missing = [u for u in source_urls if u not in sanitized_text]
    if missing:
        sanitized_text = sanitized_text + \
            "\n\nVerified sources (normalized):\n" + "\n".join(source_urls)

    return sanitized_text
