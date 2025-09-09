from groq import Client
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

# Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq = Client(api_key=GROQ_API_KEY)
# # ChromaDB client (persistent storage)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("childprotection_docs")
# client = chromadb.HttpClient(host="13.60.206.231", port=8000)
# collection = client.get_collection("childprotection_docs")

# Embedding model
embed_model = SentenceTransformer("all-mpnet-base-v2")


def generate_answer(user_question, language="en"):
    # 1) Embed the user question
    query_embedding = embed_model.encode(user_question).tolist()

    # 2) Retrieve top docs from Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # 3) Build context
    context = []
    for d, m in zip(docs, metas):
        context.append(
            f"---\nSource: {m.get('source_url')}\nType: {m.get('source_type')}\nText: {d}\n"
        )

    system_prompt = (
        f"You are a helpful assistant for the National Child Protection Authority Sri Lanka. "
        f"Answer in {language}. Use only the provided sources. If the user is asking about emergency/abuse, "
        f"display the emergency hotline and do not give diagnostic advice. Always list source URLs used at the end."
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

    return resp.choices[0].message.content
