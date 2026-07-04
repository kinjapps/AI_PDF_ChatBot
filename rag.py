from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import requests
from pathlib import Path

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Vector DB
client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection(name="pdf_chat")


def add_pdf_to_db(pdf_file):
    # Read PDF
    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    # Store in vector DB
    for i, chunk in enumerate(chunks):
        filename = Path(pdf_file).name
        embedding = model.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{filename}_{i}"]
        )

    return len(chunks)



# Step 1: Retrieve context
def retrieve_context(query, top_k=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents"]
    )

    docs = results["documents"][0]
    ids = results["ids"][0]

    return docs, ids


# Step 2: Call LLM (Ollama)
def generate_answer(question, context):
    prompt = f"""
Use ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()
    return data.get("response", str(data))

# Step 3: Full pipeline
def ask_question(question):
    context, ids = retrieve_context(question)

    answer = generate_answer(question, context)

    return answer, ids


# Test
if __name__ == "__main__":
    q = "How many rishis assembled in the forest Naimisharanya?"
    result = ask_question(q)

    print("\nAI Answer:\n")
    print(result)
