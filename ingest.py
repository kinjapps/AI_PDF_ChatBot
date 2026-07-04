from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 1. Read PDF
reader = PdfReader("data/sample.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Create ChromaDB client
client = chromadb.PersistentClient(path="vector_store")

collection = client.get_or_create_collection(name="pdf_chat")

# 5. Add chunks to vector DB
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()

    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[str(i)]
    )

print("✅ Embeddings stored in vector database!")