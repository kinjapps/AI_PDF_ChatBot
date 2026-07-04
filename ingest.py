from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# Read PDF
reader = PdfReader("data/sample.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_text(text)

print(f"Total Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n----- Chunk {i+1} -----\n")
    print(chunk)