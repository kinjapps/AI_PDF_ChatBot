import streamlit as st
import tempfile
from rag import ask_question, add_pdf_to_db

st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄")

st.title("📄 AI PDF Chatbot (RAG)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Add to vector DB
    chunks_added = add_pdf_to_db(tmp_path)

    st.success(f"PDF processed! {chunks_added} chunks added.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question from your PDF:")

if st.button("Ask"):
    if user_input:
        answer, sources = ask_question(user_input)

        st.session_state.chat_history.append((user_input, answer, sources))
# Show chat
for user, answer, sources in st.session_state.chat_history:
    st.write("**You:**", user)
    st.write("**AI:**", answer)

    st.write("📚 Sources:")
    for s in sources:
        st.write("-", s)

    st.write("---")
