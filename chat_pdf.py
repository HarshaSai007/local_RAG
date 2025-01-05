import os
import tempfile
import streamlit as st
from embedchain import App
import base64
from streamlit_chat import message


def embedchain_bot(db_path):
    return App.from_config(
        config={
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3.2:latest",
                    "max_tokens": 1000,
                    "temperature": 0.5,
                    "stream": True,
                    "base_url": "http://localhost:11434",
                },
            },
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "llama3.2:latest",
                    "base_url": "http://localhost:11434",
                },
            },
        }
    )


def display_pdf(pdf_path):
    base64_pdf = base64.b64encode(pdf_path.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


st.title("Chat With Your PDF")
st.caption("This App Lets You Chat With Your PDF")

db_path = tempfile.mkdtemp()

if "app" not in st.session_state:
    st.session_state.app = embedchain_bot(db_path)
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("PDF Upload")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    if pdf_file:
        st.subheader("PDF Preview")
        display_pdf(pdf_file)

if st.button("Submit PDF"):
    with st.spinner("adding PDF to Knowledge Base"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.getvalue())
            st.session_state.app.add(temp_pdf.name, data_type="pdf_file")
        os.remove(temp_pdf.name)
    st.success(f"Added {pdf_file.name} to Knowledge Base!")


for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(i))

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    with st.spinner("Thinking..."):
        response = st.session_state.app.chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response)

if st.button("Clear Chat"):
    st.session_state.messages = []
