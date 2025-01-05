from document_processor import DocumentProcessor
import os
import tempfile

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from datetime import datetime

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()

if "upload_status" not in st.session_state:
    st.session_state.upload_status = {}

st.set_page_config(page_title="Document Chat", layout="wide")

with st.sidebar:
    st.title("📚 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True, key="pdf_uploader"
    )

    if uploaded_files:
        progress_container = st.empty()

        total_files = len(uploaded_files)
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            if uploaded_file.name in st.session_state.upload_status:
                continue

            progress_text = f"Processing file {idx}/{total_files}: {uploaded_file.name}"
            progress_container.text(progress_text)

            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                st.session_state.processor.process_document(
                    tmp_file_path, uploaded_file.name
                )
                os.unlink(tmp_file_path)
                st.session_state.upload_status[uploaded_file.name] = "success"

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.session_state.upload_status[uploaded_file.name] = "error"
        progress_container.empty()
        success_count = sum(
            1
            for status in st.session_state.upload_status.values()
            if status == "success"
        )
        if success_count > 0:
            st.success(f"Successfully processed {success_count} new document(s)")
    add_vertical_space(2)
    st.subheader("📑 Available Documents")
    documents = st.session_state.processor.get_document_details()

    if documents:
        with st.expander("📋 Bulk Actions", expanded=False):
            if st.button("🗑️ Delete All Documents"):
                if st.checkbox("Are you sure? This cannot be undone!"):
                    for doc in documents:
                        st.session_state.processor.delete_document(doc["filename"])
                    st.session_state.upload_status = {}
                    st.rerun()

        docs_by_date = {}
        for doc in documents:
            date = doc["upload_time"].split()[0]
            if date not in docs_by_date:
                docs_by_date[date] = []
            docs_by_date[date].append(doc)
        for date in sorted(docs_by_date.keys(), reverse=True):
            st.write(f"**{date}**")
            for doc in docs_by_date[date]:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    st.write(f"**Chunks:** {doc['total_chunks']}")
                    st.write(f"**Characters:** {doc['total_characters']:,}")
                    st.write(f"**Uploaded:** {doc['upload_time']}")
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        if st.button(
                            "🗑️",
                            key=f"delete_{doc['filename']}",
                            help="Delete this document",
                        ):
                            if st.session_state.processor.delete_document(
                                doc["filename"]
                            ):
                                if doc["filename"] in st.session_state.upload_status:
                                    del st.session_state.upload_status[doc["filename"]]
                                st.success(f"Deleted {doc['filename']}")
                                st.rerun()
                            else:
                                st.error(f"Failed to delete {doc['filename']}")
    else:
        st.info("No documents uploaded yet")

st.title("💬 Document Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("View Sources", expanded=False):
                for source in message["sources"]:
                    st.info(source)

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            results = st.session_state.processor.query_documents(prompt)

            if results["documents"][0]:
                context = "\n\n".join(results["documents"][0])
                messages = [
                    {
                        "role": "system",
                        "content": f"""You are a helpful assistant using llama3.2. 
                        Analyze the following context and answer the user's question accurately.
                        If the answer cannot be found in the context, clearly state that.
                        
                        Context:
                        {context}""",
                    },
                    {"role": "user", "content": prompt},
                ]

                response_placeholder = st.empty()
                full_response = ""

                for (
                    response_chunk
                ) in st.session_state.processor.ollama_client.get_streaming_completion(
                    messages
                ):
                    full_response += response_chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "sources": results["documents"][0],
                    }
                )

                with st.expander("View Sources", expanded=False):
                    for source in results["documents"][0]:
                        st.info(source)
            else:
                response = "I couldn't find any relevant information in the uploaded documents."
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
