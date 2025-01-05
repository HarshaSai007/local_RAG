import streamlit as st
import os
import tempfile
from typing import List, Dict
import requests
import json
from chromadb import Client, Settings
import chromadb
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import concurrent.futures
import time


class OllamaClient:
    def __init__(self, embed_model="mxbai-embed-large", completion_model="llama3.2"):
        self.base_url = "http://localhost:11434"
        self.embed_model = embed_model
        self.completion_model = completion_model

    def get_embedding(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
        )
        return response.json()["embedding"]

    def get_completion(self, messages: List[Dict]) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.completion_model,
                "messages": messages,
                "stream": False,
            },
        )
        return response.json()["message"]["content"]


class DocumentProcessor:
    def __init__(self):
        # Initialize ChromaDB client with retry mechanism
        self._initialize_chroma_client()

        # Initialize Ollama client
        self.ollama_client = OllamaClient()

        # Initialize text splitter with optimized parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Smaller chunks for better retrieval
            chunk_overlap=50,  # Reduced overlap for efficiency
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

        # Cache for document processing
        self.processing_cache = {}

    def _initialize_chroma_client(self, max_retries=3):
        for attempt in range(max_retries):
            try:
                self.chroma_client = chromadb.HttpClient(
                    host="localhost",
                    port=8000,
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
                self.collection_name = "pdf_documents"
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},  # Optimized similarity metric
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed to connect to ChromaDB after {max_retries} attempts"
                    )
                time.sleep(1)

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF with improved error handling"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _process_chunk(
        self,
        chunk: str,
        filename: str,
        chunk_id: int,
        total_chunks: int,
        timestamp: str,
    ):
        """Process a single chunk with embedding"""
        try:
            embedding = self.ollama_client.get_embedding(chunk)
            return {
                "id": f"{filename}_chunk_{chunk_id}",
                "document": chunk,
                "embedding": embedding,
                "metadata": {
                    "source": filename,
                    "chunk_id": chunk_id,
                    "total_chunks": total_chunks,
                    "upload_time": timestamp,
                    "chunk_length": len(chunk),
                },
            }
        except Exception as e:
            st.warning(f"Error processing chunk {chunk_id}: {str(e)}")
            return None

    def process_document(self, file, filename: str):
        """Process document with parallel chunk processing"""
        if filename in self.processing_cache:
            st.info(f"Document {filename} already processed")
            return

        text = self.extract_text_from_pdf(file)
        if not text.strip():
            st.error("No text could be extracted from the document")
            return

        chunks = self.text_splitter.split_text(text)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_chunk, chunk, filename, i, len(chunks), timestamp
                )
                for i, chunk in enumerate(chunks)
            ]

            # Collect results
            processed_chunks = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    processed_chunks.append(result)

        # Batch add to ChromaDB
        if processed_chunks:
            self.collection.add(
                ids=[chunk["id"] for chunk in processed_chunks],
                embeddings=[chunk["embedding"] for chunk in processed_chunks],
                documents=[chunk["document"] for chunk in processed_chunks],
                metadatas=[chunk["metadata"] for chunk in processed_chunks],
            )

            self.processing_cache[filename] = {
                "chunks": len(processed_chunks),
                "timestamp": timestamp,
            }

    def query_documents(self, query: str, k: int = 5) -> dict:
        """Improved document querying with relevance scoring"""
        try:
            query_embedding = self.ollama_client.get_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["metadatas", "documents", "distances"],  # Updated parameter
            )

            # Add relevance scores
            for i in range(len(results["distances"][0])):
                similarity = (
                    1 - results["distances"][0][i]
                )  # Convert distance to similarity
                results["documents"][0][
                    i
                ] = f"[Relevance: {similarity:.2%}]\n{results['documents'][0][i]}"

            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_document_details(self) -> List[dict]:
        """Get comprehensive document details"""
        try:
            all_metadata = self.collection.get()["metadatas"]
            documents_dict = {}

            for metadata in all_metadata:
                if metadata and "source" in metadata:
                    filename = metadata["source"]
                    if filename not in documents_dict:
                        documents_dict[filename] = {
                            "filename": filename,
                            "total_chunks": metadata["total_chunks"],
                            "upload_time": metadata["upload_time"],
                            "total_characters": sum(
                                meta["chunk_length"]
                                for meta in all_metadata
                                if meta["source"] == filename
                            ),
                        }

            return list(documents_dict.values())
        except Exception as e:
            st.error(f"Error getting document details: {str(e)}")
            return []

    def delete_document(self, filename: str) -> bool:
        """Delete a document and all its chunks from ChromaDB"""
        try:
            # Get all IDs for the document's chunks
            results = self.collection.get(where={"source": filename})

            if results["ids"]:
                # Delete all chunks for this document
                self.collection.delete(ids=results["ids"])

                # Remove from processing cache if exists
                if filename in self.processing_cache:
                    del self.processing_cache[filename]

                return True
            return False
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False


# Streamlit UI
st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 Advanced Document Q&A System")

# Initialize processor in session state
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()

# Document Management Section
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Document Management")

    # File upload section
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            st.session_state.processor.process_document(
                tmp_file_path, uploaded_file.name
            )
            os.unlink(tmp_file_path)
            st.success(f"Successfully processed {uploaded_file.name}")

with col2:
    st.header("System Info")
    st.info(
        """
    - Embedding Model: mxbai-embed-large
    - Completion Model: llama3.2
    - Vector Store: ChromaDB
    """
    )

# Display document inventory
st.header("Document Inventory")
documents = st.session_state.processor.get_document_details()

if documents:
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 0.5])
    with col1:
        st.write("**Filename**")
    with col2:
        st.write("**Chunks**")
    with col3:
        st.write("**Characters**")
    with col4:
        st.write("**Upload Time**")
    with col5:
        st.write("**Action**")

    for doc in documents:
        with col1:
            st.write(doc["filename"])
        with col2:
            st.write(doc["total_chunks"])
        with col3:
            st.write(f"{doc['total_characters']:,}")
        with col4:
            st.write(doc["upload_time"])
        with col5:
            # Create a unique key for each delete button
            delete_key = f"delete_{doc['filename']}"
            if st.button("🗑️", key=delete_key, help=f"Delete {doc['filename']}"):
                if st.session_state.processor.delete_document(doc["filename"]):
                    st.success(f"Successfully deleted {doc['filename']}")
                    # Force a rerun to update the UI
                    st.rerun()
                else:
                    st.error(f"Failed to delete {doc['filename']}")
else:
    st.info("No documents have been uploaded yet.")

# Query section
st.header("Ask Questions")
query = st.text_input("Enter your question:")

if st.button("Ask") and query:
    with st.spinner("Searching and analyzing..."):
        # Get relevant chunks
        results = st.session_state.processor.query_documents(query)

        if results["documents"][0]:
            # Prepare context and query for Ollama
            context = "\n\n".join(results["documents"][0])
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant using llama3.2. Analyze the following context and answer the user's question accurately. 
                    If the answer cannot be found in the context, clearly state that. Include relevant quotes when appropriate.
                    
                    Context:
                    {context}""",
                },
                {"role": "user", "content": query},
            ]

            # Get response from Ollama
            response = st.session_state.processor.ollama_client.get_completion(messages)

            # Display results
            st.subheader("Answer")
            st.write(response)

            # Display sources
            st.subheader("Sources")
            for doc in results["documents"][0]:
                st.info(doc)
        else:
            st.warning("No relevant information found in the documents.")
