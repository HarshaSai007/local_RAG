import streamlit as st
from typing import List
from chromadb import Settings
import chromadb
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import concurrent.futures
import time
from llama_client import OllamaClient


class DocumentProcessor:
    def __init__(self):
        self._initialize_chroma_client()
        self.ollama_client = OllamaClient()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
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
                    metadata={"hnsw:space": "cosine"},
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

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_chunk, chunk, filename, i, len(chunks), timestamp
                )
                for i, chunk in enumerate(chunks)
            ]

            processed_chunks = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    processed_chunks.append(result)

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
                include=["metadatas", "documents", "distances"],
            )

            for i in range(len(results["distances"][0])):
                similarity = 1 - results["distances"][0][i]
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
            results = self.collection.get(where={"source": filename})

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                if filename in self.processing_cache:
                    del self.processing_cache[filename]

                return True
            return False
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False
