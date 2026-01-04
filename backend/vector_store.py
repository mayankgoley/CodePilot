import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

class RAGEngine:
    def __init__(self):
        print("DEBUG: Initializing RAGEngine...", flush=True)
        try:
            self.embeddings = OpenAIEmbeddings()
            self.client = QdrantClient(url="http://localhost:6333")
            self.collection_name = "codepilot_codebase"
            
            # Ensure collection exists
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=dict(size=1536, distance="Cosine")
                )
            
            self.vector_store = Qdrant(
                client=self.client, 
                collection_name=self.collection_name, 
                embeddings=self.embeddings
            )
            print("DEBUG: RAGEngine initialized successfully", flush=True)
        except Exception as e:
            print(f"RAG Engine Init Error: {e}")
            self.embeddings = None
            self.client = None
            self.vector_store = None

    def ingest_codebase(self, path: str):
        if not self.vector_store:
            print("RAG Engine not initialized, skipping ingestion.")
            return

        print(f"Ingesting codebase from {path}...")
        documents = []
        try:
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            
            for root, _, files in os.walk(path):
                if any(x in root for x in ["node_modules", ".git", "__pycache__", "venv", "qdrant_data"]):
                    continue
                    
                for file in files:
                    if file.endswith(('.py', '.js', '.html', '.css', '.md', '.txt', '.yml', '.json')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            
                            # Skip empty or very small files
                            if len(content.strip()) < 10:
                                continue
                                
                            docs = splitter.create_documents(
                                [content], 
                                metadatas=[{"source": file_path, "filename": file}]
                            )
                            documents.extend(docs)
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
            
            if documents:
                print(f"Adding {len(documents)} document chunks to Qdrant...")
                self.vector_store.add_documents(documents)
                print("Ingestion complete.")
            else:
                print("No documents found to ingest.")
                
        except Exception as e:
            print(f"Ingestion failed: {e}")

    def search(self, query: str):
        if not self.vector_store:
            return "RAG functionality is unavailable (init failed)."
        try:
            results = self.vector_store.similarity_search(query, k=5)
            return "\n\n".join([f"Source: {res.metadata['source']}\nContent:\n{res.page_content}" for res in results])
        except Exception as e:
            return f"Search error: {e}"

rag_engine = RAGEngine()