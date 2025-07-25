from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from document import DocumentProcessor
from loggerCenter import LoggerCenter
logger = LoggerCenter().get_logger()


class VectorStore:
    """Manages vector embeddings and similarity search"""
    
    def __init__(self, collection_name: str = "document_embeddings"):
        self.collection_name = collection_name
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except:
                # Create new collection if it doesn't exist
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                )
                logger.info(f"Created new collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"VectorStore initialization failed: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None, ids: List[str] = None):
        """Add documents to vector store"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
            
            if not ids:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            if not metadata:
                metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]
            
            # Ensure all lists have the same length
            if len(documents) != len(ids) or len(documents) != len(metadata):
                raise ValueError("Documents, IDs, and metadata must have the same length")
            
            self.collection.add(
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        try:
            if not query:
                logger.warning("Empty query provided")
                return []
            
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    similar_docs.append({
                        'document': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    })
            
            return similar_docs
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []


class ContextFind:
    """Context finder for RAG operations"""

    def __init__(self, pdf_path: str):
        self.pdf_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.pdf_path = pdf_path
        self.context = ""
        self._documents_loaded = False

    def return_context(self, question: str, top_k: int = 3) -> str:
        """Return context for given question"""
        try:
            if not self._documents_loaded:
                self._add_vector()
                self._documents_loaded = True
            
            self.log_info(f"Processing RAG query: {question}")
                
            # Retrieve relevant documents
            similar_docs = self.vector_store.search_similar(question, top_k)
            
            # Combine context from retrieved documents
            self.context = "\n\n".join([doc['document'] for doc in similar_docs])
            return self.context
        except Exception as e:
            self.log_error(f"Context retrieval failed: {e}")
            return ""
    
    def _add_vector(self):
        """Add PDF content to vector store"""
        try:
            text = self.pdf_processor.extract_text_from_pdf(self.pdf_path)
            
            # Split text into chunks
            chunks = self._split_text_into_chunks(text, chunk_size=1000)
            
            # Add to vector store
            self.vector_store.add_documents(
                documents=chunks,
                metadata=[{"source": self.pdf_path, "chunk_id": i} for i in range(len(chunks))],
                ids=[f"pdf_{Path(self.pdf_path).stem}_{i}" for i in range(len(chunks))]
            )
            
            logger.info(f"PDF documentation loaded: {self.pdf_path}")
        except Exception as e:
            logger.error(f"PDF loading failed: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks for vector storage"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def log_info(self, message: str):
        logger.info(f"[ContextFind] {message}")
    
    def log_error(self, message: str):
        logger.error(f"[ContextFind] {message}")

