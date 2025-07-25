import os
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
        
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except:  # Collection doesn't exist
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
            if not query.strip():
                logger.warning("Empty query provided")
                return []
            
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, 100)  
            )
            
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    similar_docs.append({
                        'document': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results.get('distances') and results['distances'][0] else 0
                    })
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

class ContextFind:
    """Context finder for RAG operations with PDF, DOCX, and TXT support"""

    def __init__(self, document_path: str, collection_name: str = None):
        self.document_processor = DocumentProcessor()
        
        # Use document name as collection name if not provided
        if collection_name is None:
            document_name = Path(document_path).stem
            collection_name = f"context_{document_name}"
        
        self.vector_store = VectorStore(collection_name)
        self.document_path = document_path
        self.context = ""
        self._documents_loaded = False

    def return_context(self, question: str, top_k: int = 3) -> str:
        """Return context for given question"""
        try:
            if not self._documents_loaded:
                self._add_vector()
                self._documents_loaded = True
            
            logger.info(f"Processing RAG query: {question}")
                
            similar_docs = self.vector_store.search_similar(question, top_k)
            
            if similar_docs:
                self.context = "\n\n".join([doc['document'] for doc in similar_docs])
                return self.context
            else:
                logger.warning(f"No relevant documents found for query: {question}")
                return ""
                
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return ""
    
    def _add_vector(self):
        """Add document content to vector store"""
        try:
            # Extract text using the appropriate processor
            text = self.document_processor.extract_text_from_documents(self.document_path)
            
            if not text.strip():
                raise ValueError(f"No text extracted from document: {self.document_path}")
            
            # Split text into chunks
            chunks = self._split_text_into_chunks(text, chunk_size=20,overlap=1)
            
            if not chunks:
                raise ValueError("No chunks created from document text")
            
            # Create metadata and IDs
            document_name = Path(self.document_path).stem
            metadata = [
                {
                    "source": self.document_path, 
                    "chunk_id": i,
                    "document_type": Path(self.document_path).suffix.lower(),
                    "document_name": document_name
                } 
                for i in range(len(chunks))
            ]
            ids = [f"{document_name}_{i}" for i in range(len(chunks))]
            
            self.vector_store.add_documents(
                documents=chunks,
                metadata=metadata,
                ids=ids
            )
            
            logger.info(f"Document loaded into vector store: {self.document_path} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 2000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for better context preservation"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:  
                chunk = ' '.join(chunk_words)
                chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks

# # USAGE EXAMPLES:
# def example_usage():
#     """Example usage with different document types"""
    
#     # Example 1: PDF document
#     try:
#         pdf_context = ContextFind("file.pdf")
#         context = pdf_context.return_context("Bireysel Kredi Kartı Var mı ?",3)
#         print(f"PDF Context: {context}")
#     except Exception as e:
#         print(f"PDF Error: {e}")
    
#     # Example 2: Word document
#     # try:
#     #     docx_context = ContextFind("manual.docx")
#     #     context = docx_context.return_context("How to configure the system?")
#     #     print(f"Word Context: {context}")
#     # except Exception as e:
#     #     print(f"Word Error: {e}")
    
#     # Example 3: Text document
#     # try:
#     #     txt_context = ContextFind("roadmap.txt")
#     #     context = txt_context.return_context("could i tell me what should i learn ?")
#     #     print(f"Text Context: {context}")
#     # except Exception as e:
#     #     print(f"Text Error: {e}")

# if __name__ == "__main__":
#     example_usage()