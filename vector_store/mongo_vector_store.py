import pymongo
import numpy as np
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List, Optional, Tuple
from vector_store.embeddings import FaissEmbeddings
import logging
import hashlib
import re
from tqdm import tqdm
import faiss
from langchain_core.embeddings import Embeddings

class MongoDBVectorStore(VectorStore):
    def __init__(self, db_name, collection_name, embeddings_model, vector_index_name, mongodb_client):
        self.chunk_size = 300  
        self.overlap_size = 20 # Number of overlapping entries
        self.mongodb_client = mongodb_client
        self.db = self.mongodb_client[db_name]
        self.logs_collection = self.db[collection_name]
        self.file_metadata_collection = self.db["file_metadata"]  # New collection for file metadata
        self.vector_index_name = vector_index_name
        self.embeddings_model = embeddings_model
        
        # Get the actual embedding dimension by generating a test embedding
        test_embedding = self.embeddings_model.get_embeddings(["test"], 0, "test")
        if isinstance(test_embedding, list):
            if isinstance(test_embedding[0], list):
                self.embedding_dim = len(test_embedding[0])  # For SentenceTransformer
            else:
                self.embedding_dim = len(test_embedding)
        elif hasattr(test_embedding, 'shape'):
            self.embedding_dim = test_embedding.shape[-1] if len(test_embedding.shape) > 0 else 768
        else:
            self.embedding_dim = 768 # Default fallback
            
        logging.info(f"Using embedding dimension: {self.embedding_dim}")
        self.faiss_idx = FaissEmbeddings(self.embedding_dim)
        
        # Create file metadata collection with proper indexes
        logging.info("Creating file_metadata collection and indexes")
        self.file_metadata_collection.create_index("file_path", unique=True)
        self.file_metadata_collection.create_index("content_hash")
        
        # logging.info("Deleting existing documents from logs collection")
        # self.logs_collection.delete_many({})
        logging.info("Creating index on logs collection")
        self.logs_collection.create_index([
            ("doc_id", pymongo.ASCENDING), 
            ("chunk_idx", pymongo.ASCENDING)
            ], unique=True
        )
        # Create text index for exact phrase matching
        self.logs_collection.create_index([("text", pymongo.TEXT)])
        logging.info("Index created successfully")

    def _calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _is_file_processed(self, file_path):
        """Check if file has already been processed by checking file_metadata collection"""
        file_hash = self._calculate_file_hash(file_path)
        existing_record = self.file_metadata_collection.find_one({
            "file_path": file_path,
            "content_hash": file_hash
        })
        return existing_record is not None

    def _mark_file_as_processed(self, file_path):
        """Mark file as processed in file_metadata collection"""
        file_hash = self._calculate_file_hash(file_path)
        file_size = self._get_file_size(file_path)
        
        self.file_metadata_collection.update_one(
            {"file_path": file_path},
            {
                "$set": {
                    "file_path": file_path,
                    "content_hash": file_hash,
                    "file_size": file_size,
                    "timestamp": pymongo.MongoClient().server_info()['localTime'] if 'localTime' in pymongo.MongoClient().server_info() else None,
                    "status": "processed"
                }
            },
            upsert=True
        )

    def _get_file_size(self, file_path):
        """Get file size in bytes"""
        import os
        return os.path.getsize(file_path)

    def _create_chunks(self, file_path, text):
        embedding_chunks = []
        text_chunks = []
        start, idx = 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunk_embedding = self.embeddings_model.get_embeddings(chunk_text, idx, file_path)
            text_chunks.append(chunk_text)
            embedding_chunks.append(chunk_embedding)
            start += self.chunk_size - self.overlap_size
            idx += 1
        return text_chunks, embedding_chunks
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to vector store."""
        doc_ids = []
        documents_to_insert = []
        
        # Calculate total chunks for progress bar
        total_chunks = 0
        for doc in documents:
            file_path = doc.metadata.get("file_path", "")
            if not self._is_file_processed(file_path):
                text_chunks, _ = self._create_chunks(file_path, doc.page_content)
                total_chunks += len(text_chunks)
        
        # Process documents with progress bar
        with tqdm(total=total_chunks, desc="Processing and indexing chunks") as pbar:
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                
                # Check if file has already been processed with same content
                if self._is_file_processed(file_path):
                    logging.info(f"Skipping {file_path} - already processed with same content")
                    continue
                    
                logging.info(f"Processing {file_path} - new or changed file")
                text_chunks, embedding_chunks = self._create_chunks(file_path, doc.page_content)
                logging.info(f"Inserting chunks into Faiss index")
                
                for i, chunk in enumerate(embedding_chunks):
                    # Add to Faiss index first to get the index position
                    self.faiss_idx.add(np.array([chunk], dtype=np.float32))
                    # Store document with the corresponding Faiss index
                    doc_data = {
                        "doc_id": file_path, 
                        "chunk_idx": i, 
                        "text": text_chunks[i], 
                        "embeddings": chunk,
                        "metadata": doc.metadata
                    }
                    documents_to_insert.append(doc_data)
                    pbar.update(1)  # Update progress for each chunk processed

                # Mark file as processed after successful embedding generation
                if documents_to_insert:  # Only mark as processed if documents were created
                    self._mark_file_as_processed(file_path)
                    logging.info(f"Marked {file_path} as processed in file_metadata collection")

        logging.info("Inserting documents into logs collection")
        if documents_to_insert:
            self.logs_collection.insert_many(documents_to_insert)
            logging.info("Documents inserted successfully")
        
        # Generate document IDs for return
        doc_ids = [str(i) for i in range(len(documents_to_insert))]
        return doc_ids

    def _create_indexes(self):
        logging.info("Creating index on text field")
        self.logs_collection.create_index([('text', pymongo.ASCENDING)], unique=True)
        logging.info("Index created successfully")
        
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Run similarity search with FAISS and exact phrase matching."""
        logging.info("Getting embeddings for query: %s", query)
        
        # Get embedding for the query text
        query_embedding = self.embeddings_model.get_embeddings([query])
        if query_embedding is None:
            logging.error("Failed to get query embeddings")
            return []
        
        query_embedding_array = np.array(query_embedding, dtype=np.float32)
        if len(query_embedding_array.shape) == 1:
            query_embedding_array = query_embedding_array.reshape(1, -1)
        
        # Search in Faiss index for semantic similarity
        distances, indices = self.faiss_idx.search(query_embedding_array, k)
        
        # First, try to find exact phrase matches in the text chunks
        exact_matches = []
        partial_matches = []
        
        # Search for exact phrase matches using MongoDB text search
        try:
            # Escape special characters for regex search
            escaped_query = re.escape(query.strip())
            exact_match_docs = list(self.logs_collection.find({
                "text": {"$regex": escaped_query, "$options": "i"}  # case-insensitive
            }).limit(k))
            
            for doc in exact_match_docs:
                doc['score'] = 1.0  # High score for exact matches
                exact_matches.append(doc)
        except Exception as e:
            logging.warning(f"Text search failed: {e}")
        
        # If we found exact matches, prioritize them
        results = exact_matches.copy()
        
        # Add semantic similarity results, avoiding duplicates
        if indices is not None and len(indices) > 0:
            all_docs = list(self.logs_collection.find())
            
            for i in indices[0]:
                if i >= 0 and i < len(all_docs):  # Check if index is valid
                    semantic_doc = all_docs[i]
                    # Avoid adding duplicate documents that were already found as exact matches
                    is_duplicate = any(
                        exact_doc['doc_id'] == semantic_doc['doc_id'] and 
                        exact_doc['chunk_idx'] == semantic_doc['chunk_idx'] 
                        for exact_doc in exact_matches
                    )
                    
                    if not is_duplicate:
                        # Add semantic score based on distance
                        semantic_doc['score'] = 0.5  # Lower priority than exact matches
                        results.append(semantic_doc)
        
        # Sort results by score (exact matches first)
        results.sort(key=lambda x: x.get('score', 0.1), reverse=True)
        
        # Convert to LangChain Document format
        langchain_docs = []
        for result in results[:k]:
            doc = Document(
                page_content=result.get('text', ''),
                metadata={
                    'doc_id': result.get('doc_id'),
                    'chunk_idx': result.get('chunk_idx'),
                    'score': result.get('score', 0.0),
                    **result.get('metadata', {})
                }
            )
            langchain_docs.append(doc)
        
        return langchain_docs

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[Tuple[Document, float]]:
        """Run similarity search and return documents with scores."""
        results = self.similarity_search(query, k, **kwargs)
        return [(doc, doc.metadata.get('score', 0.0)) for doc in results]

    @classmethod
    def from_texts(
        cls, 
        texts: List[str], 
        embedding: Embeddings, 
        metadatas: Optional[List[dict]] = None, 
        **kwargs
    ) -> "MongoDBVectorStore":
        """Construct Langchain vector store from raw texts and embeddings."""
        raise NotImplementedError("from_texts is not implemented for this custom vector store")

    def add(self, log_documents: List[Document]):
        """Legacy add method for backward compatibility."""
        return self.add_documents(log_documents)

# Example usage
# vector_store = MongoDBVectorStore("test", "test", "localhost",27017)
# vector_store.add(["This is a test", "Another test 2"])

# # Search for similar texts
# results = vector_store.similarity_search("test", 2)
# print(results)
