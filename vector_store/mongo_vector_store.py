import pymongo
import numpy as np
from llama_index.core.vector_stores.types import VectorStore
from typing import List
from llama_index.core import Document
from vector_store.embeddings import FaissEmbeddings
import logging
import hashlib

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
            self.embedding_dim = len(test_embedding)
        else:
            self.embedding_dim = len(test_embedding.flatten()) if hasattr(test_embedding, 'flatten') else 768  # Default fallback
            
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
    
    def add(self, log_documents: List[Document]):        
        documents = []
        for doc in log_documents:
            # Check if file has already been processed with same content
            if self._is_file_processed(doc.doc_id):
                logging.info(f"Skipping {doc.doc_id} - already processed with same content")
                continue
                
            logging.info(f"Processing {doc.doc_id} - new or changed file")
            text_chunks, embedding_chunks = self._create_chunks(doc.doc_id, doc.text)
            logging.info(f"Inserting chunks into Faiss index")
            for i, chunk in enumerate(embedding_chunks):
                # Add to Faiss index first to get the index position
                self.faiss_idx.add(np.array([chunk], dtype=np.float32))
                # Store document with the corresponding Faiss index
                doc_data = {"doc_id": doc.doc_id, "chunk_idx": i, "text": text_chunks[i], "embeddings": chunk}
                documents.append(doc_data)

            # Mark file as processed after successful embedding generation
            if documents:  # Only mark as processed if documents were created
                self._mark_file_as_processed(doc.doc_id)
                logging.info(f"Marked {doc.doc_id} as processed in file_metadata collection")

        logging.info("Inserting documents into logs collection")
        if documents:
            self.logs_collection.insert_many(documents)
            logging.info("Documents inserted successfully")

    def _create_indexes(self):
        logging.info("Creating index on text field")
        self.logs_collection.create_index([('text', pymongo.ASCENDING)], unique=True)
        logging.info("Index created successfully")
        
    def similarity_search(self, query: str, k: int):
        logging.info("Getting embeddings for query: %s", query)
        
        # Get embedding for the query text
        query_embedding = self.embeddings_model.get_embeddings([query])
        if query_embedding is None:
            logging.error("Failed to get query embeddings")
            return []
        
        query_embedding_array = np.array(query_embedding, dtype=np.float32)
        if len(query_embedding_array.shape) == 1:
            query_embedding_array = query_embedding_array.reshape(1, -1)
        
        # Search in Faiss index
        distances, indices = self.faiss_idx.search(query_embedding_array, k)
        
        results = []
        logging.info("Iterating through indices to retrieve documents")
        
        if indices is not None and len(indices) > 0:
            # Get all documents from the collection to match with Faiss results
            all_docs = list(self.logs_collection.find())
            
            for i in indices[0]:
                if i >= 0 and i < len(all_docs):  # Check if index is valid
                    results.append(all_docs[i])
                elif i >= 0:
                    logging.warning("Faiss index %d out of range for documents list", i)
        
        return results


# Example usage
# vector_store = MongoDBVectorStore("test", "test", "localhost",27017)
# vector_store.add(["This is a test", "Another test 2"])

# # Search for similar texts
# results = vector_store.similarity_search("test", 2)
# print(results)
