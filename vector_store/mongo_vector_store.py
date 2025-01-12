import pymongo
import numpy as np
from llama_index.core.vector_stores.types import VectorStore
from typing import List
from llama_index.core import Document
from vector_store.embeddings import FaissEmbeddings
import logging

class MongoDBVectorStore(VectorStore):
    def __init__(self, db_name, collection_name, embeddings_model, vector_index_name, mongodb_client):
        self.chunk_size = 300  
        self.overlap_size = 20  # Number of overlapping entries
        self.mongodb_client = mongodb_client
        self.db = self.mongodb_client[db_name]
        self.logs_collection = self.db[collection_name]
        self.vector_index_name=vector_index_name
        self.embeddings_model = embeddings_model
        self.faiss_idx = FaissEmbeddings(self.chunk_size)
        logging.info("Deleting existing documents from logs collection")
        self.logs_collection.delete_many({})
        logging.info("Creating index on logs collection")
        self.logs_collection.create_index([
            ("doc_id", pymongo.ASCENDING), 
            ("chunk_idx", pymongo.ASCENDING)
            ], unique=True
        )
        logging.info("Index created successfully")

    def _pad_chunk(self, chunk):
        """Pads a chunk to the specified chunk size."""
        if len(chunk) != self.chunk_size:
            pad_size = self.chunk_size - len(chunk)
            logging.info(f"Padding the chunk with {pad_size} zeros.")
            chunk = np.pad(chunk, (0, pad_size), 'constant').tolist()
        return chunk

    def _create_chunks(self, file_path, text):
        embedding_chunks = []
        text_chunks = []
        start, idx = 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunk_embedding = self._pad_chunk(self.embeddings_model.get_embeddings(chunk_text, idx, file_path))
            text_chunks.append(chunk_text)
            embedding_chunks.append(chunk_embedding)
            start += self.chunk_size - self.overlap_size
            idx += 1
        return text_chunks, embedding_chunks
    
    def add(self, log_documents: List[Document]):        
        documents = []
        for doc in log_documents:
            text_chunks, embedding_chunks = self._create_chunks(doc.doc_id, doc.text)
            logging.info(f"Inserting chunk into Faiss index")
            for i, chunk in enumerate(embedding_chunks):
                self.faiss_idx.add(np.array([chunk], dtype=np.float32))
                documents.append({"doc_id": doc.doc_id, "chunk_idx": i, "text": text_chunks[i],"embeddings": chunk})

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
        
        query_embeddings = self.embeddings_model.get_embeddings([query])
        query_text, query_chunks = self._create_chunks(query_embeddings)
        _, indices = self.faiss_idx.search(np.array(query_chunks, dtype=np.float32), k)
        
        results = []
        logging.info("Iterating through indices to retrieve documents")
        for i in indices[0]:
            logging.info("Retrieving document with _id: %s", i)
            doc = self.logs_collection.find_one({"doc_id": int(i)})
            if doc:
                results.append(doc)
            else:
                logging.warning("Document not found for _id: %d", i)
        return results


# Example usage
# vector_store = MongoDBVectorStore("test", "test", "localhost",27017)
# vector_store.add(["This is a test", "Another test", "Another test 2"])

# # Search for similar texts
# results = vector_store.similarity_search("test", 2)
# print(results)
