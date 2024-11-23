import pymongo
import numpy as np
from llama_index.core.vector_stores.types import VectorStore
from typing import List
from llama_index.core import Document
from vector_store.embeddings import FaissEmbeddings

class MongoDBVectorStore(VectorStore):
    def __init__(self, db_name, collection_name, vector_index_name, mongodb_client):
        self.mongodb_client = mongodb_client
        self.db = self.mongodb_client[db_name]
        self.logs_collection = self.db[collection_name]
        self.vector_index_name=vector_index_name
        self.faiss_idx = FaissEmbeddings(768)
        self.logs_collection.delete_many({})
        self.logs_collection.create_index([
            ("doc_id", pymongo.ASCENDING), 
            ("chunk_idx", pymongo.ASCENDING)
            ], unique=True
        )

    def add(self, log_documents: List[Document]):        
        chunk_size = 300  
        overlap_size = 20  # Number of overlapping entries
        documents = []
        for doc in log_documents:
            chunks = []
            start = 0
            while start < len(doc.embedding):
                end = min(start + chunk_size, len(doc.embedding))
                chunk = doc.embedding[start:end]
                chunks.append(chunk)
                start += chunk_size - overlap_size
                
            for i, chunk in enumerate(chunks):
                documents.append({"doc_id": doc.doc_id, "chunk_idx": i, "content": chunk})
                
        if documents:
            self.logs_collection.insert_many(documents)

    def _create_indexes(self):
        self.logs_collection.create_index([('text', pymongo.ASCENDING)], unique=True)

    def similarity_search(self, query: str, k: int):
        query_embedding = self.embedding_model.get_embeddings([query])[0].astype('float32') 
        distances, indices = self.faiss_idx.search(np.array([query_embedding]), k)
        
        results = []
        for i in indices[0]:
            doc = self.logs_collection.find_one({"_id": i})
            if doc:
                results.append(doc["text"])
        return results


# # Example usage
# vector_store = MongoVectorStore("test", "test", "localhost",27017)
# vector_store.add(["This is a test", "Another test", "Another test 2"])

# # Search for similar texts
# results = vector_store.similarity_search("test", 2)
# # print(results)
