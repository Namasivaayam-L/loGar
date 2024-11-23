import os
import time
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
# Indices and Storage
import pymongo
from vector_store.mongo_vector_store import MongoDBVectorStore
from vector_store.log_dir_reader import LogDirectoryReader
from vector_store.embeddings import CodeBERTEmbeddings, SentenceTransformerEmbeddings, FaissEmbeddings, HnswEmbeddings

load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"))

MONGODB_HOST = os.getenv("MONGODB_HOST")
MONGODB_PORT = os.getenv("MONGODB_PORT")
MONGODB_DBNAME = os.getenv("MONGODB_DBNAME")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
VECTOR_IDX = os.getenv("VECTOR_IDX")
LOGS_DIR = os.getenv("LOGS_DIR")
EMBED_MODEL = os.getenv('EMBED_MODEL')
MONGODB_CLIENT = pymongo.MongoClient(MONGODB_HOST, int(MONGODB_PORT))
EMBEDINGS_MODEL = CodeBERTEmbeddings(model_name=EMBED_MODEL)


def ingest_logs():
    print("--------------------->Ingest Logs<-----------------------")
    start = time.time()
    
    log_reader = LogDirectoryReader(    
                    directory_path=LOGS_DIR,
                    embedding_model=EMBEDINGS_MODEL,
                )
    
    documents = log_reader.load_data()
    
    vector_store  = MongoDBVectorStore(
						db_name=MONGODB_DBNAME,
						collection_name=MONGODB_COLLECTION_NAME,
						vector_index_name=VECTOR_IDX,
						mongodb_client=MONGODB_CLIENT
                    )

    vector_store.add(documents)
    
    end = time.time()
    print(
        f"  Total Time = {end-start}",
        f"Total Documents = {len(documents)}",
    )
    
ingest_logs()