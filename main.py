import os
import time
from dotenv import load_dotenv
import warnings
import logging

warnings.filterwarnings("ignore")
# Indices and Storage
import pymongo
from vector_store.mongo_vector_store import MongoDBVectorStore
from vector_store.log_dir_reader import LogDirectoryReader
from vector_store.embeddings import (
    CodeBERTEmbeddings,
    SentenceTransformerEmbeddings,
    FaissEmbeddings,
    HnswEmbeddings,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


load_dotenv(os.path.join(os.path.dirname(__file__), "./config", ".env"))

MONGODB_HOST = os.getenv("MONGODB_HOST")
MONGODB_PORT = os.getenv("MONGODB_PORT")
MONGODB_DBNAME = os.getenv("MONGODB_DBNAME")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
VECTOR_IDX = os.getenv("VECTOR_IDX")
LOGS_DIR = os.getenv("LOGS_DIR")
EMBED_MODEL = os.getenv("EMBED_MODEL")
MONGODB_CLIENT = pymongo.MongoClient(MONGODB_HOST, int(MONGODB_PORT))
EMBEDDINGS_MODEL = CodeBERTEmbeddings(model_name=EMBED_MODEL)


def initialize_vector_store_and_log_reader():
    vector_store = MongoDBVectorStore(
        db_name=MONGODB_DBNAME,
        collection_name=MONGODB_COLLECTION_NAME,
        embeddings_model=EMBEDDINGS_MODEL,
        vector_index_name=VECTOR_IDX,
        mongodb_client=MONGODB_CLIENT,
    )

    logger.info("Initialized MongoDBVectorStore")

    log_reader = LogDirectoryReader(
        directory_path=LOGS_DIR,
        embeddings_model=EMBEDDINGS_MODEL,
    )
    logger.info("Initialized LogDirectoryReader")

    return vector_store, log_reader


vector_store, log_reader = initialize_vector_store_and_log_reader()


def ingest_logs():
    logger.info("Starting log ingestion")
    start = time.time()

    documents = log_reader.load_data()
    logger.info(f"Loaded {len(documents)} documents")

    vector_store.add(documents)
    logger.info("Documents added to vector store")

    end = time.time()
    logger.info(f"Total Time = {end-start}\nTotal Documents = {len(documents)}")


def search_logs(query: str, top_k: int = 5):
    logger.info(f"Starting search for query: {query}")
    start = time.time()
    similar_chunks = vector_store.similarity_search(query, k=top_k)
    end = time.time()
    logger.info(f"Total Time = {end - start}")
    logger.info(f"Top {top_k} Document Chunks:")
    for doc in similar_chunks:
        logger.info(f"{doc['doc_id']}: {doc['text']}")



ingest_logs()
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    try:
        top_k = int(input("Enter the number of top results: "))
        search_logs(query, top_k)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
