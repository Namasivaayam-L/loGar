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

# Agentic RAG imports
from agentic_rag.langgraph_workflow import run_agentic_log_analysis

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
EMBEDDINGS_MODEL = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)


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
        logger.info(f"{doc.metadata.get('doc_id', 'N/A')}: {doc.page_content}")


def agentic_log_analysis(query: str, top_k: int = 15):
    """
    Perform agentic log analysis using Gemini LLM for detailed, contextual responses.
    """
    logger.info(f"Starting agentic analysis for query: {query}")
    start = time.time()

    try:
        # Run the agentic workflow
        result = run_agentic_log_analysis(query, vector_store)

        # Display results in a readable format
        analysis = result.get("analysis", "No analysis available")
        retrieved_chunks = result.get("retrieved_chunks", 0)

        print("\n" + "="*80)
        print("🤖 AGENTIC LOG ANALYSIS RESULTS")
        print("="*80)
        print(f"📝 Query: {query}")
        print(f"📊 Retrieved Chunks: {retrieved_chunks}")
        print("\n🔍 ANALYSIS:")
        print("-" * 50)
        print(analysis)

        # Show top relevant chunks with citations
        if result.get("top_relevant_chunks"):
            print(f"\n📎 TOP {len(result['top_relevant_chunks'])} RELEVANT CHUNKS:")
            print("-" * 50)
            for i, chunk in enumerate(result["top_relevant_chunks"], 1):
                print(f"\n[{i}] Doc ID: {chunk['doc_id']}, Chunk: {chunk['chunk_idx']}")
                print(f"Score: {chunk.get('score', 0.0):.4f}")
                print(f"Content: {chunk['content']}")

        print("\n" + "="*80)

    except Exception as e:
        logger.error(f"Error in agentic analysis: {e}")
        print(f"❌ Error in agentic analysis: {e}")

    end = time.time()
    logger.info(f"Agentic analysis completed in {end - start:.2f} seconds")


ingest_logs()

print("\n" + "="*80)
print("🔍 LoGar: Enhanced Log Analysis System")
print("="*80)
print("Available modes:")
print("1. Simple Search → Fast retrieval of similar log chunks")
print("2. Agentic RAG → Detailed analysis using Gemini LLM")
print("="*80)

while True:
    print("\n" + "-"*50)
    mode = input("Choose mode (1 for Simple Search, 2 for Agentic RAG, or 'exit'): ").strip().lower()

    if mode == "exit":
        print("👋 Goodbye!")
        break

    if mode not in ["1", "2"]:
        print("❌ Invalid mode. Please choose 1 or 2.")
        continue

    query = input("Enter your log analysis query: ").strip()
    if not query:
        print("❌ Empty query. Please try again.")
        continue

    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    try:
        if mode == "1":
            # Simple Search Mode
            top_k = int(input("Enter the number of top results (1-20): "))
            if top_k < 1 or top_k > 20:
                print("❌ Please enter a number between 1 and 20.")
                continue
            search_logs(query, top_k)

        elif mode == "2":
            # Agentic RAG Mode
            agentic_log_analysis(query)

        else:
            print("❌ Invalid mode. Please choose 1 or 2.")

    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in main loop: {e}")
