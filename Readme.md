# 🔍 LoGar: Log Analysis & QA System

**LoGar** is an open-source application for advanced **log file analysis** and **Question-Answering (QA)**. It leverages **vector search**, **embeddings**, and **Retrieval-Augmented Generation (RAG)** to extract powerful insights from your log data.

---

## ✨ Features

-   **Log Ingestion**: 📂 Process log files from directories and convert content into searchable embeddings.
-   **Flexible Embeddings**: 🧠 Supports CodeBERT, SentenceTransformer, FAISS, and HNSW for efficient embedding generation and storage.
-   **MongoDB Vector Store**: 🗄️ Stores embeddings and metadata in MongoDB for high-performance similarity searches.
-   **RAG-powered QA**: 💬 Fetches relevant log chunks to enable intelligent QA using open-source Large Language Models (LLMs).

---

## 🛠️ Tech Stack

| Component           | Tech                                    |
|---------------------|-----------------------------------------|
| Language            | Python 3.8+                             |
| Database            | MongoDB                                 |
| Embeddings          | CodeBERT, SentenceTransformer, FAISS, HNSW |
| QA Model            | Open-source LLMs (for RAG)              |
| Acceleration        | NVIDIA GPU (recommended)                |

---

## 🚀 Setup Instructions

### Prerequisites

-   Ensure **Python 3.8+** and **MongoDB** are installed.
-   An **NVIDIA GPU** is recommended for optimal performance.

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/Namasivaayam-L/loGar.git # Replace with actual repo URL if different
cd loGar
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the `config/` directory:

```env
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DBNAME=logar_db
MONGODB_COLLECTION_NAME=logar_collection
VECTOR_IDX=log_vector_index
LOGS_DIR=path/to/your/log/files # 👈 IMPORTANT: Update this path!
EMBED_MODEL=microsoft/codebert-base
```

### 3. Prepare Temporary Directory

```bash
mkdir -p temp/np_vecs
```

---

## 🧑‍💻 Usage

### 1. Ingest Logs & Generate Embeddings

Run the main script to process your logs:

```bash
python main.py
```

### 2. Retrieve Log Chunks

Example Python snippet for retrieving relevant log chunks:

```python
import vector_store # Assuming vector_store is initialized
# ... (MongoDB connection, embedding model setup)

query = "error logs from authentication service"
retrieved_chunks = vector_store.similarity_search_with_text(query=query, k=5)
for chunk in retrieved_chunks:
    print(chunk)
```

### 3. Perform Question Answering

Integrate an open-source LLM with the retrieved chunks for advanced QA.

---

## 📈 Roadmap

-   [x] Log ingestion and embedding generation
-   [x] Vector store integration with MongoDB
-   [x] Retrieve log chunks based on similarity
-   [ ] Integrate an open-source LLM for RAG-based QA
-   [ ] Optimize chunking strategy for large embeddings

---

## 📂 Project Structure

```
.
├── main.py                     # Entry point for log ingestion and processing
├── vector_store/               # Core vector store functionalities
│   ├── log_dir_reader.py       # Reads and preprocesses log files
│   ├── embeddings.py           # Handles embedding model implementations
│   └── mongo_vector_store.py   # MongoDB integration for vector storage
├── config/                     # Configuration files
│   └── .env                    # Environment variables for MongoDB and models
├── temp/                       # Temporary storage for intermediate embeddings (e.g., `temp/np_vecs`)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

Built with ❤️ by **Namasivaayam L.**
