# 🔍 LoGar: Log Analysis & QA System

**LoGar** is an open-source application for advanced **log file analysis** and **Question-Answering (QA)**. It leverages **vector search**, **embeddings**, and **Retrieval-Augmented Generation (RAG)** to extract powerful insights from your log data.

---

## ✨ Features

-   **Log Ingestion**: 📂 Process log files from directories and convert content into searchable embeddings.
-   **Flexible Embeddings**: 🧠 Supports CodeBERT, SentenceTransformer, FAISS, and HNSW for efficient embedding generation and storage.
-   **MongoDB Vector Store**: 🗄️ Stores embeddings and metadata in MongoDB for high-performance similarity searches.
-   **Dual Query Modes**: 🔄 Choose between simple vector search or Agentic RAG for advanced log analysis.
-   **Agentic RAG with Gemini**: 🤖 Uses Google's Gemini models for intelligent, context-aware log analysis and QA.
-   **LangGraph Workflow**: 🔄 Structured agent workflows for systematic log investigation and error tracking.

---

## 🛠️ Tech Stack

| Component           | Tech                                    |
|---------------------|-----------------------------------------|
| Language            | Python 3.8+                             |
| Database            | MongoDB                                 |
| Embeddings          | CodeBERT, SentenceTransformer, FAISS, HNSW |
| QA Model            | Google's Gemini (via LangChain)         |
| Agent Framework     | LangGraph                               |
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
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
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

### 2. Choose Query Mode

The system offers two query modes:

**Mode 1 - Simple Search**: Fast retrieval of similar log chunks
```bash
# Enter mode 1 when prompted
# Enter your query: "error authentication"
# Enter number of results: 5
```

**Mode 2 - Agentic RAG**: Detailed analysis using Gemini LLM
```bash
# Enter mode 2 when prompted
# Enter your query: "What caused the authentication service downtime at 3 AM?"
```

### 3. Advanced Log Analysis with Agentic RAG

For complex log analysis questions, use Agentic mode which provides:

- 🤖 **Contextual Analysis**: Gemini-powered interpretation of log patterns
- 📊 **Error Investigation**: Detailed root cause analysis
- 🎯 **Progress Tracking**: Workflow and sequence analysis
- 📎 **Citations**: References to relevant log chunks with scores

Example queries:
- "Analyze the authentication errors in the last batch"
- "What workflow steps led to this service crash?"
- "Track the progress of database connection issues"

---

## 📈 Roadmap

-   [x] Log ingestion and embedding generation
-   [x] Vector store integration with MongoDB
-   [x] Retrieve log chunks based on similarity
-   [x] Integrate Gemini LLM for Agentic RAG-based QA
-   [ ] Add multi-agent workflows for complex analysis
-   [ ] Optimize chunking strategy for large embeddings

---

## 📂 Project Structure

```
.
├── main.py                     # Enhanced entry point with dual-mode querying (simple search + agentic RAG)
├── agentic_rag/                # Agentic RAG system components
│   ├── __init__.py            # Package initialization
│   ├── agent_core.py          # LogAnalysisAgent class with Gemini integration
│   └── langgraph_workflow.py  # LangGraph workflow for structured analysis
├── vector_store/               # Core vector store functionalities
│   ├── log_dir_reader.py       # Reads and preprocesses log files
│   ├── embeddings.py           # Handles embedding model implementations
│   └── mongo_vector_store.py   # MongoDB integration for vector storage
├── config/                     # Configuration files
│   └── .env                    # Environment variables for MongoDB, models, and API keys
├── temp/                       # Temporary storage for intermediate embeddings (e.g., `temp/np_vecs`)
├── logs/                       # Sample log files for testing
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

Built with ❤️ by **Namasivaayam L.**
