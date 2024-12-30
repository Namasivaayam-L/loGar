# LoGar: Log Analysis and QA System

LoGar is an open-source application for analyzing log files and performing question-answering (QA) tasks on them. It leverages cutting-edge vector search, embeddings, and retrieval-augmented generation (RAG) techniques to extract insights from log files effectively.

---

## Key Features

- **Log File Ingestion**: Read and process log files from a directory and convert their contents into embeddings.
- **Embeddings Support**: Utilize CodeBERT, SentenceTransformer, FAISS, and HNSW for generating and storing embeddings.
- **Vector Store Integration**: Store embeddings and metadata in MongoDB, supporting efficient similarity searches.
- **Retrieval-Augmented QA**: Fetch relevant log chunks and enable QA using an open-source large language model (LLM).

---

## Prerequisites

- Python 3.8+
- MongoDB
- NVIDIA GPU (for training and inference acceleration)

### Python Dependencies
Install the required libraries using:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```plaintext
.
├── main.py                       # Entry point for log ingestion
├── vector_store/
│   ├── log_dir_reader.py         # Reads and processes log files
│   ├── embeddings.py             # Embedding model implementations
│   ├── mongo_vector_store.py     # MongoDB-based vector store
├── config/
│   └── .env                      # Environment variables configuration
├── temp/                         # Temporary storage for embeddings
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Setup

### 1. Configure Environment Variables
Create a `.env` file inside the `config/` directory with the following details:
```env
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DBNAME=logar_db
MONGODB_COLLECTION_NAME=logar_collection
VECTOR_IDX=log_vector_index
LOGS_DIR=path/to/log/files
EMBED_MODEL=microsoft/codebert-base
```

### 2. Prepare Temporary Directory
Ensure the `temp/np_vecs/` directory exists for storing intermediate embeddings:
```bash
mkdir -p temp/np_vecs
```

### 3. Run Log Ingestion
Ingest logs and generate embeddings:
```bash
python main.py
```

---

## Usage

### 1. Retrieve Relevant Log Chunks
Use the document retriever to fetch log chunks related to a specific query:
```python
retrieved_chunks = vector_store.similarity_search_with_text(query="error logs", k=5)
```

### 2. Perform Question Answering
Integrate an open-source LLM to answer questions based on the retrieved chunks.

---

## Roadmap

### LoGar Core Functionality
- [x] Log ingestion and embedding generation
- [x] Vector store integration with MongoDB
- [x] Retrieve log chunks based on similarity
- [ ] Integrate an open-source LLM for RAG-based QA
- [ ] Optimize chunking strategy for large embeddings

### Future Enhancements
- Add support for additional embedding models
- Provide an interactive CLI for queries
- Expand metadata extraction for detailed analysis

---

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements
Special thanks to:
- The creators of CodeBERT, SentenceTransformer, FAISS, and HNSW.
- Open-source communities for their invaluable tools and support.

---

## Contact
Created by **Namasivaayam L.**

For queries or collaboration, reach out via [LinkedIn](https://www.linkedin.com/in/your-profile).
