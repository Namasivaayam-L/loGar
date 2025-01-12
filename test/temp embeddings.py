import torch
from datetime import datetime
import faiss, hnswlib
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List
from tqdm import tqdm
import logging


class CodeBERTEmbeddings:
    def __init__(self, model_name="microsoft/codebert-base"):
        logging.debug("Initializing CodeBERTEmbeddings")
        self.model_name = model_name
        logging.debug("Model name: %s", self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.debug("Device: %s", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logging.debug("Tokenizer Loaded")
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        logging.debug("Model Loaded")

    def _generate_embeddings(self, texts):
        logging.debug("Generating embeddings for %d texts", len(texts))
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                return embeddings
        except Exception as e:
            logging.error("Error generating embeddings: %s", e, exc_info=True)
            return None

    def get_embeddings(self, texts: List[str], idx: int = None, file_path: str = None):
        try:
            save_path = os.path.join(os.path.dirname(file_path) if file_path else "./temp/query_embeddings", f"{idx}_chunk" if idx else f"{datetime.now().strftime('%H:%M:%S:%d:%m:%Y')}")
            if os.path.exists(save_path):
                logging.debug("Loading Embeddings from %s", save_path)
                flattened_embeddings =  np.load(save_path)
                return flattened_embeddings
            else:
                logging.debug(f"Generating Embeddings and saving at :{save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok = True)
                embeddings = self._generate_embeddings(texts)
                np.savez_compressed(save_path, embeddings.flatten())
                logging.debug("Embeddings saved to %s", save_path)
                return embeddings.flatten().tolist()
        except Exception as e:
            logging.error("Error in get_embeddings: %s", e, exc_info=True)
            return None


class SentenceTransformerEmbeddings:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logging.debug("Initializing SentenceTransformerEmbeddings")
        self.model_name = model_name
        logging.debug("Model name: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

    def get_embeddings(self, sentences):
        try:
            return self.model.encode(sentences)
        except Exception as e:
            logging.error("Error generating embeddings: %s", e, exc_info=True)
            return None


class FaissEmbeddings:

    def __init__(self, dim, metric_type=faiss.METRIC_L2, index_type="flat"):
        logging.debug("Initializing FaissEmbeddings")
        self.dim = dim
        if index_type == "flat":
            self.index = faiss.IndexFlat(self.dim, metric_type)
        elif index_type == "ivf":
            self.index = faiss.IndexIVFFlat(self.index, self.dim, 100)  # Adjust nlist as needed
        else:
            raise ValueError("Invalid index type. Choose 'flat' or 'ivf'.")

    def add(self, vectors):
        try:
            self.index.add(vectors)
        except Exception as e:
            logging.exception("Error adding vectors to Faiss index: %s", e)

    def search(self, query_vectors, k=10):
        try:
            distances, indices = self.index.search([query_vectors] if len(query_vectors.shape) == 1 else query_vectors, k)
            logging.debug("Distances: %s, Indices: %s", distances, indices)
            return distances, indices
        except Exception as e:
            logging.error("Error searching Faiss index: %s", e, exc_info=True)
            return None, None


class HnswEmbeddings:
    
    def __init__(self, dim, max_elements, ef_parameter=50, ef_construction=200, M=16):
        logging.debug("Initializing HnswEmbeddings")
        self.dim = dim
        self.index = hnswlib.Index(space="l2", dim=dim)
        # ef_construction and M, which control the construction and connectivity of the graph.
        self.index.init_index(
            max_elements=max_elements, ef_construction=ef_construction, M=M
        )
        # Set ef parameter for query time (trade-off between speed and accuracy)
        self.index.set_ef(ef_parameter)

    def add(self, vectors):
        try:
            logging.debug("Adding %d vectors to HNSW index", vectors.shape[0])
            self.index.add_items(vectors)
        except Exception as e:
            logging.error("Error adding vectors to HNSW index: %s", e, exc_info=True)

    def search(self, query_vector, k=5):
        try:
            logging.debug("Searching HNSW index for k=%d nearest neighbors", k)
            labels, distances = self.index.knn_query(query_vector, k=k)
            return labels, distances
        except Exception as e:
            logging.error("Error searching HNSW index: %s", e, exc_info=True)
            return None, None
