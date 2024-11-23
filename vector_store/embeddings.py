import torch
import faiss, hnswlib
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List
from tqdm import tqdm

class CodeBERTEmbeddings:
    def __init__(self, model_name='microsoft/codebert-base'):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        # self.model.to(self.device)
        print("Using: ","cuda" if torch.cuda.is_available() else "cpu")
        
    def get_embeddings(self, texts: List[str], file_path: str, batch_size=64):
        embeddings = np.array([])
        file_name = os.path.basename(file_path).replace('.log','.npy')
        if file_name in os.listdir('temp/np_vecs'):
            embeddings = np.load(f'temp/np_vecs/{file_name}')
            print(f"Loaded from local dir {file_name}")
        else:
            for i in tqdm(range(0, len(texts), batch_size),desc=f'Gen Embeddings {file_path}'):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    np.append(embeddings,outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
            np.save(f'temp/np_vecs/{file_name}', flat_embeddings)

        flat_embeddings = embeddings.flatten().tolist()
        return flat_embeddings

class SentenceTransformerEmbeddings:
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
    
    def get_embeddings(self, sentences):
        return self.model.encode(sentences)

class FaissEmbeddings:
    
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(self.dim)
    
    def add(self, vectors):        
        self.index.add(vectors)
    
    def search(self, query_vector, k = 10):
        distances, indices = self.index.search(np.array([query_vector]), k)
        return distances, indices

class HnswEmbeddings:
    
    def __init__(self, dim, max_elements, ef_parameter=50, ef_construction=200, M=16):
        self.dim = dim
        self.index = hnswlib.Index(space='l2', dim=dim) 
        # ef_construction and M, which control the construction and connectivity of the graph.
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        # Set ef parameter for query time (trade-off between speed and accuracy)
        self.index.set_ef(ef_parameter)
    
    def add(self, vectors):
        self.index.add_items(vectors)
        
    def search(self, query_vector, k = 5):
        labels, distances = self.index.knn_query(query_vector, k=k)
        return labels, distances
    
    
    