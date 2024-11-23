import os
import chardet
from tqdm import tqdm
from typing import List
from llama_index.core import Document

class LogDirectoryReader:
    def __init__(self, directory_path: str, embedding_model):
        self.directory_path = directory_path
        self.embedding_model=embedding_model
        
    def extract_data(self,file_path) -> str:
        with open(file_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
            
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            return str(f.readlines())

    def load_data(self)-> List[Document]:
        log_documents = []

        for root, _, files in os.walk(self.directory_path):
            log_files = [file for file in files if file.endswith(".log")]
            for filename in tqdm(log_files, desc=f"Processing Log Files:"):
                file_path = os.path.join(root, filename)
                content = self.extract_data(file_path)
                embeddings = self.embedding_model.get_embeddings(content,file_path)
                
                log_documents.append(
                    Document(
                        doc_id=file_path,
                        text=content,
                        embedding=embeddings,
                        metadata={
                            "file_name": filename,
                            "category": "log",
                            "author": "Namasivaayam L",
                        },
                        excluded_llm_metadata_keys=["file_name"],
                        metadata_seperator="::",
                        metadata_template="{key}=>{value}",
                        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
                    )
                )

        return log_documents