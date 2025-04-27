import torch
import os

class VectorDB:
    def __init__(self, path: str = "./database/") -> None:
        self.vectordb_path = path
        self.vec_data = {} # store the embedded vector
        self.vec_metadata = {} # store document id of a chunk
        if os.path.exists(os.path.join(path, "vector.pt")) and os.path.exists(os.path.join(path, "metadata.pt")):
            self.load_data()

    def load_data(self) -> None:
        data_path = os.path.join(self.vectordb_path, "vector.pt")
        metadata_path = os.path.join(self.vectordb_path, "metadata.pt")
        self.vec_data = torch.load(data_path)
        self.vec_metadata = torch.load(metadata_path)
    
    def add_vector(self, id: str, vector: torch.Tensor, metadata: tuple) -> None:
        self.vec_data[id] = vector
        self.vec_metadata[id] = metadata
    
    def get_vector(self, id: str) -> torch.Tensor:
        return self.vec_data.get(id)
    
    def get_metadata(self, id: str) -> torch.Tensor:
        return self.vec_metadata.get(id)
    
    def save_to_json(self) -> None:
        data_path = os.path.join(self.vectordb_path, "vector.pt")
        metadata_path = os.path.join(self.vectordb_path, "metadata.pt")
        torch.save(self.vec_data, data_path)
        torch.save(self.vec_metadata, metadata_path)
    
    def get_topk_similar(self, query_vector: torch.Tensor, k: int = 5, doc_id: int = None) -> list:
        similarity = []
        for id, vector in self.vec_data.items():
            if doc_id is not None and self.get_metadata(id)[0] != doc_id:
                continue
            else:
                dot_result = torch.dot(query_vector, vector) / \
                    (torch.linalg.vector_norm(query_vector) * torch.linalg.vector_norm(vector))
                similarity.append((id, dot_result))
        
        # Note that top k elements could be optimized (using heapq(), or torch.topk())
        # But for simplicity and readability, we will use sort then index first k elements
        similarity.sort(key = lambda x: x[1], reverse = True)
        return similarity[:k]