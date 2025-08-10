
from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction, Documents, Embeddings

class MovieEmbedderFunction(EmbeddingFunction):
    
    def __init__(self) -> None:
        self.model = SentenceTransformer(model_name_or_path='sentence-transformers/all-MiniLM-L6-v2', device='cuda')

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(sentences=input, device='cuda', show_progress_bar=True, batch_size=200)





