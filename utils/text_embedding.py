from transformers import AutoTokenizer, AutoModel
import torch

class TextEmbedding:
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    
    def generate_embeddings(self, chunks):
        result = []
        for chunk_list in chunks:
            inputs = self.tokenizer(chunk_list, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                result.append({"text": chunk_list, "embedding": embedding})
        return result