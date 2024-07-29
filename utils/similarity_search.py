import math
from utils.text_embedding import TextEmbedding

class SimilaritySearch:
    def __init__(self):
        self.text_embedding = TextEmbedding()
    def compare_embeddings(self, embedding1, embedding2):
        length = min(len(embedding1), len(embedding2))
        dotprod = 0

        for i in range(length):
            dotprod += (embedding1[i] or 0) * (embedding2[i] or 0)

        return dotprod

    def cosine_similarity(self, embedding1, embedding2):

        dot_product = self.compare_embeddings(embedding1, embedding2)
        magnitudeA = math.sqrt(sum(value * value for value in embedding1))
        magnitudeB = math.sqrt(sum(value * value for value in embedding2))
        if magnitudeA == 0 or magnitudeB == 0:
            return 0
        return dot_product / (magnitudeA * magnitudeB)

    def find_nearest_paragraph(self, embedding_store, target_embedding, count=5):
        scored_entries = [
            {
                "text": entry["text"],
                "embedding": entry["embedding"],
                "score": self.cosine_similarity(target_embedding, entry["embedding"]),
            }
            for entry in embedding_store
        ]
        scored_entries = sorted(scored_entries, key=lambda x: x["score"], reverse=True)
        return scored_entries[:count]

    def semantic_search(self, query, embedding_store, topN=5):
        query_embedding = self.create_testing_embedding(query)
        return self.find_nearest_paragraph(embedding_store, query_embedding, topN)

    def create_testing_embedding(self, test_chunk):
        # Since it's a single chunk, no need for a list
        embeddings = self.text_embedding.generate_embeddings([test_chunk])
        return embeddings[0]["embedding"] if embeddings else None
