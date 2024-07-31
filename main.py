from utils.text_embedding import TextEmbedding
from utils.text_extraction import TextExtraction
from utils.similarity_search import SimilaritySearch

def train():
    filenames = ['train/invoice_77073.pdf','train/Faller_8.pdf','train/invoice.pdf','train/2024.03.15_1145.pdf']
    text = TextExtraction(filenames)

    extracted_text = text.extract_text_from_pdf()

    chunk_text = text.chunk_text(extracted_text,1000)
    text_embedding = TextEmbedding()
    embeddings = text_embedding.generate_embeddings(chunk_text)
    return embeddings

embeddings = train()

# test

text = TextExtraction(['test/invoice_102857.pdf'])

extracted_text = text.extract_text_from_pdf()

# print(f"Extracted Invoice {extracted_text}")
search = SimilaritySearch()
results = search.semantic_search(extracted_text, embeddings, 10)

for result in results:
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
    print()