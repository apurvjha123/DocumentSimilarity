from PyPDF2 import PdfReader
import re

class TextExtraction:
    def __init__(self, filenames):
        self.filenames = filenames

    def extract_text_from_pdf(self):
        text = ''
        for filename in self.filenames:
            reader = PdfReader(filename)
            for page in reader.pages:
                text += page.extract_text()
        return text
    def chunk_text(self, text, chunk_size=1000):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]