import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class PaperFinder:
    """Finds foundational papers based on semantic search."""
    
    def __init__(self, df, embeddings, model):
        self.df = df
        self.embeddings = embeddings
        self.model = model

    def search(self, query, top_k=5):
        """Search for foundational papers."""
        query_vec = self.model.encode([query])
        similarities = cosine_similarity(query_vec, self.embeddings).flatten()
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = self.df.iloc[top_indices].copy()
        results['similarity'] = similarities[top_indices]
        
        return results[['title', 'authors', 'published', 'similarity', 'summary', 'pdf_url']].to_dict(orient='records')
