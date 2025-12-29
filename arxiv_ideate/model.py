import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import os
import pickle

class IdeateModel:
    """Handles embeddings and clustering."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.cluster_labels = None

    def generate_embeddings(self, df, text_col='combined_text', cache_path='data/embeddings.pkl'):
        """Generates or loads embeddings."""
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}...")
            with open(cache_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            print("Generating embeddings...")
            texts = df[text_col].tolist()
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
        return self.embeddings

    def cluster(self, embeddings=None, min_cluster_size=15):
        """Performs clustering using UMAP and HDBSCAN."""
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            raise ValueError("No embeddings found. Call generate_embeddings first.")

        print("Reducing dimensionality with UMAP...")
        reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        umap_embeddings = reducer.fit_transform(embeddings)

        print("Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
        self.cluster_labels = clusterer.fit_predict(umap_embeddings)
        return self.cluster_labels
