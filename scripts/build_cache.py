import os
import pandas as pd
from arxiv_ideate import DataManager, IdeateModel

def build():
    print("Building Arxiv Ideate cache...")
    
    manager = DataManager()
    if not os.path.exists("data/filtered_papers.csv"):
        print("Filtering papers...")
        df = manager.filter_papers()
        manager.save()
    else:
        print("Loading existing filtered papers...")
        df = pd.read_csv("data/filtered_papers.csv")
    
    # Ensure combined text for embedding
    if 'combined_text' not in df.columns:
        df['combined_text'] = df['title'].fillna('') + " " + df['summary'].fillna('')
    
    model_engine = IdeateModel()
    
    # Generate embeddings (will cache to data/embeddings.pkl)
    print("Generating/Loading embeddings...")
    embeddings = model_engine.generate_embeddings(df)
    
    # Cluster (will add 'cluster' column)
    print("Clustering...")
    df['cluster'] = model_engine.cluster()
    
    # Save final version with clusters
    df.to_csv("data/filtered_papers.csv", index=False)
    print("Build complete! Ready for use with 'python main.py'.")

if __name__ == "__main__":
    build()
