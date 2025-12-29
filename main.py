import os
import sys
import pandas as pd
from arxiv_ideate import DataManager, IdeateModel, Ideator, PaperFinder

def main():
    print("--- Arxiv Ideate Professional ---")
    
    # 1. Setup Data
    manager = DataManager()
    if not os.path.exists("data/filtered_papers.csv"):
        print("Dataset not found. Running initial download and filter...")
        df = manager.filter_papers()
        manager.save()
    else:
        df = pd.read_csv("data/filtered_papers.csv")

    # Ensure combined text for embedding
    df['combined_text'] = df['title'] + " " + df['summary']
    
    # 2. Setup Model
    model_engine = IdeateModel()
    embeddings = model_engine.generate_embeddings(df)
    
    # 3. Interactive Loop
    while True:
        print("\nWhat would you like to do?")
        print("1. Generate New Ideas (Ideate)")
        print("2. Find Foundational Papers (Search)")
        print("3. Run Analysis (Clustering)")
        print("4. Exit")
        
        choice = input("> ")
        
        if choice == '1':
            concept = input("Enter a concept or keyword for ideation (optional): ")
            # Ensure clusters exist
            if 'cluster' not in df.columns:
                print("Running clustering first...")
                df['cluster'] = model_engine.cluster()
                df.to_csv("data/filtered_papers.csv", index=False)
            
            ideator = Ideator(df)
            ideator.analyze_clusters()
            ideas = ideator.generate_ideas(count=5, concept=concept)
            print("\nGenerated Ideas:")
            for i, idea in enumerate(ideas):
                print(f"{i+1}. {idea}")
                
        elif choice == '2':
            query = input("Enter a topic to find foundational papers: ")
            finder = PaperFinder(df, embeddings, model_engine.model)
            results = finder.search(query)
            print("\nFoundational Papers Found:")
            for i, res in enumerate(results):
                print(f"{i+1}. [{res['similarity']:.2f}] {res['title']}")
                print(f"   PDF: {res['pdf_url']}")
                
        elif choice == '3':
            print("Performing HDBSCAN clustering analysis...")
            df['cluster'] = model_engine.cluster()
            df.to_csv("data/filtered_papers.csv", index=False)
            print("Clustering complete. Data updated.")
            
        elif choice == '4':
            break

if __name__ == "__main__":
    main()
