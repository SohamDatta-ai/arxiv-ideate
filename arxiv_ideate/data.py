import pandas as pd
import ast
import os
import kagglehub

class DataManager:
    """Handles data acquisition, filtering, and cleaning."""
    
    DEFAULT_CATEGORIES = [
        'cs.AI', 'cs.LG', 'cs.HC', 
        'cs.RO', 'cs.CV', 'cs.CL', 
        'eess.SP', 'cs.AR'
    ]
    
    DEFAULT_KEYWORDS = [
        'edge', 'wearable', 'emotion', 'personalization',
        'on-device', 'tinyml', 'neuromorphic', 'federated',
        'context-aware', 'multimodal', 'low-power'
    ]

    def __init__(self, dataset_id="yasirabdaali/arxivorg-ai-research-papers-dataset"):
        self.dataset_id = dataset_id
        self.raw_df = None
        self.filtered_df = None

    def download(self):
        """Downloads the dataset using kagglehub."""
        print(f"Downloading dataset {self.dataset_id}...")
        path = kagglehub.dataset_download(self.dataset_id)
        csv_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    csv_file = os.path.join(root, file)
                    break
            if csv_file: break
        
        if not csv_file:
            raise FileNotFoundError("CSV file not found in downloaded dataset.")
        
        print(f"Loading {csv_file}...")
        self.raw_df = pd.read_csv(csv_file)
        return self.raw_df

    def filter_papers(self, categories=None, keywords=None):
        """Filters papers based on categories and keywords."""
        if self.raw_df is None:
            self.download()
            
        categories = categories or self.DEFAULT_CATEGORIES
        keywords = keywords or self.DEFAULT_KEYWORDS
        
        df = self.raw_df.copy()
        
        def safe_literal_eval(val):
            try:
                return ast.literal_eval(val) if isinstance(val, str) else val
            except:
                return []

        df['categories_list'] = df['categories'].apply(safe_literal_eval)
        
        # Filter by categories
        df = df[df['categories_list'].apply(lambda c: any(cat in categories for cat in c))]
        
        # Identify high relevance
        def is_high_relevance(row):
            text = f"{row['title']} {row['summary']}".lower()
            return any(kw.lower() in text for kw in keywords)
        
        df['high_relevance'] = df.apply(is_high_relevance, axis=1)
        self.filtered_df = df
        
        print(f"Filtered to {len(df)} papers ({df['high_relevance'].sum()} high relevance).")
        return df

    def save(self, path="data/filtered_papers.csv"):
        """Saves the filtered dataset."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.filtered_df is not None:
            self.filtered_df.to_csv(path, index=False)
            print(f"Saved to {path}")
