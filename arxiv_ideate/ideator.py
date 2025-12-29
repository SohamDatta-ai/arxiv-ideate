import random
import json

class Ideator:
    """Synthesizes research ideas based on clusters."""
    
    def __init__(self, df):
        self.df = df
        self.clusters_info = {}

    def analyze_clusters(self):
        """Analyze clusters to extract themes."""
        if 'cluster' not in self.df.columns:
            raise ValueError("DataFrame must have a 'cluster' column.")
        
        for cluster_id in self.df['cluster'].unique():
            if cluster_id == -1: continue
            
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            self.clusters_info[str(cluster_id)] = {
                "size": len(cluster_data),
                "titles": cluster_data['title'].head(5).tolist(),
                "high_relevance": cluster_data['high_relevance'].sum()
            }
        return self.clusters_info

    def generate_ideas(self, count=5, concept=None):
        """Generates synthetic ideas."""
        # If concept is provided, we'd ideally find the nearest cluster.
        # For this professional version, let's implement a smarter synthesis.
        
        ideas = []
        cluster_ids = list(self.clusters_info.keys())
        
        if not cluster_ids:
            return ["Analyze data first to identify clusters."]

        # Simple combinatorial synthesis
        for _ in range(count):
            c1 = random.choice(cluster_ids)
            c2 = random.choice(cluster_ids)
            
            t1 = random.choice(self.clusters_info[c1]['titles'])
            t2 = random.choice(self.clusters_info[c2]['titles'])
            
            if concept:
                idea = f"Integrating '{concept}' with the principles of '{t1}' and '{t2}' to create a future-proof research direction."
            else:
                idea = f"Cross-pollinating '{t1}' and '{t2}' for next-generation research innovation."
            
            ideas.append(idea)
            
        return ideas
