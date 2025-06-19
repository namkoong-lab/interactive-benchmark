import json
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import openai
from typing import List, Dict, Tuple, Optional
from item_index import ItemIndex
import streamlit as st
from tqdm import tqdm
import pickle

class EmbeddingVisualizer:
    def __init__(self, openai_api_key: str = None, cache_dir: str = "embedding_cache"):
        """
        Initialize the embedding visualizer.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            cache_dir: Directory to cache embeddings
        """
        self.openai_api_key = openai_api_key
        self.cache_dir = cache_dir
        self.embeddings_cache = {}
        self.item_index = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set up OpenAI client if API key is provided
        if openai_api_key:
            openai.api_key = openai_api_key
            self.client = openai.OpenAI(api_key=openai_api_key)
        else:
            self.client = None
    
    def load_item_index(self):
        """Load the item index."""
        if self.item_index is None:
            self.item_index = ItemIndex()
            self.item_index.load_items()
        return self.item_index
    
    def get_product_text(self, item: Dict) -> str:
        """
        Extract and combine text content from a product item.
        
        Args:
            item: Product item dictionary
            
        Returns:
            Combined text content for embedding
        """
        text_parts = []
        
        # Add title
        if item['metadata'].get('title'):
            text_parts.append(item['metadata']['title'])
        
        # Add features
        if item['metadata'].get('features'):
            text_parts.extend(item['metadata']['features'])
        
        # Add description
        if item['metadata'].get('description'):
            text_parts.extend(item['metadata']['description'])
        
        # Add store/brand
        if item['metadata'].get('store'):
            text_parts.append(f"Brand: {item['metadata']['store']}")
        
        # Add details if available
        if item['metadata'].get('details'):
            try:
                details = json.loads(item['metadata']['details'])
                for key, value in details.items():
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
            except:
                pass
        
        return " ".join(text_parts)
    
    def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
        """
        Get embedding for text using OpenAI API.
        
        Args:
            text: Text to embed
            model: OpenAI embedding model to use
            
        Returns:
            Embedding vector
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def get_cached_embedding(self, item_id: int, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache or compute and cache it.
        
        Args:
            item_id: Product item ID
            text: Text content to embed
            
        Returns:
            Embedding vector
        """
        cache_file = os.path.join(self.cache_dir, f"embedding_{item_id}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data['text_hash'] == hash(text):
                        return cached_data['embedding']
            except:
                pass
        
        # Compute new embedding
        embedding = self.get_embedding(text)
        if embedding is not None:
            # Cache the embedding
            cache_data = {
                'text_hash': hash(text),
                'embedding': embedding
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        
        return embedding
    
    def get_category_embeddings(self, category_path: List[str] = None) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Get embeddings for all products in a category.
        
        Args:
            category_path: Category path to filter by (None for all products)
            
        Returns:
            Tuple of (embeddings, item_ids, texts)
        """
        index = self.load_item_index()
        
        # Get items in category
        if category_path:
            items = []
            for path in index.category_indices.keys():
                if all(cat in path for cat in category_path):
                    items.extend([index.items[item_id] for item_id in index.category_indices[path]])
        else:
            items = list(index.items.values())
        
        embeddings = []
        item_ids = []
        texts = []
        
        print(f"Computing embeddings for {len(items)} products...")
        
        for item in tqdm(items):
            text = self.get_product_text(item)
            if text.strip():
                embedding = self.get_cached_embedding(item['item_id'], text)
                if embedding is not None:
                    embeddings.append(embedding)
                    item_ids.append(item['item_id'])
                    texts.append(text)
        
        if not embeddings:
            raise ValueError("No valid embeddings found")
        
        return np.array(embeddings), item_ids, texts
    
    def compute_diversity_metrics(self, embeddings: np.ndarray) -> Dict:
        """
        Compute diversity metrics for a set of embeddings.
        
        Args:
            embeddings: Array of embedding vectors
            
        Returns:
            Dictionary of diversity metrics
        """
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(embeddings)
        
        # Remove self-similarities (diagonal)
        np.fill_diagonal(similarities, 0)
        
        # Average similarity
        avg_similarity = np.mean(similarities)
        
        # Standard deviation of similarities
        std_similarity = np.std(similarities)
        
        # Minimum similarity (most different pair)
        min_similarity = np.min(similarities)
        
        # Maximum similarity (most similar pair)
        max_similarity = np.max(similarities)
        
        # Diversity score (1 - average similarity)
        diversity_score = 1 - avg_similarity
        
        return {
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'diversity_score': diversity_score,
            'num_products': len(embeddings)
        }
    
    def reduce_dimensions(self, embeddings: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            method: Dimensionality reduction method ("pca" or "tsne")
            n_components: Number of components to reduce to
            
        Returns:
            Reduced embeddings
        """
        if method.lower() == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
        elif method.lower() == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        return reducer.fit_transform(embeddings)
    
    def create_2d_visualization(self, embeddings: np.ndarray, item_ids: List[int], 
                               texts: List[str], method: str = "pca") -> go.Figure:
        """
        Create 2D visualization of embeddings.
        
        Args:
            embeddings: Embedding vectors
            item_ids: List of item IDs
            texts: List of text descriptions
            method: Dimensionality reduction method
            
        Returns:
            Plotly figure
        """
        # Reduce to 2D
        reduced_embeddings = self.reduce_dimensions(embeddings, method, 2)
        
        # Create hover text
        hover_text = []
        for i, (item_id, text) in enumerate(zip(item_ids, texts)):
            # Truncate text for hover
            short_text = text[:100] + "..." if len(text) > 100 else text
            hover_text.append(f"Item {item_id}<br>{short_text}")
        
        fig = go.Figure(data=[go.Scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=np.arange(len(embeddings)),
                colorscale='Viridis',
                opacity=0.7
            ),
            text=hover_text,
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f"Product Embeddings - {method.upper()} Visualization",
            xaxis_title=f"{method.upper()} Component 1",
            yaxis_title=f"{method.upper()} Component 2",
            showlegend=False
        )
        
        return fig
    
    def create_3d_visualization(self, embeddings: np.ndarray, item_ids: List[int], 
                               texts: List[str], method: str = "pca") -> go.Figure:
        """
        Create 3D visualization of embeddings.
        
        Args:
            embeddings: Embedding vectors
            item_ids: List of item IDs
            texts: List of text descriptions
            method: Dimensionality reduction method
            
        Returns:
            Plotly figure
        """
        # Reduce to 3D
        reduced_embeddings = self.reduce_dimensions(embeddings, method, 3)
        
        # Create hover text
        hover_text = []
        for i, (item_id, text) in enumerate(zip(item_ids, texts)):
            short_text = text[:100] + "..." if len(text) > 100 else text
            hover_text.append(f"Item {item_id}<br>{short_text}")
        
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            z=reduced_embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=np.arange(len(embeddings)),
                colorscale='Viridis',
                opacity=0.7
            ),
            text=hover_text,
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f"Product Embeddings - 3D {method.upper()} Visualization",
            scene=dict(
                xaxis_title=f"{method.upper()} Component 1",
                yaxis_title=f"{method.upper()} Component 2",
                zaxis_title=f"{method.upper()} Component 3"
            ),
            showlegend=False
        )
        
        return fig
    
    def create_cluster_visualization(self, embeddings: np.ndarray, item_ids: List[int], 
                                   texts: List[str], n_clusters: int = 5) -> go.Figure:
        """
        Create visualization with clustering.
        
        Args:
            embeddings: Embedding vectors
            item_ids: List of item IDs
            texts: List of text descriptions
            n_clusters: Number of clusters
            
        Returns:
            Plotly figure
        """
        # Reduce to 2D for visualization
        reduced_embeddings = self.reduce_dimensions(embeddings, "pca", 2)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Create hover text
        hover_text = []
        for i, (item_id, text) in enumerate(zip(item_ids, texts)):
            short_text = text[:100] + "..." if len(text) > 100 else text
            hover_text.append(f"Item {item_id}<br>Cluster {clusters[i]}<br>{short_text}")
        
        fig = go.Figure()
        
        # Add scatter plot with different colors for each cluster
        for cluster_id in range(max(clusters) + 1):
            mask = clusters == cluster_id
            fig.add_trace(go.Scatter(
                x=reduced_embeddings[mask, 0],
                y=reduced_embeddings[mask, 1],
                mode='markers',
                marker=dict(size=8, opacity=0.7),
                text=[hover_text[i] for i in range(len(hover_text)) if mask[i]],
                hoverinfo='text',
                name=f'Cluster {cluster_id}'
            ))
        
        fig.update_layout(
            title=f"Product Clusters (K={n_clusters})",
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2"
        )
        
        return fig
    
    def compare_categories(self, category_paths: List[List[str]]) -> go.Figure:
        """
        Compare diversity across multiple categories.
        
        Args:
            category_paths: List of category paths to compare
            
        Returns:
            Plotly figure with comparison
        """
        categories_data = []
        
        for category_path in category_paths:
            try:
                embeddings, item_ids, texts = self.get_category_embeddings(category_path)
                metrics = self.compute_diversity_metrics(embeddings)
                categories_data.append({
                    'category': ' > '.join(category_path) if category_path else 'All Products',
                    'diversity_score': metrics['diversity_score'],
                    'avg_similarity': metrics['avg_similarity'],
                    'num_products': metrics['num_products']
                })
            except Exception as e:
                print(f"Error processing category {category_path}: {e}")
        
        if not categories_data:
            raise ValueError("No valid categories to compare")
        
        df = pd.DataFrame(categories_data)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Diversity Score by Category', 'Average Similarity by Category'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Diversity score plot
        fig.add_trace(
            go.Bar(x=df['category'], y=df['diversity_score'], name='Diversity Score'),
            row=1, col=1
        )
        
        # Average similarity plot
        fig.add_trace(
            go.Bar(x=df['category'], y=df['avg_similarity'], name='Avg Similarity'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Category Diversity Comparison",
            height=500,
            showlegend=False
        )
        
        return fig

def main():
    """Main function for command-line usage."""
    # You'll need to set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    visualizer = EmbeddingVisualizer(api_key)
    
    # Example usage
    print("Loading data...")
    index = visualizer.load_item_index()
    
    # Get available categories
    categories = index.get_category_hierarchy()
    print("\nAvailable categories:")
    for cat, data in categories.items():
        print(f"- {cat} ({data['count']} items)")
    
    # Example: Analyze a specific category
    category_path = ["All Beauty", "Hair Care"]
    print(f"\nAnalyzing category: {' > '.join(category_path)}")
    
    try:
        embeddings, item_ids, texts = visualizer.get_category_embeddings(category_path)
        metrics = visualizer.compute_diversity_metrics(embeddings)
        
        print(f"Diversity metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Create visualizations
        fig_2d = visualizer.create_2d_visualization(embeddings, item_ids, texts, "pca")
        fig_2d.show()
        
        fig_cluster = visualizer.create_cluster_visualization(embeddings, item_ids, texts, 5)
        fig_cluster.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 