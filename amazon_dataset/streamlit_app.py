import streamlit as st
import os
import json
from embedding_visualizer import EmbeddingVisualizer
from item_index import ItemIndex
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Product Diversity Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_visualizer():
    """Load the embedding visualizer with caching."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set OPENAI_API_KEY environment variable")
        return None
    
    return EmbeddingVisualizer(api_key)

@st.cache_data
def load_categories():
    """Load category hierarchy with caching."""
    index = ItemIndex()
    index.load_items()
    return index.get_category_hierarchy()

def get_category_options(categories, prefix=""):
    """Recursively get all category options for selection."""
    options = []
    for category, data in categories.items():
        full_path = f"{prefix}{category}" if prefix else category
        options.append(full_path)
        if data['subcategories']:
            options.extend(get_category_options(data['subcategories'], f"{full_path} > "))
    return options

def parse_category_path(category_string):
    """Parse category string into path list."""
    if not category_string:
        return None
    return [cat.strip() for cat in category_string.split(" > ")]

def main():
    st.markdown('<h1 class="main-header">Product Diversity Analyzer</h1>', unsafe_allow_html=True)
    
    # Load data
    visualizer = load_visualizer()
    if visualizer is None:
        return
    
    categories = load_categories()
    category_options = get_category_options(categories)
    
    # Sidebar configuration
    st.sidebar.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
    
    # Category selection
    st.sidebar.subheader("Category Selection")
    selected_category = st.sidebar.selectbox(
        "Choose a category to analyze:",
        ["All Products"] + category_options,
        help="Select a category to analyze product diversity"
    )
    
    # Visualization type
    st.sidebar.subheader("Visualization Type")
    viz_type = st.sidebar.selectbox(
        "Choose visualization type:",
        ["2D Scatter Plot", "3D Scatter Plot", "Clustering", "Diversity Metrics Only"],
        help="Select the type of visualization to display"
    )
    
    # Dimensionality reduction method
    if viz_type in ["2D Scatter Plot", "3D Scatter Plot"]:
        st.sidebar.subheader("Dimensionality Reduction")
        reduction_method = st.sidebar.selectbox(
            "Choose reduction method:",
            ["PCA", "t-SNE"],
            help="PCA is faster, t-SNE preserves local structure better"
        )
    
    # Clustering parameters
    if viz_type == "Clustering":
        st.sidebar.subheader("Clustering Parameters")
        n_clusters = st.sidebar.slider(
            "Number of clusters:",
            min_value=2,
            max_value=10,
            value=5,
            help="Number of clusters to identify in the data"
        )
    
    # Analysis button
    if st.sidebar.button("Analyze Products", type="primary"):
        with st.spinner("Analyzing products..."):
            try:
                # Parse category path
                category_path = parse_category_path(selected_category) if selected_category != "All Products" else None
                
                # Get embeddings
                embeddings, item_ids, texts = visualizer.get_category_embeddings(category_path)
                
                # Compute diversity metrics
                metrics = visualizer.compute_diversity_metrics(embeddings)
                
                # Display metrics
                st.subheader("Diversity Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Diversity Score", f"{metrics['diversity_score']:.3f}")
                with col2:
                    st.metric("Avg Similarity", f"{metrics['avg_similarity']:.3f}")
                with col3:
                    st.metric("Products Analyzed", metrics['num_products'])
                with col4:
                    st.metric("Similarity Std Dev", f"{metrics['std_similarity']:.3f}")
                
                # Create visualizations based on selection
                if viz_type == "2D Scatter Plot":
                    fig = visualizer.create_2d_visualization(
                        embeddings, item_ids, texts, reduction_method.lower()
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "3D Scatter Plot":
                    fig = visualizer.create_3d_visualization(
                        embeddings, item_ids, texts, reduction_method.lower()
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Clustering":
                    fig = visualizer.create_cluster_visualization(
                        embeddings, item_ids, texts, n_clusters
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed metrics
                if viz_type != "Diversity Metrics Only":
                    st.subheader("Detailed Metrics")
                    metrics_df = pd.DataFrame([metrics])
                    st.dataframe(metrics_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Category comparison section
    st.markdown("---")
    st.subheader("Compare Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Select categories to compare:")
        selected_categories_compare = st.multiselect(
            "Choose categories:",
            category_options,
            default=category_options[:3] if len(category_options) >= 3 else category_options,
            help="Select multiple categories to compare their diversity"
        )
    
    with col2:
        if st.button("Compare Categories", type="secondary"):
            if len(selected_categories_compare) < 2:
                st.warning("Please select at least 2 categories to compare")
            else:
                with st.spinner("Comparing categories..."):
                    try:
                        category_paths = [parse_category_path(cat) for cat in selected_categories_compare]
                        fig = visualizer.compare_categories(category_paths)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error during comparison: {str(e)}")

if __name__ == "__main__":
    main() 