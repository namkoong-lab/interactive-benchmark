import plotly.express as px
import plotly.graph_objects as go
from item_index import ItemIndex
import json
from typing import List, Tuple, Dict

def print_category_tree(categories: dict, level: int = 0) -> None:
    """Print the category hierarchy in a tree-like structure."""
    for category, data in categories.items():
        print('  ' * level + f"- {category} ({data['count']} items)")
        if data['subcategories']:
            print_category_tree(data['subcategories'], level + 1)

def get_user_category_selection(categories: Dict) -> List[str]:
    """Get user's category selection."""
    print("\nAvailable Categories:")
    print_category_tree(categories)
    
    while True:
        selection = input("\nEnter category path (e.g., 'Headphones') or 'all' for all categories: ").strip()
        if selection.lower() == 'all':
            return []
        
        # Split the path and traverse the hierarchy
        path = [cat.strip() for cat in selection.split('>')]
        current = categories
        valid_path = True
        
        for category in path:
            if category not in current:
                print(f"Invalid category: {category}")
                valid_path = False
                break
            current = current[category]
        
        if valid_path:
            return path

def get_user_dimensions(dimensions: List[str]) -> Tuple[str, str, str]:
    """Get user's dimension selection."""
    print("\nAvailable Dimensions:")
    for i, dim in enumerate(dimensions, 1):
        print(f"{i}. {dim}")
    
    while True:
        try:
            print("\nEnter three dimension numbers (e.g., '1 2 3'):")
            selections = input().strip().split()
            if len(selections) != 3:
                raise ValueError("Please enter exactly three numbers")
            
            dims = [dimensions[int(s) - 1] for s in selections]
            return tuple(dims)
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            print("Please try again.")

def visualize_3d(data: dict, dimensions: Tuple[str, str, str]) -> None:
    """Create an interactive 3D visualization."""
    fig = go.Figure(data=[go.Scatter3d(
        x=data['x'],
        y=data['y'],
        z=data['z'],
        mode='markers',
        marker=dict(
            size=5,
            color=data['z'],  # Color by z-dimension
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=f"3D Product Visualization<br>{dimensions[0]} vs {dimensions[1]} vs {dimensions[2]}",
        scene=dict(
            xaxis_title=dimensions[0],
            yaxis_title=dimensions[1],
            zaxis_title=dimensions[2]
        ),
        showlegend=False
    )
    
    fig.show()

def main():
    # Initialize and load data
    print("Loading product data...")
    index = ItemIndex()
    index.load_items()
    
    # Get available dimensions
    dimensions = index.get_available_dimensions()
    
    # Get category hierarchy
    categories = index.get_category_hierarchy()
    
    # Get user selections
    selected_categories = get_user_category_selection(categories)
    selected_dimensions = get_user_dimensions(dimensions)
    
    # Get data for visualization
    print("\nGenerating visualization...")
    data = index.get_3d_data(
        x_dim=selected_dimensions[0],
        y_dim=selected_dimensions[1],
        z_dim=selected_dimensions[2],
        categories=selected_categories
    )
    
    # Create visualization
    visualize_3d(data, selected_dimensions)

if __name__ == "__main__":
    main() 