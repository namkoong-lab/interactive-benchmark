import json
import os
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from datetime import datetime

class ItemIndex:
    def __init__(self, categorized_dir: str = "categorized_products"):
        self.categorized_dir = categorized_dir
        self.items = {}  # item_id -> full item data
        self.metadata_indices = defaultdict(set)  # field -> value -> set of item_ids
        self.category_indices = defaultdict(set)  # category_path -> set of item_ids
        self.vectorized_data = {}  # field -> numpy array of values
        
    def load_items(self):
        """Load all categorized items and build indices."""
        print("Loading items and building indices...")
        for filename in tqdm(os.listdir(self.categorized_dir)):
            if not filename.startswith('item_') or not filename.endswith('.json'):
                continue
                
            with open(os.path.join(self.categorized_dir, filename), 'r', encoding='utf-8') as f:
                item = json.load(f)
                item_id = item['item_id']
                
                # Preprocess numeric fields
                metadata = item['metadata']
                if 'price' in metadata:
                    try:
                        metadata['price'] = float(str(metadata['price']).replace('$', '').strip())
                    except (ValueError, TypeError):
                        metadata['price'] = None
                
                if 'date_first_available' in metadata:
                    try:
                        date = datetime.strptime(metadata['date_first_available'], '%Y-%m-%d')
                        metadata['year_of_production'] = date.year
                    except (ValueError, TypeError):
                        metadata['year_of_production'] = None
                
                if 'ratings_total' in metadata:
                    try:
                        metadata['ratings_total'] = int(str(metadata['ratings_total']).replace(',', ''))
                    except (ValueError, TypeError):
                        metadata['ratings_total'] = None
                
                if 'bundle_size' in metadata:
                    try:
                        metadata['bundle_size'] = int(str(metadata['bundle_size']).split()[0])
                    except (ValueError, TypeError):
                        metadata['bundle_size'] = 1
                
                self.items[item_id] = item
                
                # Index metadata fields
                for field, value in metadata.items():
                    if isinstance(value, (str, int, float)) and value is not None:
                        self.metadata_indices[field].add((value, item_id))
                
                # Index category paths
                if 'categorization' in item and 'category_path' in item['categorization']:
                    path = tuple(item['categorization']['category_path'])
                    self.category_indices[path].add(item_id)
        
        # Convert sets to sorted lists for faster iteration
        for field in self.metadata_indices:
            self.metadata_indices[field] = sorted(self.metadata_indices[field])
            
        print(f"Loaded {len(self.items)} items")
        print(f"Indexed {len(self.metadata_indices)} metadata fields")
        print(f"Indexed {len(self.category_indices)} category paths")
    
    def get_available_dimensions(self) -> List[str]:
        """Get list of available dimensions for visualization."""
        return list(self.metadata_indices.keys())
    
    def get_3d_data(self, x_dim: str, y_dim: str, z_dim: str, categories: List[str] = None) -> Dict:
        """Get data for 3D visualization."""
        # Vectorize the fields
        x_data = self.vectorize_field(x_dim)
        y_data = self.vectorize_field(y_dim)
        z_data = self.vectorize_field(z_dim)
        
        # Get valid indices for all dimensions
        valid_indices = set(range(len(x_data['values'])))
        valid_indices &= set(range(len(y_data['values'])))
        valid_indices &= set(range(len(z_data['values'])))
        
        # Apply category filtering if specified
        if categories:
            category_indices = set()
            for path in self.category_indices.keys():
                if all(cat in path for cat in categories):
                    category_indices.update(self.category_indices[path])
            valid_indices &= category_indices
        
        # Convert to sorted array for consistent indexing
        indices = np.array(sorted(list(valid_indices)))
        
        if len(indices) == 0:
            raise ValueError("No valid data points found for the selected dimensions and categories")
        
        return {
            'x': x_data['values'][indices],
            'y': y_data['values'][indices],
            'z': z_data['values'][indices],
            'labels': {
                'x': x_dim,
                'y': y_dim,
                'z': z_dim
            }
        }
    
    def get_category_hierarchy(self) -> Dict:
        """Get the full category hierarchy for visualization."""
        hierarchy = {}
        for path in self.category_indices.keys():
            current = hierarchy
            for category in path:
                if category not in current:
                    current[category] = {'count': 0, 'subcategories': {}}
                current[category]['count'] += len(self.category_indices[path])
                current = current[category]['subcategories']
        return hierarchy
    
    def vectorize_field(self, field: str) -> np.ndarray:
        """Convert a field's values to a numpy array for fast operations."""
        if field not in self.vectorized_data:
            values = []
            item_ids = []
            for value, item_id in self.metadata_indices[field]:
                values.append(value)
                item_ids.append(item_id)
            self.vectorized_data[field] = {
                'values': np.array(values),
                'item_ids': np.array(item_ids)
            }
        return self.vectorized_data[field]
    
    def get_items_by_metadata(self, field: str, value: Any) -> List[Dict]:
        """Get all items with a specific metadata value."""
        if field not in self.metadata_indices:
            return []
        
        # Use binary search for fast lookup
        values = [v[0] for v in self.metadata_indices[field]]
        item_ids = [v[1] for v in self.metadata_indices[field]]
        
        # Find all matching indices
        matches = [i for i, v in enumerate(values) if v == value]
        return [self.items[item_ids[i]] for i in matches]
    
    def get_items_by_category(self, category_path: tuple) -> List[Dict]:
        """Get all items in a specific category path."""
        if category_path not in self.category_indices:
            return []
        return [self.items[item_id] for item_id in self.category_indices[category_path]]
    
    def get_field_distribution(self, field: str) -> Dict[Any, int]:
        """Get the distribution of values for a specific field."""
        if field not in self.metadata_indices:
            return {}
        
        distribution = defaultdict(int)
        for value, _ in self.metadata_indices[field]:
            distribution[value] += 1
        return dict(distribution)
    
    def get_category_distribution(self) -> Dict[tuple, int]:
        """Get the distribution of items across categories."""
        return {path: len(items) for path, items in self.category_indices.items()}
    
    def get_items_by_range(self, field: str, min_val: float, max_val: float) -> List[Dict]:
        """Get items where a numeric field falls within a range."""
        if field not in self.vectorized_data:
            self.vectorize_field(field)
        
        data = self.vectorized_data[field]
        mask = (data['values'] >= min_val) & (data['values'] <= max_val)
        return [self.items[item_id] for item_id in data['item_ids'][mask]]

def main():
    # Example usage
    index = ItemIndex()
    index.load_items()
    
    # Get available dimensions
    dimensions = index.get_available_dimensions()
    print("\nAvailable dimensions:", dimensions)
    
    # Example: Get 3D data for price, year, and ratings
    data = index.get_3d_data('price', 'year_of_production', 'ratings_total')
    print(f"\n3D data points: {len(data['x'])}")
    
    # Example: Get category hierarchy
    hierarchy = index.get_category_hierarchy()
    print("\nCategory hierarchy sample:", list(hierarchy.items())[:2])

if __name__ == "__main__":
    main() 