import numpy as np

class ConceptWrapper:
    """Wrapper class to standardize concept handling"""
    def __init__(self, concept):
        if isinstance(concept, dict):
            try:
                # First try to use direct id or name
                if 'id' in concept:
                    self.id = str(concept['id'])
                elif 'name' in concept:
                    self.id = str(concept['name'])
                else:
                    # Create a stable string representation instead of hashing
                    items_str = str(sorted([(str(k), str(v)) for k, v in concept.items()]))
                    self.id = str(hash(items_str))
                
                self.content = str(concept.get('content', ''))
                self.metadata = concept
            except TypeError:
                # Handle unhashable types more gracefully
                print(f"DEBUG: Handling unhashable type in concept dict")
                self.id = str(id(concept))  # Use object's memory address as fallback
                self.content = str(concept.get('content', ''))
                self.metadata = concept
        else:
            self.id = str(concept)
            self.content = str(concept)
            self.metadata = {'content': str(concept)}
        
        # Always ensure we have a hashable_id
        self.hashable_id = self.id

    def get(self, key, default=None):
        """Safely get attributes from either string or dict concepts"""
        if key in ('id', 'content'):
            return getattr(self, key)
        return self.metadata.get(key, default)

    def __str__(self):
        return str(self.id)

    def __hash__(self):
        return hash(self.hashable_id)

    def __eq__(self, other):
        if isinstance(other, ConceptWrapper):
            return self.hashable_id == other.hashable_id
        return str(self.id) == str(other)