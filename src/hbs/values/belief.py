from collections import defaultdict
import numpy as np
from hbs.consciousness.concept import ConceptWrapper

class BeliefSystem:
    def __init__(self, size, layer_type):
        """Initialize belief system for specific consciousness layer
        
        Args:
            size: Size of the system
            layer_type: One of 'conscious', 'subconscious', 'unconscious'
        """
        # Normalize layer type to lowercase
        self.layer_type = layer_type.lower()
        
        # Layer-specific belief influence weights
        self.influence_weights = {
            'conscious': {
                'rational': 0.4,
                'emotional': 0.2,
                'instinctive': 0.1
            },
            'subconscious': {
                'rational': 0.2,
                'emotional': 0.4,
                'instinctive': 0.2
            },
            'unconscious': {
                'rational': 0.1,
                'emotional': 0.2,
                'instinctive': 0.4
            }
        }[self.layer_type]
        
        # Initialize with regular dicts instead of defaultdict
        self.belief_influence = {
            'opportunity': {},
            'respect': {},
            'security': {}
        }

        # Generic prediction categories with layer-specific weights
        self.predictions = {
            'logical': {},      # Rational/analytical predictions
            'emotional': {},    # Feeling-based predictions
            'pattern': {},      # Pattern-based predictions
            'instinctive': {}   # Gut/survival predictions
        }
        
        # Initialize belief categories
        self.belief_categories = {
            'foundational': {},
            'experiential': {},
            'social': {},
            'personal': {}
        }

    def process_belief_update(self, concept, context, emotional_value):
        """Process belief update for specific layer"""
        # Wrap concept for standardized handling
        wrapped_concept = ConceptWrapper(concept)
        
        belief_updates = {}
        activation = context.get('activation', 0.0)
        belief_conflict = context.get('belief_conflict', 0.0)
        layer_emotional_state = context.get('emotional_state', 0.0)
        
        # Update categories using layer's emotional state
        for category, beliefs in self.belief_categories.items():
            belief_strength = self._calculate_belief_strength(
                wrapped_concept.id,
                category,
                activation,
                belief_conflict,
                layer_emotional_state
            )
            belief_updates[category] = belief_strength
            beliefs[wrapped_concept.id] = belief_strength
        
        return belief_updates

    def integrate_with_desires(self, desire_system):
        motivation_updates = {}
        
        # Match core_motivations structure
        for motivation, subcategories in desire_system.core_motivations.items():
            subcategory_updates = {}
            for subcategory in subcategories:
                weighted_belief_influence = 0.0
                
                # Get relevant belief categories
                belief_categories = self._get_relevant_beliefs(motivation, subcategory)
                
                # Calculate weighted influence
                for layer, categories in self.belief_categories.items():
                    layer_weight = desire_system.motivation_weights[layer][motivation]
                    for category in belief_categories:
                        if category in categories:
                            belief_strength = np.mean(list(categories[category].values()))
                            weighted_belief_influence += belief_strength * layer_weight
                
                subcategory_updates[subcategory] = weighted_belief_influence
            
            motivation_updates[motivation] = subcategory_updates
            
        return motivation_updates

    def _calculate_belief_strength(self, concept, category, activation, belief_conflict):
        """Calculate belief strength based on activation and conflicts"""
        base_strength = activation * (1.0 - belief_conflict)
        existing_strength = self.belief_categories[category].get(concept, 0.0)
        
        # Integrate new with existing belief strength
        return existing_strength * 0.7 + base_strength * 0.3

    def predict_outcome(self, concept, context):
        """Predict outcome based on beliefs and desires"""
        layer = context.get('layer', 'conscious')
        
        # Get belief-based prediction
        belief_prediction = self._calculate_belief_prediction(concept, layer)
        
        # Get desire influence
        desire_influence = context.get('desire_strength', 0.0)
        
        # Weight predictions based on layer
        if layer == 'conscious':
            return belief_prediction * 0.7 + desire_influence * 0.3
        elif layer == 'subconscious':
            return belief_prediction * 0.5 + desire_influence * 0.5
        else:  # unconscious
            return belief_prediction * 0.3 + desire_influence * 0.7

    def _calculate_belief_prediction(self, concept, layer):
        """Calculate prediction based on beliefs in specific layer"""
        predictions = self.predictions[layer]
        belief_categories = self.belief_categories[layer]
        
        prediction_value = 0.0
        for pred_type, pred_values in predictions.items():
            if concept in pred_values:
                prediction_value += pred_values[concept]
                
        # Weight with belief strength
        belief_strength = np.mean([
            beliefs[concept] 
            for beliefs in belief_categories.values()
            if concept in beliefs
        ])
        
        return prediction_value * belief_strength
