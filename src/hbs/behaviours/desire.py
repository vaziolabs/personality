import numpy as np
from collections import defaultdict
from hbs.consciousness.concept import ConceptWrapper

class DesireSystem:
    def __init__(self, size):
        """Initialize desire system"""
        self.size = size
        # Replace lambda defaultdicts with regular dicts
        self.layer_desires = {}
        self.desire_strengths = {}
        self.motivation_weights = {}
        
        # Simplified core motivations
        self.core_motivations = {
            'opportunity': {},
            'respect': {}, 
            'security': {}
        }
        
        self.emotional_memory_weight = 0.3
        self.desire_levels = np.zeros((size, size))
        self.past_outcomes = {}
        
        # How much we care about different kinds of rewards
        self.reward_weights = {
            'pleasure': 0.7,
            'safety': 0.8,
            'growth': 0.5
        }
        
        self.drive_desires = {
            'survival': {},
            'social': {},
            'mastery': {},
            'autonomy': {},
            'purpose': {}
        }
        
        self.personality_weights = {}
        
        # Add knowledge desires
        self.knowledge_desires = {
            'understanding': {},
            'mastery': {},
            'curiosity': {}
        }
        
        # Layer-specific emotional memory
        self.emotional_memory = {
            'positive': [],
            'negative': [],
            'neutral': [],
            'ambivalent': []
        }
        
        self.emotional_memory_capacity = 1000
        self.emotional_threshold = 0.3
        self.history_ptr = 0
        
        # Add drives list
        self.drives = ['survival', 'social', 'mastery', 'autonomy', 'purpose']

    def update_desire(self, stimulus, outcome, belief_system):
        # Change how much we want something based on what happened
        location = self._get_location(stimulus)
        current_desire = self.desire_levels[location]
        
        # Add belief influence
        belief_influence = belief_system.get_belief_influence(stimulus)
        
        if outcome > 0:  # Good thing happened
            # Want it a bit more (but not too much more)
            new_desire = (current_desire + (1 - current_desire) * 0.2) * (1 + belief_influence * 0.3)
        else:  # Bad thing happened
            # Want it less
            new_desire = current_desire * 0.8 * (1 - belief_influence * 0.2)
        
        self.desire_levels[location] = new_desire
        self.past_outcomes[str(location)].append(outcome)

    def get_desire_strength(self, concept, context):
        """Get integrated desire strength with context"""
        # Convert string context into proper dictionary format
        if isinstance(context, str):
            context = {'layer': context}
        
        base_desire = self.desire_levels[self._get_location(concept)]
        
        # Add drive influence
        drive_influence = self._calculate_drive_influence(concept, context)
        
        # Add emotional influence
        emotional_value = context.get('emotional_state', 0.0)
        emotional_influence = self._calculate_emotional_influence(
            concept, emotional_value
        )
        
        return base_desire * (1 + drive_influence + emotional_influence)

    def store_emotional_memory(self, stimulus, response, emotional_value, impact, context=None):
        """Store emotional memory with optional context"""
        try:
            # Categorize emotional value
            if isinstance(emotional_value, (float, np.floating, int)):
                if emotional_value > 0.2:
                    category = 'positive'
                elif emotional_value < -0.2:
                    category = 'negative'
                else:
                    category = 'neutral'
            else:
                category = 'neutral'
            
            memory = {
                'value': emotional_value,
                'timestamp': self.history_ptr,
                'concept': stimulus,
                'response': response,
                'impact': impact,
                'context': context
            }
            
            # Store in appropriate category
            self.emotional_memory[category].append(memory)
            
            # Maintain memory capacity
            if len(self.emotional_memory[category]) > self.emotional_memory_capacity:
                self.emotional_memory[category].pop(0)
                
            self.history_ptr += 1
            
        except Exception as e:
            print(f"Error in store_emotional_memory: {str(e)}")
    
    def _get_location(self, stimulus):
        """Get grid location for stimulus"""
        try:
            if isinstance(stimulus, str):
                # Hash string to get consistent numeric value
                hash_val = hash(stimulus)
                # Map to grid coordinates
                x = hash_val % self.size
                y = (hash_val // self.size) % self.size
                return (x, y)
            elif isinstance(stimulus, (tuple, list)) and len(stimulus) == 2:
                # If already coordinates, ensure within bounds
                x = int(stimulus[0]) % self.size
                y = int(stimulus[1]) % self.size
                return (x, y)
            else:
                # Default to center if invalid input
                return (self.size // 2, self.size // 2)
        except Exception as e:
            print(f"Error in _get_location: {str(e)}")
            return (0, 0)

    def _calculate_drive_influence(self, concept, context):
        """Calculate drive influence based on core motivations and drive desires"""
        try:
            wrapped_concept = concept if isinstance(concept, ConceptWrapper) else ConceptWrapper(concept)
            total_influence = 0.0
            
            # Get drive influences from core motivations
            for motivation, values in self.core_motivations.items():
                motivation_strength = values[wrapped_concept.id] if wrapped_concept.id in values else 0.0
                total_influence += motivation_strength * 0.3
                
            # Get influences from drive desires
            for drive, values in self.drive_desires.items():
                drive_strength = values[wrapped_concept.id] if wrapped_concept.id in values else 0.0
                
                # Weight different drives based on context
                if drive == 'survival' and context.get('threat_level', 0.0) > 0.5:
                    total_influence += drive_strength * 0.4
                elif drive == 'social' and context.get('social_presence', False):
                    total_influence += drive_strength * 0.3
                elif drive == 'mastery' and context.get('learning_opportunity', False):
                    total_influence += drive_strength * 0.3
                elif drive == 'autonomy':
                    total_influence += drive_strength * 0.2
                elif drive == 'purpose':
                    total_influence += drive_strength * 0.2
                    
            return np.clip(total_influence, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in _calculate_drive_influence: {str(e)}")
            return 0.0

    def _calculate_emotional_influence(self, concept, emotional_value):
        """Calculate emotional influence on desire strength"""
        try:
            # Get emotional memories for concept
            relevant_memories = []
            for category in self.emotional_memory:
                relevant_memories.extend([
                    m for m in self.emotional_memory[category] 
                    if m['concept'] == concept
                ])
            
            if not relevant_memories:
                return emotional_value * self.emotional_memory_weight
            
            # Calculate weighted influence from memories
            total_influence = 0.0
            decay_factor = 0.95  # Older memories have less influence
            
            for memory in sorted(relevant_memories, key=lambda x: x['timestamp']):
                age = self.history_ptr - memory['timestamp']
                memory_weight = decay_factor ** age
                total_influence += memory['value'] * memory['impact'] * memory_weight
            
            # Combine with current emotional value
            return np.clip(
                (total_influence + emotional_value) * self.emotional_memory_weight,
                -1.0, 1.0
            )
            
        except Exception as e:
            print(f"Error in _calculate_emotional_influence: {str(e)}")
            return 0.0

    def _calculate_survival_relevance(self, concept_id):
        """Calculate survival relevance of concept"""
        try:
            return self.drive_desires['survival'].get(concept_id, 0.0)
        except Exception as e:
            print(f"Error in _calculate_survival_relevance: {str(e)}")
            return 0.0

    def _calculate_social_relevance(self, concept_id):
        """Calculate social relevance of concept"""
        try:
            return self.drive_desires['social'].get(concept_id, 0.0)
        except Exception as e:
            print(f"Error in _calculate_social_relevance: {str(e)}")
            return 0.0

    def _calculate_mastery_relevance(self, concept_id):
        """Calculate mastery relevance of concept"""
        try:
            return self.drive_desires['mastery'].get(concept_id, 0.0)
        except Exception as e:
            print(f"Error in _calculate_mastery_relevance: {str(e)}")
            return 0.0

    def _calculate_autonomy_relevance(self, concept_id):
        """Calculate autonomy relevance of concept"""
        try:
            return self.drive_desires['autonomy'].get(concept_id, 0.0)
        except Exception as e:
            print(f"Error in _calculate_autonomy_relevance: {str(e)}")
            return 0.0

    def _calculate_purpose_relevance(self, concept_id):
        """Calculate purpose relevance of concept"""
        try:
            return self.drive_desires['purpose'].get(concept_id, 0.0)
        except Exception as e:
            print(f"Error in _calculate_purpose_relevance: {str(e)}")
            return 0.0
    