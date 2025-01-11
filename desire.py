import numpy as np
from collections import defaultdict

class DesireSystem:
    def __init__(self, size):
        # How much we want different things (0-1)
        self.desire_levels = np.zeros((size, size))
        # What happened when we got things before
        self.past_outcomes = defaultdict(list)
        # How much we care about different kinds of rewards
        self.reward_weights = {
            'pleasure': 0.7,
            'safety': 0.8,
            'growth': 0.5
        }
        
        # Add drive-based desires
        self.drive_desires = {
            'survival': defaultdict(float),
            'social': defaultdict(float),
            'mastery': defaultdict(float),
            'autonomy': defaultdict(float),
            'purpose': defaultdict(float)
        }
        
        # Add personality influence
        self.personality_weights = defaultdict(float)
        
        # Add emotional memory influence
        self.emotional_memory_weight = 0.3
        
        # Add reward memory systems
        self.reward_memory = {
            'conscious': {
                'achievement': [],    # Goal completion
                'mastery': [],       # Skill improvement
                'recognition': []    # External validation
            },
            'subconscious': {
                'emotional': [],     # Emotional satisfaction
                'social': [],        # Social feedback
                'habitual': []       # Routine rewards
            },
            'unconscious': {
                'safety': [],        # Security feelings
                'comfort': [],       # Physical comfort
                'instinctive': []    # Primal satisfaction
            }
        }
        
        # Add knowledge desires
        self.knowledge_desires = {
            'understanding': defaultdict(float),
            'mastery': defaultdict(float),
            'curiosity': defaultdict(float)
        }
    
    def update_desire(self, stimulus, outcome):
        # Change how much we want something based on what happened
        location = self._get_location(stimulus)
        current_desire = self.desire_levels[location]
        
        if outcome > 0:  # Good thing happened
            # Want it a bit more (but not too much more)
            new_desire = current_desire + (1 - current_desire) * 0.2
        else:  # Bad thing happened
            # Want it less
            new_desire = current_desire * 0.8
        
        self.desire_levels[location] = new_desire
        self.past_outcomes[str(location)].append(outcome)
    
    def get_desire_strength(self, stimulus, context):
        base_desire = self.desire_levels[self._get_location(stimulus)]
        
        # Integrate drives
        drive_influence = sum(
            context['drives'][drive] * self.drive_desires[drive][str(stimulus)]
            for drive in self.drive_desires
        ) / len(self.drive_desires)
        
        # Add personality influence
        personality_mod = sum(
            context['personality'][trait] * self.personality_weights[trait]
            for trait in context['personality']
        ) / len(context['personality'])
        
        # Include emotional memory
        emotional_influence = self._get_emotional_memory_influence(stimulus)
        
        return (base_desire * 0.4 + 
                drive_influence * 0.3 + 
                personality_mod * 0.2 + 
                emotional_influence * 0.1)
    
    def _get_location(self, stimulus):
        # Simplified location mapping
        return (0, 0)

    def process_reward(self, reward, context):
        conscious_impact = reward * self.personality_weights['conscientiousness']
        emotional_impact = reward * np.exp(abs(context['emotional_state']))
        primal_impact = reward * self.drive_desires['survival'][str(context)]
        
        return {
            'conscious': conscious_impact,
            'subconscious': emotional_impact,
            'unconscious': primal_impact
        }

    def process_layered_reward(self, experience, context, time_horizon='short'):
        """Process rewards at different consciousness levels with time-based weighting"""
        # Get base reward evaluations
        reward_impact = {
            'conscious': self._evaluate_conscious_reward(experience, context),
            'subconscious': self._evaluate_subconscious_reward(experience, context),
            'unconscious': self._evaluate_unconscious_reward(experience, context)
        }
        
        # Adjust weights based on time horizon
        if time_horizon == 'long':
            weights = {
                'conscious': 0.2,
                'subconscious': 0.4,
                'unconscious': 0.4  # Stronger unconscious/subconscious for long-term
            }
        else:  # short-term
            weights = {
                'conscious': 0.5,
                'subconscious': 0.3,
                'unconscious': 0.2  # Conscious dominates short-term
            }
        
        # Apply dissonance modulation
        dissonance = self._calculate_layer_dissonance(reward_impact)
        if dissonance > 0.3:  # High dissonance
            # Shift towards unconscious processing
            weights = self._adjust_weights_for_dissonance(weights, dissonance)
        
        # Weight and combine rewards
        weighted_impact = {
            layer: impact * weights[layer]
            for layer, impact in reward_impact.items()
        }
        
        return weighted_impact

    def process_knowledge_reward(self, concepts_learned, relationships_formed):
        """Process rewards from knowledge acquisition"""
        understanding_reward = len(concepts_learned) * 0.1
        mastery_reward = len(relationships_formed) * 0.15
        curiosity_satisfaction = understanding_reward + mastery_reward
        
        return {
            'conscious': understanding_reward,
            'subconscious': mastery_reward,
            'unconscious': curiosity_satisfaction
        }

class FuturePrediction:
    def __init__(self, size):
        # How far ahead we look (in steps)
        self.lookahead_steps = 3
        # How much we care about far vs near future
        self.time_discount = 0.8
        # Remember what led to what
        self.outcome_memory = defaultdict(list)
        
    def predict_outcome(self, action, current_state, time_steps=3):
        """Predict outcomes with layered consciousness influence"""
        predicted_states = []
        current = current_state
        
        for step in range(time_steps):
            # Adjust layer weights based on prediction distance
            layer_weights = self._get_time_based_weights(step)
            
            # Generate layered predictions
            next_state = self._generate_layered_prediction(
                current, 
                action,
                layer_weights
            )
            
            importance = self.time_discount ** step
            predicted_states.append((next_state, importance))
            current = next_state
        
        return predicted_states

    def _get_time_based_weights(self, step):
        """Get consciousness layer weights based on prediction distance"""
        if step < 1:  # Immediate future
            return {
                'conscious': 0.6,
                'subconscious': 0.3,
                'unconscious': 0.1
            }
        elif step < 2:  # Near future
            return {
                'conscious': 0.3,
                'subconscious': 0.4,
                'unconscious': 0.3
            }
        else:  # Distant future
            return {
                'conscious': 0.2,
                'subconscious': 0.4,
                'unconscious': 0.4
            }

    def _simulate_next_state(self, current_state, action):
        # Simple prediction based on past patterns
        key = str((current_state, action))
        if key in self.outcome_memory:
            return np.mean(self.outcome_memory[key])
        return current_state + np.random.randn() * 0.1
