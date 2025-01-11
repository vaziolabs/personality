import numpy as np
from collections import defaultdict
from numba import jit
from desire import DesireSystem, FuturePrediction

class ConsciousnessLayer:
    def __init__(self, size, name):
        self.state = np.zeros((size, size), dtype=np.float32)
        self.activation_history = np.zeros((1000, size, size), dtype=np.float32)
        self.history_index = 0
        self.patterns = {}
        self.emotional_weights = {}
        self.belief_contexts = defaultdict(self._create_belief_matrix)
        self.context_relationships = defaultdict(dict)  # Track relationships between contexts
        self.dissonance_by_context = defaultdict(list)  # Track dissonance per context
        self.size = size
        self.name = name
        self.layer_above = None
        self.layer_below = None
        # Add desire and prediction systems
        self.desire = DesireSystem(size)
        self.prediction = FuturePrediction(size)
        
        # Add psychological state tracking
        self.active_beliefs = {}  # Explicitly held beliefs
        self.behavioral_patterns = defaultdict(list)  # Habit/pattern tracking
        self.emotional_associations = {}  # Emotional memory links
        self.primal_drives = defaultdict(float)  # Basic motivations
        
        # Layer-specific processing weights
        self.processing_weights = self._init_layer_weights(name)
        
    def _create_belief_matrix(self):
        """Create a new belief matrix of the correct size"""
        return np.zeros((self.size, self.size), dtype=np.float32)
    
    def store_pattern(self, pattern, location, emotional_weight=0):
        """Store pattern with emotional weight in dictionaries"""
        pattern_key = hash(str(pattern.flatten())) % 1000  # Limit key size
        self.patterns[pattern_key] = location
        self.emotional_weights[pattern_key] = emotional_weight
    
    def find_similar_patterns(self, pattern, threshold=0.8):
        pattern_flat = pattern.ravel()
        matches = []
        
        for key in self.patterns:
            stored_pattern = np.asarray(key).ravel()
            if len(stored_pattern) == len(pattern_flat):
                similarity = self._calculate_similarity(pattern_flat, stored_pattern)
                if similarity > threshold:
                    matches.append((
                        self.patterns[key], 
                        similarity, 
                        self.emotional_weights.get(key, 0.0)
                    ))
        return matches
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_similarity(pattern1, pattern2):
        return np.sum(pattern1 * pattern2) / (
            np.sqrt(np.sum(pattern1**2)) * np.sqrt(np.sum(pattern2**2))
        )

    def update_history(self, activation):
        self.activation_history[self.history_index] = activation
        self.history_index = (self.history_index + 1) % 1000

    def update_belief(self, new_input, context, related_contexts, dissonance_level):
        """Update belief state for specific context and handle context relationships"""
        confidence = 1.0 - dissonance_level
        current_belief = self.belief_contexts[context]
        
        # Update primary context
        self.belief_contexts[context] = (
            current_belief * (0.8 * confidence) + 
            new_input * (0.2 * (2.0 - confidence))
        )
        
        # Process related contexts
        for related_context, relationship_strength in related_contexts.items():
            if related_context in self.belief_contexts:
                context_dissonance = np.mean(
                    np.abs(self.belief_contexts[context] - 
                          self.belief_contexts[related_context])
                )
                self.context_relationships[context][related_context] = {
                    'strength': relationship_strength,
                    'dissonance': context_dissonance
                }
                
        self.dissonance_by_context[context].append(dissonance_level)

    def evaluate_action(self, action, current_context):
        """Enhanced action evaluation based on psychological model"""
        # Rational evaluation (conscious-dominant)
        rational_eval = self.prediction.predict_outcome(action, self.belief_contexts[current_context])
        
        # Emotional evaluation (subconscious-dominant)
        emotional_eval = self.desire.get_desire_strength(
            action,
            context=current_context
        )
        
        # Instinctive evaluation (unconscious-dominant)
        instinctive_eval = self._evaluate_primal_response(action)
        
        # Weight evaluations based on layer type
        return (
            rational_eval * self.processing_weights['rational_weight'] +
            emotional_eval * self.processing_weights['emotional_weight'] +
            instinctive_eval * self.processing_weights['instinctive_weight']
        )

    def _init_layer_weights(self, name):
        """Initialize processing weights based on layer type"""
        if name == "Conscious":
            return {
                'rational_weight': 0.7,
                'emotional_weight': 0.2,
                'instinctive_weight': 0.1
            }
        elif name == "Subconscious":
            return {
                'rational_weight': 0.3,
                'emotional_weight': 0.5,
                'instinctive_weight': 0.2
            }
        else:  # Unconscious
            return {
                'rational_weight': 0.1,
                'emotional_weight': 0.3,
                'instinctive_weight': 0.6
            }

class ConsciousnessSystem:
    def __init__(self, size=5):
        self.conscious = ConsciousnessLayer(size, "Conscious")
        self.subconscious = ConsciousnessLayer(size, "Subconscious")
        self.unconscious = ConsciousnessLayer(size, "Unconscious")
        
        # Connect layers
        self.conscious.layer_below = self.subconscious
        self.subconscious.layer_above = self.conscious
        self.subconscious.layer_below = self.unconscious
        self.unconscious.layer_above = self.subconscious
        
        self.history = []
        self.thought_paths = np.zeros((1000, 6, size, size), dtype=np.float32)
        self.path_index = 0
        
        # Pattern learning parameters
        self.learning_rate = 0.1
        self.pattern_threshold = 0.7
        self.emotional_decay = 0.99
        self.dissonance_threshold = 0.3
        self.belief_integration_rate = 0.2
        
        self.context_map = defaultdict(set)  # Track related contexts
        self.context_dissonance_threshold = 0.3
        
        # Add modulation parameters
        self.conscious_override_threshold = 0.7
        self.integration_weights = {
            'conscious': 0.4,
            'subconscious': 0.35,
            'unconscious': 0.25
        }
        self.emotional_modulation = 0.3
        
    def process_impulse(self, impulse, context=None, related_contexts=None, emotional_value=0):
        # Process through layers with psychological hierarchy
        unconscious_response = self._process_in_layer(
            impulse, 
            self.unconscious,
            context,
            related_contexts,
            emotional_value
        )
        
        subconscious_response = self._process_in_layer(
            unconscious_response,  # Bottom-up processing
            self.subconscious,
            context,
            related_contexts,
            emotional_value
        )
        
        conscious_response = self._process_in_layer(
            subconscious_response,  # Bottom-up processing
            self.conscious,
            context,
            related_contexts,
            emotional_value
        )
        
        # Top-down modulation
        modulated_response = self._apply_conscious_modulation(
            conscious_response,
            subconscious_response,
            unconscious_response
        )
        
        return modulated_response

    def _apply_conscious_modulation(self, conscious, subconscious, unconscious):
        """Apply top-down conscious control over lower responses"""
        conscious_strength = np.mean(conscious)
        emotional_intensity = np.mean(abs(unconscious))
        
        # Strong conscious override when above threshold
        if conscious_strength > self.conscious_override_threshold:
            return conscious
        
        # Emotional state influences integration weights
        if emotional_intensity > 0.8:  # High emotional state
            weights = {
                'conscious': 0.3,
                'subconscious': 0.3,
                'unconscious': 0.4  # More unconscious influence
            }
        else:  # Normal state
            weights = self.integration_weights
            
        # Integrate all layers with dynamic weighting
        integrated = (
            conscious * weights['conscious'] +
            subconscious * weights['subconscious'] +
            unconscious * weights['unconscious']
        )
        
        return integrated
    
    def _learn_from_history(self):
        if len(self.history) > 0:
            recent_history = self.history[-10:]  # Look at last 10 experiences
            for memory in recent_history:
                # Strengthen patterns that led to high emotional responses
                if abs(memory['emotional_value']) > self.pattern_threshold:
                    self._strengthen_patterns(memory)
    
    def _strengthen_patterns(self, memory):
        # Strengthen patterns in each layer based on emotional impact
        for layer, response in [
            (self.conscious, memory['conscious']),
            (self.subconscious, memory['subconscious']),
            (self.unconscious, memory['unconscious'])
        ]:
            similar_patterns = layer.find_similar_patterns(response, 
                                                         threshold=self.pattern_threshold)
            for locations, similarity, stored_emotion in similar_patterns:
                # Update emotional weight based on experience
                new_weight = stored_emotion * self.emotional_decay + \
                           memory['emotional_value'] * self.learning_rate
                layer.emotional_weights[hash(str(response.flatten()))] = new_weight
    
    def _store_thought_path(self, impulse, emotional_value, *responses):
        # Store complete thought path with metadata
        path_data = {
            'impulse': impulse,
            'emotional_value': emotional_value,
            'conscious': responses[0],
            'subconscious': responses[1],
            'unconscious': responses[2],
            'final_conscious': responses[3],
            'thought_path': self.thought_paths[self.path_index - 6:self.path_index],
            'timestamp': len(self.history)
        }
        self.history.append(path_data)
        
        # Maintain history size
        if len(self.history) > 1000:
            self.history.pop(0)
    
    def _process_in_layer(self, input_signal, layer, context=None, 
                         related_contexts=None, emotional_value=0, 
                         dissonance_level=0):
        """Process information within a single layer with context awareness"""
        if related_contexts is None:
            related_contexts = {}
            
        # Find similar patterns
        similar_patterns = layer.find_similar_patterns(input_signal)
        
        if similar_patterns:
            responses = np.zeros_like(input_signal)
            weights = np.zeros(len(similar_patterns))
            
            for i, (locations, similarity, stored_emotion) in enumerate(similar_patterns):
                pattern_response = self._recall_pattern(locations, layer)
                emotional_weight = np.exp(stored_emotion * emotional_value)
                confidence = 1.0 - dissonance_level
                
                responses += pattern_response * similarity * emotional_weight * confidence
                weights[i] = similarity * emotional_weight * confidence
            
            if np.sum(weights) > 0:
                return responses / np.sum(weights)
                
        # If no patterns or weights, create new response
        return self._create_new_response(input_signal, layer)
    
    def _feedback_up(self, signal, layer, emotional_value):
        # Complex feedback incorporating emotional weight
        layer_influence = np.random.rand(*signal.shape) * 0.2 * np.exp(emotional_value)
        
        # Fix the array comparison issue
        if layer.activation_history.size > 0:  # Check if history exists
            last_three = layer.activation_history[max(0, layer.history_index-3):layer.history_index]
            resonance = np.mean(last_three) if last_three.size > 0 else 0
        else:
            resonance = 0
            
        processed_signal = signal + layer_influence + (resonance * 0.1)
        return np.clip(processed_signal, 0, 1)

    def visualize_thought_flow(self):
        """Return the complete thought propagation path"""
        return {
            'thought_paths': self.thought_paths,
            'layer_activations': {
                'conscious': self.conscious.activation_history,
                'subconscious': self.subconscious.activation_history,
                'unconscious': self.unconscious.activation_history
            }
        }

    def _create_new_response(self, input_signal, layer):
        """Create a new response when no similar patterns are found"""
        # Add slight randomness to input while maintaining general pattern
        noise = np.random.randn(*input_signal.shape) * 0.1
        response = np.clip(input_signal + noise, 0, 1)
        return response

    @staticmethod
    @jit(nopython=True)
    def _process_signal(input_signal, emotional_value):
        # Move numerical computations here
        return np.clip(input_signal * np.exp(emotional_value), 0, 1)
    
    @staticmethod
    @jit(nopython=True)
    def _combine_responses(responses, weights):
        return responses / (np.sum(weights) + 1e-10)

    def process_rest_state(self):
        """Process unconscious activity during rest periods"""
        # Process pattern consolidation in deeper layers
        unconscious_activity = self._process_unconscious_rest()
        subconscious_activity = self._process_subconscious_rest(unconscious_activity)
        conscious_activity = self._process_conscious_rest(subconscious_activity)
        
        return {
            'unconscious': unconscious_activity,
            'subconscious': subconscious_activity,
            'conscious': conscious_activity
        }
    
    def _process_unconscious_rest(self):
        """Process deep pattern recognition during rest"""
        recent_patterns = self.unconscious.activation_history[
            max(0, self.unconscious.history_index-10):self.unconscious.history_index
        ]
        
        if recent_patterns.size > 0:
            pattern_mean = np.mean(recent_patterns, axis=0)
            pattern_std = np.std(recent_patterns, axis=0)
            rest_activity = pattern_mean * (1 + 0.2 * pattern_std)
            return np.clip(rest_activity, 0, 1)
        return np.zeros_like(self.unconscious.state)

    def _process_subconscious_rest(self, unconscious_activity):
        """Process subconscious integration during rest"""
        recent_patterns = self.subconscious.activation_history[
            max(0, self.subconscious.history_index-10):self.subconscious.history_index
        ]
        
        if recent_patterns.size > 0:
            pattern_mean = np.mean(recent_patterns, axis=0)
            # Integrate unconscious influence
            integrated_activity = (pattern_mean + unconscious_activity) * 0.5
            return np.clip(integrated_activity, 0, 1)
        return unconscious_activity

    def _process_conscious_rest(self, subconscious_activity):
        """Process conscious integration during rest"""
        recent_patterns = self.conscious.activation_history[
            max(0, self.conscious.history_index-5):self.conscious.history_index
        ]
        
        if recent_patterns.size > 0:
            pattern_mean = np.mean(recent_patterns, axis=0)
            # Light integration of subconscious activity
            integrated_activity = pattern_mean * 0.8 + subconscious_activity * 0.2
            return np.clip(integrated_activity, 0, 1)
        return subconscious_activity * 0.2  # Minimal conscious activity during rest

    def _measure_system_dissonance(self, context, related_contexts):
        """Measure dissonance between layers and contexts"""
        # Layer dissonance within primary context
        conscious_sub = np.mean(
            np.abs(self.conscious.belief_contexts[context] - 
                  self.subconscious.belief_contexts[context])
        )
        
        sub_unconscious = np.mean(
            np.abs(self.subconscious.belief_contexts[context] - 
                  self.unconscious.belief_contexts[context])
        )
        
        # Context relationship dissonance
        context_dissonance = 0.0
        if related_contexts:
            context_differences = []
            for related_context, strength in related_contexts.items():
                for layer in [self.conscious, self.subconscious, self.unconscious]:
                    if related_context in layer.belief_contexts:
                        diff = np.mean(
                            np.abs(layer.belief_contexts[context] - 
                                  layer.belief_contexts[related_context])
                        )
                        context_differences.append(diff * strength)
            
            if context_differences:
                context_dissonance = np.mean(context_differences)
        
        return {
            'total': (conscious_sub + sub_unconscious + context_dissonance) / 3,
            'conscious_subconscious': conscious_sub,
            'subconscious_unconscious': sub_unconscious,
            'context': context_dissonance
        }
    
    def _resolve_belief_conflicts(self, context, related_contexts):
        """Resolve conflicts between contexts and layers"""
        # Get average belief state across layers for primary context
        primary_average = (
            self.conscious.belief_contexts[context] +
            self.subconscious.belief_contexts[context] +
            self.unconscious.belief_contexts[context]
        ) / 3.0
        
        # Process related contexts
        for related_context, strength in related_contexts.items():
            related_average = sum(
                layer.belief_contexts.get(related_context, np.zeros_like(primary_average))
                for layer in [self.conscious, self.subconscious, self.unconscious]
            ) / 3.0
            
            # Weighted integration based on relationship strength
            integrated_belief = (
                primary_average * (1 - strength) +
                related_average * strength
            )
            
            # Update beliefs across layers with context awareness
            for layer in [self.conscious, self.subconscious, self.unconscious]:
                layer.belief_contexts[context] += (
                    (integrated_belief - layer.belief_contexts[context]) *
                    self.belief_integration_rate
                )

    def _calculate_outcome(self, response, emotional_value):
        """Calculate actual outcome of action"""
        base_outcome = np.mean(response)
        emotional_factor = np.exp(abs(emotional_value))
        return base_outcome * emotional_factor

    def _analyze_behavior_patterns(self):
        """Analyze patterns across consciousness levels"""
        patterns = {
            'conscious': self._identify_conscious_patterns(),
            'subconscious': self._identify_subconscious_patterns(),
            'unconscious': self._identify_unconscious_patterns()
        }
        
        # Identify cross-layer patterns
        cross_patterns = self._identify_cross_layer_patterns(patterns)
        
        # Update pattern influence on behavior
        self._update_pattern_influence(cross_patterns)

    def process_text_input(self, text_data, context):
        """Process text input across consciousness layers"""
        # Conscious text processing
        conscious_response = self.conscious.process_text(
            text_data,
            self.desire_layers['conscious']
        )
        
        # Subconscious pattern recognition
        subconscious_response = self.subconscious.process_patterns(
            text_data,
            conscious_response
        )
        
        # Unconscious integration
        unconscious_response = self.unconscious.integrate_knowledge(
            conscious_response,
            subconscious_response
        )
        
        return self._integrate_text_responses(
            conscious_response,
            subconscious_response,
            unconscious_response
        )