import numpy as np
from collections import defaultdict
from numba import jit
from hbs.behaviours.desire import DesireSystem
from hbs.consciousness.cl import ConsciousnessLayer
from hbs.values.belief import BeliefSystem
from hbs.behaviours.imagine import Imagination

class ConsciousnessSystem:
    def __init__(self, size):
        """Initialize consciousness system"""
        self.size = size
        
        # Initialize layers first
        self.conscious = ConsciousnessLayer(size, "Conscious")
        self.subconscious = ConsciousnessLayer(size, "Subconscious") 
        self.unconscious = ConsciousnessLayer(size, "Unconscious")
        self.conscious.layer_below = self.subconscious
        self.subconscious.layer_above = self.conscious
        self.subconscious.layer_below = self.unconscious
        self.unconscious.layer_above = self.subconscious
        
        # Required components for imagination
        self.layers = {
            'conscious': self.conscious,
            'subconscious': self.subconscious,
            'unconscious': self.unconscious
        }
        
        # Initialize semantic memory
        self.semantic_memory = {
            'concepts': {},
            'relationships': {},
            'hierarchies': {}
        }
        
        # Initialize personality traits BEFORE belief system initialization
        self.personality_traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5,
            'resilience': 0.5,
            'adaptability': 0.5,
            'curiosity': 0.5
        }
        
        # Initialize belief systems ONCE
        try:
            self._initialize_belief_systems()
        except Exception as e:
            print(f"Error initializing belief systems: {e}")
            raise
        
        # Create main belief system (using same system as conscious layer)
        # This provides the interface expected by HBS
        self.belief_system = self.conscious.belief
        
        # Create main desire system (using same system as conscious layer)
        # This provides the interface expected by HBS
        self.desire_system = self.conscious.desire
        
        # Add error handling for imagination initialization
        try:
            self.imagination = Imagination(self)
            # Initialize imagination state
            self.imagination.imagination_state = {
                'pattern_flexibility': 0.5,
                'reflection_depth': 0.5,
                'associative_strength': 0.5,
                'creative_divergence': 0.5,
                'analytical_convergence': 0.5
            }
        except Exception as e:
            print(f"Error initializing imagination: {e}")
            raise
        
        # Initialize remaining systems
        self.desire = DesireSystem(size)
        
        # Layer-specific weights
        self.layer_weights = {
            'conscious': 0.4,
            'subconscious': 0.3,
            'unconscious': 0.3
        }
        
        # Initialize state
        self.state = np.zeros((size, size))
        
        # Initialize desire layers using regular dict
        self.desire_layers = {
            'conscious': {},
            'subconscious': {},
            'unconscious': {}
        }
        
        # Connect layers
        self.conscious.layer_below = self.subconscious
        self.subconscious.layer_above = self.conscious
        self.subconscious.layer_below = self.unconscious
        self.unconscious.layer_above = self.subconscious

        # Integration parameters
        self.belief_integration_rate = 0.1
        self.pattern_recognition_threshold = 0.7
        self.emotional_integration_weight = 0.3
        
        self.history = []
        self.thought_paths = np.zeros((1000, 6, size, size), dtype=np.float32)
        self.path_index = 0
        
        # Pattern learning parameters
        self.learning_rate = 0.1
        self.pattern_threshold = 0.7
        self.emotional_decay = 0.99
        self.dissonance_threshold = 0.3
        
        # Use regular dict for context map
        self.context_map = {}
        
        # Add modulation parameters
        self.conscious_override_threshold = 0.7
        self.integration_weights = {
            'conscious': 0.4,
            'subconscious': 0.35,
            'unconscious': 0.25
        }
        self.emotional_modulation = 0.3
        
        # Initialize pattern tracking at system level
        self.pattern_threshold = 0.7
        self.learning_rate = 0.1
        
        # Add missing attributes
        self.belief_integration_rate = 0.1
        
        # Add DMN-related parameters
        self.dmn_state = {
            'active': False,
            'intensity': 0.0,
            'duration': 0.0
        }
        
        # Add imagination integration parameters
        self.imagination_weights = {
            'conscious': 0.3,
            'subconscious': 0.4,
            'unconscious': 0.3
        }

    def _initialize_belief_systems(self):
        """Initialize core belief structures with philosophical framework"""
        belief_categories = {
            'foundational': {
                'epistemological': {},  # Knowledge/truth beliefs
                'ontological': {},      # Nature of reality beliefs
                'axiological': {},      # Value/purpose beliefs
                'logical': {}           # Reasoning/rationality beliefs
            },
            'self_concept': {
                'ego': {},             # Core identity
                'super-ego': {},       # Higher-order identity
                'personality': {                       # Personality traits
                    trait: value for trait, value in self.personality_traits.items()
                },
                'abilities': {}        # Self-perceived capabilities
            },
            'value_systems': {
                'ethical': {},         # Moral framework
                'aesthetic': {},       # Beauty/art appreciation
                'social': {}           # Interpersonal values
            },
            'contextual': {
                'cultural': {},        # Cultural context
                'temporal': {},        # Time-based context
                'environmental': {}    # Situational context
            }
        }

        # Initialize for each consciousness layer
        for layer in [self.conscious, self.subconscious, self.unconscious]:
            layer.belief_systems = belief_categories.copy()
            
            # Set layer-specific confidence weights
            confidence_weights = {
                'Conscious': {
                    'foundational': 0.7,
                    'self_concept': 0.5,
                    'value_systems': 0.6,
                    'contextual': 0.8
                },
                'Subconscious': {
                    'foundational': 0.5,
                    'self_concept': 0.7,
                    'value_systems': 0.8,
                    'contextual': 0.6
                },
                'Unconscious': {
                    'foundational': 0.3,
                    'self_concept': 0.9,
                    'value_systems': 0.7,
                    'contextual': 0.4
                }
            }[layer.name]
            
            layer.belief_confidence = confidence_weights

    def _initialize_base_beliefs(self, layer):
        """Initialize foundational beliefs for each layer"""
        layer.belief_contexts = defaultdict(lambda: np.zeros(self.size))
        layer.active_beliefs = defaultdict(float)
        
        # Set initial belief strengths
        if layer.name == "Conscious":
            confidence_base = 0.6
        elif layer.name == "Subconscious":
            confidence_base = 0.4
        else:  # Unconscious
            confidence_base = 0.8
            
        # Initialize core beliefs with layer-appropriate confidence
        for category in layer.belief_systems:
            layer.belief_systems[category]['confidence']['base'] = confidence_base

    def process_impulse(self, impulse, context=None, related_contexts=None):
        """Process impulse through consciousness layers with integrated learning and rewards"""
        # Process from unconscious up through layers with emotional states
        layer_responses = {
            'unconscious': self._process_in_layer(
                impulse, 
                self.unconscious,
                context,
                self.unconscious.emotional_state
            ),
            'subconscious': None,
            'conscious': None
        }
        
        # Bottom-up processing with emotional states
        layer_responses['subconscious'] = self._process_in_layer(
            layer_responses['unconscious']['activation'],
            self.subconscious,
            context,
            self.subconscious.emotional_state,
            previous_state=layer_responses['unconscious']
        )
        
        layer_responses['conscious'] = self._process_in_layer(
            layer_responses['subconscious']['activation'],
            self.conscious,
            context,
            self.conscious.emotional_state,
            previous_state=layer_responses['subconscious']
        )

        # Update beliefs and integrate with desires
        belief_updates = self.belief_system.process_belief_update(
            str(impulse),
            context,
            layer_responses
        )
        
        # Update desires based on belief influence
        motivation_updates = self.belief_system.integrate_with_desires(self.desire)
        
        # Update core motivations
        self._update_core_motivations(motivation_updates, impulse)
        
        return self._integrate_responses(layer_responses, context)

    def _process_in_layer(self, signal, layer, context, emotional_value, previous_state=None):
        """Unified layer processing with belief and desire integration"""
        try:
            # Handle ambivalent emotional state
            if isinstance(emotional_value, str) and emotional_value == 'ambivalent':
                # Create oscillating emotional value for ambivalent state
                time_factor = len(layer.emotional_history) if hasattr(layer, 'emotional_history') else 0
                emotional_value = np.sin(time_factor * 0.5) * 0.5  # Oscillate between -0.5 and 0.5
            elif isinstance(emotional_value, (np.ndarray, np.generic)):
                emotional_value = float(np.mean(emotional_value))
            
            # Calculate reward and belief expectations
            reward_expectation = layer.desire.get_desire_strength(signal, context)
            belief_consistency = self._check_belief_consistency(signal, layer.active_beliefs)
            
            # Get pattern matches considering rewards and beliefs
            pattern_matches = layer.find_similar_patterns(
                signal, 
                threshold=self.pattern_threshold
            )
            
            # Process matches with reward and belief weighting
            if pattern_matches:
                response = self._process_pattern_matches(
                    pattern_matches,
                    layer,
                    context,
                    emotional_value,
                    reward_expectation,
                    belief_consistency
                )
            else:
                response = self._create_new_response(signal, layer)

            # Update layer state with integrated learning
            self._update_layer_state(
                layer,
                response,
                reward_expectation,
                belief_consistency,
                emotional_value
            )
            
            return {
                'activation': response,
                'reward': reward_expectation,
                'belief_conflict': 1.0 - belief_consistency,
                'pattern_key': hash(str(response.flatten())) % 1000
            }
            
        except Exception as e:
            print(f"Error in _process_in_layer: {str(e)}")
            return {'activation': np.zeros_like(signal), 'reward': 0.0, 'belief_conflict': 1.0, 'pattern_key': None}

    def _update_layer_state(self, layer, response, reward, belief_consistency, emotional_value):
        """Update layer state with integrated learning"""
        pattern_key = hash(str(response.flatten())) % 1000
        
        # Calculate strengthen amount with belief influence
        strengthen_amount = (
            reward * 0.4 +
            belief_consistency * 0.3 +
            abs(emotional_value) * 0.3
        ) * self.learning_rate
        
        # Update pattern strength
        layer.pattern_strengths[pattern_key] = min(
            1.0,
            layer.pattern_strengths.get(pattern_key, 0) + strengthen_amount
        )
        
        # Update connected patterns and beliefs
        self._strengthen_connected_patterns(layer, pattern_key, strengthen_amount)
        self._update_beliefs(layer, response, belief_consistency)

    def _integrate_responses(self, layer_responses, context):
        """Integrate responses from all layers with proper weighting"""
        # Calculate integration weights based on context and states
        weights = self._calculate_integration_weights(
            layer_responses,
            context
        )
        
        # Integrate activations
        integrated_activation = sum(
            state['activation'] * weights[layer_name]
            for layer_name, state in layer_responses.items()
        )
        
        # Update pattern connections across layers
        self._update_cross_layer_patterns(layer_responses)
        
        return {
            'activation': integrated_activation,
            'weights': weights,
            'states': layer_responses
        }

    def _calculate_integration_weights(self, layer_responses, context):
        """Calculate dynamic integration weights based on layer states"""
        # Get base weights for context
        weights = self._get_base_weights(context)
        
        # Adjust weights based on emotional intensity
        emotional_intensity = max(
            abs(state['emotional_value']) 
            for state in layer_responses.values()
        )
        
        if emotional_intensity > 0.8:
            weights = {
                'conscious': weights['conscious'] * 0.7,
                'subconscious': weights['subconscious'] * 1.1,
                'unconscious': weights['unconscious'] * 1.2
            }
            
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _update_cross_layer_patterns(self, layer_responses):
        """Update pattern connections between layers"""
        for layer_name, state in layer_responses.items():
            if state['pattern_key']:
                # Connect patterns across layers
                for other_name, other_state in layer_responses.items():
                    if other_name != layer_name and other_state['pattern_key']:
                        self._connect_patterns(
                            layer_name, state['pattern_key'],
                            other_name, other_state['pattern_key']
                        )

    def _get_layer_weights(self, layer_name, emotional_value):
        """Get layer-specific processing weights"""
        base_weights = {
            'conscious': 0.4,
            'subconscious': 0.35,
            'unconscious': 0.25
        }
        
        # Adjust weights based on emotional value
        if abs(emotional_value) > 0.7:
            base_weights[layer_name] *= 1.2
            
        # Normalize weights
        total = sum(base_weights.values())
        return {k: v/total for k, v in base_weights.items()}

    def _get_base_weights(self, context):
        """Get base weights for context"""
        weights = {
            'subconscious': 0.35,
            'unconscious': 0.25
        }
        
        # Adjust weights based on context type
        if context in self.context_map:
            related_contexts = self.context_map[context]
            context_weight = len(related_contexts) * 0.1
            weights = {k: v * (1 + context_weight) for k, v in weights.items()}
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _connect_patterns(self, layer_name, pattern_key, other_name, other_pattern_key):
        """Connect patterns between layers"""
        # Get layers
        layer = getattr(self, layer_name)
        other_layer = getattr(self, other_name)
        
        # Create bidirectional connections
        layer.pattern_connections[pattern_key].append((other_pattern_key, 0.5))
        other_layer.pattern_connections[other_pattern_key].append((pattern_key, 0.5))

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

    def process_text_input(self, content, metadata):
        """Unified consciousness processing"""
        try:
            patterns = metadata['patterns']
            concepts = metadata['concepts']
            relationships = metadata['relationships']
            
            # Convert numpy arrays to tuples for hashing
            hashable_patterns = {
                concept: tuple(pattern.flatten()) if isinstance(pattern, np.ndarray) else pattern
                for concept, pattern in patterns.items()
            }
            
            layer_results = {}
            for layer in [self.conscious, self.subconscious, self.unconscious]:
                layer_results[layer.name] = self._process_layer(
                    layer, hashable_patterns, concepts, relationships
                )
            
            self._update_semantic_memory(concepts, relationships)
            return self._integrate_results(layer_results)
            
        except Exception as e:
            print(f"Error in process_text_input: {str(e)}")
            return {'integrated_strength': 0.0}

    def _process_layer(self, layer, patterns, concepts, relationships):
        """Process patterns in a single layer with belief integration"""
        try:
            # Initialize pattern storage if needed
            if not hasattr(layer, 'patterns'):
                layer.patterns = {}
                layer.pattern_strengths = {}
                layer.pattern_connections = {}
                
            # Initialize semantic memory if needed
            if not hasattr(layer, 'semantic_memory'):
                layer.semantic_memory = {
                    'concepts': {},
                    'relationships': {},
                    'hierarchies': {}
                }
                
            layer_patterns = {}
            strength = 0.0
            
            # Process each concept
            for concept in concepts:
                # Initialize concept in semantic memory if needed
                if concept not in layer.semantic_memory['concepts']:
                    layer.semantic_memory['concepts'][concept] = 0.0
                    
                # Get or create pattern for concept
                if concept in patterns:
                    pattern = np.array(patterns[concept])
                    pattern_key = hash(str(pattern.tobytes())) % 1000
                    
                    # Store pattern
                    layer.patterns[pattern_key] = pattern
                    
                    # Initialize or update pattern strength
                    if pattern_key not in layer.pattern_strengths:
                        layer.pattern_strengths[pattern_key] = 0.1
                    layer.pattern_strengths[pattern_key] = min(
                        1.0,
                        layer.pattern_strengths[pattern_key] + 0.1
                    )
                    
                    # Initialize pattern connections if needed
                    if pattern_key not in layer.pattern_connections:
                        layer.pattern_connections[pattern_key] = []
                        
                    # Update semantic memory
                    layer.semantic_memory['concepts'][concept] = min(
                        1.0,
                        layer.semantic_memory['concepts'].get(concept, 0) + 0.1
                    )
                    
                    # Update strength based on relationships
                    strength += len(relationships.get(concept, [])) / 10.0
                    layer_patterns[concept] = pattern_key
                    
                    # Process relationships
                    if concept in relationships:
                        # Initialize relationships if needed
                        if concept not in layer.semantic_memory['relationships']:
                            layer.semantic_memory['relationships'][concept] = []
                            
                        # Update relationships
                        layer.semantic_memory['relationships'][concept] = list(set(
                            layer.semantic_memory['relationships'].get(concept, []) + 
                            relationships[concept]
                        ))
                        
                        # Update belief context if concept is significant
                        if layer.semantic_memory['concepts'][concept] > 0.3:
                            self._update_belief_context(layer, concept, relationships[concept])
                            
                        # Create pattern connections for related concepts
                        for related in relationships[concept]:
                            if related in patterns:
                                related_key = hash(str(patterns[related].tobytes())) % 1000
                                if related_key not in layer.pattern_connections[pattern_key]:
                                    layer.pattern_connections[pattern_key].append((related_key, 0.5))
                
            return {
                'patterns': layer_patterns,
                'strength': strength / max(1, len(concepts)),
                'learning_strength': 0.1,
                'patterns_processed': len(patterns)
            }
            
        except Exception as e:
            print(f"Error in _process_layer: {str(e)}")
            return {
                'patterns': {},
                'strength': 0.0,
                'learning_strength': 0.0,
                'patterns_processed': 0
            }

    def _update_semantic_memory(self, concepts, relationships):
        """Update semantic memory across all layers"""
        # Initialize semantic memory structure if not exists
        if 'concepts' not in self.semantic_memory:
            self.semantic_memory = {
                'concepts': {},
                'relationships': {},
                'hierarchies': {}
            }
        
        for concept in concepts:
            # Update system-level semantic memory directly
            self.semantic_memory['concepts'][concept] = max(
                self.semantic_memory['concepts'].get(concept, 0) + 0.1,
                self.conscious.semantic_memory['concepts'].get(concept, 0),
                self.subconscious.semantic_memory['concepts'].get(concept, 0),
                self.unconscious.semantic_memory['concepts'].get(concept, 0)
            )
            
            # Update relationships at system level
            if concept in relationships:
                self.semantic_memory['relationships'][concept] = list(set(
                    self.semantic_memory['relationships'].get(concept, []) + 
                    relationships[concept]
                ))

    def _integrate_results(self, layer_results):
        """Integrate results from all layers"""
        try:
            if isinstance(layer_results, dict):
                # Calculate average strength across layers
                total_strength = sum(
                    result.get('learning_strength', 0.0)
                    for result in layer_results.values()
                    if isinstance(result, dict)
                )
                num_layers = len(layer_results)
                
                return {
                    'integrated_strength': total_strength / max(1, num_layers),
                    'layer_results': layer_results
                }
            else:
                return {'integrated_strength': 0.0, 'layer_results': {}}
            
        except Exception as e:
            print(f"Error in _integrate_results: {str(e)}")
            return {'integrated_strength': 0.0, 'layer_results': {}}
    
    def _update_belief_context(self, layer, concept, relationships):
        """Update belief context using existing layer belief update logic"""
        context = 'general'
        related_contexts = {rel: 0.5 for rel in relationships}
        dissonance = 0.1  # Default low dissonance for new concepts
        
        layer.update_belief(
            concept,
            context, 
            related_contexts,
            dissonance
        )

    def _process_layer_beliefs(self, concept, context, layer_name):
        """Process beliefs for specific layer"""
        layer = self.layers[layer_name]
        layer_response = context.get('layer_response', {})
        
        belief_updates = layer['belief'].process_belief_update(
            concept,
            {
                'activation': layer_response.get('activation', 0.0),
                'belief_conflict': layer_response.get('belief_conflict', 0.0),
                'emotional_state': layer['emotional_state']
            },
            layer['emotional_state']
        )
        
        return belief_updates

    def _identify_conscious_patterns(self):
        """Identify patterns in conscious layer"""
        try:
            return {
                'activation': self.conscious.state.copy(),
                'patterns': self.conscious.find_similar_patterns(
                    self.conscious.state,
                    threshold=0.7
                ),
                'strength': np.mean([
                    p['strength'] for p in self.conscious.find_similar_patterns(
                        self.conscious.state,
                        threshold=0.7
                    )
                ]) if self.conscious.patterns else 0.0
            }
        except Exception as e:
            print(f"Error in _identify_conscious_patterns: {str(e)}")
            return {
                'activation': np.zeros_like(self.conscious.state),
                'patterns': [],
                'strength': 0.0
            }

    def _identify_subconscious_patterns(self):
        """Identify patterns in subconscious layer"""
        try:
            return {
                'activation': self.subconscious.state.copy(),
                'patterns': self.subconscious.find_similar_patterns(
                    self.subconscious.state,
                    threshold=0.6
                ),
                'strength': np.mean([
                    p['strength'] for p in self.subconscious.find_similar_patterns(
                        self.subconscious.state,
                        threshold=0.6
                    )
                ]) if self.subconscious.patterns else 0.0
            }
        except Exception as e:
            print(f"Error in _identify_subconscious_patterns: {str(e)}")
            return {
                'activation': np.zeros_like(self.subconscious.state),
                'patterns': [],
                'strength': 0.0
            }

    def _identify_unconscious_patterns(self):
        """Identify patterns in unconscious layer"""
        try:
            return {
                'activation': self.unconscious.state.copy(),
                'patterns': self.unconscious.find_similar_patterns(
                    self.unconscious.state,
                    threshold=0.5
                ),
                'strength': np.mean([
                    p['strength'] for p in self.unconscious.find_similar_patterns(
                        self.unconscious.state,
                        threshold=0.5
                    )
                ]) if self.unconscious.patterns else 0.0
            }
        except Exception as e:
            print(f"Error in _identify_unconscious_patterns: {str(e)}")
            return {
                'activation': np.zeros_like(self.unconscious.state),
                'patterns': [],
                'strength': 0.0
            }

    def _identify_cross_layer_patterns(self, patterns):
        """Identify patterns that exist across consciousness layers"""
        try:
            cross_patterns = {}
            for layer_name, layer_patterns in patterns.items():
                for pattern in layer_patterns.get('patterns', []):
                    pattern_key = hash(str(pattern['pattern'].tobytes()))
                    if pattern_key not in cross_patterns:
                        cross_patterns[pattern_key] = {
                            'pattern': pattern['pattern'],
                            'layers': [],
                            'strength': 0.0
                        }
                    cross_patterns[pattern_key]['layers'].append(layer_name)
                    cross_patterns[pattern_key]['strength'] += pattern['strength']

            return cross_patterns
        except Exception as e:
            print(f"Error in _identify_cross_layer_patterns: {str(e)}")
            return {}

    def _update_pattern_influence(self, cross_patterns):
        """Update pattern influence across consciousness layers"""
        try:
            for pattern_key, pattern_data in cross_patterns.items():
                pattern_strength = pattern_data['strength']
                affected_layers = pattern_data['layers']
                
                # Update pattern strengths in each affected layer
                for layer_name in affected_layers:
                    layer = getattr(self, layer_name)
                    
                    # Update pattern strength
                    layer.pattern_strengths[pattern_key] = min(
                        1.0,
                        layer.pattern_strengths.get(pattern_key, 0) + 
                        pattern_strength * self.learning_rate
                    )
                    
                    # Update connected patterns
                    if pattern_key in layer.pattern_connections:
                        for connected_key, connection_strength in layer.pattern_connections[pattern_key]:
                            layer.pattern_strengths[connected_key] = min(
                                1.0,
                                layer.pattern_strengths.get(connected_key, 0) + 
                                pattern_strength * connection_strength * 0.5 * self.learning_rate
                            )
                    
                    # Update belief systems based on pattern activation
                    if pattern_strength > self.pattern_threshold:
                        layer.belief.update_from_pattern(
                            pattern_data['pattern'],
                            pattern_strength
                        )
                        
                    # Update semantic memory connections
                    pattern_concepts = self._get_pattern_concepts(pattern_data['pattern'])
                    for concept in pattern_concepts:
                        layer.semantic_memory['concepts'][concept] = min(
                            1.0,
                            layer.semantic_memory['concepts'].get(concept, 0) + 
                            pattern_strength * 0.1
                        )
                        
        except Exception as e:
            print(f"Error in _update_pattern_influence: {str(e)}")

    def _calculate_consolidation_strength(self, learning_results):
        """Calculate memory consolidation strength"""
        try:
            if isinstance(learning_results, dict):
                # Extract metrics
                depth = learning_results.get('depth', 0.0)
                breadth = learning_results.get('breadth', 0.0)
                cognitive_load = learning_results.get('cognitive_load', 0.0)
                understanding = learning_results.get('understanding', 0.0)
                
                # Calculate consolidation strength
                consolidation = (
                    depth * 0.3 +
                    breadth * 0.2 +
                    (1 - cognitive_load) * 0.2 +  # Lower cognitive load = better consolidation
                    understanding * 0.3
                )
                
                return float(np.clip(consolidation, 0.0, 1.0))
            else:
                return 0.0
            
        except Exception as e:
            print(f"Error in _calculate_consolidation_strength: {str(e)}")
            return 0.0

    def process_rest_period(self, duration=1.0):
        """Process rest period with consciousness integration and imagination"""
        # Process base consciousness state
        consciousness_state = self.process_rest_state()
        
        # Run imagination/reflection process
        discoveries = self.imagination.reflect(duration)
        
        # Integrate discoveries with belief system
        self._integrate_imagination_discoveries(discoveries)
        
        return {
            'discoveries': discoveries,
            'consciousness_state': consciousness_state
        }

    def _integrate_imagination_discoveries(self, discoveries):
        """Integrate imagination discoveries into belief and goal systems"""
        for discovery in discoveries:
            if discovery['type'] == 'direct':
                # Update direct concept relationships
                connection = discovery['connection']
                concepts = connection['concepts']
                
                # Strengthen or weaken beliefs based on relationship
                if connection['opposition']:
                    self._strengthen_opposing_beliefs(concepts[0], concepts[1])
                else:
                    self._strengthen_supporting_beliefs(concepts[0], concepts[1])
                    
            elif discovery['type'] == 'indirect':
                # Process indirect connections
                path = discovery['path']
                strength = discovery['strength']
                
                # Update semantic relationships
                for i in range(len(path)-1):
                    self._update_semantic_connection(
                        path[i], 
                        path[i+1], 
                        strength
                    )
                    
            elif discovery['type'] == 'cluster':
                # Process concept clusters
                concepts = discovery['concepts']
                self._strengthen_concept_cluster(concepts)
                
            elif discovery['type'] == 'central_concepts':
                # Update importance of central concepts
                concepts = discovery['concepts']
                self._update_concept_centrality(concepts)

    def _strengthen_opposing_beliefs(self, concept_a, concept_b):
        """Strengthen beliefs when concepts oppose each other"""
        for layer in self.layers.values():
            for category in layer.belief_systems.values():
                for subcategory in category.values():
                    if concept_a in subcategory:
                        subcategory[concept_a] = min(
                            1.0, 
                            subcategory[concept_a] * 1.1
                        )
                    if concept_b in subcategory:
                        subcategory[concept_b] = min(
                            1.0, 
                            subcategory[concept_b] * 1.1
                        )

    def _strengthen_supporting_beliefs(self, concept_a, concept_b):
        """Strengthen beliefs when concepts support each other"""
        for layer in self.layers.values():
            belief_a = self._get_belief_strength(concept_a)
            belief_b = self._get_belief_strength(concept_b)
            
            avg_strength = (belief_a + belief_b) / 2
            for category in layer.belief_systems.values():
                for subcategory in category.values():
                    if concept_a in subcategory:
                        subcategory[concept_a] = min(
                            1.0, 
                            subcategory[concept_a] + avg_strength * 0.1
                        )
                    if concept_b in subcategory:
                        subcategory[concept_b] = min(
                            1.0, 
                            subcategory[concept_b] + avg_strength * 0.1
                        )

    def process_imagination(self, input_stimulus, context=None):
        """Process imagination with DMN influence"""
        # Get current emotional and personality influence
        emotional_state = self._get_normalized_emotional_value()
        personality_influence = self._calculate_personality_influence()
        
        # Adjust thresholds based on DMN state
        thresholds = self._get_adjusted_thresholds()
        
        # Process through layers
        return self.imagination.process_imagination(
            input_stimulus,
            context=context,
            emotional_state=emotional_state,
            personality_influence=personality_influence,
            thresholds=thresholds
        )

    def _get_adjusted_thresholds(self):
        """Get thresholds adjusted for DMN state"""
        base_thresholds = self.imagination.thresholds.copy()
        if self.dmn_state['active']:
            # Lower thresholds during DMN activity
            return {k: v * 0.7 for k, v in base_thresholds.items()}
        return base_thresholds

    def _get_normalized_emotional_value(self):
        """Get normalized emotional value across all layers"""
        # Get emotional values from each layer
        emotional_values = {
            'conscious': self.conscious.emotional_state,
            'subconscious': self.subconscious.emotional_state,
            'unconscious': self.unconscious.emotional_state
        }
        
        # Calculate weighted average based on layer weights
        weighted_sum = sum(
            value * self.layer_weights[layer]
            for layer, value in emotional_values.items()
        )
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, weighted_sum))

    def _calculate_personality_influence(self):
        """Calculate personality influence on imagination"""
        # Create personality influence dictionary
        return {
            'openness': self.personality_traits['openness'],
            'conscientiousness': self.personality_traits['conscientiousness'],
            'extraversion': self.personality_traits['extraversion'],
            'agreeableness': self.personality_traits['agreeableness'],
            'neuroticism': self.personality_traits['neuroticism'],
            'resilience': self.personality_traits['resilience'],
            'adaptability': self.personality_traits['adaptability'],
            'curiosity': self.personality_traits['curiosity']
        }