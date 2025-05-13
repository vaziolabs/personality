from collections import defaultdict
from numba import jit
from hbs.behaviours.desire import DesireSystem
import numpy as np
from collections import defaultdict
from hbs.values.belief import BeliefSystem
from hbs.consciousness.concept import ConceptWrapper

class ConsciousnessLayer:
    def __init__(self, size, name, dissonance_threshold=0.3):
        """Initialize consciousness layer"""
        self.size = size
        self.name = name
        self.dissonance_threshold = dissonance_threshold
        self.state = np.zeros((size, size), dtype=np.float32)
        
        # Initialize emotional state tracking
        self.emotional_state = 0.0
        self.emotional_history = []
        self.emotional_decay_rate = 0.1
        
        # Initialize semantic memory structure
        self.semantic_memory = {
            'concepts': {},
            'relationships': {},
            'hierarchies': {}
        }
        
        # Initialize all dictionaries without lambdas
        self.belief_contexts = {}
        self.active_beliefs = {}
        self.patterns = {}
        self.pattern_locations = {}
        self.emotional_weights = {}
        self.pattern_strengths = {}
        self.pattern_connections = {}
        self.context_relationships = {}
        self.dissonance_by_context = {}
        self.familiarity_scores = {}
        
        # Initialize belief systems
        self.belief_systems = {
            'foundational': {
                'epistemological': {},
                'ontological': {},
                'axiological': {},
                'logical': {}
            },
            'value_systems': {
                'ethical': {},
                'aesthetic': {},
                'social': {}
            },
            'self_concept': {
                'personality': {},
                'identity': {},
                'values': {},
                'abilities': {}
            },
            'contextual': {}
        }
        
        # Initialize emotional memory
        self.emotional_memory = {
            'positive': {},
            'negative': {},
            'neutral': {},
            'ambivalent': {}
        }
        
        # Layer state tracking
        self.activation_history = np.zeros((1000, size, size), dtype=np.float32)
        self.history_index = 0
        
        # Initialize belief and desire systems
        self.belief = BeliefSystem(size, self.name)
        self.belief_confidence = {
            'Conscious': 0.7,
            'Subconscious': 0.5, 
            'Unconscious': 0.3
        }[self.name]
        
        self.desire = DesireSystem(size)
        self.desire_weights = {
            'Conscious': {
                'logical': 0.4,
                'emotional': 0.2,
                'pattern': 0.3,
                'instinctive': 0.1
            },
            'Subconscious': {
                'logical': 0.2,
                'emotional': 0.4,
                'pattern': 0.3,
                'instinctive': 0.1
            },
            'Unconscious': {
                'logical': 0.1,
                'emotional': 0.2,
                'pattern': 0.3,
                'instinctive': 0.4
            }
        }[self.name]

        self.motivation_weights = {
            'Conscious': {
                'opportunity': 0.5,
                'respect': 0.3,
                'security': 0.2
            },
            'Subconscious': {
                'opportunity': 0.3,
                'respect': 0.4,
                'security': 0.3
            },
            'Unconscious': {
                'opportunity': 0.2,
                'respect': 0.3,
                'security': 0.5
            }
        }[self.name]
        
        self.layer_above = None
        self.layer_below = None
        
        # Initialize pattern connections for each pattern
        def init_pattern_connection(pattern_key):
            if pattern_key not in self.pattern_connections:
                self.pattern_connections[pattern_key] = []
    
    def _create_belief_matrix(self):
        """Create a new belief matrix of the correct size"""
        return np.zeros((self.size, self.size), dtype=np.float32)
    
    def store_pattern(self, pattern, location, emotional_weight=0):
        """Store pattern with consistent structure"""
        # Make location hashable if it's a numpy array
        if isinstance(location, np.ndarray):
            location_key = str(location.tobytes())
        else:
            location_key = location
        
        # Convert pattern to bytes for hashing
        pattern_key = hash(pattern.tobytes()) % 1000
        
        # Store core pattern data
        self.patterns[pattern_key] = pattern.copy()
        self.pattern_locations[pattern_key] = location_key
        self.emotional_weights[pattern_key] = emotional_weight
        
        # Initialize pattern strength if new
        if pattern_key not in self.pattern_strengths:
            self.pattern_strengths[pattern_key] = 0.1
            
        # Initialize pattern connections if new
        if pattern_key not in self.pattern_connections:
            self.pattern_connections[pattern_key] = []
            
        # Store concept-specific pattern if location is a concept
        if isinstance(location_key, str):
            concept_key = hash(location_key) % 1000
            self.patterns[concept_key] = pattern
            self.pattern_locations[concept_key] = location_key
            self.emotional_weights[concept_key] = emotional_weight
            self.pattern_strengths[concept_key] = 0.1
            
            # Initialize concept connections if new
            if concept_key not in self.pattern_connections:
                self.pattern_connections[concept_key] = []
            
            # Connect concept pattern to original pattern
            self.pattern_connections[pattern_key].append((concept_key, 1.0))
            self.pattern_connections[concept_key].append((pattern_key, 1.0))
        
        return pattern_key

    def find_similar_patterns(self, pattern, threshold=0.8):
        """Find similar patterns with consistent structure"""
        pattern_flat = pattern.ravel()
        matches = []
        
        for key, stored_pattern in self.patterns.items():
            stored_flat = stored_pattern.ravel()
            if len(stored_flat) == len(pattern_flat):
                similarity = self._calculate_similarity(pattern_flat, stored_flat)
                if similarity > threshold:
                    matches.append({
                        'pattern': stored_pattern,
                        'location': self.pattern_locations[key],
                        'similarity': similarity,
                        'emotional_weight': self.emotional_weights.get(key, 0.0),
                        'strength': self.pattern_strengths.get(key, 0.0)
                    })
        return matches

    def strengthen_pattern(self, pattern_key, amount=0.1):
        """Strengthen a pattern and its connections"""
        if pattern_key in self.pattern_strengths:
            self.pattern_strengths[pattern_key] = min(
                1.0, 
                self.pattern_strengths[pattern_key] + amount
            )
            
            # Strengthen connections proportionally
            for connected_key, connection_strength in self.pattern_connections[pattern_key]:
                self.pattern_strengths[connected_key] = min(
                    1.0,
                    self.pattern_strengths[connected_key] + (amount * connection_strength * 0.5)
                )

    @staticmethod
    @jit(nopython=True)
    def _calculate_similarity(pattern1, pattern2):
        return np.sum(pattern1 * pattern2) / (np.sqrt(np.sum(pattern1**2)) * np.sqrt(np.sum(pattern2**2)))

    def update_history(self, activation):
        self.activation_history[self.history_index] = activation
        self.history_index = (self.history_index + 1) % 1000

    def update_belief(self, new_input, context, related_contexts, dissonance_level):
        """Update belief state for specific context and handle context relationships"""
        confidence = 1.0 - dissonance_level
        current_belief = self.belief_contexts[context]
        
        # Convert text input to activation pattern if needed
        input_pattern = (self._create_text_pattern(new_input) 
                        if isinstance(new_input, str) 
                        else new_input)
        
        # Update primary context
        self.belief_contexts[context] = (
            current_belief * (0.8 * confidence) + 
            input_pattern * (0.2 * (2.0 - confidence))
        )
        
        # Process related contexts
        if related_contexts:
            self._update_related_contexts(context, related_contexts)
        
        self.dissonance_by_context[context].append(dissonance_level)

    def _update_related_contexts(self, context, related_contexts):
        """Update relationships between contexts"""
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

    def evaluate_action(self, action, current_context):
        """Enhanced action evaluation based on psychological model"""
        # Use belief system's prediction instead
        prediction = self.belief.predict_outcome(
            action, 
            {
                'layer': self.name.lower(),
                'context': current_context,
                'desire_strength': self.desire.get_desire_strength(action, current_context)
            }
        )
        
        return prediction

    def _init_layer_weights(self, name):
        """Initialize processing weights based on layer type"""
        weights = {
            'rational_weight': 0.0,
            'emotional_weight': 0.0,
            'instinctive_weight': 0.0
        }
        
        if name == "Conscious":
            weights.update({
                'rational_weight': 0.7,
                'emotional_weight': 0.2,
                'instinctive_weight': 0.1,
                name: np.ones((self.size, self.size)) * 0.7  # Layer-specific processing matrix
            })
        elif name == "Subconscious":
            weights.update({
                'rational_weight': 0.3,
                'emotional_weight': 0.5,
                'instinctive_weight': 0.2,
                name: np.ones((self.size, self.size)) * 0.5  # Layer-specific processing matrix
            })
        else:  # Unconscious
            weights.update({
                'rational_weight': 0.1,
                'emotional_weight': 0.3,
                'instinctive_weight': 0.6,
                name: np.ones((self.size, self.size)) * 0.3  # Layer-specific processing matrix
            })
        
        return weights

    def process_text(self, text_data, desires):
        """Process text at conscious level"""
        # Extract semantic features
        features = self._extract_semantic_features(text_data)
        
        # Calculate activation pattern
        activation = self._calculate_text_activation(features)
        
        return {
            'activation': activation,
            'features': features,
            'desires': desires
        }

    def _extract_semantic_features(self, text_data):
        """Extract semantic features from text"""
        # Create fixed-size pattern matrix
        text_pattern = self._create_text_pattern(text_data)
        
        return {
            'content': text_data,
            'complexity': len(text_data.split()),
            'emotional_valence': self._estimate_emotional_content(text_data),
            'pattern_matches': self.find_similar_patterns(text_pattern)
        }

    def _create_text_pattern(self, text_data):
        """Create a fixed-size pattern matrix from text"""
        # Convert text to ASCII values and pad/truncate to match size
        ascii_values = [ord(c) for c in str(text_data)]
        target_size = self.size  # This is 256
        
        if len(ascii_values) > target_size:
            ascii_values = ascii_values[:target_size]
        else:
            ascii_values.extend([0] * (target_size - len(ascii_values)))
        
        # Create 1D array and normalize
        pattern = np.array(ascii_values, dtype=np.float32)
        if pattern.max() > 0:
            pattern = pattern / pattern.max()
        
        return pattern

    def _calculate_text_activation(self, features):
        """Calculate activation pattern from text features"""
        base_activation = np.zeros((self.size, self.size))
        
        # Modulate by complexity
        complexity_factor = min(1.0, features['complexity'] / 100.0)
        base_activation += complexity_factor * self.processing_weights.get(self.name, np.ones((self.size, self.size)) * 0.5)
        
        # Add emotional influence
        emotional_factor = features['emotional_valence']
        base_activation *= (1.0 + emotional_factor * 0.2)
        
        return base_activation

    def _process_text_beliefs(self, features):
        """Process text through belief system"""
        belief_update = {}
        
        # Update beliefs based on pattern matches
        for location, similarity, emotional_weight in features['pattern_matches']:
            context_key = f"text_{hash(str(location)) % 100}"
            self.belief_contexts[context_key] += similarity * self.processing_weights[self.name]
            belief_update[context_key] = similarity
            
        return belief_update

    def _identify_text_patterns(self, features):
        """Identify patterns in text processing"""
        return {
            'semantic': features['pattern_matches'],
            'activation': self.activation_history[max(0, self.history_index-10):self.history_index].mean(axis=0)
        }

    def _calculate_emotional_response(self, emotional_input):
        """Calculate emotional response from either features dict or direct value"""
        if isinstance(emotional_input, dict):
            # Extract emotional valence from features
            emotional_value = emotional_input.get('emotional_valence', 0.0)
        else:
            # Use direct emotional value
            emotional_value = float(emotional_input)
        
        # Apply layer-specific processing weight
        layer_weight = self.processing_weights.get(self.name, 1.0)
        
        # Calculate final emotional response
        return emotional_value * layer_weight

    def _estimate_emotional_content(self, text):
        """Estimate emotional content of text"""
        # Simple estimation based on pattern matching
        emotional_value = 0.0
        for pattern_key, location in self.patterns.items():
            if str(pattern_key) in text.lower():
                emotional_value += self.emotional_weights.get(pattern_key, 0.0)
        return np.tanh(emotional_value)  # Normalize to [-1, 1]

    def process_patterns(self, text_data, conscious_response):
        """Process patterns at subconscious level"""
        # Extract text patterns
        text_pattern = self._create_text_pattern(text_data)
        
        # Find similar stored patterns
        pattern_matches = self.find_similar_patterns(text_pattern)
        
        # Calculate pattern-based activation
        pattern_activation = np.zeros_like(self.state)
        total_weight = 0.0
        
        for location, similarity, emotion in pattern_matches:
            weight = similarity * (1 + abs(emotion))
            pattern_activation += self.state[location] * weight
            total_weight += weight
        
        # Normalize pattern activation
        if total_weight > 0:
            pattern_activation /= total_weight
        
        # Integrate with conscious response
        conscious_activation = (
            conscious_response['activation'] 
            if isinstance(conscious_response, dict) 
            else conscious_response
        )
        
        integrated_response = {
            'activation': pattern_activation * 0.6 + conscious_activation * 0.4,
            'patterns': pattern_matches,
            'emotional_weight': total_weight
        }
        
        # Update state and history
        self.state = integrated_response['activation']
        self.update_history(integrated_response['activation'])
        
        return integrated_response

    def integrate_knowledge(self, conscious_response, subconscious_response):
        """Integrate knowledge while maintaining beliefs and semantic memory"""
        # Extract activations and features
        conscious_activation = conscious_response['activation']
        conscious_features = conscious_response.get('features', {})
        subconscious_activation = subconscious_response['activation']
        
        # Calculate belief consistency and update semantic memory
        emotional_factor = np.exp(abs(self.emotional_state))
        belief_conflict = self._check_belief_consistency(
            conscious_features.get('content', ''),
            self.active_beliefs,
            emotional_factor
        )
        
        # Update semantic memory with new patterns
        pattern_key = self._store_semantic_pattern(
            conscious_activation,
            conscious_features.get('content', ''),
            conscious_features.get('context', 'general')
        )
        
        # Calculate integrated dissonance
        dissonance = np.mean([
            belief_conflict,
            abs(np.mean(conscious_activation - subconscious_activation)),
            abs(np.mean(self.state - subconscious_activation))
        ])
        
        # Update beliefs and semantic memory if dissonance is acceptable
        if dissonance < self.dissonance_threshold:
            self._update_belief_and_memory(
                pattern_key,
                conscious_features,
                dissonance
            )
        else:
            self._record_cognitive_dissonance(conscious_features, dissonance)
            
        return self._create_integrated_response(
            conscious_activation,
            subconscious_activation,
            pattern_key,
            dissonance,
            belief_conflict
        )

    def _store_semantic_pattern(self, pattern, content, context):
        """Store pattern in semantic memory with proper connections"""
        pattern_key = hash(str(pattern.flatten())) % 1000
        
        # Store core pattern data
        self.patterns[pattern_key] = pattern
        self.pattern_locations[pattern_key] = context
        
        # Create semantic links
        if isinstance(content, str):
            # Extract concepts and store relationships
            concepts = set(content.lower().split())
            for concept in concepts:
                concept_key = hash(concept) % 1000
                
                # Initialize pattern connections if needed
                if pattern_key not in self.pattern_connections:
                    self.pattern_connections[pattern_key] = []
                self.pattern_connections[pattern_key].append((concept_key, 1.0))
                
                # Initialize concept in semantic memory if needed
                if concept not in self.semantic_memory['concepts']:
                    self.semantic_memory['concepts'][concept] = 0.0
                self.semantic_memory['concepts'][concept] += 0.1
                
                # Initialize relationships if needed
                if concept not in self.semantic_memory['relationships']:
                    self.semantic_memory['relationships'][concept] = []
                self.semantic_memory['relationships'][concept].append(pattern_key)
        
        return pattern_key

    def _update_belief_and_memory(self, pattern_key, features, dissonance):
        """Update beliefs and semantic memory together"""
        context = features.get('context', 'general')
        content = features.get('content', '')
        
        # Update belief state
        self.update_belief(
            self.patterns[pattern_key],
            context,
            features.get('related_contexts', {}),
            dissonance
        )
        
        # Strengthen semantic connections
        if pattern_key in self.pattern_connections:
            for concept_key, strength in self.pattern_connections[pattern_key]:
                # Strengthen connections based on belief confidence
                new_strength = strength * (1.0 - dissonance)
                self.pattern_strengths[concept_key] = min(
                    1.0,
                    self.pattern_strengths.get(concept_key, 0) + new_strength * 0.1
                )
                
                # Update semantic hierarchies if strong connection
                if new_strength > 0.7:
                    self.semantic_memory['hierarchies'][context].add(
                        self.pattern_locations.get(concept_key, '')
                    )

    def _check_belief_consistency(self, signal, beliefs, emotional_factor=None):
        """Check consistency between signal and beliefs with emotional influence"""
        try:
            if emotional_factor is None:
                emotional_factor = abs(self.emotional_state)
            
            # Get active beliefs for signal
            active_beliefs = {
                k: v for k, v in beliefs.items() 
                if isinstance(v, (int, float)) and v > 0.1
            }
            
            if not active_beliefs:
                return 1.0  # No beliefs to check against
            
            # Calculate consistency score
            total_score = 0.0
            belief_count = len(active_beliefs)
            
            for belief, strength in active_beliefs.items():
                # Check if belief exists in belief systems
                belief_value = self.belief_systems['foundational']['logical'].get(belief, 0.0)
                
                # Higher emotional states reduce logical consistency
                consistency = belief_value * (1.0 - emotional_factor * 0.3)
                
                # Weight by belief strength
                total_score += consistency * strength
            
            # Normalize score
            consistency_score = total_score / belief_count if belief_count > 0 else 1.0
            
            return np.clip(consistency_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in _check_belief_consistency: {str(e)}")
            return 1.0

    def _record_cognitive_dissonance(self, features, dissonance):
        """Record cognitive dissonance for later processing"""
        self.dissonance_by_context[features.get('context', 'general')].append({
            'content': features.get('content', ''),
            'strength': dissonance,
            'time': len(self.dissonance_by_context)  # Simple timestamp
        })
        
    def _calculate_philosophical_influence(self, concept):
        """Helper method to calculate philosophical influence"""
        influences = {
            'epistemological': self._calculate_belief_influence(concept, 'foundational', 'epistemological'),
            'ontological': self._calculate_belief_influence(concept, 'foundational', 'ontological'),
            'axiological': self._calculate_belief_influence(concept, 'foundational', 'axiological'),
            'logical': self._calculate_belief_influence(concept, 'foundational', 'logical')
        }
        return sum(influences.values()) / len(influences)

    def _calculate_reward(self, pattern, emotional_value):
        """Helper method to calculate reward"""
        novelty = self._calculate_novelty_reward(str(pattern.tobytes()))
        familiarity = self._calculate_familiarity_reward(str(pattern.tobytes()))
        return (novelty + familiarity) * (1 + abs(emotional_value))

    def _calculate_belief_influence(self, concept, category, subcategory):
        """Calculate influence of specific belief category on concept processing"""
        belief_strength = self.belief_systems[category][subcategory].get(str(concept), 0.0)
        confidence = self.belief_confidence[category]
        return belief_strength * confidence

    def _create_concept_pattern(self, concept):
        """Create pattern from concept string with dynamic sizing"""
        # Convert concept to array of ASCII values
        concept_array = np.array([ord(c) for c in str(concept)], dtype=np.float32)
        
        # Calculate dimensions that will fit the array
        array_size = len(concept_array)
        dim = int(np.ceil(np.sqrt(array_size)))
        
        # Create square array padded with zeros
        square_size = dim * dim
        padded_array = np.zeros(square_size, dtype=np.float32)
        padded_array[:array_size] = concept_array
        
        # Reshape into 2D square matrix
        pattern = padded_array.reshape((dim, dim))
        
        return pattern

    def _calculate_self_relevance(self, concept):
        """Calculate how relevant a concept is to self-concept"""
        relevance = 0.0
        
        # Check if personality traits exist
        if hasattr(self, 'personality_traits'):
            for trait in self.personality_traits:
                if trait in str(concept).lower():
                    relevance += 0.2
        
        # Check if self_concept and its categories exist
        if 'self_concept' in self.belief_systems:
            for category in ['identity', 'values', 'abilities']:
                if (category in self.belief_systems['self_concept'] and 
                    concept in self.belief_systems['self_concept'][category]):
                    relevance += 0.3
        
        return min(1.0, relevance)

    def _calculate_novelty_reward(self, concept):
        """Calculate reward for concept novelty with philosophical weighting"""
        base_novelty = 1.0 if concept not in self.familiarity_scores else max(0, 1.0 - self.familiarity_scores[concept])
        
        # Weight novelty based on epistemological beliefs
        epistemic_weight = self.belief_systems['foundational']['epistemological'].get('novelty_value', 0.5)
        return base_novelty * (1 + epistemic_weight)

    def _calculate_familiarity_reward(self, concept):
        """Calculate reward based on concept familiarity and ontological beliefs"""
        base_familiarity = min(0.5, self.familiarity_scores.get(concept, 0) * 0.5)
        
        # Weight familiarity based on ontological stability preference
        ontological_weight = self.belief_systems['foundational']['ontological'].get('stability_value', 0.5)
        return base_familiarity * (1 + ontological_weight)

    def _calculate_goal_alignment_penalty(self, concept, goals):
        """Calculate penalty based on goal alignment and axiological beliefs"""
        if not goals:
            return 0.0
        
        axiological_weight = self.belief_systems['foundational']['axiological'].get('goal_importance', 0.5)
        goal_misalignment = sum(1 for goal in goals if concept not in goal) / len(goals)
        return goal_misalignment * axiological_weight

    def _calculate_repetition_factor(self, pattern_key):
        """Calculate strengthening factor based on pattern repetition and logical beliefs"""
        recent_occurrences = sum(
            1 for memory in self.history[-10:]
            for response in [memory['conscious'], memory['subconscious'], memory['unconscious']]
            if hash(str(response.flatten())) % 1000 == pattern_key
        )
        
        logical_weight = self.belief_systems['foundational']['logical'].get('pattern_value', 0.5)
        return min(1.0, recent_occurrences * 0.2 * (1 + logical_weight))

    def _calculate_context_relevance(self, pattern_location, current_context):
        """Calculate relevance between pattern context and current context with social beliefs"""
        if pattern_location == current_context:
            return 1.0
        
        social_weight = self.belief_systems['value_systems']['social'].get('context_importance', 0.5)
        if current_context in self.context_relationships.get(pattern_location, {}):
            base_relevance = self.context_relationships[pattern_location][current_context]
            return base_relevance * (1 + social_weight)
        
        return 0.2 * (1 + social_weight)

    def _strengthen_connected_patterns(self, layer, pattern_key, strengthen_amount):
        """Strengthen patterns with aesthetic and logical influences"""
        aesthetic_weight = self.belief_systems['value_systems']['aesthetic'].get('pattern_beauty', 0.5)
        logical_weight = self.belief_systems['foundational']['logical'].get('connection_value', 0.5)
        
        if pattern_key in layer.pattern_connections:
            for connected_key, connection_strength in layer.pattern_connections[pattern_key]:
                # Apply philosophical weights to strengthening
                strengthen_factor = (1 + aesthetic_weight) * (1 + logical_weight)
                
                layer.pattern_strengths[connected_key] = min(
                    1.0,
                    layer.pattern_strengths.get(connected_key, 0) + 
                    (strengthen_amount * connection_strength * 0.5 * strengthen_factor)
                )
                
                connection_idx = next(
                    (i for i, (k, _) in enumerate(layer.pattern_connections[pattern_key])
                     if k == connected_key),
                    None
                )
                if connection_idx is not None:
                    layer.pattern_connections[pattern_key][connection_idx] = (
                        connected_key,
                        min(1.0, connection_strength + strengthen_amount * 0.1 * strengthen_factor)
                    )

    def _update_semantic_strength(self, concept, strengthen_amount):
        """Update semantic memory with epistemological influence"""
        # Initialize concept if not exists
        if concept not in self.semantic_memory['concepts']:
            self.semantic_memory['concepts'][concept] = 0.0
        
        self.semantic_memory['concepts'][concept] = min(
            1.0,
            self.semantic_memory['concepts'].get(concept, 0) + strengthen_amount
        )
        
        # Initialize relationships if needed
        if concept not in self.semantic_memory['relationships']:
            self.semantic_memory['relationships'][concept] = []
        
        for related_concept in self.semantic_memory['relationships'].get(concept, []):
            if related_concept not in self.semantic_memory['concepts']:
                self.semantic_memory['concepts'][related_concept] = 0.0
            self.semantic_memory['concepts'][related_concept] = min(
                1.0,
                self.semantic_memory['concepts'].get(related_concept, 0) + 
                (strengthen_amount * 0.3)
            )

    def _update_emotional_state(self, emotional_value):
        """Update emotional state with decay and ambivalence handling"""
        try:
            if isinstance(emotional_value, str) and emotional_value == 'ambivalent':
                time_factor = len(self.emotional_history) % 1000
                self.emotional_state = 0.5 * np.sin(time_factor * 0.1)
            else:
                self.emotional_state = (
                    self.emotional_state * 0.9 + 
                    float(emotional_value) * 0.1
                )
            self.emotional_history.append(self.emotional_state)
        except Exception as e:
            print(f"Error in _update_emotional_state: {str(e)}")

    def _create_integrated_response(self, pattern, context=None, emotional_value=0.0, 
                                   belief_consistency=1.0, reward_expectation=0.0, 
                                   philosophical_influence=0.5, context_influence=0.5):
        """Create integrated response with full philosophical framework"""
        try:
            # Convert string pattern to numpy array if needed
            if isinstance(pattern, str):
                pattern = self._create_text_pattern(pattern)
            
            # Ensure pattern is properly shaped
            pattern_reshaped = np.zeros((self.size, self.size), dtype=np.float32)
            
            # Handle pattern dimensions safely
            if isinstance(pattern, np.ndarray):
                # Flatten and resize pattern safely to match our output dimensions
                pattern_flat = pattern.flatten()
                max_idx = min(pattern_flat.size, self.size*self.size)
                pattern_reshaped.flat[:max_idx] = pattern_flat[:max_idx]
            
            # Apply philosophical modulation
            emotional_factor = np.exp(abs(emotional_value * 0.5))  # Dampen to prevent overflow
            cognitive_factor = max(0.1, belief_consistency * 0.6 + philosophical_influence * 0.4)
            reward_factor = 1.0 + max(-0.5, min(0.5, reward_expectation)) * 0.3
            context_factor = max(0.1, context_influence)
            
            # Combine factors 
            combined_pattern = pattern_reshaped * emotional_factor * cognitive_factor * reward_factor * context_factor
            
            # Normalize values
            if np.sum(combined_pattern) > 0:
                combined_pattern = combined_pattern / max(1.0, np.max(combined_pattern))
            
            return combined_pattern
            
        except Exception as e:
            print(f"Error in _create_integrated_response: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros((self.size, self.size), dtype=np.float32)

    def process_learning(self, concepts, knowledge_item, emotional_state, previous_layer_result=None):
        """Optimized learning process with parallel processing and caching"""
        try:
            # Update emotional state once
            self._update_emotional_state(emotional_state)
            
            # Pre-calculate shared values
            base_reward = abs(emotional_state) * (1.0 if previous_layer_result else 0.5)
            emotional_category = self._get_emotional_category(emotional_state)
            belief_weights = self._get_belief_weights()
            
            # Batch process concepts
            concept_batch = self._prepare_concept_batch(concepts)
            
            # Process concepts in parallel using numpy operations
            concept_vectors = np.array([
                self._create_concept_vector(concept) 
                for concept in concept_batch
            ])
            
            # Vectorized learning calculations
            learning_strengths = np.zeros(len(concept_batch))
            semantic_updates = defaultdict(float)
            pattern_updates = defaultdict(float)
            
            # Parallel processing of philosophical metrics
            metrics = {
                'epistemological': np.zeros(len(concept_batch)),
                'ontological': np.zeros(len(concept_batch)),
                'axiological': np.zeros(len(concept_batch)),
                'logical': np.zeros(len(concept_batch))
            }
            
            # Vectorized belief consistency check
            belief_consistencies = self._check_belief_consistency_batch(
                concept_vectors,
                self.active_beliefs,
                abs(emotional_state)
            )
            
            # Batch update semantic memory and patterns
            for idx, (concept, vector) in enumerate(zip(concept_batch, concept_vectors)):
                concept_key = concept['key']
                
                # Calculate learning metrics in parallel
                metrics['epistemological'][idx] = belief_weights['epistemological'] * self._evaluate_knowledge_validity(concept_key)
                metrics['ontological'][idx] = belief_weights['ontological'] * self._evaluate_concept_reality(concept_key)
                metrics['axiological'][idx] = belief_weights['axiological'] * self._evaluate_value_alignment(concept_key)
                metrics['logical'][idx] = belief_weights['logical'] * belief_consistencies[idx]
                
                # Calculate learning strength
                learning_strengths[idx] = np.mean([
                    metrics[m][idx] for m in metrics
                ])
                
                # Batch updates for semantic memory
                semantic_updates[concept_key] += learning_strengths[idx]
                
                # Batch updates for patterns
                if 'pattern_key' in concept:
                    pattern_updates[concept['pattern_key']] += learning_strengths[idx]
            
            # Bulk update semantic memory
            for concept_key, strength in semantic_updates.items():
                self.semantic_memory['concepts'][concept_key] = min(
                    1.0,
                    self.semantic_memory['concepts'].get(concept_key, 0.0) + strength
                )
            
            # Bulk update pattern strengths
            for pattern_key, strength in pattern_updates.items():
                self._strengthen_connected_patterns_batch(pattern_key, strength)
            
            # Update belief contexts with learning results
            for concept in concepts:
                belief_matrix = np.random.rand(10, 10)  # Generate belief pattern
                self.belief_contexts[concept] = belief_matrix
                
                # Update belief systems
                for category in self.belief_systems:
                    for subcategory in self.belief_systems[category]:
                        confidence = self.belief_confidence[category]
                        self.belief_systems[category][subcategory][concept] = confidence * learning_strengths[0]
            
            return {
                'learning_strength': np.mean(learning_strengths),
                'metrics': metrics,
                'patterns': pattern_updates
            }
            
        except Exception as e:
            print(f"Error in process_learning: {str(e)}")
            return {'learning_strength': 0.0, 'metrics': {}, 'patterns': {}}

    def _prepare_concept_batch(self, concepts):
        """Prepare concepts for batch processing"""
        concept_batch = []
        for concept in concepts:
            if isinstance(concept, dict):
                concept_key = str(concept.get('id') or concept.get('name') or hash(frozenset(concept.items())))
                concept_batch.append({
                    'key': concept_key,
                    'content': str(concept.get('content', '')),
                    'pattern_key': concept.get('pattern_key')
                })
            else:
                concept_key = str(concept)
                concept_batch.append({
                    'key': concept_key,
                    'content': str(concept),
                    'pattern_key': None
                })
        return concept_batch

    def _create_concept_vector(self, concept):
        """Create vectorized representation of concept for parallel processing"""
        # Convert concept to fixed-size vector using semantic features
        vector_size = 100  # Adjust based on needs
        vector = np.zeros(vector_size)
        
        # Extract features from concept content
        content = concept['content'].lower()
        words = set(content.split())
        
        # Use semantic memory to create feature vector
        for idx, word in enumerate(words):
            if idx >= vector_size:
                break
            vector[idx] = self.semantic_memory['concepts'].get(word, 0.0)
        
        return vector

    def _check_belief_consistency_batch(self, concept_vectors, beliefs, emotional_factor):
        """Vectorized belief consistency check"""
        # Convert beliefs to matrix for batch operations
        belief_matrix = np.array([
            beliefs.get(str(k), 0.0) 
            for k in beliefs 
            if isinstance(beliefs[k], (int, float))
        ])
        
        # Calculate consistency scores in parallel
        consistency_scores = np.zeros(len(concept_vectors))
        if len(belief_matrix) > 0:
            # Vectorized dot product for similarity
            similarities = np.dot(concept_vectors, belief_matrix)
            # Apply emotional modulation
            consistency_scores = similarities * (1.0 - emotional_factor * 0.3)
        
        return np.clip(consistency_scores, 0.0, 1.0)

    def _strengthen_connected_patterns_batch(self, pattern_keys, strengthen_amounts):
        """Batch update of connected patterns"""
        if not isinstance(pattern_keys, (list, np.ndarray)):
            pattern_keys = [pattern_keys]
            strengthen_amounts = [strengthen_amounts]
        
        # Pre-calculate weights
        aesthetic_weight = self.belief_systems['value_systems']['aesthetic'].get('pattern_beauty', 0.5)
        logical_weight = self.belief_systems['foundational']['logical'].get('connection_value', 0.5)
        strengthen_factor = (1 + aesthetic_weight) * (1 + logical_weight)
        
        # Batch update pattern strengths
        updates = defaultdict(float)
        for pattern_key, amount in zip(pattern_keys, strengthen_amounts):
            if pattern_key in self.pattern_connections:
                for connected_key, connection_strength in self.pattern_connections[pattern_key]:
                    updates[connected_key] += amount * connection_strength * 0.5 * strengthen_factor
        
        # Apply updates in bulk
        for key, update in updates.items():
            self.pattern_strengths[key] = min(1.0, self.pattern_strengths.get(key, 0) + update)

    def _get_belief_weights(self):
        """Cache and return belief weights"""
        return {
            'epistemological': self.belief.influence_weights.get('rational', 0.3),
            'ontological': self.belief.influence_weights.get('emotional', 0.3),
            'axiological': self.belief.influence_weights.get('instinctive', 0.3),
            'logical': self.belief.influence_weights.get('rational', 0.3)
        }

    def _get_default_learning_result(self):
        """Return default learning result structure"""
        return {
            'learning_strength': 0.0,
            'metrics': {
                'complexity': 0.0,
                'comprehension': 0.0,
                'epistemological': 0.0,
                'ontological': 0.0,
                'axiological': 0.0,
                'logical': 0.0
            }
        }

    def _evaluate_knowledge_validity(self, concept_key):
        """Evaluate epistemological validity of knowledge"""
        try:
            # Get cached belief values
            epistemological_beliefs = self.belief_systems['foundational']['epistemological']
            
            # Vectorized validity checks
            validity_metrics = np.array([
                epistemological_beliefs.get('empirical', 0.3),
                epistemological_beliefs.get('rational', 0.3),
                epistemological_beliefs.get('coherence', 0.4)
            ])
            
            # Get concept features
            concept_vector = self._create_concept_vector({'key': concept_key, 'content': str(concept_key)})
            
            # Calculate validity score using dot product
            validity_score = np.dot(concept_vector[:3], validity_metrics)
            
            return np.clip(validity_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in _evaluate_knowledge_validity: {str(e)}")
            return 0.3  # Default moderate validity

    def _evaluate_concept_reality(self, concept_key):
        """Evaluate ontological reality of concept"""
        try:
            # Get cached ontological beliefs
            ontological_beliefs = self.belief_systems['foundational']['ontological']
            
            # Vectorized reality metrics
            reality_metrics = np.array([
                ontological_beliefs.get('existence', 0.4),
                ontological_beliefs.get('causality', 0.3),
                ontological_beliefs.get('stability', 0.3)
            ])
            
            # Get concept stability from pattern history
            pattern_stability = self.pattern_strengths.get(concept_key, 0.0)
            
            # Calculate reality score
            reality_base = np.mean([
                self.familiarity_scores.get(concept_key, 0.0),
                pattern_stability,
                self.semantic_memory['concepts'].get(concept_key, 0.0)
            ])
            
            # Apply ontological weights
            reality_score = reality_base * np.mean(reality_metrics)
            
            return np.clip(reality_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in _evaluate_concept_reality: {str(e)}")
            return 0.3  # Default moderate reality

    def _evaluate_value_alignment(self, concept_key):
        """Evaluate axiological value alignment"""
        try:
            # Get cached value systems
            value_systems = self.belief_systems['value_systems']
            
            # Vectorized value metrics
            value_metrics = np.array([
                value_systems['ethical'].get(concept_key, 0.3),
                value_systems['aesthetic'].get(concept_key, 0.3),
                value_systems['social'].get(concept_key, 0.4)
            ])
            
            # Get emotional associations
            emotional_weight = self.emotional_weights.get(concept_key, 0.0)
            
            # Calculate alignment score
            alignment_score = np.mean(value_metrics) * (1.0 + abs(emotional_weight))
            
            return np.clip(alignment_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in _evaluate_value_alignment: {str(e)}")
            return 0.3  # Default moderate alignment

    def _evaluate_logical_consistency(self, concept_key):
        """Evaluate logical consistency of concept"""
        try:
            # Get cached logical beliefs
            logical_beliefs = self.belief_systems['foundational']['logical']
            
            # Get concept relationships
            relationships = self.semantic_memory['relationships'].get(concept_key, [])
            
            if not relationships:
                return logical_beliefs.get('default_consistency', 0.3)
            
            # Vectorized consistency check
            relationship_vectors = np.array([
                self._create_concept_vector({'key': rel, 'content': str(rel)})
                for rel in relationships[:10]  # Limit to recent relationships
            ])
            
            # Calculate consistency scores
            consistency_matrix = np.dot(relationship_vectors, relationship_vectors.T)
            consistency_scores = np.mean(consistency_matrix, axis=1)
            
            # Weight by logical belief factors
            logical_weights = np.array([
                logical_beliefs.get('coherence', 0.4),
                logical_beliefs.get('consistency', 0.3),
                logical_beliefs.get('completeness', 0.3)
            ])
            
            weighted_consistency = np.mean(consistency_scores) * np.mean(logical_weights)
            
            return np.clip(weighted_consistency, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in _evaluate_logical_consistency: {str(e)}")
            return 0.3  # Default moderate consistency

    def process_concept(self, concept, context=None, emotional_value=0.0):
        """Process concept through consciousness layer with optimized processing"""
        try:
            # Wrap concept for standardized handling
            wrapped_concept = ConceptWrapper(concept)
            
            # Ensure concept ID is hashable
            if isinstance(wrapped_concept.id, np.ndarray):
                # Create a hashable representation of the array
                wrapped_concept.hashable_id = str(wrapped_concept.id.tobytes())
            else:
                wrapped_concept.hashable_id = str(wrapped_concept.id)
            
            # Debug logging
            # print(f"DEBUG: Processing in {self.name}. Concept type: {type(concept)}, ID type: {type(wrapped_concept.id)}")
            
            # Create and store pattern - vectorized operation
            concept_pattern = self._create_text_pattern(wrapped_concept.content)
            pattern_key = self.store_pattern(concept_pattern, wrapped_concept.hashable_id, emotional_value)
            
            # Batch update emotional state and memory
            self._batch_update_emotional_state(wrapped_concept, emotional_value)
            
            # Get active beliefs using vectorized operations
            active_beliefs = self._get_active_beliefs_batch(wrapped_concept.hashable_id)
            
            # Calculate all metrics in parallel
            metrics = self._calculate_concept_metrics(
                wrapped_concept,
                context,
                emotional_value,
                active_beliefs
            )
            
            # Process influences and rewards in parallel
            influences = self._process_concept_influences(
                wrapped_concept,
                context,
                metrics,
                emotional_value
            )
            
            # Update semantic memory and relationships in batch
            self._update_concept_relationships(wrapped_concept, context)
            
            # Create final response with all components
            response = self._create_final_response(
                wrapped_concept,
                concept_pattern,
                metrics,
                influences,
                context
            )
            
            return response

        except Exception as e:
            print(f"Error in process_concept for {self.name} layer: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_default_concept_response(emotional_value)

    def _batch_update_emotional_state(self, wrapped_concept, emotional_value):
        """Batch update emotional states and memory"""
        try:
            # Ensure concept_id is hashable (extra safeguard)
            concept_id = wrapped_concept.hashable_id
            
            # Update desire system - check if layer exists
            layer_name = self.name.lower()
            if hasattr(self.desire, 'layer_desires') and layer_name in self.desire.layer_desires:
                self.desire.layer_desires[layer_name]['emotional_state'] = emotional_value
                self.desire.store_emotional_memory(
                    layer_name, 
                    concept_id, 
                    self.state, 
                    emotional_value, 
                    0.0
                )
            
            # Update emotional memory
            category = self._get_emotional_category(emotional_value)
            self.emotional_memory[category][concept_id] = {
                'value': emotional_value,
                'timestamp': len(self.emotional_history),
                'concept': wrapped_concept.content
            }
        except Exception as e:
            print(f"Error in _batch_update_emotional_state: {str(e)} for concept_id: {concept_id if 'concept_id' in locals() else 'unknown'}")
            import traceback
            traceback.print_exc()

    def _get_default_concept_response(self, emotional_value=0.0):
        """Return default concept response when processing fails"""
        return {
            'activation': np.zeros((self.size, self.size)),
            'features': {
                'content': '',
                'emotional_value': emotional_value,
                'context': 'default'
            },
            'metrics': {
                'belief_consistency': 0.5,
                'rewards': np.zeros(3),
                'desire_strength': 0.0
            }
        }

    def _get_active_beliefs_batch(self, concept_id):
        """Get active beliefs using vectorized operations"""
        # Create belief matrix for faster lookup
        belief_matrix = {}
        for category, subcats in self.belief_systems.items():
            belief_matrix[category] = {}
            for subcat, beliefs in subcats.items():
                belief_matrix[category][subcat] = []
                for belief_key, belief_value in beliefs.items():
                    # Check if concept_id exists in beliefs
                    has_concept = str(concept_id) in str(belief_key)
                    belief_matrix[category][subcat].append(
                        (has_concept, beliefs.get(str(concept_id), 0.0))
                    )
        
        return belief_matrix

    def _calculate_concept_metrics(self, wrapped_concept, context, emotional_value, active_beliefs):
        """Calculate all concept metrics in parallel"""
        try:
            # Prepare arrays for vectorized calculations
            belief_consistency = self._check_belief_consistency(
                wrapped_concept.hashable_id,
                active_beliefs,
                abs(emotional_value)
            )
            
            # Calculate rewards in parallel with error prevention
            rewards = np.zeros(3)
            rewards[0] = self._calculate_novelty_reward(wrapped_concept.hashable_id)
            rewards[1] = self._calculate_familiarity_reward(wrapped_concept.hashable_id)
            
            # Handle potentially empty goals dictionary
            goal_penalty = 0.0
            if isinstance(context, dict) and 'goals' in context:
                goal_penalty = self._calculate_goal_alignment_penalty(
                    wrapped_concept.hashable_id, 
                    context.get('goals', {})
                )
            rewards[2] = -goal_penalty
            
            # Prevent NaN values
            rewards = np.nan_to_num(rewards, nan=0.0)
            
            # Fix desire strength calculation by providing required arguments
            desire_strength = 0.0
            if hasattr(self, 'desire'):
                default_context = {'layer': self.name, 'emotional_state': emotional_value}
                concept_context = context or default_context
                desire_strength = float(self.desire.get_desire_strength(wrapped_concept.hashable_id, concept_context))
            
            return {
                'belief_consistency': float(belief_consistency),
                'rewards': rewards,
                'desire_strength': desire_strength
            }
        except Exception as e:
            print(f"Error in _calculate_concept_metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'belief_consistency': 0.5,
                'rewards': np.zeros(3),
                'desire_strength': 0.0
            }

    def _get_emotional_category(self, emotional_value):
        """Determine emotional category from value"""
        try:
            if isinstance(emotional_value, (float, np.floating, int)):
                if emotional_value > 0.2:
                    return 'positive'
                elif emotional_value < -0.2:
                    return 'negative'
                else:
                    return 'neutral'
            elif isinstance(emotional_value, str) and emotional_value == 'ambivalent':
                return 'ambivalent'
            else:
                return 'neutral'
        except Exception as e:
            print(f"Error in _get_emotional_category: {str(e)}")
            return 'neutral'

    def _process_concept_influences(self, wrapped_concept, context, metrics, emotional_value):
        """Process philosophical and psychological influences on concept processing"""
        try:
            # Extract concept ID and ensure it's hashable
            concept_id = wrapped_concept.hashable_id
            
            # Calculate philosophical influences
            philosophical_influence = {
                'epistemological': self._evaluate_knowledge_validity(concept_id),
                'ontological': self._evaluate_concept_reality(concept_id),
                'axiological': self._evaluate_value_alignment(concept_id),
                'logical': self._evaluate_logical_consistency(concept_id)
            }
            
            # Calculate reward expectations
            reward_expectation = np.mean(metrics['rewards'])
            
            # Calculate contextual influence
            context_name = context if isinstance(context, str) else 'general'
            context_influence = 0.5  # Default value
            if context_name in self.belief_contexts:
                context_influence = np.mean(self.belief_contexts[context_name])
            
            return {
                'philosophical': philosophical_influence,
                'reward': reward_expectation,
                'context': context_influence,
                'emotional': emotional_value
            }
            
        except Exception as e:
            print(f"Error in _process_concept_influences: {str(e)}")
            return {
                'philosophical': {'epistemological': 0.3, 'ontological': 0.3, 'axiological': 0.3, 'logical': 0.3},
                'reward': 0.0,
                'context': 0.5,
                'emotional': 0.0
            }

    def _create_final_response(self, wrapped_concept, concept_pattern, metrics, influences, context):
        """Create final response with all processed components"""
        try:
            # Create activation pattern
            final_activation = self._create_integrated_response(
                concept_pattern,
                context if isinstance(context, str) else 'general',
                influences['emotional'],
                metrics['belief_consistency'],
                influences['reward'],
                np.mean([v for v in influences['philosophical'].values()]),
                influences['context']
            )
            
            # Return structured response
            return {
                'activation': final_activation,
                'features': {
                    'content': wrapped_concept.content,
                    'emotional_value': influences['emotional'],
                    'context': context if isinstance(context, str) else 'general'
                },
                'metrics': {
                    'belief_consistency': metrics['belief_consistency'],
                    'rewards': metrics['rewards'],
                    'philosophical': influences['philosophical']
                }
            }
            
        except Exception as e:
            print(f"Error in _create_final_response: {str(e)}")
            return self._get_default_concept_response(influences['emotional'])

    def _update_concept_relationships(self, wrapped_concept, context):
        """Update concept relationships in semantic memory"""
        try:
            # Get concept ID and ensure it's hashable
            concept_id = wrapped_concept.hashable_id
            
            # Update familiarity score
            if concept_id not in self.familiarity_scores:
                self.familiarity_scores[concept_id] = 0.0
            self.familiarity_scores[concept_id] = min(1.0, self.familiarity_scores[concept_id] + 0.1)
            
            # Update relationships
            context_name = context if isinstance(context, str) else 'general'
            
            # Initialize relationships if needed
            if concept_id not in self.semantic_memory['relationships']:
                self.semantic_memory['relationships'][concept_id] = []
            
            # Add context relationship
            if context_name not in self.semantic_memory['relationships'][concept_id]:
                self.semantic_memory['relationships'][concept_id].append(context_name)
            
            # Update hierarchies
            if context_name not in self.semantic_memory['hierarchies']:
                self.semantic_memory['hierarchies'][context_name] = set()
            self.semantic_memory['hierarchies'][context_name].add(concept_id)
            
        except Exception as e:
            print(f"Error in _update_concept_relationships: {str(e)} for concept_id: {wrapped_concept.hashable_id if hasattr(wrapped_concept, 'hashable_id') else 'unknown'}")
            import traceback
            traceback.print_exc()