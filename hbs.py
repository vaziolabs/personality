import numpy as np
from cs import ConsciousnessSystem
from numba import jit
from collections import defaultdict
from learning import LearningContext

class SleepCycle:
    def __init__(self):
        self.sleep_pressure = 0
        self.circadian_phase = 0
        self.rem_cycle = 0
        
    def update(self, hour, is_asleep):
        # Update circadian rhythm (24-hour cycle)
        self.circadian_phase = np.sin(2 * np.pi * hour / 24)
        
        if is_asleep:
            # Sleep pressure decreases during sleep
            self.sleep_pressure = max(0, self.sleep_pressure - 0.3)
            # REM cycles occur roughly every 90 minutes during sleep
            self.rem_cycle = np.sin(2 * np.pi * hour / 1.5)
        else:
            # Sleep pressure builds during wakefulness
            self.sleep_pressure = min(10, self.sleep_pressure + 0.1)
            self.rem_cycle = 0
            
        return self.get_state()
    
    def get_state(self):
        return {
            'pressure': self.sleep_pressure,
            'circadian': self.circadian_phase,
            'rem': self.rem_cycle
        }

class HumanBehaviorSystem:
    def __init__(self):
        # Base parameters
        self.energy = 50.0  # Initialize as float
        self.responsiveness = 0.3
        self.resistance = 0.2
        self.recovery_rate = 0.1
        
        # Memory system
        self.memory = np.zeros(5, dtype=np.float32)
        self.memory_ptr = 0
        self.memory_influence = 0.15
        
        # History tracking
        self.energy_history = np.zeros(1000, dtype=np.float32)
        self.history_ptr = 0
        
        # Enhanced consciousness system
        self.consciousness = ConsciousnessSystem(size=5)
        self.emotional_state = 0.0  # Track emotional state
        self.adaptation_rate = 0.1  # How quickly we adapt to experiences
        
        # Experience tracking
        self.experience_buffer = []
        self.max_experiences = 100
        
        # Add learning context initialization
        self.learning_context = LearningContext()
        
        # Add learning state tracking
        self.learning_state = {
            'active_skills': set(),
            'skill_patterns': defaultdict(list),
            'learning_momentum': 0.0
        }
        
        # Add core drives and motivations
        self.drives = {
            'survival': 0.5,    # Basic needs (food, sleep, safety)
            'social': 0.5,      # Connection, belonging
            'mastery': 0.5,     # Competence, achievement
            'autonomy': 0.5,    # Independence, control
            'purpose': 0.5      # Meaning, goals
        }
        
        # Add personality traits that influence behavior
        self.personality = {
            'openness': 0.5,        # Curiosity and creativity
            'conscientiousness': 0.5,  # Organization and responsibility
            'extraversion': 0.5,    # Social energy and assertiveness
            'agreeableness': 0.5,   # Cooperation and empathy
            'neuroticism': 0.5      # Emotional sensitivity and anxiety
        }
        
        # Personality influence weights
        self.personality_weights = {
            'decision_making': 0.3,
            'emotional_response': 0.4,
            'social_interaction': 0.5,
            'learning_rate': 0.2,
            'stress_tolerance': 0.3
        }
        
        # Add emotional memory system
        self.emotional_memory = {
            'positive': defaultdict(list),  # Success experiences
            'negative': defaultdict(list),  # Failure/threat experiences
            'neutral': defaultdict(list)    # Routine experiences
        }
        self.emotional_memory_capacity = 100  # Per category
        self.emotional_threshold = 0.3  # Threshold for emotional significance
        
        # Add context tracking
        self.current_context = {
            'time': 0,
            'location': 'default',
            'activity': 'none',
            'social': [],
            'environmental': {},
            'internal_state': {}
        }
        self.context_history = []
        self.context_associations = defaultdict(dict)
        
        # Add layered desire tracking
        self.desire_layers = {
            'conscious': {
                'goals': defaultdict(float),      # Explicit objectives
                'values': defaultdict(float),     # Personal values
                'intentions': defaultdict(float)  # Planned actions
            },
            'subconscious': {
                'emotional_needs': defaultdict(float),  # Emotional desires
                'social_needs': defaultdict(float),     # Social validation
                'habits': defaultdict(float)           # Learned patterns
            },
            'unconscious': {
                'survival_drives': defaultdict(float),  # Basic needs
                'safety_needs': defaultdict(float),     # Security
                'primal_urges': defaultdict(float)     # Deep instincts
            }
        }
        
    def respond_to_stimulus(self, stimulus_strength):
        # Get current context relationships
        related_contexts = self._get_related_contexts()
        
        # Get current motivational state
        drive_state = self._evaluate_drive_states()
        
        # Create richer context including personality and drives
        context = {
            'drives': drive_state,
            'personality': self.personality,
            'emotional_state': self.emotional_state,
            'energy': self.energy
        }
        
        # Process through consciousness with enhanced context
        consciousness_response = self.consciousness.process_impulse(
            self._create_stimulus_pattern(stimulus_strength),
            context=context,
            related_contexts=related_contexts,
            emotional_value=self.emotional_state
        )
        
        # Update drives based on response
        self._update_drives(consciousness_response)
        
        # Update emotional memory
        self._store_emotional_memory(stimulus_strength, consciousness_response)
        
        return self.energy, {
            'response': consciousness_response,
            'emotional_state': self.emotional_state,
            'drives': self.drives,
            'personality_influence': self._get_personality_influence()
        }

    def _get_desire_states(self):
        """Get current desire states from all layers"""
        return {
            'conscious': self.consciousness.conscious.desire.desire_levels,
            'subconscious': self.consciousness.subconscious.desire.desire_levels,
            'unconscious': self.consciousness.unconscious.desire.desire_levels
        }

    def _get_predictions(self):
        """Get current predictions from all layers"""
        return {
            'conscious': self.consciousness.conscious.prediction.predict_outcome,
            'subconscious': self.consciousness.subconscious.prediction.predict_outcome,
            'unconscious': self.consciousness.unconscious.prediction.predict_outcome
        }
    
    def _calculate_emotional_impact(self, stimulus_strength):
        """Calculate emotional impact based on stimulus and current state"""
        base_impact = stimulus_strength / 100.0
        energy_state = (self.energy - 50) / 50.0  # Normalize energy level
        return base_impact * (1 + abs(energy_state))
    
    def _create_stimulus_pattern(self, stimulus_strength):
        """Create a rich stimulus pattern with context"""
        pattern = np.zeros((5, 5), dtype=np.float32)
        
        # Center represents current stimulus
        pattern[2, 2] = stimulus_strength / 100.0
        
        # Corners represent context from memory
        pattern[0, 0] = self.memory[self.memory_ptr-1] / 100.0
        pattern[0, 4] = self.emotional_state
        pattern[4, 0] = self.energy / 100.0
        pattern[4, 4] = self.responsiveness
        
        return pattern
    
    def _store_experience(self, stimulus, response):
        """Store and learn from experiences"""
        experience = {
            'stimulus': stimulus,
            'response': response,
            'energy': self.energy,
            'emotional_state': self.emotional_state,
            'timestamp': self.history_ptr
        }
        
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_experiences:
            self.experience_buffer.pop(0)
            
        # Adapt parameters based on experience
        self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adapt system parameters based on recent experiences"""
        if len(self.experience_buffer) > 10:
            recent_experiences = self.experience_buffer[-10:]
            
            # Calculate average emotional impact
            avg_emotional_impact = np.mean([exp['emotional_state'] 
                                          for exp in recent_experiences])
            
            # Adapt responsiveness
            self.responsiveness += self.adaptation_rate * (
                avg_emotional_impact - self.responsiveness
            )
            self.responsiveness = np.clip(self.responsiveness, 0.1, 0.5)
            
            # Adapt resistance based on energy stability
            energy_variance = np.var([exp['energy'] for exp in recent_experiences])
            if energy_variance > 400:  # High variance
                self.resistance += self.adaptation_rate * 0.1
            else:
                self.resistance -= self.adaptation_rate * 0.1
            self.resistance = np.clip(self.resistance, 0.1, 0.4)
    
    def _calculate_consciousness_influence(self, consciousness_response, stimulus_strength):
        """Calculate consciousness influence with pattern recognition"""
        base_influence = float(np.mean(consciousness_response))
        emotional_factor = np.exp(abs(self.emotional_state))
        pattern_factor = 1.0
        
        if len(self.experience_buffer) > 0:
            # Look for similar patterns in recent experiences
            current_pattern = self._create_stimulus_pattern(stimulus_strength)
            similar_experiences = [
                exp for exp in self.experience_buffer[-10:]
                if np.mean(abs(exp['response'] - current_pattern)) < 0.3
            ]
            if similar_experiences:
                pattern_factor = 1.2  # Boost influence for recognized patterns
        
        return base_influence * emotional_factor * pattern_factor * self.responsiveness
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_response(energy, memory, responsiveness, resistance, 
                          memory_influence, recovery_rate, stimulus_strength, 
                          consciousness_influence):
        memory_effect = np.mean(memory) * memory_influence
        raw_response = stimulus_strength * responsiveness
        dampened_response = raw_response * (1 - resistance * abs(raw_response))
        
        new_energy = (energy + dampened_response + memory_effect + 
                     consciousness_influence)
        
        distance_from_baseline = 50 - new_energy
        new_energy += distance_from_baseline * recovery_rate
        
        # Replace np.clip with manual min/max operations for Numba compatibility
        if new_energy < 0:
            return 0.0
        elif new_energy > 100:
            return 100.0
    
    def _create_learning_pattern(self):
        """Create a pattern representing current learning state"""
        pattern = np.zeros((5, 5), dtype=np.float32)
        
        # Center represents current learning momentum
        pattern[2, 2] = self.learning_state['learning_momentum']
        
        # Corners represent active skills and their mastery
        for i, skill in enumerate(self.learning_state['active_skills']):
            if i < 4 and skill in self.learning_context.skills:  # Add safety check
                row, col = [(0,0), (0,4), (4,0), (4,4)][i]
                pattern[row, col] = self.learning_context.skills[skill]['mastery']
        
        return pattern
    
    def process_rest_period(self, duration=1.0):
        """Process rest period with consciousness integration"""
        consciousness_state = self.consciousness.process_rest_state()
        
        # Create consolidated state for learning
        rest_state = {
            'unconscious': consciousness_state['unconscious'],
            'consolidation_factor': duration,
            'emotional_state': self.emotional_state,
            'patterns': self.consciousness.thought_paths[self.consciousness.path_index-10:self.consciousness.path_index]
        }
        
        # Update learning context
        self.learning_context.process_rest_period(rest_state)
        
        return {
            'memory_consolidation': np.mean([
                skill['consolidation'] 
                for skill in self.learning_context.skills.values()
            ])
        }

    def _evaluate_drive_states(self):
        """Evaluate current state of all drives"""
        # Update drive states based on current system state
        self.drives['survival'] = self._calculate_survival_drive()
        self.drives['social'] = self._calculate_social_drive()
        self.drives['mastery'] = self._calculate_mastery_drive()
        self.drives['autonomy'] = self._calculate_autonomy_drive()
        self.drives['purpose'] = self._calculate_purpose_drive()
        
        return self.drives
        
    def _calculate_survival_drive(self):
        """Calculate survival drive based on energy and basic needs"""
        energy_factor = 1.0 - (self.energy / 100.0)  # Higher when energy is low
        return np.clip(
            0.5 + (energy_factor * 0.5) + 
            (self.consciousness.unconscious.primal_drives['safety'] * 0.3),
            0, 1
        )
        
    def _calculate_social_drive(self):
        """Calculate social drive based on emotional state and experiences"""
        social_experiences = [
            exp['emotional_state'] 
            for exp in self.experience_buffer 
            if 'social' in exp.get('context', {})
        ]
        social_satisfaction = np.mean(social_experiences) if social_experiences else 0.5
        return np.clip(
            0.5 + (self.emotional_state * 0.3) - (social_satisfaction * 0.2),
            0, 1
        )
        
    def _calculate_mastery_drive(self):
        """Calculate mastery drive based on learning progress"""
        if not self.learning_state['active_skills']:
            return self.drives['mastery']
            
        recent_improvements = []
        for skill in self.learning_state['active_skills']:
            if skill in self.learning_context.skills:
                improvements = self.learning_context.skills[skill]['recent_improvements']
                if improvements:
                    recent_improvements.extend(improvements)
                    
        progress_factor = np.mean(recent_improvements) if recent_improvements else 0
        return np.clip(
            self.drives['mastery'] + (progress_factor * 0.3) - 
            (self.learning_state['learning_momentum'] * 0.2),
            0, 1
        )
        
    def _calculate_autonomy_drive(self):
        """Calculate autonomy drive based on control and independence"""
        control_factor = self.responsiveness / (self.resistance + 0.1)
        return np.clip(
            0.5 + (control_factor * 0.3) - 
            (abs(self.emotional_state) * 0.2),
            0, 1
        )
        
    def _calculate_purpose_drive(self):
        """Calculate purpose drive based on goal progress and meaning"""
        # Use consciousness system's belief contexts to evaluate purpose
        belief_strength = np.mean([
            np.mean(layer.belief_contexts['purpose'])
            for layer in [
                self.consciousness.conscious,
                self.consciousness.subconscious,
                self.consciousness.unconscious
            ]
        ])
        return np.clip(
            self.drives['purpose'] + (belief_strength * 0.3) - 
            (self.emotional_state * 0.2),
            0, 1
        )

    def _update_drives(self, response):
        """Update drive states based on response and outcomes"""
        # Calculate response impact
        response_strength = np.mean(response)
        
        # Update survival drive
        energy_cost = response_strength * 0.1
        self.energy = max(0, min(100, self.energy - energy_cost))
        self.drives['survival'] = self._calculate_survival_drive()
        
        # Update mastery drive based on learning outcomes
        if self.learning_state['active_skills']:
            learning_progress = sum(
                self.learning_context.skills[skill]['recent_improvements'][-1]
                for skill in self.learning_state['active_skills']
                if skill in self.learning_context.skills
                and self.learning_context.skills[skill]['recent_improvements']
            ) / len(self.learning_state['active_skills'])
            
            self.drives['mastery'] = np.clip(
                self.drives['mastery'] + learning_progress * 0.2,
                0, 1
            )
        
        # Update social drive based on interaction outcome
        social_impact = response_strength * (1 + self.emotional_state)
        self.drives['social'] = np.clip(
            self.drives['social'] + social_impact * 0.1,
            0, 1
        )
        
        # Update autonomy drive based on control
        control_factor = self.responsiveness / (self.resistance + 0.1)
        self.drives['autonomy'] = np.clip(
            self.drives['autonomy'] + (control_factor - 0.5) * 0.1,
            0, 1
        )
        
        # Update purpose drive based on goal alignment
        purpose_alignment = np.mean([
            np.mean(layer.belief_contexts['purpose'])
            for layer in [
                self.consciousness.conscious,
                self.consciousness.subconscious,
                self.consciousness.unconscious
            ]
        ])
        self.drives['purpose'] = np.clip(
            self.drives['purpose'] + (purpose_alignment - 0.5) * 0.1,
            0, 1
        )
        
        # Natural decay of all drives
        for drive in self.drives:
            self.drives[drive] *= 0.95  # 5% decay per update

    def _store_emotional_memory(self, stimulus, response):
        """Store experience in emotional memory with context"""
        # Calculate emotional impact
        emotional_impact = self._calculate_emotional_impact(stimulus)
        
        # Create memory entry
        memory = {
            'stimulus': stimulus,
            'response': response,
            'emotional_state': self.emotional_state,
            'energy': self.energy,
            'context': self._get_current_context(),
            'timestamp': self.history_ptr,
            'impact': emotional_impact
        }
        
        # Determine memory category
        if abs(emotional_impact) < self.emotional_threshold:
            category = 'neutral'
        else:
            category = 'positive' if emotional_impact > 0 else 'negative'
            
        # Store memory with capacity management
        memory_key = str(hash(str(stimulus)))
        self.emotional_memory[category][memory_key].append(memory)
        
        # Maintain capacity limits
        if len(self.emotional_memory[category][memory_key]) > self.emotional_memory_capacity:
            self.emotional_memory[category][memory_key].pop(0)
        
        # Update emotional state based on memory storage
        self.emotional_state = self.emotional_state * 0.9 + emotional_impact * 0.1

    def _get_emotional_memory_influence(self, stimulus):
        """Calculate influence of emotional memories on current situation"""
        memory_key = str(hash(str(stimulus)))
        total_influence = 0.0
        count = 0
        
        # Weight recent memories more heavily
        for category in ['positive', 'negative', 'neutral']:
            memories = self.emotional_memory[category][memory_key]
            for i, memory in enumerate(memories):
                recency_weight = np.exp(-0.1 * (self.history_ptr - memory['timestamp']))
                total_influence += memory['impact'] * recency_weight
                count += 1
                
        return total_influence / max(1, count)

    def _get_current_context(self):
        """Get current context with all relevant state information"""
        return {
            'time': self.current_context['time'],
            'location': self.current_context['location'],
            'activity': self.current_context['activity'],
            'social': self.current_context['social'].copy(),
            'environmental': self.current_context['environmental'].copy(),
            'internal_state': {
                'energy': self.energy,
                'emotional_state': self.emotional_state,
                'drives': self.drives.copy(),
                'active_skills': self.learning_state['active_skills'].copy() if 'active_skills' in self.learning_state else [],
                'responsiveness': self.responsiveness,
                'resistance': self.resistance
            }
        }
        
    def _get_related_contexts(self):
        """Get contexts related to current context"""
        related = {}
        current_key = self._context_to_key(self.current_context)
        
        for context_key, associations in self.context_associations.items():
            if context_key != current_key and associations:
                similarity = self._calculate_context_similarity(
                    self.current_context,
                    self._key_to_context(context_key)
                )
                if similarity > 0.3:  # Threshold for relatedness
                    related[context_key] = similarity
                    
        return related
        
    def _context_to_key(self, context):
        """Convert context dict to string key"""
        return f"{context['location']}:{context['activity']}"
        
    def _key_to_context(self, key):
        """Convert string key back to basic context"""
        location, activity = key.split(':')
        return {
            'location': location,
            'activity': activity,
            'time': 0,  # Default values
            'social': [],
            'environmental': {},
            'internal_state': {}
        }
        
    def _calculate_context_similarity(self, context1, context2):
        """Calculate similarity between two contexts"""
        # Location and activity exact match
        base_similarity = float(
            context1['location'] == context2['location'] and
            context1['activity'] == context2['activity']
        )
        
        # Social context overlap
        social_overlap = len(
            set(context1['social']) & 
            set(context2['social'])
        ) / max(1, len(set(context1['social']) | set(context2['social'])))
        
        # Environmental factors similarity
        env_similarity = self._calculate_dict_similarity(
            context1['environmental'],
            context2['environmental']
        )
        
        return (base_similarity * 0.4 + 
                social_overlap * 0.3 + 
                env_similarity * 0.3)
                
    def _calculate_dict_similarity(self, dict1, dict2):
        """Calculate similarity between two dictionaries"""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        if not all_keys:
            return 1.0
            
        matches = sum(
            1 for k in all_keys
            if k in dict1 and k in dict2 and dict1[k] == dict2[k]
        )
        return matches / len(all_keys)

    def _get_personality_influence(self):
        """Calculate current personality influence on behavior"""
        influences = {
            'decision_making': self._calculate_decision_influence(),
            'emotional_response': self._calculate_emotional_influence(),
            'social_interaction': self._calculate_social_influence(),
            'learning_rate': self._calculate_learning_influence(),
            'stress_tolerance': self._calculate_stress_influence()
        }
        
        return influences
        
    def _calculate_decision_influence(self):
        """Calculate personality influence on decision making"""
        return (
            self.personality['conscientiousness'] * 0.4 +
            self.personality['openness'] * 0.3 +
            (1 - self.personality['neuroticism']) * 0.3
        )
        
    def _calculate_emotional_influence(self):
        """Calculate personality influence on emotional responses"""
        return (
            self.personality['neuroticism'] * 0.5 +
            self.personality['extraversion'] * 0.3 +
            self.personality['agreeableness'] * 0.2
        )
        
    def _calculate_social_influence(self):
        """Calculate personality influence on social interactions"""
        return (
            self.personality['extraversion'] * 0.4 +
            self.personality['agreeableness'] * 0.4 +
            self.personality['openness'] * 0.2
        )
        
    def _calculate_learning_influence(self):
        """Calculate personality influence on learning"""
        return (
            self.personality['openness'] * 0.4 +
            self.personality['conscientiousness'] * 0.4 +
            (1 - self.personality['neuroticism']) * 0.2
        )
        
    def _calculate_stress_influence(self):
        """Calculate personality influence on stress handling"""
        return (
            (1 - self.personality['neuroticism']) * 0.5 +
            self.personality['conscientiousness'] * 0.3 +
            self.personality['extraversion'] * 0.2
        )

    def _resolve_desire_conflicts(self):
        conscious_desires = self.consciousness.conscious.desire.desire_levels
        subconscious_desires = self.consciousness.subconscious.desire.desire_levels
        unconscious_desires = self.consciousness.unconscious.desire.desire_levels
        
        dissonance = self._calculate_desire_dissonance(
            conscious_desires,
            subconscious_desires,
            unconscious_desires
        )
        
        return self._adjust_behavior_for_dissonance(dissonance)

    def _evaluate_desire_conflicts(self):
        """Evaluate conflicts between different layers of desires"""
        conflicts = {
            'conscious_sub': self._calculate_layer_conflict('conscious', 'subconscious'),
            'conscious_uncon': self._calculate_layer_conflict('conscious', 'unconscious'),
            'sub_uncon': self._calculate_layer_conflict('subconscious', 'unconscious')
        }
        
        # Calculate internal tension
        tension = sum(conflicts.values()) / len(conflicts)
        self.emotional_state = self.emotional_state * 0.8 + tension * 0.2

    def _resolve_cognitive_dissonance(self):
        """Handle conflicts between beliefs and behaviors"""
        # Identify conflicting desires/beliefs
        conflicts = self._identify_conflicts()
        
        # Calculate dissonance pressure
        pressure = sum(
            conflict['strength'] * conflict['importance']
            for conflict in conflicts
        )
        
        # Attempt resolution strategies
        resolutions = {
            'belief_change': self._try_belief_adjustment(conflicts),
            'behavior_change': self._try_behavior_adjustment(conflicts),
            'rationalization': self._try_rationalization(conflicts)
        }
        
        # Update internal state based on resolution success
        resolution_success = max(resolutions.values())
        self.emotional_state = self.emotional_state * 0.9 + (pressure - resolution_success) * 0.1

    def process_text_knowledge(self, knowledge_item):
        """Process text knowledge across consciousness levels"""
        try:
            # Extract concepts and relationships
            concepts = self._extract_concepts(knowledge_item['content'])
            relationships = self._extract_relationships(
                knowledge_item['content'],
                knowledge_item.get('links', {})
            )
            
            # Process through consciousness system
            consciousness_response = self.consciousness.process_text_input(
                knowledge_item['content'],
                context={'concepts': concepts, 'relationships': relationships}
            )
            
            # Initialize learning metrics with safe defaults
            learning_results = {
                'depth': 0.0,
                'breadth': 0.0,
                'cognitive_load': 0.0,  # Changed from 'load'
                'understanding': 0.0
            }
            
            # Calculate metrics safely
            if concepts and relationships:
                learning_results.update({
                    'depth': float(len(relationships)) / max(len(concepts), 1),
                    'breadth': float(len(concepts)) / 100.0,
                    'cognitive_load': float(consciousness_response.get('dissonance', {}).get('total', 0.0)),
                    'understanding': float(consciousness_response.get('weights', {}).get('conscious', 0.0))
                })
            
            # Process through learning layers
            if hasattr(self, 'learning_context'):
                self.learning_context.learning_layers['conscious']['active_concepts'].update(
                    {concept: 1.0 for concept in concepts}
                )
                self._process_semantic_associations(knowledge_item)
                self._integrate_deep_patterns(knowledge_item)
                self._update_knowledge_desires(concepts)
            
            return learning_results
            
        except Exception as e:
            print(f"Error in process_text_knowledge: {str(e)}")
            return {
                'depth': 0.0,
                'breadth': 0.0,
                'cognitive_load': 0.0,  # Changed from 'load'
                'understanding': 0.0
            }

    def _extract_concepts(self, text):
        """Extract key concepts from text content"""
        # Initialize NLTK if not already done
        try:
            from nltk import word_tokenize, pos_tag
            from nltk.corpus import stopwords
        except:
            import nltk
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('stopwords')
            from nltk import word_tokenize, pos_tag
            from nltk.corpus import stopwords
        
        # Tokenize and tag parts of speech
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        
        # Extract nouns and important concepts
        stop_words = set(stopwords.words('english'))
        concepts = []
        
        for word, tag in tagged:
            if (tag.startswith(('NN', 'VB', 'JJ')) and 
                word not in stop_words and 
                len(word) > 2):
                concepts.append(word)
        
        return list(set(concepts))  # Remove duplicates

    def _process_semantic_associations(self, knowledge_item):
        """Process semantic associations from text knowledge"""
        # Extract relationships from content and links
        relationships = self._extract_relationships(
            knowledge_item['content'],
            knowledge_item.get('links', {})
        )
        
        # Update subconscious learning layers
        for source, targets in relationships.items():
            self.learning_context.learning_layers['subconscious']['semantic_associations'][source].extend(targets)
            
            # Update pattern recognition weights
            pattern_strength = len(targets) / 10.0  # Normalize by max expected connections
            self.learning_context.learning_layers['subconscious']['pattern_recognition'][source] += pattern_strength

    def _extract_relationships(self, content, links):
        """Extract semantic relationships from text content and links"""
        relationships = defaultdict(list)
        
        # Process main content for co-occurrence relationships
        words = content.lower().split()
        window_size = 5
        
        for i in range(len(words)):
            window = words[max(0, i-window_size):min(len(words), i+window_size)]
            for word in window:
                if word != words[i]:
                    relationships[words[i]].append(word)
        
        # Process links for hierarchical relationships
        for link, summary in links.items():
            link_words = link.lower().split()
            summary_words = summary.lower().split()
            
            for word in link_words:
                relationships[word].extend(summary_words)
        
        return relationships

    def _integrate_deep_patterns(self, knowledge_item):
        """Integrate knowledge into unconscious patterns"""
        # Extract core concepts and relationships
        concepts = self._extract_concepts(knowledge_item['content'])
        relationships = self._extract_relationships(
            knowledge_item['content'],
            knowledge_item.get('links', {})
        )
        
        # Update unconscious learning layers
        for concept in concepts:
            # Update deep abstractions based on concept frequency
            self.learning_context.learning_layers['unconscious']['deep_abstractions'][concept] += 0.1
            
            # Update intuitive models based on relationships
            if concept in relationships:
                self.learning_context.learning_layers['unconscious']['intuitive_models'][concept].extend(
                    relationships[concept]
                )

    def _update_knowledge_desires(self, concepts):
        """Update knowledge-related desires based on learned concepts"""
        # Calculate base desire updates
        understanding_increase = len(concepts) * 0.05
        mastery_increase = len(concepts) * 0.03
        curiosity_boost = len(concepts) * 0.08
        
        # Apply personality modifiers
        understanding_mod = 1.0 + self.personality['openness'] * 0.5
        mastery_mod = 1.0 + self.personality['conscientiousness'] * 0.5
        curiosity_mod = 1.0 + (self.personality['openness'] + 
                              self.personality['extraversion']) * 0.25
        
        # Update knowledge desires with personality influence
        for concept in concepts:
            self.consciousness.desire.knowledge_desires['understanding'][concept] += (
                understanding_increase * understanding_mod
            )
            self.consciousness.desire.knowledge_desires['mastery'][concept] += (
                mastery_increase * mastery_mod
            )
            self.consciousness.desire.knowledge_desires['curiosity'][concept] += (
                curiosity_boost * curiosity_mod
            )
            
            # Decay older knowledge desires slightly
            for desire_type in self.consciousness.desire.knowledge_desires:
                for existing_concept in self.consciousness.desire.knowledge_desires[desire_type]:
                    if existing_concept != concept:
                        self.consciousness.desire.knowledge_desires[desire_type][existing_concept] *= 0.95
