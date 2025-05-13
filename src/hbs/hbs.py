import numpy as np
from hbs.consciousness.cs import ConsciousnessSystem
from numba import jit
from collections import defaultdict
from hbs.consciousness.learning import LearningContext, TextLearningContext
from concurrent.futures import ThreadPoolExecutor
from hbs.consciousness.concept import ConceptWrapper
from hbs.behaviours.imagine import Imagination
from hbs.behaviours.perception import Perception
from hbs.consciousness.cohesion import ThoughtProcess
from hbs.interaction import HumanBehaviorInterface
from hbs.behaviours.impulse import ImpulseSystem

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
        # Initialize core systems
        self.consciousness = ConsciousnessSystem(size=256)
        self.perception = Perception(self.consciousness)
        self.thought_process = ThoughtProcess(self.consciousness)
        self.imagination = Imagination(self.consciousness)
        self.interface = HumanBehaviorInterface(
            self.consciousness,
            self.consciousness.belief_system,
            self.consciousness.desire_system
        )
        
        # Initialize learning contexts
        self.learning_context = LearningContext()
        self.text_learning_context = TextLearningContext()
        
        # Base parameters
        self.energy = 50.0  # Initialize as float
        self.responsiveness = 0.3
        self.resistance = 0.2
        self.recovery_rate = 0.1
        
        # Memory system
        self.memory = np.zeros(64, dtype=np.float32)
        self.memory_ptr = 0
        self.memory_influence = 0.15
        
        # History tracking
        self.energy_history = np.zeros(100, dtype=np.float32)
        self.history_ptr = 0
        
        # Enhanced consciousness system
        self.emotional_state = 0.0  # Track emotional state
        self.adaptation_rate = 0.1  # How quickly we adapt to experiences
        
        # Experience tracking
        self.experience_buffer = []
        self.max_experiences = 10000
        
        # Add learning context initialization
        self.learning_context = LearningContext()
        
        # Add learning state tracking
        # TODO: would be nice to have 
        #           'information_gather',   # Taking in New Information         - Learning by exposure to new information
        #           'skills_practice',      # Practicing Skills                 - Learning by reinforcing knowledge with experience
        #           'memory_recall',        # Reviewing Memorized Information   - Learning by recalling information and testing correctness
        #           'memory_consolidation', # Consolidating Memory              - Learning by consolidating memory and evaluating relationships
        #           'creative_thinking',    # Generating New Ideas              - Learning by generating new ideas
        #           'problem_solving',      # Solving Problems                  - Learning by testing hypothesis with uncertainty
        #           'decision_making',      # Making Decisions                  - Learning by making pre-emptive choices with certainty
        #           'social_interaction',   # Interacting with Others           - Different type of learning
        #           'emotional_response',   # Responding to Emotions            - Learning from impulse decision making
        #           'stress_tolerance'      # Stress Tolerance                  - Inability to learn creates new stress thresholds
        self.learning_state = {
            'active_skills': set(),
            'skill_patterns': defaultdict(list),
            'learning_momentum': 0.0
        }
        
        # Initialize core drives
        self.drives = {
            'survival': 0.0,
            'social': 0.0,
            'mastery': 0.0,
            'autonomy': 0.0,
            'purpose': 0.0
        }
        
        # Add drive-specific learning tracking
        self.drive_learning = {
            'survival': defaultdict(float),
            'social': defaultdict(float),
            'mastery': defaultdict(float),
            'autonomy': defaultdict(float),
            'purpose': defaultdict(float)
        }
        
        # Add personality traits that influence behavior
        self.personality = {
            'openness': 0.5,                # Curiosity and creativity
            'conscientiousness': 0.5,       # Organization and responsibility
            'extraversion': 0.5,            # Social energy and assertiveness
            'agreeableness': 0.5,           # Cooperation and empathy
            'neuroticism': 0.5              # Emotional sensitivity and anxiety
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
            'neutral': defaultdict(list),   # Routine experiences
            'ambivalent': defaultdict(list) # Mixed experiences
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
        
        self.goals = []
        self.reward_history = []
        
        # Add missing emotional state propagation
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
        
        # Update emotional value handling
        emotional_value = self._get_normalized_emotional_value()
        
        # Process through consciousness with enhanced context
        consciousness_response = self.consciousness.process_impulse(
            self._create_stimulus_pattern(stimulus_strength),
            context=context,
            related_contexts=related_contexts,
            emotional_value=emotional_value
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
        emotional_value = self._get_normalized_emotional_value()
        return base_impact * (1 + abs(energy_state)) * (1 + abs(emotional_value))
    
    def _create_stimulus_pattern(self, stimulus_strength):
        """Create a rich stimulus pattern with context"""
        pattern = np.zeros((5, 5), dtype=np.float32)
        
        # Convert emotional state to numerical value
        emotional_value = self._get_normalized_emotional_value()
        
        # Center represents current stimulus
        pattern[2, 2] = stimulus_strength / 100.0
        
        # Corners represent context from memory
        pattern[0, 0] = self.memory[self.memory_ptr-1] / 100.0
        pattern[0, 4] = emotional_value
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
        """Process imagination and reflection during quiet periods"""
        # Use consciousness system's imagination process
        results = self.consciousness.process_rest_period(duration)
        
        # Update learning context based on discoveries
        if results['discoveries']:
            self.learning_context.process_rest_period({
                'duration': duration,
                'emotional_state': self._get_normalized_emotional_value(),
                'discoveries': results['discoveries']
            })
        
        return results

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
            exp['emotional_state'] if not isinstance(exp['emotional_state'], str)
            else self._get_normalized_emotional_value()
            for exp in self.experience_buffer 
            if 'social' in exp.get('context', {})
        ]
        social_satisfaction = np.mean(social_experiences) if social_experiences else 0.5
        emotional_value = self._get_normalized_emotional_value()
        return np.clip(
            0.5 + (emotional_value * 0.3) - (social_satisfaction * 0.2),
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
        emotional_value = self._get_normalized_emotional_value()
        return np.clip(
            0.5 + (control_factor * 0.3) - 
            (abs(emotional_value) * 0.2),
            0, 1
        )
        
    def _calculate_purpose_drive(self):
        """Calculate purpose drive based on goal progress and meaning"""
        belief_strength = np.mean([
            np.mean(layer.belief_contexts['purpose'])
            for layer in [
                self.consciousness.conscious,
                self.consciousness.subconscious,
                self.consciousness.unconscious
            ]
        ])
        emotional_value = self._get_normalized_emotional_value()
        return np.clip(
            self.drives['purpose'] + (belief_strength * 0.3) - 
            (emotional_value * 0.2),
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
            'emotional_state': (
                0.5 * np.sin(len(self.emotional_memory.get('neutral', {})) * 0.1)
                if isinstance(self.emotional_state, str) and self.emotional_state == 'ambivalent'
                else self.emotional_state
            ),
            'energy': self.energy,
            'context': self._get_current_context(),
            'timestamp': self.history_ptr,
            'impact': emotional_impact
        }
        
        # Determine category including ambivalent
        if isinstance(self.emotional_state, str) and self.emotional_state == 'ambivalent':
            category = 'ambivalent'
        elif abs(emotional_impact) < self.emotional_threshold:
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
        emotional_value = self._get_normalized_emotional_value()
        self.emotional_state = emotional_value * 0.9 + (pressure - resolution_success) * 0.1

    def process_text_knowledge(self, knowledge_item):
        """Process text-based knowledge with integrated consciousness"""
        try:
            # Extract concepts from knowledge
            concepts = self._extract_concepts(knowledge_item['content'])
            
            # Process through learning layers
            learning_results = self._process_learning_layers(
                concepts,
                knowledge_item
            )
            
            # Update learning context with new knowledge
            self.learning_context.update_semantic_memory(
                concepts,
                knowledge_item['content'],
                learning_results['depth']
            )
            
            # Update active skills if knowledge relates to any
            for skill_name, skill_data in self.learning_context.skills.items():
                if any(self._is_concept_related_to_skill(concept, skill_name) for concept in concepts):
                    self.learning_context.practice_skill(
                        skill_name, 
                        learning_results['understanding']
                    )
            
            # Extract and store learning patterns
            for concept in concepts:
                patterns = self._extract_learning_patterns(concept)
                if patterns:
                    self.learning_context.store_patterns(concept, patterns)
            
            # Update drives based on learning outcome
            for concept in concepts:
                self.update_drives(concept, learning_results['understanding'])
            
            return {
                'depth': learning_results['depth'],
                'breadth': learning_results['breadth'],
                'cognitive_load': learning_results['cognitive_load'],
                'understanding': learning_results['understanding'],
                'patterns': len(self._extract_learning_patterns(concepts[0])) if concepts else 0
            }
            
        except Exception as e:
            print(f"Error in process_text_knowledge: {str(e)}")
            return {
                'depth': 0.0,
                'breadth': 0.0,
                'cognitive_load': 0.0,
                'understanding': 0.0,
                'patterns': 0
            }

    @staticmethod
    def _extract_relationships_vectorized(content, links):
        # Vectorized relationship extraction using numpy
        words = np.array(content.lower().split())
        relationships = defaultdict(list)
        
        # Use numpy operations for window processing
        window_size = 5
        for i in range(len(words)):
            window = words[max(0, i-window_size):min(len(words), i+window_size)]
            relationships[words[i]].extend(window[window != words[i]])
        
        return relationships

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


    def _process_concept_through_layers(self, concept, context=None):
        try:
            # Wrap concept for standardized handling
            wrapped_concept = ConceptWrapper(concept)
            
            # Initialize context
            if context is None:
                context = {}
            context['goals'] = getattr(self, 'goals', [])
            
            # Process through consciousness layers
            layer_responses = {}
            patterns = []
            
            for layer_name, weight in self.consciousness.layer_weights.items():
                layer = self.consciousness.layers[layer_name]
                
                # Process with layer-specific systems using concept id
                response = {
                    'belief': layer['belief'].process_belief_update(
                        wrapped_concept.id, context, layer['emotional_state']
                    ),
                    'desire': layer['desire'].get_desire_strength(
                        wrapped_concept.id, context
                    ),
                    'emotional': layer['emotional_state']
                }
                
                response['total_reward'] = response['belief'] * weight
                
                if hasattr(self.learning_context, 'semantic_memory'):
                    patterns.extend(self._process_semantic_associations(wrapped_concept.id))
                
                layer_responses[layer_name] = response
            
            learning_outcome = sum(
                response['total_reward'] 
                for response in layer_responses.values()
            )
            
            self.update_drives(wrapped_concept.id, learning_outcome)
            if context and 'related_concepts' in context:
                for related in context['related_concepts']:
                    related_wrapped = ConceptWrapper(related)
                    if hasattr(self.learning_context, 'semantic_memory'):
                        self.learning_context.semantic_memory['relationships'][wrapped_concept.id].append(related_wrapped.id)
            
            if learning_outcome > 0.5:
                self._integrate_core_belief(wrapped_concept.id)
            
            return layer_responses
            
        except Exception as e:
            print(f"Error in process_concept: {str(e)}")
            return {}

   
    def _integrate_core_belief(self, concept):
        """Integrate a concept as a core belief in the unconscious layer"""
        self.consciousness.unconscious.core_beliefs[concept] = {
            'strength': 1.0,
            'connections': [],
            'integration_time': len(self.reward_history)
        }
    
    def add_goal(self, goal):
        """Add a new goal to guide learning and belief formation"""
        self.goals.append(goal)
    
    def update_learning_state(self, concept, reward):
        """Update learning state based on rewards"""
        self.reward_history.append({
            'concept': concept,
            'reward': reward,
            'time': len(self.reward_history)
        })
        
        # Update active concepts across layers
        for layer in ['conscious', 'subconscious', 'unconscious']:
            if layer not in self.learning_context.learning_layers:
                self.learning_context.learning_layers[layer] = {'active_concepts': {}}
            
            layer_data = self.learning_context.learning_layers[layer]
            if concept in layer_data['active_concepts']:
                layer_data['active_concepts'][concept] += reward
            else:
                layer_data['active_concepts'][concept] = reward

    def _empty_learning_results(self):
        """Return empty results structure when processing fails"""
        return {
            'depth': 0.0,
            'breadth': 0.0,
            'cognitive_load': 0.0,
            'understanding': 0.0
        }
        
    def _process_learning_layers(self, concepts, knowledge_item):
        """Process learning across consciousness layers"""
        try:
            layer_results = {}
            emotional_value = self._get_normalized_emotional_value()
            
            # Extract hashable concept keys
            concept_keys = []
            for concept in concepts:
                if isinstance(concept, dict):
                    key = concept.get('id') or concept.get('name') or hash(frozenset(concept.items()))
                    concept_keys.append(str(key))  # Convert all keys to strings
                else:
                    concept_keys.append(str(concept))
            
            # Process through layers with hashable keys
            layer_results['unconscious'] = self.consciousness.unconscious.process_learning(
                concept_keys,
                knowledge_item,
                emotional_value
            )
            
            layer_results['subconscious'] = self.consciousness.subconscious.process_learning(
                concept_keys,
                knowledge_item,
                emotional_value,
                previous_layer_result=layer_results['unconscious']
            )
            
            layer_results['conscious'] = self.consciousness.conscious.process_learning(
                concept_keys,
                knowledge_item,
                emotional_value,
                previous_layer_result=layer_results['subconscious']
            )
            
            return {
                'depth': np.mean([r['learning_strength'] for r in layer_results.values()]),
                'breadth': len(concept_keys) / 100,
                'cognitive_load': sum(r['metrics'].get('complexity', 0) for r in layer_results.values()) / 3,
                'understanding': np.mean([r['metrics'].get('comprehension', 0) for r in layer_results.values()]),
                'layer_results': layer_results
            }
            
        except Exception as e:
            print(f"Error in _process_learning_layers: {str(e)}")
            return {
                'depth': 0.0,
                'breadth': 0.0,
                'cognitive_load': 0.0,
                'understanding': 0.0,
                'layer_results': {}
            }

    def _extract_patterns(self, concept):
        """Extract patterns from concept across consciousness layers"""
        patterns = []
        
        for layer_name, layer in {
            'conscious': self.consciousness.conscious,
            'subconscious': self.consciousness.subconscious,
            'unconscious': self.consciousness.unconscious
        }.items():
            if concept in layer.patterns:
                patterns.append({
                    'layer': layer_name,
                    'pattern': layer.patterns[concept],
                    'emotional_weight': layer.emotional_weights.get(concept, 0)
                })
        
        return patterns
    
    def _get_concept_drive_relevance(self, concept, drive):
        """Calculate how relevant a concept is to a core drive"""
        if drive == 'survival':
            return self._calculate_survival_relevance(concept)
        elif drive == 'social':
            return self._calculate_social_relevance(concept)
        elif drive == 'mastery':
            return self._calculate_mastery_relevance(concept)
        elif drive == 'autonomy':
            return self._calculate_autonomy_relevance(concept)
        elif drive == 'purpose':
            return self._calculate_purpose_relevance(concept)
        return 0.0

    def _is_concept_related_to_skill(self, concept, skill_name):
        """Determine if a concept is related to a particular skill"""
        # Check semantic relationships in learning context
        if hasattr(self.learning_context, 'semantic_memory'):
            relationships = self.learning_context.semantic_memory['relationships']
            if concept in relationships and skill_name in str(relationships[concept]):
                return True
        
        # Check active concepts in skill context
        skill_concepts = self.learning_context.learning_layers['conscious']['active_concepts']
        return skill_name in str(skill_concepts.get(concept, ''))

    def _process_semantic_associations(self, concept):
        """Process semantic associations for a concept"""
        associations = []
        
        if hasattr(self.learning_context, 'semantic_memory'):
            # Get direct relationships
            relationships = self.learning_context.semantic_memory['relationships']
            if concept in relationships:
                associations.extend(relationships[concept])
            
            # Get hierarchical relationships
            hierarchies = self.learning_context.semantic_memory['hierarchies']
            if concept in hierarchies:
                associations.extend(list(hierarchies[concept]))
        
        return associations

    def _calculate_drive_relevance(self, concept):
        """Calculate concept relevance to each drive"""
        relevance = {}
        for drive in self.drives:
            if drive == 'survival':
                relevance[drive] = self._calculate_survival_relevance(concept)
            elif drive == 'social':
                relevance[drive] = self._calculate_social_relevance(concept)
            elif drive == 'mastery':
                relevance[drive] = self._calculate_mastery_relevance(concept)
            elif drive == 'autonomy':
                relevance[drive] = self._calculate_autonomy_relevance(concept)
            elif drive == 'purpose':
                relevance[drive] = self._calculate_purpose_relevance(concept)
        return relevance

    def _calculate_survival_relevance(self, concept):
        """Calculate how relevant a concept is to survival"""
        survival_keywords = {'safety', 'health', 'security', 'protection', 'risk'}
        return self._calculate_semantic_relevance(concept, survival_keywords)

    def _calculate_social_relevance(self, concept):
        """Calculate how relevant a concept is to social drives"""
        social_keywords = {'relationship', 'communication', 'community', 'cooperation', 'empathy'}
        return self._calculate_semantic_relevance(concept, social_keywords)

    def _calculate_mastery_relevance(self, concept):
        """Calculate how relevant a concept is to mastery"""
        mastery_keywords = {'skill', 'learning', 'improvement', 'achievement', 'expertise'}
        return self._calculate_semantic_relevance(concept, mastery_keywords)

    def _calculate_autonomy_relevance(self, concept):
        """Calculate how relevant a concept is to autonomy"""
        autonomy_keywords = {'independence', 'choice', 'freedom', 'self-direction', 'control'}
        return self._calculate_semantic_relevance(concept, autonomy_keywords)

    def _calculate_purpose_relevance(self, concept):
        """Calculate how relevant a concept is to purpose"""
        purpose_keywords = {'meaning', 'goal', 'vision', 'mission', 'value'}
        return self._calculate_semantic_relevance(concept, purpose_keywords)

    def _calculate_semantic_relevance(self, concept, keywords):
        """Calculate semantic similarity between concept and keywords"""
        if hasattr(self.learning_context, 'semantic_memory'):
            relationships = self.learning_context.semantic_memory['relationships']
            if concept in relationships:
                concept_words = set(str(relationships[concept]).lower().split())
                return len(concept_words.intersection(keywords)) / len(keywords)
        return 0.0

    def _extract_learning_patterns(self, concept):
        """Extract patterns related to learning and skill development"""
        patterns = []
        
        # Check active learning contexts
        for layer_name, layer_data in self.learning_context.learning_layers.items():
            if concept in layer_data['active_concepts']:
                strength = layer_data['active_concepts'][concept]
                patterns.append({
                    'type': 'learning',
                    'layer': layer_name,
                    'strength': strength,
                    'drive_relevance': self._calculate_drive_relevance(concept)
                })
        
        # Check skill relationships
        for skill_name, skill_data in self.learning_context.skills.items():
            if self._is_concept_related_to_skill(concept, skill_name):
                patterns.append({
                    'type': 'skill',
                    'skill': skill_name,
                    'mastery': skill_data['mastery'],
                    'momentum': self.learning_context.skill_momentum[skill_name]
                })
        
        return patterns

    def update_drives(self, concept, learning_outcome):
        """Update drive strengths based on concept processing"""
        relevance = self._calculate_drive_relevance(concept)
        
        for drive, value in relevance.items():
            # Update drive strength based on relevance and learning outcome
            self.drives[drive] = min(1.0, self.drives[drive] + 
                                   value * learning_outcome * 0.1)
            
            # Update drive-specific learning
            self.drive_learning[drive][concept] += learning_outcome * value

    def _get_normalized_emotional_value(self):
        """Convert emotional state to normalized value considering consciousness layers and beliefs"""
        try:
            # Get base emotional values from each layer
            layer_emotions = {
                'conscious': self.consciousness.conscious.emotional_state,
                'subconscious': self.consciousness.subconscious.emotional_state,
                'unconscious': self.consciousness.unconscious.emotional_state
            }
            
            # Get belief influence weights for each layer
            belief_weights = {
                'conscious': self.consciousness.conscious.belief.influence_weights,
                'subconscious': self.consciousness.subconscious.belief.influence_weights,
                'unconscious': self.consciousness.unconscious.belief.influence_weights
            }
            
            # Calculate weighted emotional value for each layer
            weighted_emotions = {}
            for layer_name, emotion in layer_emotions.items():
                if isinstance(emotion, str) and emotion == 'ambivalent':
                    time_factor = len(self.consciousness.conscious.emotional_history) % 1000
                    emotion = 0.5 * np.sin(time_factor * 0.1)
                elif isinstance(emotion, (np.ndarray, np.generic)):
                    emotion = float(np.mean(emotion))
                    
                # Apply belief-based emotional weighting
                rational_weight = belief_weights[layer_name]['rational']
                emotional_weight = belief_weights[layer_name]['emotional']
                instinctive_weight = belief_weights[layer_name]['instinctive']
                
                weighted_emotions[layer_name] = emotion * (
                    rational_weight * 0.3 +
                    emotional_weight * 0.5 +
                    instinctive_weight * 0.2
                )
            
            # Integrate across layers with consciousness-specific weights
            normalized_value = (
                weighted_emotions['conscious'] * 0.4 +
                weighted_emotions['subconscious'] * 0.35 +
                weighted_emotions['unconscious'] * 0.25
            )
            
            return float(np.clip(normalized_value, -1.0, 1.0))
            
        except Exception as e:
            print(f"Error in _get_normalized_emotional_value: {str(e)}")
            return 0.0

    def process_imagination(self, stimulus, context=None):
        """Process imagination with personality and emotional influence"""
        # Update imagination state based on personality
        self._update_imagination_state()
        
        # Process through consciousness system
        imagination_result = self.consciousness.process_imagination(
            stimulus,
            context=self._get_current_context()
        )
        
        # Integrate results with learning and emotional systems
        self._integrate_imagination_results(imagination_result)
        
        return imagination_result

    def _update_imagination_state(self):
        """Update imagination parameters based on personality"""
        self.consciousness.imagination.imagination_state['pattern_flexibility'] = (
            0.5 + (self.personality['openness'] - 0.5) * 0.3
        )
        self.consciousness.imagination.imagination_state['reflection_depth'] = (
            0.5 + (self.personality['conscientiousness'] - 0.5) * 0.2
        )

    def _integrate_imagination_results(self, results):
        """Integrate imagination results with learning and emotional systems"""
        # Update learning state
        if 'patterns' in results:
            self.learning_state['pattern_recognition'] = results['patterns']
        
        # Update emotional memory
        if 'emotional_value' in results:
            self._update_emotional_memory(results['emotional_value'])
        
        # Update drive learning
        if 'discoveries' in results:
            for discovery in results['discoveries']:
                self.update_drives(
                    discovery['concepts'][0],
                    discovery['strength']
                )

    def process_input(self, input_data, input_type):
        """Process input through perception and thought systems"""
        # Check if perception system exists, create if not
        if not hasattr(self, 'perception'):
            self.perception = Perception(self.consciousness)
        
        # Perceptual processing
        perceived_data = self.perception.process_input(input_data, input_type)
        
        if perceived_data:
            # Check if thought_process exists, create if not
            if not hasattr(self, 'thought_process'):
                self.thought_process = ThoughtProcess(self.consciousness)
            
            # Generate thought
            thought = self.thought_process.process_thought(
                perceived_data,
                context=self._get_current_context()
            )
            
            # Check if interface exists, create if not
            if not hasattr(self, 'interface'):
                self.interface = HumanBehaviorInterface(
                    self.consciousness,
                    self.consciousness.belief_system,
                    self.consciousness.desire_system
                )
            
            # Generate response through interface
            return self.interface._formulate_response(thought)
        
        return None

    def process_rest_period(self, duration):
        """Process during rest/reflection periods"""
        # Get current state
        context = self._get_current_context()
        
        # Process imagination/reflection
        imagination_result = self.process_imagination(
            None,  # No direct stimulus
            context=context
        )
        
        # Generate autonomous thought if discoveries made
        if imagination_result.get('discoveries'):
            thought = self.thought_process.process_thought(
                imagination_result['discoveries'][0],
                context={'type': 'autonomous'}
            )
            
            return {
                'response': self.interface._formulate_response(thought),
                'discoveries': imagination_result['discoveries']
            }
            
        return {'discoveries': []}

    def process_impulse(self, stimulus_strength, context=None):
        """Process potential spontaneous impulse"""
        if not hasattr(self, 'impulse_system'):
            self.impulse_system = ImpulseSystem(self.consciousness)
        
        impulse_result = self.impulse_system.process_impulse(
            stimulus_strength,
            context or self._get_current_context()
        )
        
        if impulse_result:
            # Generate autonomous thought from impulse
            thought = self.thought_process.process_thought(
                impulse_result,
                context={'type': 'impulse'}
            )
            
            return {
                'response': self.interface._formulate_response(thought),
                'impulse_data': impulse_result
            }
        
        return None

    def initialize_perception(self):
        self.perception = Perception(self.consciousness)
        
    def initialize_thought_process(self):
        self.thought_process = ThoughtProcess(self.consciousness)
        
    def initialize_imagination(self):
        self.imagination = Imagination(self.consciousness)
        
    def initialize_interface(self):
        self.interface = HumanBehaviorInterface(
            self.consciousness,
            self.consciousness.belief_system,
            self.consciousness.desire_system
        )
        
    def initialize_impulse_system(self):
        self.impulse_system = ImpulseSystem(self.consciousness)
        
    def initialize_sleep_cycle(self):
        self.sleep_cycle = SleepCycle()
