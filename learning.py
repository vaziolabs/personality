from collections import defaultdict
import numpy as np

class LearningContext:
    def __init__(self):
        self.skills = {}
        self.knowledge = defaultdict(float)
        self.experiences = []
        self.daily_routine = {}
        self.skill_momentum = defaultdict(float)
        self.rest_learning = defaultdict(float)
        # Add layered learning tracking
        self.learning_layers = {
            'conscious': {
                'explicit_knowledge': defaultdict(float),
                'strategies': defaultdict(list),
                'goals_progress': defaultdict(float)
            },
            'subconscious': {
                'emotional_patterns': defaultdict(list),
                'social_scripts': defaultdict(list),
                'behavioral_habits': defaultdict(float)
            },
            'unconscious': {
                'emotional_memories': defaultdict(list),
                'survival_patterns': defaultdict(float),
                'core_beliefs': defaultdict(float)
            }
        }
        
        # Add text processing states
        self.learning_layers['conscious'].update({
            'active_concepts': defaultdict(float),
            'reasoning_patterns': defaultdict(list)
        })
        
        self.learning_layers['subconscious'].update({
            'semantic_associations': defaultdict(list),
            'pattern_recognition': defaultdict(float)
        })
        
        self.learning_layers['unconscious'].update({
            'deep_abstractions': defaultdict(float),
            'intuitive_models': defaultdict(list)
        })
        
    def add_skill(self, skill_name, difficulty, mastery=0.0):
        """Initialize a skill with proper structure"""
        self.skills[skill_name] = {
            'mastery': mastery,
            'difficulty': difficulty,
            'practice_count': 0,
            'recent_improvements': [],
            'consolidation': 0.0
        }
    
    def practice_skill(self, skill_name):
        if skill_name not in self.skills:
            return 0.0
            
        skill = self.skills[skill_name]
        momentum = self.skill_momentum[skill_name]
        base_improvement = (1.0 - skill['mastery']) * 0.1 * (1.0 / skill['difficulty'])
        
        # Apply momentum with dampening
        improvement = base_improvement * (1.0 + momentum * 0.5)
        
        # Update momentum
        self.skill_momentum[skill_name] = min(
            2.0,
            momentum * 0.8 + improvement * 0.2
        )
        
        # Update skill
        skill['mastery'] = min(1.0, skill['mastery'] + improvement)
        skill['practice_count'] += 1
        skill['recent_improvements'].append(improvement)
        if len(skill['recent_improvements']) > 10:
            skill['recent_improvements'].pop(0)
            
        return improvement * 100.0
    
    def process_rest_period(self, consciousness_state):
        """Enhanced rest period processing with consciousness integration"""
        for skill_name, skill in self.skills.items():
            if skill['recent_improvements']:
                # Calculate consolidation with enhanced consciousness influence
                avg_improvement = np.mean(skill['recent_improvements'])
                unconscious_influence = consciousness_state['unconscious'].mean()
                consolidation_factor = consciousness_state['consolidation_factor']
                
                # Apply emotional modulation
                emotional_factor = np.exp(abs(consciousness_state['emotional_state']))
                
                # Process patterns from consciousness system
                pattern_factor = self._process_learning_patterns(
                    consciousness_state['patterns'],
                    skill['mastery']
                )
                
                # Enhanced consolidation calculation
                consolidation = (
                    avg_improvement * 0.3 * unconscious_influence * 
                    consolidation_factor * emotional_factor * pattern_factor
                )
                
                skill['consolidation'] += consolidation
                skill['mastery'] = min(1.0, skill['mastery'] + consolidation)
                
                # Adaptive momentum decay based on consolidation
                decay_rate = 0.95 - (consolidation * 0.1)
                self.skill_momentum[skill_name] *= max(0.8, decay_rate)

    def _process_learning_patterns(self, patterns, current_mastery):
        """Process consciousness patterns for learning enhancement"""
        if not patterns:
            return 1.0
            
        pattern_similarities = []
        for pattern in patterns:
            flat_pattern = pattern.flatten()
            avg_activation = np.mean(flat_pattern)
            pattern_similarities.append(avg_activation)
            
        # Calculate pattern influence factor
        pattern_strength = np.mean(pattern_similarities)
        mastery_factor = 1 - current_mastery  # Harder to improve at higher mastery
        
        return 1.0 + (pattern_strength * mastery_factor * 0.5)

    def update_learning(self, reward_state, consciousness_state):
        for skill_name, skill in self.skills.items():
            conscious_learning = reward_state['conscious'] * skill['practice_count']
            emotional_learning = reward_state['subconscious'] * consciousness_state['emotional_state']
            implicit_learning = reward_state['unconscious'] * consciousness_state['unconscious'].mean()
            
            total_learning = (
                conscious_learning * 0.4 +
                emotional_learning * 0.3 +
                implicit_learning * 0.3
            )
            
            skill['mastery'] = min(1.0, skill['mastery'] + total_learning)

    def integrate_learning_experience(self, experience, context):
        """Process learning across all consciousness levels"""
        # Conscious learning
        self._process_conscious_learning(experience, context)
        
        # Subconscious pattern formation
        self._process_subconscious_learning(experience, context)
        
        # Unconscious conditioning
        self._process_unconscious_learning(experience, context)
        
        # Update cross-layer integration
        self._integrate_learning_layers()

class TextLearningContext(LearningContext):
    def __init__(self):
        super().__init__()
        self.semantic_memory = {
            'concepts': defaultdict(float),
            'relationships': defaultdict(list),
            'hierarchies': defaultdict(set)
        }
        
    def process_text_knowledge(self, knowledge_item):
        """Process text-based knowledge"""
        # Extract concepts
        concepts = self._extract_concepts(knowledge_item['content'])
        
        # Process relationships
        relationships = self._extract_relationships(
            knowledge_item['content'],
            knowledge_item['links']
        )
        
        # Update semantic memory
        self._update_semantic_memory(concepts, relationships)
        
        return {
            'concepts_learned': len(concepts),
            'relationships_formed': len(relationships)
        }