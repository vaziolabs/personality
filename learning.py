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
        """Optimized skill practice with adaptive learning rate"""
        if skill_name not in self.skills:
            return 0.0
            
        skill = self.skills[skill_name]
        momentum = self.skill_momentum[skill_name]
        
        # Adaptive learning rate based on mastery level
        learning_rate = 0.1 * np.exp(-skill['mastery'] * skill['difficulty'])
        base_improvement = (1.0 - skill['mastery']) * learning_rate
        
        # Optimized momentum calculation
        improvement = base_improvement * (1.0 + np.tanh(momentum))
        self.skill_momentum[skill_name] = min(2.0, momentum * 0.8 + improvement * 0.2)
        
        # Efficient skill update
        skill['mastery'] = min(1.0, skill['mastery'] + improvement)
        skill['practice_count'] += 1
        
        # Maintain fixed-size improvement history
        skill['recent_improvements'] = (
            skill['recent_improvements'][-9:] + [improvement] 
            if skill['recent_improvements'] 
            else [improvement]
        )
        
        return improvement * 100.0
    
    def process_rest_period(self, consciousness_state):
        """Optimized rest period processing"""
        unconscious_influence = consciousness_state['unconscious'].mean()
        emotional_factor = np.exp(abs(consciousness_state['emotional_state']))
        consolidation_factor = consciousness_state['consolidation_factor']
        
        # Pre-calculate pattern factor
        pattern_factor = self._process_learning_patterns(
            consciousness_state['patterns'],
            max(skill['mastery'] for skill in self.skills.values())
        )
        
        # Batch process all skills
        for skill_name, skill in self.skills.items():
            if skill['recent_improvements']:
                consolidation = (
                    np.mean(skill['recent_improvements']) * 
                    0.3 * unconscious_influence * 
                    consolidation_factor * emotional_factor * 
                    pattern_factor
                )
                
                # Single update for mastery and consolidation
                skill['mastery'] = min(1.0, skill['mastery'] + consolidation)
                skill['consolidation'] += consolidation
                
                # Optimized momentum decay
                self.skill_momentum[skill_name] *= max(
                    0.8, 
                    0.95 - (consolidation * 0.1)
                )

    @staticmethod
    def _process_learning_patterns(patterns, current_mastery):
        """Optimized pattern processing"""
        if not patterns:
            return 1.0
            
        # Vectorized pattern processing
        pattern_similarities = np.mean([
            pattern.flatten() for pattern in patterns
        ], axis=1)
        
        pattern_strength = np.mean(pattern_similarities)
        mastery_factor = 1 - current_mastery
        
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