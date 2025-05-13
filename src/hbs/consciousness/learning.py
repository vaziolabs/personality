from collections import defaultdict
import numpy as np
from hbs.consciousness.concept import ConceptWrapper

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
            'conscious': {'active_concepts': {}},
            'subconscious': {'active_concepts': {}},
            'unconscious': {'active_concepts': {}}
        }
        self.reward_threshold = {
            'conscious': 0.3,
            'subconscious': 0.6,
            'unconscious': 0.9
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
        
        self.semantic_memory = {
            'concepts': defaultdict(float),
            'relationships': defaultdict(list),
            'hierarchies': defaultdict(set)
        }
        
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
        
        # Vectorized calculations
        mastery_factor = np.exp(-skill['mastery'] * skill['difficulty'])
        learning_rate = 0.1 * mastery_factor
        improvement = (1.0 - skill['mastery']) * learning_rate * (1.0 + np.tanh(momentum))
        
        # Single update for momentum and mastery
        self.skill_momentum[skill_name] = min(2.0, momentum * 0.8 + improvement * 0.2)
        skill['mastery'] = min(1.0, skill['mastery'] + improvement)
        skill['practice_count'] += 1
        skill['recent_improvements'] = skill['recent_improvements'][-9:] + [improvement] if skill['recent_improvements'] else [improvement]
        
        return improvement * 100.0
    
    def process_rest_period(self, state):
        """Process rest period for learning consolidation"""
        # Process consciousness state
        unconscious_influence = state['emotional_state']
        consolidation_factor = 0.1 * len(state.get('discoveries', []))
        
        # Process discoveries
        for discovery in state.get('discoveries', []):
            if discovery['type'] == 'direct':
                # Strengthen related skills
                self._strengthen_related_skills(
                    discovery['connection']['concepts'],
                    discovery['connection']['strength']
                )
                
            elif discovery['type'] == 'cluster':
                # Update skill relationships
                self._update_skill_cluster(discovery['concepts'])
                
        # Consolidate existing skills
        for skill_name, skill in self.skills.items():
            if skill['recent_improvements']:
                consolidation = (
                    np.mean(skill['recent_improvements']) * 
                    consolidation_factor * 
                    unconscious_influence
                )
                skill['mastery'] = min(1.0, skill['mastery'] + consolidation)
                skill['consolidation'] += consolidation
        
        return {
            'consolidation_factor': consolidation_factor,
            'skills_updated': len(self.skills)
        }

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
        """Process learning updates with proper skill mastery tracking"""
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
            skill['consolidation'] += total_learning * 0.1

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

    def process_concept(self, concept, layer, reward):
        """Process concept with reward in specific layer"""
        try:
            # Handle both string and dict concepts
            concept_key = concept if isinstance(concept, str) else concept.get('id', str(concept))
            
            if reward >= self.reward_threshold[layer]:
                self.learning_layers[layer]['active_concepts'][concept_key] = reward
                return True
            return False
        except Exception as e:
            print(f"Error in process_concept: {str(e)}")
            return False

    def update_semantic_memory(self, concepts, knowledge, links=None):
        """Optimized semantic memory updates"""
        if not isinstance(concepts, list):
            concepts = [concepts]
        
        if isinstance(knowledge, str):
            knowledge = {'content': knowledge}
        
        # Batch process concepts
        wrapped_concepts = [ConceptWrapper(c) for c in concepts]
        extracted_concepts = self._extract_concepts(knowledge.get('content', ''))
        relationships = self._extract_relationships(knowledge.get('content', ''), links or knowledge.get('links', {}))
        
        # Vectorized concept strength updates
        for concept in extracted_concepts:
            wrapped = ConceptWrapper(concept)
            self.semantic_memory['concepts'][wrapped.id] = min(1.0, self.semantic_memory['concepts'].get(wrapped.id, 0) + 0.1)
        
        # Batch relationship updates
        for wrapped in wrapped_concepts:
            if wrapped.id in relationships:
                new_rels = [ConceptWrapper(rel).id for rel in relationships[wrapped.id]]
                existing = set(self.semantic_memory['relationships'][wrapped.id])
                self.semantic_memory['relationships'][wrapped.id].extend([r for r in new_rels if r not in existing])
        
        # Batch hierarchy updates
        if isinstance(knowledge, dict) and 'hierarchies' in knowledge:
            for parent, children in knowledge['hierarchies'].items():
                wrapped_parent = ConceptWrapper(parent)
                self.semantic_memory['hierarchies'][wrapped_parent.id].update(
                    {ConceptWrapper(child).id for child in children}
                )

    def _extract_concepts(self, text):
        """Extract concepts from text using basic tokenization"""
        try:
            # Basic tokenization and cleaning
            words = text.lower().split()
            # Remove punctuation and common words
            cleaned_words = [
                word.strip('.,!?()[]{}":;') 
                for word in words 
                if len(word) > 3  # Skip short words
            ]
            
            # Convert to string format for consistent hashing
            return [str(word) for word in cleaned_words]
            
        except Exception as e:
            print(f"Error in _extract_concepts: {str(e)}")
            return []

    def _extract_relationships(self, content, links):
        """Optimized relationship extraction"""
        relationships = defaultdict(list)
        words = content.lower().split()
        window_size = 5
        
        # Vectorized window processing
        word_windows = [words[max(0, i-window_size):min(len(words), i+window_size)] for i in range(len(words))]
        for i, window in enumerate(word_windows):
            current_word = str(words[i])
            relationships[current_word].extend([str(w) for w in window if w != words[i]])
        
        # Batch process links
        if isinstance(links, dict):
            for link, summary in links.items():
                summary_str = str(summary) if not isinstance(summary, dict) else summary.get('content', '')
                summary_words = [str(w) for w in summary_str.lower().split()]
                for word in str(link).lower().split():
                    relationships[word].extend(summary_words)
        
        return relationships

class TextLearningContext(LearningContext):
    def __init__(self):
        super().__init__()
        self.semantic_memory = defaultdict(float)
        self.reward_history = []
        self.goals = []

    def process_text_knowledge(self, knowledge_item):
        """Single entry point for text processing"""
        try:
            # Extract concepts and relationships once
            concepts = self._extract_concepts(knowledge_item['content'])
            relationships = self._extract_relationships(
                knowledge_item['content'],
                knowledge_item.get('links', {})
            )
            
            # Generate stable patterns
            patterns = self._generate_patterns(concepts)
            
            # Update semantic memory with the processed knowledge
            for concept in concepts:
                self.update_semantic_memory(concept, knowledge_item)
            
            return {
                'concepts': concepts,
                'relationships': relationships,
                'patterns': patterns
            }
        except Exception as e:
            print(f"Error in process_text_knowledge: {str(e)}")
            return {'concepts': [], 'relationships': {}, 'patterns': {}}

    def _generate_patterns(self, concepts):
        """Unified pattern generation"""
        patterns = {}
        for concept in concepts:
            concept_array = np.array([ord(c) for c in str(concept)], dtype=np.float32)
            pattern = concept_array.reshape((int(np.sqrt(len(concept_array))+1), -1))
            patterns[concept] = pattern.tobytes()
        return patterns
