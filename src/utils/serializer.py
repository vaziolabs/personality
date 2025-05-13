import pickle
import os
from datetime import datetime
import numpy as np
from collections import defaultdict
from hbs.hbs import HumanBehaviorSystem
from hbs.consciousness.learning import LearningContext

class SystemSerializer:
    def __init__(self, save_dir='../saved_states'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def save_system(self, state, filepath):
        """Save system state to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert state components to serializable format
            serialized_state = {
                'hbs': self._serialize_hbs(state['hbs']),
                'learning_context': self._serialize_learning(state['learning_context']),
                'metrics': state['metrics'],
                'timestamp': state.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
            }
            
            # Use protocol=4 for better compatibility
            with open(filepath, 'wb') as f:
                pickle.dump(serialized_state, f, protocol=4)
                
            return filepath
                
        except Exception as e:
            print(f"Error saving state: {str(e)}")
            raise

    def _serialize_hbs(self, hbs):
        """Serialize HBS and all its components"""
        return {
            # Base parameters
            'energy': getattr(hbs, 'energy', 50.0),
            'responsiveness': getattr(hbs, 'responsiveness', 0.3),
            'resistance': getattr(hbs, 'resistance', 0.2),
            'recovery_rate': getattr(hbs, 'recovery_rate', 0.1),
            'adaptation_rate': getattr(hbs, 'adaptation_rate', 0.1),
            
            # Memory systems
            'memory': getattr(hbs, 'memory', np.zeros(64)).tolist(),
            'memory_ptr': getattr(hbs, 'memory_ptr', 0),
            'memory_influence': getattr(hbs, 'memory_influence', 0.15),
            'emotional_memory': self._convert_nested_defaultdict_to_dict(getattr(hbs, 'emotional_memory', {})),
            
            # History tracking
            'energy_history': getattr(hbs, 'energy_history', np.zeros(100)).tolist(),
            'history_ptr': getattr(hbs, 'history_ptr', 0),
            'experience_buffer': getattr(hbs, 'experience_buffer', []),
            'context_history': getattr(hbs, 'context_history', []),
            'reward_history': getattr(hbs, 'reward_history', []),
            
            # State tracking
            'emotional_state': getattr(hbs, 'emotional_state', 0.0),
            'current_context': getattr(hbs, 'current_context', {}),
            'context_associations': self._convert_defaultdict_to_dict(getattr(hbs, 'context_associations', {})),
            
            # Core systems
            'drives': dict(getattr(hbs, 'drives', {})),
            'drive_learning': self._convert_nested_defaultdict_to_dict(getattr(hbs, 'drive_learning', {})),
            'personality': dict(getattr(hbs, 'personality', {})),
            'personality_weights': dict(getattr(hbs, 'personality_weights', {})),
            'goals': list(getattr(hbs, 'goals', [])),
            
            # Learning systems
            'learning_state': self._convert_nested_defaultdict_to_dict(getattr(hbs, 'learning_state', {})),
            'learning_context': self._serialize_learning(hbs.learning_context),
            'text_learning_context': self._serialize_learning(hbs.text_learning_context),
            
            # Consciousness system
            'consciousness': self._serialize_consciousness(hbs.consciousness),
            
            # Add missing systems
            'perception_initialized': hasattr(hbs, 'perception'),
            'thought_process_initialized': hasattr(hbs, 'thought_process'),
            'imagination_initialized': hasattr(hbs, 'imagination'),
            'interface_initialized': hasattr(hbs, 'interface'),
            'impulse_system_initialized': hasattr(hbs, 'impulse_system'),
            'sleep_cycle': self._serialize_sleep_cycle(hbs.sleep_cycle) if hasattr(hbs, 'sleep_cycle') else None,
        }

    def _serialize_sleep_cycle(self, sleep_cycle):
        """Serialize sleep cycle state"""
        return {
            'state': getattr(sleep_cycle, 'state', 'awake'),
            'sleep_pressure': getattr(sleep_cycle, 'sleep_pressure', 0.0),
            'last_sleep_time': getattr(sleep_cycle, 'last_sleep_time', 0),
            'dream_state': getattr(sleep_cycle, 'dream_state', None),
            'sleep_stages': getattr(sleep_cycle, 'sleep_stages', []),
            'memory_consolidation': getattr(sleep_cycle, 'memory_consolidation', 0.0)
        }

    def _serialize_consciousness(self, cs):
        """Serialize consciousness system state"""
        return {
            'layers': {
                name: self._serialize_layer(layer)
                for name, layer in [
                    ('conscious', cs.conscious),
                    ('subconscious', cs.subconscious),
                    ('unconscious', cs.unconscious)
                ]
            },
            'semantic_memory': self._convert_nested_defaultdict_to_dict(cs.semantic_memory),
            'thought_paths': getattr(cs, 'thought_paths', np.zeros((1000, 6))).tolist(),
            'path_index': getattr(cs, 'path_index', 0),
            'layer_weights': dict(getattr(cs, 'layer_weights', {})),
            'personality_traits': dict(getattr(cs, 'personality_traits', {}))
        }

    def _serialize_layer(self, layer):
        """Serialize consciousness layer"""
        return {
            'state': layer.state.tolist(),
            'name': layer.name,
            'size': layer.size,
            'dissonance_threshold': layer.dissonance_threshold,
            'belief_contexts': dict(layer.belief_contexts),
            'semantic_memory': {
                'concepts': dict(layer.semantic_memory['concepts']),
                'relationships': dict(layer.semantic_memory['relationships']),
                'hierarchies': dict(layer.semantic_memory['hierarchies'])
            },
            'belief_systems': {
                category: {
                    subcat: dict(beliefs)
                    for subcat, beliefs in subcats.items()
                }
                for category, subcats in layer.belief_systems.items()
            },
            'emotional_memory': {
                category: dict(memories)
                for category, memories in layer.emotional_memory.items()
            },
            'belief': self._serialize_belief_system(layer.belief),
            'desire': self._serialize_desire_system(layer.desire)
        }

    def _serialize_belief_system(self, belief):
        """Serialize belief system"""
        return {
            'influence_weights': dict(belief.influence_weights),
            'belief_influence': self._convert_nested_defaultdict_to_dict(belief.belief_influence),
            'predictions': self._convert_nested_defaultdict_to_dict(belief.predictions)
        }

    def _serialize_desire_system(self, desire):
        """Serialize desire system"""
        return {
            'size': desire.size,
            'layer_desires': dict(desire.layer_desires),
            'desire_strengths': dict(desire.desire_strengths),
            'motivation_weights': dict(desire.motivation_weights),
            'core_motivations': {
                k: dict(v) for k, v in desire.core_motivations.items()
            },
            'emotional_memory_weight': desire.emotional_memory_weight,
            'desire_levels': desire.desire_levels.tolist(),
            'past_outcomes': dict(desire.past_outcomes),
            'reward_weights': dict(desire.reward_weights),
            'drive_desires': {
                k: dict(v) for k, v in desire.drive_desires.items()
            },
            'personality_weights': dict(desire.personality_weights),
            'knowledge_desires': {
                k: dict(v) for k, v in desire.knowledge_desires.items()
            },
            'emotional_memory': desire.emotional_memory,
            'emotional_memory_capacity': desire.emotional_memory_capacity,
            'emotional_threshold': desire.emotional_threshold,
            'history_ptr': desire.history_ptr,
            'drives': desire.drives
        }
        
    def load_system(self, filepath):
        """Load system state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Add basic validation
            if not isinstance(state, dict):
                raise ValueError("Invalid state format: not a dictionary")
                
            # Convert serialized state back to proper objects
            hbs = HumanBehaviorSystem()
            self._deserialize_hbs(hbs, state['hbs'])
            
            # Reconstruct learning context
            learning_context = LearningContext()
            self._deserialize_learning(learning_context, state['learning_context'])
            
            return {
                'hbs': hbs,
                'learning_context': learning_context,
                'metrics': state['metrics'],
                'timestamp': state.get('timestamp')
            }
            
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            raise

    def _deserialize_hbs(self, hbs, state):
        """Deserialize HBS state back into object"""
        # Base parameters
        for key in ['energy', 'responsiveness', 'resistance', 'recovery_rate', 'adaptation_rate']:
            setattr(hbs, key, state[key])
            
        # Memory systems
        hbs.memory = np.array(state['memory'])
        hbs.memory_ptr = state['memory_ptr']
        hbs.memory_influence = state['memory_influence']
        hbs.emotional_memory = self._convert_dict_to_defaultdict(state['emotional_memory'], float)
        
        # History systems
        hbs.energy_history = np.array(state['energy_history'])
        hbs.history_ptr = state['history_ptr']
        hbs.experience_buffer = state['experience_buffer']
        hbs.context_history = state['context_history']
        hbs.reward_history = state['reward_history']
        
        # State tracking
        hbs.emotional_state = state['emotional_state']
        hbs.current_context = state['current_context']
        hbs.context_associations = self._convert_dict_to_defaultdict(state['context_associations'], float)
        
        # Core systems
        hbs.drives = state['drives']
        hbs.drive_learning = self._convert_dict_to_defaultdict(state['drive_learning'], float)
        hbs.personality = state['personality']
        hbs.personality_weights = state['personality_weights']
        hbs.goals = state['goals']
        
        # Learning systems
        hbs.learning_state = self._convert_dict_to_defaultdict(state['learning_state'], float)
        
        # Deserialize learning contexts
        self._deserialize_learning(hbs.learning_context, state['learning_context'])
        self._deserialize_learning(hbs.text_learning_context, state['text_learning_context'])
        
        # Consciousness system
        self._deserialize_consciousness(hbs.consciousness, state['consciousness'])
        
        # Initialize missing systems if they were present in the original
        if state.get('perception_initialized', False) and not hasattr(hbs, 'perception'):
            hbs.initialize_perception()
        
        if state.get('thought_process_initialized', False) and not hasattr(hbs, 'thought_process'):
            hbs.initialize_thought_process()
        
        if state.get('imagination_initialized', False) and not hasattr(hbs, 'imagination'):
            hbs.initialize_imagination()
        
        if state.get('interface_initialized', False) and not hasattr(hbs, 'interface'):
            hbs.initialize_interface()
        
        if state.get('impulse_system_initialized', False) and not hasattr(hbs, 'impulse_system'):
            hbs.initialize_impulse_system()
        
        if state.get('sleep_cycle') and not hasattr(hbs, 'sleep_cycle'):
            hbs.initialize_sleep_cycle()
            self._deserialize_sleep_cycle(hbs.sleep_cycle, state['sleep_cycle'])

    def _deserialize_consciousness(self, cs, state):
        """Deserialize consciousness system state"""
        # Restore layers
        for name, layer_state in state['layers'].items():
            layer = getattr(cs, name)
            self._deserialize_layer(layer, layer_state)
            
        cs.semantic_memory = self._convert_dict_to_defaultdict(state['semantic_memory'], float)
        cs.thought_paths = np.array(state['thought_paths'])
        cs.path_index = state['path_index']
        cs.layer_weights = state['layer_weights']
        cs.personality_traits = state['personality_traits']

    def _deserialize_layer(self, layer, state):
        """Deserialize consciousness layer"""
        layer.state = np.array(state['state'])
        layer.name = state['name']
        layer.size = state['size']
        layer.dissonance_threshold = state['dissonance_threshold']
        layer.belief_contexts = state['belief_contexts']
        layer.semantic_memory = state['semantic_memory']
        layer.belief_systems = state['belief_systems']
        layer.emotional_memory = state['emotional_memory']
        
        # Deserialize subsystems
        self._deserialize_belief_system(layer.belief, state['belief'])
        self._deserialize_desire_system(layer.desire, state['desire'])

    def _deserialize_belief_system(self, belief, state):
        """Deserialize belief system"""
        belief.influence_weights = dict(state['influence_weights'])
        belief.belief_influence = self._convert_dict_to_defaultdict(state['belief_influence'], float)
        belief.predictions = self._convert_dict_to_defaultdict(state['predictions'], float)

    def _deserialize_desire_system(self, desire, state):
        """Deserialize desire system"""
        desire.size = state['size']
        desire.layer_desires = state['layer_desires']
        desire.desire_strengths = state['desire_strengths']
        desire.motivation_weights = state['motivation_weights']
        desire.core_motivations = state['core_motivations']
        desire.emotional_memory_weight = state['emotional_memory_weight']
        desire.desire_levels = np.array(state['desire_levels'])
        desire.past_outcomes = state['past_outcomes']
        desire.reward_weights = state['reward_weights']
        desire.drive_desires = state['drive_desires']
        desire.personality_weights = state['personality_weights']
        desire.knowledge_desires = state['knowledge_desires']
        desire.emotional_memory = state['emotional_memory']
        desire.emotional_memory_capacity = state['emotional_memory_capacity']
        desire.emotional_threshold = state['emotional_threshold']
        desire.history_ptr = state['history_ptr']
        desire.drives = state['drives']

    def _serialize_learning(self, learning):
        """Serialize learning context state"""
        return {
            'skills': dict(learning.skills),
            'knowledge': dict(learning.knowledge),
            'experiences': learning.experiences,
            'daily_routine': dict(learning.daily_routine),
            'skill_momentum': dict(learning.skill_momentum),
            'rest_learning': dict(learning.rest_learning),
            'learning_layers': self._convert_nested_defaultdict_to_dict(learning.learning_layers)
        }
        
    @staticmethod
    def _convert_defaultdict_to_dict(d):
        """Convert defaultdict to regular dict"""
        if isinstance(d, defaultdict):
            return {k: SystemSerializer._convert_defaultdict_to_dict(v) 
                   if isinstance(v, (dict, defaultdict)) else v
                   for k, v in d.items()}
        return d
        
    @staticmethod
    def _convert_nested_defaultdict_to_dict(d):
        """Convert nested defaultdicts to regular dicts"""
        return {
            k: SystemSerializer._convert_defaultdict_to_dict(v)
            if isinstance(v, (dict, defaultdict)) else v
            for k, v in d.items()
        }
        
    @staticmethod
    def _convert_dict_to_defaultdict(d, default_factory):
        """Convert dict back to defaultdict"""
        result = defaultdict(default_factory)
        result.update(d)
        return result
        
    @staticmethod
    def _convert_dict_to_nested_defaultdict(d):
        """Convert nested dicts back to nested defaultdicts"""
        result = defaultdict(lambda: defaultdict(float))
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = SystemSerializer._convert_dict_to_defaultdict(v, float)
            else:
                result[k] = v
        return result

    @staticmethod
    def save_system(system, filepath):
        """Save system state to file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Validate state before saving
        if isinstance(system, dict):
            required_keys = ['hbs', 'learning_context', 'metrics']
            if not all(key in system for key in required_keys):
                raise ValueError(f"Missing required keys in system state: {required_keys}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(system, f)
    
    @staticmethod
    def load_system(filepath):
        """Load system state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Validate loaded state
            if isinstance(state, dict):
                # Check for minimum required keys
                if 'hbs' in state:
                    # Initialize missing components with defaults
                    state.setdefault('metrics', {
                        'knowledge_depth': [],
                        'knowledge_breadth': [],
                        'cognitive_load': [],
                        'understanding': []
                    })
                    state.setdefault('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
                    return state
                else:
                    print("Invalid state file: missing HBS")
            else:
                print("Invalid state file: not a dictionary")
            
            return None
            
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"Could not load state from {filepath}: {str(e)}")
            return None

    def deserialize_state(self, state):
        """Deserialize complete system state"""
        try:
            deserialized = {
                'metrics': state['metrics'],
                'hbs': self._deserialize_hbs(state['hbs']),
                'learning_context': self._deserialize_learning(state['learning_context']),
                'timestamp': state.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
            }
            return deserialized
        except Exception as e:
            print(f"Error deserializing state: {str(e)}")
            return self.create_empty_state()

    def create_empty_state(self):
        """Create a new empty state with proper initialization"""
        hbs = HumanBehaviorSystem()
        
        # Initialize metrics
        metrics = {
            'knowledge_depth': [0.1],
            'knowledge_breadth': [0.1],
            'cognitive_load': [0.2],
            'understanding': [0.1],
            'processing_times': [0.0]
        }
        
        return {
            'hbs': hbs,
            'learning_context': hbs.learning_context,
            'metrics': metrics,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

    def _deserialize_sleep_cycle(self, sleep_cycle, state):
        """Deserialize sleep cycle state back into object"""
        sleep_cycle.state = state['state']
        sleep_cycle.sleep_pressure = state['sleep_pressure']
        sleep_cycle.last_sleep_time = state['last_sleep_time']
        sleep_cycle.dream_state = state['dream_state']
        sleep_cycle.sleep_stages = state['sleep_stages']
        sleep_cycle.memory_consolidation = state['memory_consolidation']

