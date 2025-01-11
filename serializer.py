import pickle
import os
from datetime import datetime
import numpy as np
from collections import defaultdict

class SystemSerializer:
    def __init__(self, save_dir='./saved_states'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def save_system(self, simulation_state, filename=None):
        """Save the entire system state"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'system_state_{timestamp}.pkl'
            
        filepath = os.path.join(self.save_dir, filename)
        
        # Handle both direct HBS objects and dictionary state
        if isinstance(simulation_state, dict):
            # Extract HBS from simulation state if it exists
            hbs = simulation_state.get('hbs')
            if isinstance(hbs, dict):
                # Already in correct format
                system_state = simulation_state
            else:
                # Convert HBS object to serializable format
                system_state = {
                    'hbs': {
                        'energy': getattr(hbs, 'energy', 0.0),
                        'responsiveness': getattr(hbs, 'responsiveness', 0.0),
                        'resistance': getattr(hbs, 'resistance', 0.0),
                        'recovery_rate': getattr(hbs, 'recovery_rate', 0.0),
                        'emotional_state': getattr(hbs, 'emotional_state', 0.0),
                        'adaptation_rate': getattr(hbs, 'adaptation_rate', 0.0),
                        'learning_state': dict(getattr(hbs, 'learning_state', {})),
                        'drives': dict(getattr(hbs, 'drives', {})),
                        'personality': dict(getattr(hbs, 'personality', {}))
                    },
                    'context': simulation_state.get('context', {}),
                    'metrics': simulation_state.get('metrics', {})
                }
        else:
            # Direct HBS object
            system_state = {
                'hbs': {
                    'energy': simulation_state.energy,
                    'responsiveness': simulation_state.responsiveness,
                    'resistance': simulation_state.resistance,
                    'recovery_rate': simulation_state.recovery_rate,
                    'emotional_state': simulation_state.emotional_state,
                    'adaptation_rate': simulation_state.adaptation_rate,
                    'learning_state': dict(simulation_state.learning_state),
                    'drives': dict(simulation_state.drives),
                    'personality': dict(simulation_state.personality)
                }
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_state, f)
            
        return filepath
        
    def load_system(self, filepath, hbs):
        """Load and restore system state"""
        with open(filepath, 'rb') as f:
            system_state = pickle.load(f)
            
        # Restore HBS state
        hbs_state = system_state['hbs']
        for key, value in hbs_state.items():
            if key in ['memory', 'energy_history']:
                setattr(hbs, key, np.array(value))
            elif key in ['learning_state', 'drives', 'personality', 
                        'personality_weights', 'current_context']:
                setattr(hbs, key, dict(value))
            elif key == 'emotional_memory':
                hbs.emotional_memory = self._convert_dict_to_defaultdict(value, list)
            elif key == 'desire_layers':
                hbs.desire_layers = self._convert_dict_to_nested_defaultdict(value)
            else:
                setattr(hbs, key, value)
                
        # Restore consciousness system
        self._deserialize_consciousness(system_state['consciousness'], hbs.consciousness)
        
        # Restore learning context
        self._deserialize_learning(system_state['learning'], hbs.learning_context)
        
        return hbs
        
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
            'history': cs.history,
            'thought_paths': cs.thought_paths.tolist(),
            'path_index': cs.path_index
        }
        
    def _serialize_layer(self, layer):
        """Serialize individual consciousness layer"""
        return {
            'state': layer.state.tolist(),
            'activation_history': layer.activation_history.tolist(),
            'history_index': layer.history_index,
            'patterns': dict(layer.patterns),
            'emotional_weights': dict(layer.emotional_weights),
            'belief_contexts': self._convert_defaultdict_to_dict(layer.belief_contexts),
            'context_relationships': dict(layer.context_relationships),
            'dissonance_by_context': dict(layer.dissonance_by_context),
            'active_beliefs': dict(layer.active_beliefs),
            'behavioral_patterns': dict(layer.behavioral_patterns),
            'emotional_associations': dict(layer.emotional_associations),
            'primal_drives': dict(layer.primal_drives)
        }
        
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
