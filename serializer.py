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
            
    def save_system(self, hbs, filename=None):
        """Save the entire system state"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'system_state_{timestamp}.pkl'
            
        filepath = os.path.join(self.save_dir, filename)
        
        # Convert defaultdict to dict for serialization
        system_state = {
            'hbs': {
                'energy': hbs.energy,
                'responsiveness': hbs.responsiveness,
                'resistance': hbs.resistance,
                'recovery_rate': hbs.recovery_rate,
                'memory': hbs.memory.tolist(),
                'memory_ptr': hbs.memory_ptr,
                'memory_influence': hbs.memory_influence,
                'energy_history': hbs.energy_history.tolist(),
                'history_ptr': hbs.history_ptr,
                'emotional_state': hbs.emotional_state,
                'adaptation_rate': hbs.adaptation_rate,
                'experience_buffer': hbs.experience_buffer,
                'learning_state': dict(hbs.learning_state),
                'drives': dict(hbs.drives),
                'personality': dict(hbs.personality),
                'personality_weights': dict(hbs.personality_weights),
                'emotional_memory': self._convert_defaultdict_to_dict(hbs.emotional_memory),
                'current_context': dict(hbs.current_context),
                'context_history': hbs.context_history,
                'context_associations': dict(hbs.context_associations),
                'desire_layers': self._convert_nested_defaultdict_to_dict(hbs.desire_layers)
            },
            'consciousness': self._serialize_consciousness(hbs.consciousness),
            'learning': self._serialize_learning(hbs.learning_context)
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
