class Perception:
    def __init__(self, consciousness_system):
        self.consciousness = consciousness_system
        self.sensory_buffer = []
        self.attention_threshold = 0.3
        
    def process_input(self, input_data, input_type):
        """Process raw sensory input through perception layers"""
        # Initial sensory processing
        sensory_data = self._process_sensory_input(input_data, input_type)
        
        # Filter through RAS (Reticular Activating System)
        if self._check_attention_threshold(sensory_data):
            # Process through consciousness layers
            conscious_data = self._process_conscious_perception(sensory_data)
            subconscious_data = self._process_subconscious_perception(sensory_data) 
            unconscious_data = self._process_unconscious_perception(sensory_data)
            
            return {
                'conscious': conscious_data,
                'subconscious': subconscious_data,
                'unconscious': unconscious_data,
                'emotional_value': self._evaluate_emotional_significance(sensory_data),
                'attention_level': self._calculate_attention_level(sensory_data)
            }
        
        return None

    def _process_sensory_input(self, input_data, input_type):
        """Initial sensory processing"""
        return {
            'data': input_data,
            'type': input_type,
            'patterns': self._extract_patterns(input_data),
            'timestamp': len(self.sensory_buffer)
        }

    def _check_attention_threshold(self, sensory_data):
        """Check if input requires attention"""
        significance = self._calculate_significance(sensory_data)
        return significance > self.attention_threshold

    def _calculate_significance(self, sensory_data):
        """Calculate input significance based on emotional and survival relevance"""
        emotional_value = abs(self._evaluate_emotional_significance(sensory_data))
        pattern_familiarity = self._evaluate_pattern_familiarity(sensory_data)
        
        return emotional_value * 0.6 + pattern_familiarity * 0.4

    def _evaluate_emotional_significance(self, sensory_data):
        """Evaluate emotional significance of input"""
        return self.consciousness.evaluate_emotional_response(sensory_data)

    def _evaluate_pattern_familiarity(self, sensory_data):
        """Evaluate pattern familiarity"""
        return self.consciousness.evaluate_pattern_recognition(sensory_data)
