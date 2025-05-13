class ThoughtProcess:
    def __init__(self, consciousness_system):
        self.consciousness = consciousness_system
        self.thoughts = []
        self.context = {}
        self.emotional_state = 0.0
        
    def process_thought(self, input_data, context=None):
        """Process input into cohesive thought"""
        # Update context
        self.context.update(context or {})
        
        # Process through consciousness layers
        layer_responses = self._process_consciousness_layers(input_data)
        
        # Integrate responses into cohesive thought
        thought = self._integrate_thought(layer_responses)
        self.thoughts.append(thought)
        
        # Update emotional state
        self._update_emotional_state(thought)
        
        return thought

    def _process_consciousness_layers(self, input_data):
        """Process through consciousness layers"""
        return {
            'unconscious': self.consciousness._process_unconscious_rest(),
            'subconscious': self.consciousness._process_subconscious_rest(),
            'conscious': self.consciousness._process_conscious_rest()
        }
        
    def _integrate_thought(self, layer_responses):
        """Integrate layer responses into cohesive thought"""
        # Extract patterns across layers
        patterns = self.consciousness._analyze_behavior_patterns()
        
        # Integrate imagination discoveries
        discoveries = self.consciousness.process_imagination(
            layer_responses['conscious'],
            context=self.context
        )
        
        return {
            'layer_responses': layer_responses,
            'patterns': patterns,
            'discoveries': discoveries,
            'emotional_state': self.emotional_state,
            'context': self.context.copy()
        }

    def _update_emotional_state(self, thought):
        """Update emotional state based on thought"""
        # Calculate emotional influence from each layer
        conscious_emotion = thought['layer_responses']['conscious'] * 0.4
        subconscious_emotion = thought['layer_responses']['subconscious'] * 0.4  
        unconscious_emotion = thought['layer_responses']['unconscious'] * 0.2
        
        # Update emotional state with decay
        self.emotional_state = (
            self.emotional_state * 0.7 +
            (conscious_emotion + subconscious_emotion + unconscious_emotion) * 0.3
        )