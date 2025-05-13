class ImpulseSystem:
    def __init__(self, consciousness_system):
        self.consciousness = consciousness_system
        self.impulse_threshold = 0.4
        self.impulse_decay = 0.85
        self.current_impulse = 0.0
        
    def process_impulse(self, stimulus_strength, context):
        """Process incoming impulse and determine if action needed"""
        # Calculate impulse strength
        impulse_strength = self._calculate_impulse_strength(stimulus_strength)
        
        # Update current impulse with decay
        self.current_impulse = (self.current_impulse * self.impulse_decay) + impulse_strength
        
        if self.current_impulse > self.impulse_threshold:
            return self._generate_impulse_response(context)
        return None
        
    def _calculate_impulse_strength(self, stimulus_strength):
        """Calculate impulse strength based on stimulus"""
        emotional_factor = self.consciousness.evaluate_emotional_response({
            'strength': stimulus_strength
        })
        return stimulus_strength * (1 + abs(emotional_factor))

    def _generate_impulse_response(self, context):
        """Generate response when impulse threshold exceeded"""
        # Get current consciousness state
        state = self.consciousness.process_rest_state()
        
        return {
            'strength': self.current_impulse,
            'state': state,
            'context': context
        }
