class HumanBehaviorSystem:
    def __init__(self):
        # Base energy level (represents normal state)
        self.energy = 50
        
        # How quickly we respond to stimuli
        self.responsiveness = 0.3
        
        # How strongly we resist change (dampening)
        self.resistance = 0.2
        
        # Memory of past states (hysteresis)
        self.memory = []
        self.memory_influence = 0.15
        
        # Recovery rate towards baseline
        self.recovery_rate = 0.1
        
        # Keep track of history for plotting
        self.energy_history = []
    
    def respond_to_stimulus(self, stimulus_strength):
        # Remember current state
        self.memory.append(self.energy)
        if len(self.memory) > 5:  # Keep last 5 states
            self.memory.pop(0)
        
        # Calculate memory effect (past experiences influence current response)
        memory_effect = 0
        if self.memory:
            memory_effect = sum(self.memory) / len(self.memory) * self.memory_influence
        
        # Calculate response with dampening
        raw_response = stimulus_strength * self.responsiveness
        dampened_response = raw_response * (1 - self.resistance * abs(raw_response))
        
        # Update energy level with memory effect
        self.energy += dampened_response + memory_effect
        
        # Apply recovery towards baseline (homeostasis)
        distance_from_baseline = 50 - self.energy
        self.energy += distance_from_baseline * self.recovery_rate
        
        # Keep energy within realistic bounds (0-100)
        self.energy = max(0, min(100, self.energy))
        
        # Record history
        self.energy_history.append(self.energy)
        
        return self.energy
