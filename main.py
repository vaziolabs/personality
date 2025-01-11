import matplotlib.pyplot as plt
import numpy as np
from hbs import HumanBehaviorSystem

def simulate_day():
    # Create our system
    human = HumanBehaviorSystem()
    
    # Simulate a day with various stimuli
    hours = range(24)
    stimuli = [
        30 * np.sin(hour/4) +  # Regular daily rhythm
        20 * np.random.randn() +  # Random events
        (30 if hour in [8, 14] else 0) +  # Work/activity spikes
        (-20 if hour in [12, 22] else 0)   # Rest periods
        for hour in hours
    ]
    
    # Run simulation
    energy_levels = []
    for stimulus in stimuli:
        energy = human.respond_to_stimulus(stimulus)
        energy_levels.append(energy)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.plot(hours, energy_levels, '-o', label='Energy Level')
    plt.plot(hours, stimuli, '--', label='External Stimuli')
    plt.axhline(y=50, color='r', linestyle=':', label='Baseline')
    plt.xlabel('Hour of Day')
    plt.ylabel('Level')
    plt.title('Human Energy Levels vs External Stimuli')
    plt.legend()
    plt.grid(True)
    return plt

if __name__ == "__main__":
    plot = simulate_day()
    plot.show()
