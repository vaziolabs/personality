import os
import nltk

# Set up NLTK data path
nltk_data_dir = os.path.expanduser('./nltk_data')
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

import matplotlib.pyplot as plt
import numpy as np
from hbs import HumanBehaviorSystem, SleepCycle
from collections import defaultdict
from numba import jit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from learning import LearningContext, TextLearningContext
from wiki_scraper import WikiKnowledgeBase
import networkx as nx
from serializer import SystemSerializer

def create_learning_scenario():
    context = LearningContext()
    
    # Add various skills and knowledge domains
    context.add_skill('programming', difficulty=3.0)
    context.add_skill('mathematics', difficulty=2.5)
    context.add_skill('writing', difficulty=2.0)
    context.add_skill('public_speaking', difficulty=2.8)
    
    return context

def create_daily_routine(hour, day, sleep_state, context, season_factor=1.0):
    """Comprehensive daily routine with both activities and learning"""
    # Add active skills tracking
    active_skills = set()
    
    # Regular weekday schedule with both activities and learning
    weekday_schedule = {
        6: ('wake_up', None, 40, ['physical_recovery']),
        8: ('morning_study', 'mathematics', 30, ['mathematics', 'focus']),
        10: ('work_focus', 'programming', 35, ['programming', 'problem_solving']),
        12: ('lunch_break', None, -10, ['physical_recovery']),
        13: ('writing_task', 'writing', 25, ['writing', 'creativity']),
        15: ('meeting', 'public_speaking', 30, ['public_speaking', 'social']),
        17: ('exercise', None, 35, ['physical_fitness']),
        19: ('dinner', None, -15, ['physical_recovery']),
        20: ('evening_study', 'programming', 20, ['programming', 'focus']),
        22: ('sleep', None, -30, ['rest'])
    }
    
    # Weekend schedule
    weekend_schedule = {
        9: ('wake_up', None, 20, ['physical_recovery']),
        10: ('hobby_coding', 'programming', 30, ['programming', 'creativity']),
        12: ('social_lunch', None, 15, ['social']),
        14: ('public_event', 'public_speaking', 25, ['public_speaking', 'social']),
        16: ('creative_writing', 'writing', 20, ['writing', 'creativity']),
        18: ('exercise', None, 35, ['physical_fitness']),
        20: ('relaxation', None, -10, ['rest']),
        23: ('sleep', None, -30, ['rest'])
    }
    
    # Select schedule based on day
    is_weekend = (day % 7) >= 5
    schedule = weekend_schedule if is_weekend else weekday_schedule
    
    # Process current hour's activity
    learning_stimulus = 0
    activity_stimulus = 0
    
    if hour in schedule:
        activity, skill, base_impact, involved_skills = schedule[hour]
        active_skills.update(involved_skills)
        
        # Apply seasonal and time-of-day adjustments
        adjusted_impact = base_impact * season_factor
        
        # If it's a learning activity, process skill improvement
        if skill:
            improvement = context.practice_skill(skill)
            learning_stimulus = improvement * (1 + season_factor)
        
        activity_stimulus = adjusted_impact
    
    # Random life events
    life_events = {
        'unexpected_challenge': ((-20, 40), 0.05),
        'social_interaction': ((10, 30), 0.1),
        'learning_opportunity': ((20, 40), 0.05),
        'stress_event': ((-30, -10), 0.08),
        'achievement': ((20, 50), 0.03)
    }
    
    # Process random events
    random_stimulus = 0
    for event, (impact_range, prob) in life_events.items():
        if np.random.random() < prob:
            random_stimulus += np.random.uniform(*impact_range)
    
    # Environmental factors
    weather_impact = 10 * np.sin(2 * np.pi * day / 365)  # Seasonal weather
    noise = 5 * np.random.randn()  # Random environmental noise
    
    # Combine all stimuli
    total_stimulus = (
        activity_stimulus +
        learning_stimulus +
        random_stimulus +
        weather_impact +
        noise
    )
    
    return {
        'total': total_stimulus,
        'activity': schedule.get(hour, (None, None, 0, []))[0],
        'skill': schedule.get(hour, (None, None, 0, []))[1],
        'learning': learning_stimulus,
        'random_events': random_stimulus != 0,
        'active_skills': active_skills
    }

def process_day(args):
    day, sleep_cycle, learning_context, human, season_factor = args
    
    # Initialize day results dictionary
    day_results = {
        'energy': np.zeros(24),
        'consciousness': [],
        'learning': np.zeros(24),
        'emotional': np.zeros(24),
        'sleep': np.zeros(24),
        'active_skills': [set() for _ in range(24)],
        'activities': [None] * 24,
        'rest_consolidation': []
    }
    
    # Track consciousness state throughout day
    is_asleep = np.zeros(24, dtype=bool)
    is_asleep[22:] = True  # Sleeping hours
    is_asleep[:6] = True   # Sleeping hours
    
    for hour in range(24):
        # Update sleep cycle
        sleep_state = sleep_cycle.update(hour, is_asleep[hour])
        day_results['sleep'][hour] = sleep_state['pressure']
        
        # Get routine data with active skills
        routine_data = create_daily_routine(hour, day, sleep_state, learning_context, season_factor)
        
        # Update active skills in human behavior system
        human.learning_state['active_skills'] = routine_data['active_skills']
        day_results['active_skills'][hour] = routine_data['active_skills']
        day_results['activities'][hour] = routine_data['activity']
        
        # Process stimulus and get detailed state
        energy, state = human.respond_to_stimulus(routine_data['total'])
        
        # Record results
        day_results['energy'][hour] = energy
        day_results['emotional'][hour] = human.emotional_state
        day_results['learning'][hour] = routine_data['learning']
        day_results['consciousness'].append(state)
        
        # Process rest periods
        if is_asleep[hour]:
            rest_state = human.process_rest_period(duration=1.0)
            day_results['rest_consolidation'].append(rest_state)
    
    return day_results

def simulate_year():
    # Initialize systems
    sleep_cycle = SleepCycle()
    learning_context = create_learning_scenario()
    human = HumanBehaviorSystem()
    
    # Track yearly results
    yearly_results = {
        'energy': [],
        'emotional': [],
        'learning': [],
        'sleep': [],
        'consciousness': [],
        'active_skills': [],
        'activities': [],
        'rest_consolidation': []
    }
    
    # Setup parallel processing
    num_processes = mp.cpu_count()
    process_args = [
        (day, sleep_cycle, learning_context, human, 
         1 + 0.2 * np.sin(2 * np.pi * day / 365))  # Season factor
        for day in range(365)
    ]
    
    # Process days in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_day, process_args))
        
        # Combine results
        for day_results in results:
            yearly_results['energy'].extend(day_results['energy'])
            yearly_results['emotional'].extend(day_results['emotional'])
            yearly_results['learning'].extend(day_results['learning'])
            yearly_results['sleep'].extend(day_results['sleep'])
            yearly_results['consciousness'].extend(day_results['consciousness'])
            yearly_results['active_skills'].extend(day_results['active_skills'])
            yearly_results['activities'].extend(day_results['activities'])
            yearly_results['rest_consolidation'].extend(day_results['rest_consolidation'])
    
    # Create visualization
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
    # Plot energy levels
    axes[0,0].plot(yearly_results['energy'])
    axes[0,0].set_title('Energy Levels')
    axes[0,0].grid(True)
    
    # Plot emotional state
    axes[0,1].plot(yearly_results['emotional'])
    axes[0,1].set_title('Emotional State')
    axes[0,1].grid(True)
    
    # Plot learning progress
    axes[1,0].plot(yearly_results['learning'])
    axes[1,0].set_title('Learning Progress')
    axes[1,0].grid(True)
    
    # Plot sleep pressure
    axes[1,1].plot(yearly_results['sleep'])
    axes[1,1].set_title('Sleep Pressure')
    axes[1,1].grid(True)
    
    # Plot consciousness activity - Fixed visualization
    consciousness_states = yearly_results['consciousness']
    if consciousness_states:
        # Reshape consciousness data for visualization
        consciousness_data = np.array([
            state['response'].flatten() for state in consciousness_states
        ])
        # Plot as heatmap over time
        axes[2,0].imshow(consciousness_data.T, aspect='auto', cmap='viridis')
        axes[2,0].set_title('Consciousness Activity')
        axes[2,0].set_ylabel('Consciousness Components')
        axes[2,0].set_xlabel('Time')
    
    # Plot skill development
    skill_data = []
    for skills in yearly_results['active_skills']:
        if skills:
            skill_data.append(len(skills))
        else:
            skill_data.append(0)
    axes[2,1].plot(skill_data)
    axes[2,1].set_title('Active Skills')
    axes[2,1].grid(True)
    
    # Plot rest consolidation
    consolidation_data = [rest['memory_consolidation'] for rest in yearly_results['rest_consolidation']]
    axes[3,0].plot(consolidation_data)
    axes[3,0].set_title('Memory Consolidation')
    axes[3,0].grid(True)
    
    # Plot activity patterns
    activity_data = [1 if activity else 0 for activity in yearly_results['activities']]
    axes[3,1].plot(activity_data)
    axes[3,1].set_title('Activity Patterns')
    axes[3,1].grid(True)
    
    plt.tight_layout()
    return fig

def visualize_context_and_memory(hbs, recent_experiences=10):
    fig = plt.figure(figsize=(20, 10))
    
    # Memory Map (Left subplot)
    ax1 = plt.subplot(121)
    memory_values = hbs.memory
    memory_positions = np.array([
        [0, 0],   # Oldest memory
        [-1, 1],
        [0, 1.5],
        [1, 1],
        [0, 0.5]  # Most recent memory
    ])
    
    # Create bubble chart for memory
    sizes = np.abs(memory_values) * 1000  # Scale for visibility
    colors = plt.cm.viridis(np.linspace(0, 1, len(memory_values)))
    
    for pos, size, color, value in zip(memory_positions, sizes, colors, memory_values):
        circle = plt.Circle(pos, size/5000, color=color, alpha=0.6)
        ax1.add_artist(circle)
        ax1.annotate(f'{value:.2f}', pos, ha='center', va='center')
    
    # Context Map (Right subplot)
    ax2 = plt.subplot(122)
    
    # Get recent experiences
    recent_exp = hbs.experience_buffer[-recent_experiences:]
    
    # Create network graph of experiences
    positions = {}
    connections = []
    
    # Central node (current state)
    positions['current'] = np.array([0, 0])
    
    # Position experiences in a circle around current state
    for i, exp in enumerate(recent_exp):
        angle = 2 * np.pi * i / len(recent_exp)
        pos = np.array([np.cos(angle), np.sin(angle)])
        positions[i] = pos
        connections.append((i, 'current'))
    
    # Draw connections
    for start, end in connections:
        start_pos = positions[start]
        end_pos = positions['current']
        plt.plot([start_pos[0], end_pos[0]], 
                [start_pos[1], end_pos[1]], 
                'gray', alpha=0.3)
    
    # Draw nodes
    current_state = {
        'energy': hbs.energy,
        'emotional': hbs.emotional_state,
        'responsiveness': hbs.responsiveness
    }
    
    # Central node
    plt.scatter([0], [0], s=500, c='red', alpha=0.6, label='Current State')
    
    # Experience nodes
    for i, exp in enumerate(recent_exp):
        pos = positions[i]
        size = 300 * (1 + abs(exp['emotional_state']))
        color = plt.cm.RdYlBu(exp['energy']/100)
        plt.scatter([pos[0]], [pos[1]], s=size, c=[color], alpha=0.6)
        plt.annotate(f"E:{exp['energy']:.1f}\nEm:{exp['emotional_state']:.1f}", 
                    pos, ha='center', va='center')
    
    # Formatting
    ax1.set_title('Memory Map')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.axis('equal')
    
    ax2.set_title('Experience Context Network')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis('equal')
    
    plt.tight_layout()
    return fig

def test_learning_sequence(stimulus_sequence, duration=100):
    """Test system learning with a specific stimulus sequence"""
    hbs = HumanBehaviorSystem()
    results = {
        'energy': [],
        'emotional': [],
        'responsiveness': [],
        'resistance': []
    }
    
    for i in range(duration):
        # Get stimulus from sequence
        stimulus = stimulus_sequence(i)
        
        # Process stimulus
        energy, state = hbs.respond_to_stimulus(stimulus)
        
        # Record results
        results['energy'].append(energy)
        results['emotional'].append(hbs.emotional_state)
        results['responsiveness'].append(hbs.responsiveness)
        results['resistance'].append(hbs.resistance)
        
        # Periodically visualize context
        if i % 20 == 0:
            fig = visualize_context_and_memory(hbs)
            plt.savefig(f'context_map_{i}.png')
            plt.close()
    
    return results, hbs

# Example usage:
def example_stimulus_sequence(t):
    """Example stimulus sequence with patterns"""
    base = 30 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    weekly = 20 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly pattern
    noise = np.random.normal(0, 10)  # Random noise
    return base + weekly + noise

def simulate_knowledge_acquisition(topics):
    """Main simulation function"""
    # Check for existing saved state
    serializer = SystemSerializer()
    latest_state = None
    
    # Try to find most recent save file
    if os.path.exists('./saved_states'):
        save_files = sorted(os.listdir('./saved_states'))
        if save_files:
            latest_save = os.path.join('./saved_states', save_files[-1])
            try:
                latest_state = serializer.load_system(latest_save)
                print(f"Loaded previous state from: {latest_save}")
            except Exception as e:
                print(f"Could not load previous state: {e}")
    
    # Initialize or restore system
    if latest_state:
        hbs = latest_state['hbs']
        context = latest_state['learning_context']
        metrics = latest_state['metrics']
    else:
        # Initialize new system
        hbs = HumanBehaviorSystem()
        context = create_learning_scenario()
        metrics = defaultdict(list)
    
    try:
        # Run simulation
        wiki_kb = WikiKnowledgeBase()
        
        for topic in topics:
            print(f"Learning about: {topic}")
            knowledge = wiki_kb.fetch_topic(topic)
            
            if not knowledge:
                continue
                
            # Process knowledge
            learning_results = hbs.process_text_knowledge(knowledge)
            
            # Update metrics
            metrics['concepts_learned'].append(len(learning_results.get('concepts', [])))
            metrics['relationships_formed'].append(len(learning_results.get('relationships', [])))
            metrics['understanding_level'].append(hbs.learning_state['learning_momentum'])
        
        # Save final state
        save_path = save_simulation_state(hbs, context, metrics)
        print(f"Saved final state to: {save_path}")
        
        return metrics, context, hbs
        
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        # Save state on error for recovery
        save_path = save_simulation_state(hbs, context, metrics, 'error_state.pkl')
        print(f"Saved error state to: {save_path}")
        raise

def visualize_knowledge_state(learning_context, human):
    """Visualize current knowledge state"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot concept network
    concepts = list(learning_context.semantic_memory['concepts'].keys())
    relationships = learning_context.semantic_memory['relationships']
    
    # Create network visualization
    G = nx.Graph()
    for concept in concepts:
        G.add_node(concept)
    for source, targets in relationships.items():
        for target in targets:
            G.add_edge(source, target)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=axes[0,0], node_size=100, alpha=0.6)
    axes[0,0].set_title('Knowledge Graph')
    
    # Plot learning metrics
    axes[0,1].plot(human.learning_state['learning_momentum'])
    axes[0,1].set_title('Learning Momentum')
    
    plt.tight_layout()
    return fig

def save_simulation_state(hbs, context, metrics, filename=None):
    """Save simulation state to file"""
    serializer = SystemSerializer()
    
    # Create a state dictionary that matches the expected format
    simulation_state = {
        'hbs': hbs,  # HumanBehaviorSystem instance
        'context': context.__dict__,
        'metrics': {
            'learning_progress': metrics.get('learning_progress', []),
            'consciousness_states': metrics.get('consciousness_states', []),
            'emotional_states': metrics.get('emotional_states', []),
            'knowledge_acquisition': metrics.get('knowledge_acquisition', {})
        }
    }
    
    return serializer.save_system(simulation_state, filename)

def load_simulation_state(filepath):
    """Load a previously saved simulation state"""
    serializer = SystemSerializer()
    loaded_state = serializer.load_system(filepath)
    
    return (
        loaded_state['hbs'],
        loaded_state['learning_context'],
        loaded_state['metrics']
    )

# Example usage
if __name__ == "__main__":
    # First ensure NLTK data is downloaded
    from setup_nltk import download_nltk_data
    download_nltk_data()

    topics = [
        "Logic",
        "Proof",
        "Constructive proof",
        "Proof by Induction",
        "Proof by Contradiction",
        "Proof Theory",
        "Aesthetics",
        "Ethics",
        "Linguistics",
        "Poetry"
        "Philosophy",
        "Epistemology",
        "Set Theory",
        "Intuitionistic type theory",
        "Math Theory",
        "Data Science",
        "Computer Science",
        "Physics",
        "Metaphysics",
        "Spirituality",
        "History",
        "Sutras",
        "Meditation",
        "Pranayama",
        "Tao",
        "Tai chi",
        "Five precepts",
        "The arts",
        "Art",
        "Psychology of art",
        "Quantum mechanics",
        "Introduction to quantum mechanics",
        "Interpretations of quantum mechanics",
        "Philosophy of physics",
        "Sustainability",
        "Ecological systems theory",
        "Sustainable development",
        "Ecosystem management",
        "Ecological literacy",
        "Ethics of technology",
        "Technology and society",
        "Philosophy of technology",
        "Scientific method",
        "History of scientific method",
        "Systems thinking",
        "Critical systems thinking",
        "Creative entrepreneurship",
        "Entrepreneurial economics",
        "Sustainable living",
        "Minimalism",
        "Sustainable consumption",
        "The Philosophy of Money",
        "Personal finance",
        "Finance",
        "Investment",
        "Financial literacy",
        "Investment strategy",
        "Return (finance)",
        "Exchange rate",
        "Economics",
        "Dividend",
        "Interest",
        "Asset",
        "Cash flow",
        "Diversification (finance)",
        "Volatility (finance)",
        "Financial risk",
        "Personal boundaries",
        "Professional boundaries",
        "Interpersonal relationship",
        "Proxemics",
        "Assertiveness",
        "Nonviolent communication",
        "Morality",
        "Right and wrong",
        "Sociology of culture",
        "Culture",
        "Sociology",
        "Cultural analysis",
        "Cultural identity",
        "Sociocultural evolution",
        "Attachment theory",
        "Attachment in adults",
        "Biology of romantic love",
        "Theories of love",
        "Affectional bond",
        "Conflict resolution",
        "Mediation",
        "Dispute resolution",
        "Active listening",
        "Listening",
        "Listening behaviour types",
        "Communication",
        "Reflective listening",
        "Sarcasm",
        "Humor",
        "Empathy",
        "History of geometry",
        "Sacred geometry",
        "Platonic solids",
        "Tetractys",
        "Cymatics",
        "Sri Yantra",
        "Tree of life (Kabbalah)",
        "Theology",
        "Subtle body",
        "Chakra",
        "Kundalini",
        "Kundalini yoga",
        "Dantian",
        "Consciousness",
        "Universal mind",
        "Panpsychism",
        "Electromagnetic theories of consciousness",
        "Inner peace",
        "Contentment",
        "Peace of Mind",
        "Peace",
        "Santosha",
        "Calmness",
        "List of religions and spiritual traditions",
        "Comparative religion",
        "Religion",
        "World religions",
        "Ontology",
        "Ontological argument",
        "Existence",
        "Flow (psychology)",
        "Motivation",
        "Motivated reasoning",
        "Motivation and Personality",
        "Maslow's hierarchy of needs",
        "Neuroplasticity",
        "Addiction-related structural neuroplasticity",
        "Brain healing",
        "Brain health and pollution",
        "Memory improvement",
        "Activity-dependent plasticity",
        "Creativity",
        "Innovation",
        "Creativity techniques",
        "Memory improvement",
        "Memorization",
        "Memory and retention in learning",
        "Memory technique",
        "List of cognitive biases",
        "Cognitive bias",
        "Heuristic (psychology)",
        "Dunning–Kruger effect",
        "Motivated Reasoning",
        "Cognitive bias mitigation",
        "Anchoring effect",
        "Psychological resilience",
        "Family resilience",
        "Resilience",
        "Community resilience",
        "Mental toughness",
        "Goal setting",
        "Goal",
        "Creative visualization",
        "Objectives and key results",
        "Goal orientation",
        "Data and information visualization",
        "Time management",
        "Self-reflection",
        "Reflective practice",
        "Emotional intelligence",
        "Empathy quotient",
    ]
    
    # Run simulation
    metrics, context, final_hbs = simulate_knowledge_acquisition(topics)
    
    # Save simulation state
    save_path = save_simulation_state(final_hbs, context, metrics)
    print(f"Simulation state saved to: {save_path}")
    
    # Plot results
    plot_learning_results(metrics)
    plt.show()
    