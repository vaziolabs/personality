import os
import nltk
from datetime import datetime
import argparse

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
from wordcloud import WordCloud
import seaborn as sns

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

def save_simulation_state(hbs, context, metrics, timestamp=None):
    """Save the complete simulation state"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure all components are present
    state = {
        'hbs': hbs,
        'learning_context': context,
        'metrics': metrics or {
            'knowledge_depth': [],
            'knowledge_breadth': [],
            'cognitive_load': [],
            'understanding': []
        },
        'timestamp': timestamp
    }
    
    filepath = f'./saved_states/system_state_{timestamp}.pkl'
    try:
        SystemSerializer.save_system(state, filepath)
        return filepath
    except Exception as e:
        print(f"Error saving state: {str(e)}")
        return None

def load_simulation_state():
    """Load the most recent simulation state"""
    import glob
    import os
    
    # Find most recent state file
    state_files = glob.glob('./saved_states/system_state_*.pkl')
    if not state_files:
        return None
    
    latest_file = max(state_files, key=os.path.getctime)
    print(f"Loading previous state from: {latest_file}")
    
    state = SystemSerializer.load_system(latest_file)
    if state and isinstance(state, dict):
        return state
    return None

def simulate_knowledge_acquisition(topics):
    """Simulate knowledge acquisition process"""
    # Initialize metrics structure
    metrics = {
        'knowledge_depth': [],
        'knowledge_breadth': [],
        'cognitive_load': [],  # Changed from 'load'
        'understanding': []
    }
    
    # Try to load previous state
    previous_state = load_simulation_state()
    if previous_state:
        try:
            hbs = previous_state.get('hbs')
            context = previous_state.get('learning_context', TextLearningContext())
            # Update metrics from previous state if they exist
            for key in metrics:
                if key in previous_state.get('metrics', {}):
                    metrics[key] = previous_state['metrics'][key]
            if not isinstance(hbs, HumanBehaviorSystem):
                hbs = HumanBehaviorSystem()
        except Exception as e:
            print(f"Error loading previous state: {e}")
            hbs = HumanBehaviorSystem()
            context = TextLearningContext()
    else:
        hbs = HumanBehaviorSystem()
        context = TextLearningContext()

    # Process topics
    wiki = WikiKnowledgeBase()
    for topic in topics:
        print(f"Learning about: {topic}")
        try:
            knowledge = wiki.get_article(topic)
            if knowledge:
                learning_results = hbs.process_text_knowledge(knowledge)
                # Map learning results to metrics
                metrics['knowledge_depth'].append(learning_results['depth'])
                metrics['knowledge_breadth'].append(learning_results['breadth'])
                metrics['cognitive_load'].append(learning_results['cognitive_load'])  # Changed from 'load'
                metrics['understanding'].append(learning_results['understanding'])
        except Exception as e:
            print(f"Error processing topic {topic}: {str(e)}")
            save_error_state(hbs, context, metrics)
            continue

    return metrics, context, hbs

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

def plot_learning_results(metrics):
    """Plot learning metrics over time"""
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot knowledge acquisition
    if 'knowledge_depth' in metrics:
        ax1.plot(metrics['knowledge_depth'], label='Knowledge Depth')
        ax1.plot(metrics['knowledge_breadth'], label='Knowledge Breadth')
        ax1.set_title('Knowledge Acquisition Over Time')
        ax1.set_xlabel('Learning Steps')
        ax1.set_ylabel('Knowledge Level')
        ax1.legend()
    
    # Plot cognitive metrics
    if 'cognitive_load' in metrics:
        ax2.plot(metrics['cognitive_load'], label='Cognitive Load')
        ax2.plot(metrics['understanding'], label='Understanding')
        ax2.set_title('Cognitive Metrics Over Time')
        ax2.set_xlabel('Learning Steps')
        ax2.set_ylabel('Level')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('./plots/learning_results.png')
    plt.close()

def save_error_state(hbs, context, metrics):
    """Save system state when an error occurs"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    error_state = {
        'hbs': hbs,
        'learning_context': context,
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    filepath = f'./saved_states/error_state.pkl'
    SystemSerializer.save_system(error_state, filepath)
    print(f"Saved error state to: {filepath}")

class WikiKnowledgeBase:
    def __init__(self):
        import wikipedia
        self.wiki = wikipedia
        
    def get_article(self, topic):
        """Retrieve article content from Wikipedia"""
        try:
            # Search for the page
            page = self.wiki.page(topic, auto_suggest=True)
            return {
                'title': page.title,
                'content': page.content,
                'links': {link: '' for link in page.links},  # Convert links list to dict
                'references': page.references,
                'url': page.url
            }
        except self.wiki.DisambiguationError as e:
            print(f"Disambiguation for {topic}. Trying most relevant option...")
            try:
                # Try first option from disambiguation
                page = self.wiki.page(e.options[0], auto_suggest=False)
                return {
                    'title': page.title,
                    'content': page.content,
                    'links': {link: '' for link in page.links},  # Convert links list to dict
                    'references': page.references,
                    'url': page.url
                }
            except:
                print(f"No relevant disambiguation option found for: {topic}")
                return None
        except Exception as e:
            print(f"Error fetching topic {topic}: {str(e)}")
            return None

def visualize_belief_systems(hbs):
    """Visualize belief systems and consciousness layers"""
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import seaborn as sns
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Belief Context Map (Top Left)
    ax1 = plt.subplot(221)
    belief_data = {}
    for context, value in hbs.consciousness.conscious.belief_contexts.items():
        if isinstance(value, np.ndarray):
            belief_data[str(context)] = float(np.mean(value))
        else:
            belief_data[str(context)] = float(value)
    
    # Create heatmap of belief strengths with safety checks
    if belief_data:
        belief_values = list(belief_data.values())
        num_cols = 4
        num_rows = max(1, len(belief_values) // num_cols)
        if num_rows * num_cols < len(belief_values):
            num_rows += 1
        belief_values.extend([0] * (num_rows * num_cols - len(belief_values)))
        belief_matrix = np.array(belief_values).reshape(num_rows, num_cols)
        sns.heatmap(belief_matrix, ax=ax1, cmap='viridis',
                   xticklabels=False, yticklabels=False)
    else:
        ax1.text(0.5, 0.5, 'No belief data available',
                ha='center', va='center')
    ax1.set_title('Belief Context Strengths')
    
    # 2. Consciousness Layer Activity (Top Right)
    ax2 = plt.subplot(222)
    layer_activities = {
        'Conscious': np.mean(hbs.consciousness.conscious.activation_history),
        'Subconscious': np.mean(hbs.consciousness.subconscious.activation_history),
        'Unconscious': np.mean(hbs.consciousness.unconscious.activation_history)
    }
    ax2.bar(layer_activities.keys(), layer_activities.values())
    ax2.set_title('Consciousness Layer Activity')
    
    # 3. Knowledge Word Cloud (Bottom Left)
    ax3 = plt.subplot(223)
    word_weights = {}
    
    # Get concepts from learning layers
    if hasattr(hbs, 'learning_context'):
        for concept, weight in hbs.learning_context.learning_layers['conscious']['active_concepts'].items():
            word_weights[str(concept)] = float(weight)
    
    if word_weights:
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white'
        ).generate_from_frequencies(word_weights)
        ax3.imshow(wordcloud, interpolation='bilinear')
    else:
        ax3.text(0.5, 0.5, 'No concept data available',
                ha='center', va='center')
    ax3.axis('off')
    ax3.set_title('Knowledge Concepts')
    
    # 4. Relationship Network (Bottom Right)
    ax4 = plt.subplot(224)
    if hasattr(hbs, 'learning_context'):
        G = nx.Graph()
        # Get relationships from learning layers
        semantic_assoc = hbs.learning_context.learning_layers['subconscious']['semantic_associations']
        edge_count = 0
        for source, targets in semantic_assoc.items():
            if isinstance(targets, list):
                for target in targets[:3]:  # Limit to top 3 connections
                    if edge_count < 20:  # Limit total edges
                        G.add_edge(str(source), str(target))
                        edge_count += 1
        
        if G.number_of_edges() > 0:
            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=ax4, 
                    node_color='lightblue',
                    node_size=500,
                    font_size=8,
                    with_labels=True)
        else:
            ax4.text(0.5, 0.5, 'No relationship data available',
                    ha='center', va='center')
    ax4.set_title('Concept Relationships')
    
    plt.tight_layout()
    
    # Ensure plots directory exists
    os.makedirs('./plots', exist_ok=True)
    
    # Save the visualization
    plt.savefig('./plots/belief_systems.png')
    plt.close()
    
    return fig

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run personality simulation')
    parser.add_argument('--plot', action='store_true', help='Only load and plot results')
    args = parser.parse_args()
    
    if args.plot:
        # Load most recent state and plot
        previous_state = load_simulation_state()
        if previous_state and 'metrics' in previous_state:
            plot_learning_results(previous_state['metrics'])
            if 'hbs' in previous_state:
                visualize_belief_systems(previous_state['hbs'])
            plt.show()
        else:
            print("No previous state found to plot")
        return

    # Original simulation logic
    topics = [
        "Logic",
        "Proof",
        # "Constructive proof",
        "Proof by Induction",
        "Proof by Contradiction",
        # "Proof Theory",
        "Aesthetics",
        # "Ethics",
        "Linguistics",
        "Poetry",
        "Philosophy",
        "Epistemology",
        "Set Theory",
        "Intuitionistic type theory",
        "Math Theory",
        "Data Science",
        # "Computer Science",
        # "Physics",
        "Metaphysics",
        # "Spirituality",
        "History",
        # "Sutras",
        "Meditation",
        "Pranayama",
        "Tao",
        # "Tai chi",
        "Five precepts",
        # "The arts",
        "Art",
        "Psychology of art",
        # "Quantum mechanics",
        "Introduction to quantum mechanics",
        # "Interpretations of quantum mechanics",
        "Philosophy of physics",
        "Sustainability",
        # "Ecological systems theory",
        # "Sustainable development",
        "Ecosystem management",
        # "Ecological literacy",
        "Ethics of technology",
        "Technology and society",
        "Philosophy of technology",
        "Scientific method",
        # "History of scientific method",
        "Systems thinking",
        "Critical systems thinking",
        # "Creative entrepreneurship",
        "Entrepreneurial economics",
        # "Sustainable living",
        # "Minimalism",
        "Sustainable consumption",
        # "The Philosophy of Money",
        # "Personal finance",
        # "Finance",
        # "Investment",
        # "Financial literacy",
        "Investment strategy",
        "Return (finance)",
        "Exchange rate",
        # "Economics",
        # "Dividend",
        # "Interest",
        # "Asset",
        # "Cash flow",
        # "Diversification (finance)",
        # "Volatility (finance)",
        # "Financial risk",
        # "Personal boundaries",
        # "Professional boundaries",
        # "Interpersonal relationship",
        "Proxemics",
        # "Assertiveness",
        "Nonviolent communication",
        # "Morality",
        # "Right and wrong",
        "Sociology of culture",
        "Culture",
        # "Sociology",
        # "Cultural analysis",
        "Cultural identity",
        "Sociocultural evolution",
        "Attachment theory",
        # "Attachment in adults",
        # "Biology of romantic love",
        "Theories of love",
        "Affectional bond",
        # "Conflict resolution",
        "Mediation",
        "Dispute resolution",
        "Active listening",
        # "Listening",
        "Listening behaviour types",
        # "Communication",
        # "Reflective listening",
        # "Sarcasm",
        # "Humor",
        # "Empathy",
        "History of geometry",
        "Sacred geometry",
        # "Platonic solids",
        # "Tetractys",
        # "Cymatics",
        # "Sri Yantra",
        "Tree of life (Kabbalah)",
        "Theology",
        "Subtle body",
        # "Chakra",
        "Kundalini",
        # "Kundalini yoga",
        "Dantian",
        "Consciousness",
        "Universal mind",
        # "Panpsychism",
        "Electromagnetic theories of consciousness",
        "Inner peace",
        # "Contentment",
        # "Peace of Mind",
        "Peace",
        "Santosha",
        # "Calmness",
        # "List of religions and spiritual traditions",
        "Comparative religion",
        # "Religion",
        "World religions",
        "Ontology",
        "Ontological argument",
        "Existence",
        "Flow (psychology)",
        # "Motivation",
        "Motivated reasoning",
        "Motivation and Personality",
        # "Maslow's hierarchy of needs",
        "Neuroplasticity",
        "Addiction-related structural neuroplasticity",
        "Brain healing",
        # "Brain health and pollution",
        "Memory improvement",
        # "Activity-dependent plasticity",
        # "Creativity",
        # "Innovation",
        "Creativity techniques",
        "Memory improvement",
        # "Memorization",
        # "Memory and retention in learning",
        # "Memory technique",
        "List of cognitive biases",
        "Cognitive bias",
        "Heuristic (psychology)",
        # "Dunning–Kruger effect",
        "Motivated Reasoning",
        # "Cognitive bias mitigation",
        # "Anchoring effect",
        # "Psychological resilience",
        # "Family resilience",
        # "Resilience",
        "Community resilience",
        # "Mental toughness",
        "Goal setting",
        # "Goal",
        # "Creative visualization",
        "Objectives and key results",
        "Goal orientation",
        "Data and information visualization",
        "Time management",
        # "Self-reflection",
        "Reflective practice",
        # "Emotional intelligence",
        "Empathy quotient",
    ]
    
    try:
        metrics, context, final_hbs = simulate_knowledge_acquisition(topics)
        # Save simulation state
        save_path = save_simulation_state(final_hbs, context, metrics)
        print(f"Simulation state saved to: {save_path}")
        
        # Plot results if metrics exist
        if metrics and any(metrics.values()):
            plot_learning_results(metrics)
            visualize_belief_systems(final_hbs)
            plt.show()
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    