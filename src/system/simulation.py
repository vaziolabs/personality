import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from hbs.hbs import HumanBehaviorSystem, SleepCycle
from hbs.consciousness.learning import LearningContext, TextLearningContext
from utils.wiki_scraper import WikiKnowledgeBase
from concurrent.futures import ThreadPoolExecutor
from utils.plot import visualize_context_and_memory, visualize_belief_systems, plot_learning_results, visualize_concept_network
from utils.file import save_simulation_state, load_simulation_state, repair_state_file
from datetime import datetime
import time
from datetime import timedelta

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
    """Simulate knowledge acquisition process with integrated reflection periods"""
    start_time = time.time()
    metrics = {
        'knowledge_depth': [],
        'knowledge_breadth': [],
        'cognitive_load': [],
        'understanding': [],
        'processing_times': [],
        'emotional_states': {
            'conscious': [],
            'subconscious': [],
            'unconscious': []
        },
        'belief_strengths': {
            'rational': [],
            'emotional': [],
            'instinctive': []
        },
        'pattern_activations': [],
        'semantic_memory_growth': [],
        'layer_integration': [],
        'consciousness_states': [],
        'learning_consolidation': [],
        'discoveries': [],  # Add tracking for imagination discoveries
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Initialize systems
    hbs = HumanBehaviorSystem()
    context = TextLearningContext()
    wiki = WikiKnowledgeBase()
    
    # Process topics in batches
    batch_size = 10
    total_topics = len(topics)
    processed_topics = 0
    
    for batch_start in range(0, total_topics, batch_size):
        batch_end = min(batch_start + batch_size, total_topics)
        batch_topics = topics[batch_start:batch_end]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(wiki.fetch_topic, topic): topic 
                for topic in batch_topics
            }
            
            for future in futures:
                topic = futures[future]
                topic_start = time.time()
                
                try:
                    knowledge = future.result()
                    if knowledge:
                        # Pre-processing state
                        pre_state = {
                            'emotional': hbs._get_normalized_emotional_value(),
                            'consciousness': hbs.consciousness._analyze_behavior_patterns(),
                            'semantic': len(hbs.consciousness.semantic_memory.get('concepts', {}))
                        }
                        
                        # Process knowledge
                        learning_results = hbs.process_text_knowledge(knowledge)
                        
                        # Post-processing state
                        post_state = {
                            'emotional': hbs._get_normalized_emotional_value(),
                            'consciousness': hbs.consciousness._analyze_behavior_patterns(),
                            'semantic': len(hbs.consciousness.semantic_memory.get('concepts', {}))
                        }
                        
                        # Update metrics with learning results
                        _update_simulation_metrics(metrics, hbs, learning_results, pre_state, post_state, topic_start)
                        processed_topics += 1
                        
                        # After each topic, allow for reflection/imagination
                        reflection_results = hbs.process_rest_period(duration=0.5)
                        
                        # Integrate discoveries into metrics
                        if 'discoveries' in reflection_results:
                            metrics['discoveries'].extend(reflection_results['discoveries'])
                            metrics['learning_consolidation'].append(
                                reflection_results.get('learning_consolidation', 0.0)
                            )
                        
                except Exception as e:
                    print(f'\rError processing topic {topic}: {str(e)}')
                    continue
                
                update_progress(processed_topics, total_topics, topic, start_time)
        
        # After each batch, longer reflection period
        batch_reflection = hbs.process_rest_period(duration=1.0)
        if 'discoveries' in batch_reflection:
            metrics['discoveries'].extend(batch_reflection['discoveries'])
            metrics['learning_consolidation'].append(
                batch_reflection.get('learning_consolidation', 0.0)
            )
    
    print('\n\nProcessing complete.')
    print(f'Processed {processed_topics} out of {total_topics} topics')
    return metrics, context, hbs

def _update_simulation_metrics(metrics, hbs, learning_results, pre_state, post_state, topic_start):
    """Update all simulation metrics"""
    # Track layer-specific metrics
    for layer in ['conscious', 'subconscious', 'unconscious']:
        metrics['emotional_states'][layer].append(
            getattr(hbs.consciousness, layer).emotional_state
        )
    
    # Track belief system metrics
    for belief_type in ['rational', 'emotional', 'instinctive']:
        metrics['belief_strengths'][belief_type].append(
            hbs.consciousness.conscious.belief.influence_weights[belief_type]
        )
    
    # Update standard metrics
    metrics['knowledge_depth'].append(learning_results['depth'])
    metrics['knowledge_breadth'].append(learning_results['breadth'])
    metrics['cognitive_load'].append(learning_results['cognitive_load'])
    metrics['understanding'].append(learning_results['understanding'])
    metrics['processing_times'].append(time.time() - topic_start)
    metrics['pattern_activations'].append(post_state['consciousness'])
    metrics['semantic_memory_growth'].append(
        post_state['semantic'] - pre_state['semantic']
    )
    metrics['layer_integration'].append(
        hbs.consciousness._integrate_results(learning_results)['integrated_strength']
    )
    metrics['consciousness_states'].append(post_state['consciousness'])
    metrics['learning_consolidation'].append(
        hbs.consciousness._calculate_consolidation_strength(learning_results)
    )

def update_progress(processed, total, current_topic, start_time):
    """Update progress bar with detailed metrics"""
    progress = (processed / total) * 100
    elapsed = time.time() - start_time
    avg_time = elapsed / processed if processed > 0 else 0
    eta = avg_time * (total - processed)
    
    bar_length = 40
    filled_length = int(bar_length * processed // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    eta_str = str(timedelta(seconds=int(eta)))
    
    print(f'\rProgress: |{bar}| {progress:.1f}% ({processed}/{total}) Topic: {current_topic:<30} Elapsed: {elapsed_str} ETA: {eta_str}', end='')
