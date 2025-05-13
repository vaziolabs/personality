import os
import nltk
from datetime import datetime
import argparse
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import sys
import traceback
from utils.serializer import SystemSerializer

training_iterations = 10

# Set up NLTK data path
nltk_data_dir = os.path.expanduser('./nltk_data')
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

import matplotlib.pyplot as plt
from utils.plot import visualize_belief_systems, plot_learning_results, visualize_concept_network
from utils.file import save_simulation_state, load_simulation_state, repair_state_file
from system.simulation import simulate_knowledge_acquisition
from utils.tests import test_imagination, test_interaction

def main():
    parser = argparse.ArgumentParser(description='Run personality simulation')
    parser.add_argument('--plot', nargs='?', const='all', choices=['all', 'concepts', 'beliefs', 'network'], 
                        help='Load and plot results. Optionally specify which plot: concepts, beliefs, network, or all (default)')
    parser.add_argument('--repair', action='store_true', help='Attempt to repair corrupted state files')
    parser.add_argument('--state', type=str, help='Load specific state file')
    parser.add_argument('--imagine', action='store_true', help='Test imagination system')
    parser.add_argument('--cycles', type=int, default=5, help='Number of imagination test cycles')
    parser.add_argument('--duration', type=float, default=1.0, help='Duration of each reflection period')
    parser.add_argument('--chat', action='store_true', help='Start interactive chat')
    parser.add_argument('--autonomous', action='store_true', help='Enable autonomous responses')
    args = parser.parse_args()
    
    try:
        # Create plots directory early
        os.makedirs('./plots', exist_ok=True)
        
        if args.repair:
            state_dir = '../saved_states'
            if os.path.exists(state_dir):
                for file in os.listdir(state_dir):
                    if file.startswith('system_state_') and file.endswith('.pkl'):
                        file_path = os.path.join(state_dir, file)
                        print(f"Attempting to repair: {file_path}")
                        repair_state_file(file_path)
            return

        if args.plot:
            # Load specific state file if provided
            state = None
            if args.state:
                serializer = SystemSerializer()
                state = serializer.load_system(args.state)
            else:
                state = load_simulation_state()
                
            if state and isinstance(state, dict):
                print("\nCreating plots from loaded state...")
                timestamp = state['metrics'].get('timestamp', 
                    datetime.now().strftime("%Y%m%d_%H%M%S"))
                    
                try:
                    if args.plot in ['all', 'learning']:
                        print("Plotting learning results...")
                        plot_learning_results(state['metrics'])
                        plt.close('all')
                    
                    if args.plot in ['all', 'beliefs']:
                        print("Visualizing belief systems...")
                        visualize_belief_systems(state['hbs'])
                        plt.close('all')
                    
                    if args.plot in ['all', 'concepts']:
                        print("Visualizing concept network...")
                        visualize_concept_network(
                            state['hbs'],
                            f'./plots/concept_network_{timestamp}.png'
                        )
                        plt.close('all')
                    
                    print("\nAll requested plots created successfully")
                except Exception as e:
                    print(f"Error creating plots: {str(e)}")
                    traceback.print_exc()
            else:
                print("No valid simulation state found to plot")
            return

        if args.imagine:
            print("\nLoading system state for imagination testing...")
            state = None
            if args.state:
                serializer = SystemSerializer()
                state = serializer.load_system(args.state)
            else:
                state = load_simulation_state()
                
            if state and isinstance(state, dict):
                discoveries = test_imagination(
                    state['hbs'],
                    duration=args.duration,
                    num_cycles=args.cycles
                )
                
                # Save discoveries to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f'./imagination_test_{timestamp}.txt'
                
                with open(save_path, 'w') as f:
                    f.write("Imagination Test Results\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, discovery in enumerate(discoveries, 1):
                        f.write(f"Discovery {i}:\n")
                        f.write(f"Type: {discovery['type']}\n")
                        
                        if discovery['type'] == 'direct':
                            conn = discovery['connection']
                            concepts = conn['concepts']
                            f.write(f"Concepts: {concepts[0]} {'<->' if not conn['opposition'] else '><'} {concepts[1]}\n")
                            f.write(f"Strength: {conn['strength']:.2f}\n")
                            
                        elif discovery['type'] == 'indirect':
                            f.write(f"Path: {' -> '.join(discovery['path'])}\n")
                            f.write(f"Strength: {discovery['strength']:.2f}\n")
                            f.write(f"Type: {discovery.get('relationship_type', 'unknown')}\n")
                            
                        elif discovery['type'] in ['cluster', 'central_concepts']:
                            f.write(f"Concepts: {', '.join(discovery['concepts'])}\n")
                            
                        f.write("\n")
                
                print(f"\nDetailed results saved to: {save_path}")
            else:
                print("No valid simulation state found for testing")
            return

        if args.chat or args.autonomous:
            print("\nLoading system state for interaction testing...")
            state = None
            if args.state:
                serializer = SystemSerializer()
                state = serializer.load_system(args.state)
            else:
                state = load_simulation_state()
                
            if state and isinstance(state, dict):
                test_interaction(
                    state['hbs'],
                    duration=args.duration if args.duration else 1.0,
                    interactive=args.chat
                )
            else:
                print("No valid simulation state found for testing")
            return

        # Filter out commented topics and create final topics list
        topics = [
            topic for topic in [
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
                # "Peace",
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
            ] if not topic.startswith('#')
        ]
        
        # Print total topics for verification
        print(f"\nTotal topics to process: {len(topics)}")
        
        # Run simulation 10 times
        final_metrics = None
        final_context = None 
        final_hbs = None

        for i in range(training_iterations):
            print(f"\nRunning simulation iteration {i+1}/{training_iterations}")
            metrics, context, hbs = simulate_knowledge_acquisition(topics)
            
            # Store results from last iteration
            final_metrics = metrics
            final_context = context
            final_hbs = hbs

        # Save final state with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_simulation_state(final_hbs, final_context, final_metrics, timestamp)
        if save_path:
            print(f"\nSimulation state saved to: {save_path}")
        else:
            print("\nWarning: Failed to save simulation state")
        
        # Plot results sequentially with immediate cleanup after each plot
        if final_metrics and any(final_metrics.values()):
            print("\nCreating plots...")
            
            print("Plotting learning results...")
            plot_learning_results(final_metrics)
            plt.close('all')
            
            print("Visualizing belief systems...")
            visualize_belief_systems(final_hbs)
            plt.close('all')
            
            print("Visualizing concept network...")
            timestamp = final_metrics['timestamp']
            visualize_concept_network(
                final_hbs, 
                f'./plots/concept_network_{timestamp}.png'
            )
            plt.close('all')
            
            print("\nAll plots created successfully")
            
    except Exception as e:
        print(f"\nSimulation failed: {str(e)}")
        traceback.print_exc()
    finally:
        plt.close('all')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        traceback.print_exc()
    finally:
        plt.close('all')
        sys.exit(0)
    