import time
from hbs.interaction import HumanBehaviorInterface
from hbs.consciousness.cohesion import ThoughtProcess


def test_imagination(hbs, duration=1.0, num_cycles=5):
    """Test imagination system during quiet/processing periods"""
    print("\nTesting Imagination System...")
    print("-" * 50)
    
    all_discoveries = []
    
    for cycle in range(num_cycles):
        print(f"\nReflection Cycle {cycle + 1}/{num_cycles}")
        print("-" * 30)
        
        # Process imagination period
        results = hbs.process_rest_period(duration)
        discoveries = results['discoveries']
        all_discoveries.extend(discoveries)
        
        # Print discoveries by type
        for discovery in discoveries:
            if discovery['type'] == 'direct':
                print(f"\nDirect Connection:")
                conn = discovery['connection']
                concepts = conn['concepts']
                print(f"  {concepts[0]} {'<->' if not conn['opposition'] else '><'} {concepts[1]}")
                print(f"  Strength: {conn['strength']:.2f}")
                
            elif discovery['type'] == 'indirect':
                print(f"\nIndirect Connection:")
                print(f"  Path: {' -> '.join(discovery['path'])}")
                print(f"  Strength: {discovery['strength']:.2f}")
                print(f"  Type: {discovery.get('relationship_type', 'unknown')}")
                
            elif discovery['type'] == 'cluster':
                print(f"\nConcept Cluster:")
                print(f"  Concepts: {', '.join(discovery['concepts'][:5])}")
                if len(discovery['concepts']) > 5:
                    print(f"  ...and {len(discovery['concepts'])-5} more")
                    
            elif discovery['type'] == 'central_concepts':
                print(f"\nCentral Concepts:")
                print(f"  {', '.join(discovery['concepts'][:5])}")
                if len(discovery['concepts']) > 5:
                    print(f"  ...and {len(discovery['concepts'])-5} more")
    
    # Print summary statistics
    print("\nImagination Test Summary")
    print("-" * 50)
    print(f"Total Discoveries: {len(all_discoveries)}")
    
    discovery_types = {}
    for d in all_discoveries:
        discovery_types[d['type']] = discovery_types.get(d['type'], 0) + 1
    
    print("\nDiscoveries by Type:")
    for dtype, count in discovery_types.items():
        print(f"  {dtype}: {count}")
    
    return all_discoveries

def test_interaction(hbs, duration=1.0, interactive=True):
    """Test interactive chat or autonomous responses"""
    print("\nTesting Interaction System...")
    print("-" * 50)
    
    # Use HBS's built-in methods for processing
    thought_process = ThoughtProcess(hbs.consciousness)
    
    while True:
        if interactive:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
                
            # Process user input using HBS's built-in method
            response = hbs.process_input(user_input, 'text')
            
            if response:
                print(f"\nBot: {response['content']}")
                print(f"Emotional Value: {response['emotional_value']:.2f}")
                print(f"Belief Alignment: {response['belief_alignment']:.2f}")
                print(f"Motivation: {response['motivation_strength']:.2f}")
            else:
                print("\nBot: Sorry, I couldn't process that input.")
            
        else:
            # Generate autonomous response based on internal state
            results = hbs.process_rest_period(duration)
            
            if results.get('discoveries'):
                if 'response' in results:
                    response = results['response']
                    print(f"\nBot (autonomous): {response['content']}")
                else:
                    print("\nBot made some discoveries but couldn't formulate a response.")
                
            # Pause between autonomous responses
            time.sleep(duration)