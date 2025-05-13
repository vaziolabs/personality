import os
import pickle
from datetime import datetime
from hbs.hbs import HumanBehaviorSystem
from hbs.consciousness.learning import LearningContext, TextLearningContext
from utils.serializer import SystemSerializer

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
    
    filepath = f'../saved_states/error_state.pkl'
    SystemSerializer.save_system(error_state, filepath)
    print(f"Saved error state to: {filepath}")

def save_simulation_state(hbs, context, metrics, timestamp=None):
    """Save the complete simulation state"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory if it doesn't exist
    os.makedirs('../saved_states', exist_ok=True)
    
    filepath = f'../saved_states/system_state_{timestamp}.pkl'
    try:
        serializer = SystemSerializer()
        # First serialize the state components
        serialized_state = serializer.save_system({
            'hbs': hbs,
            'learning_context': context,
            'metrics': metrics,
            'timestamp': timestamp
        }, filepath)
        return filepath
    except Exception as e:
        print(f"Error saving state: {str(e)}")
        save_error_state(hbs, context, metrics)  # Save error state as backup
        return None

def create_empty_state():
    """Create a minimal valid state structure"""
    return {
        'metrics': {
            'knowledge_depth': [],
            'knowledge_breadth': [],
            'cognitive_load': [],
            'understanding': [],
            'processing_times': [],
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        },
        'hbs': HumanBehaviorSystem(),
        'learning_context': TextLearningContext()
    }

def repair_state_file(file_path):
    """Attempt to repair or recreate a corrupted state file"""
    try:
        # Try to partially load the file
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Create backup of corrupted file
        backup_path = file_path + '.backup'
        with open(backup_path, 'wb') as f:
            f.write(data)
            
        print(f"Created backup of corrupted file: {backup_path}")
        
        # Create new valid state
        new_state = create_empty_state()
        
        # Save new valid state
        with open(file_path, 'wb') as f:
            pickle.dump(new_state, f)
            
        print(f"Created new valid state file: {file_path}")
        return new_state
        
    except Exception as e:
        print(f"Error repairing state file: {str(e)}")
        return None

def load_simulation_state():
    """Load the most recent simulation state"""
    try:
        state_dir = '../saved_states'
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
            return create_empty_state()
            
        state_files = sorted([
            f for f in os.listdir(state_dir) 
            if f.startswith('system_state_') and f.endswith('.pkl')
        ], reverse=True)
        
        if not state_files:
            return create_empty_state()
            
        serializer = SystemSerializer()
        
        for state_file in state_files:
            file_path = os.path.join(state_dir, state_file)
            try:
                # Load and deserialize in one step
                state = serializer.load_system(file_path)
                if state and isinstance(state, dict) and 'hbs' in state:
                    return state
            except Exception as e:
                print(f"Error loading {state_file}: {str(e)}")
                continue
                
        return create_empty_state()
        
    except Exception as e:
        print(f"Error accessing saved states: {str(e)}")
        return create_empty_state()