import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from collections import defaultdict
from datetime import datetime
import numpy as np

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
    """Plot learning metrics with proper cleanup"""
    try:
        # Validate metrics data
        if not all(len(metrics.get(key, [])) > 0 for key in ['knowledge_depth', 'knowledge_breadth', 'cognitive_load', 'understanding']):
            print("Warning: Some metrics data is empty")
            return
            
        # Create figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Learning Results', fontsize=16)
        
        # Plot with indices for x-axis
        for i, (metric_name, ax) in enumerate([
            ('knowledge_depth', axes[0,0]),
            ('knowledge_breadth', axes[0,1]),
            ('cognitive_load', axes[1,0]),
            ('understanding', axes[1,1])
        ]):
            data = metrics[metric_name]
            x = range(len(data))
            ax.plot(x, data, label=metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('Learning Steps')
            ax.set_ylabel('Value')
            
        plt.tight_layout()
        
        # Save plot with timestamp
        timestamp = metrics.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        save_path = f'./plots/learning_results_{timestamp}.png'
        plt.savefig(save_path)
        print(f"Saved learning results plot to: {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"Error plotting learning results: {e}")
        plt.close('all')

def visualize_concept_network(hbs, filename):
    """Visualize concept network across consciousness layers with clear relationships"""
    try:
        # Create a single graph for better community detection
        G = nx.Graph()
        
        # Track nodes by layer for coloring
        node_layers = {}
        node_weights = {}
        
        # Extract concepts and relationships from each layer
        for layer_name in ['conscious', 'subconscious', 'unconscious']:
            layer = getattr(hbs.consciousness, layer_name)
            if hasattr(layer, 'semantic_memory'):
                # Add concepts by weight
                if 'concepts' in layer.semantic_memory:
                    sorted_concepts = sorted(
                        layer.semantic_memory['concepts'].items(),
                        key=lambda x: float(x[1]),
                        reverse=True
                    )[:150]  # More concepts per layer
                    
                    for concept, weight in sorted_concepts:
                        if concept and isinstance(concept, str):
                            G.add_node(concept)
                            node_layers[concept] = layer_name
                            node_weights[concept] = float(weight)
                
                # Add relationships
                if 'relationships' in layer.semantic_memory:
                    for source in layer.semantic_memory['relationships']:
                        targets = layer.semantic_memory['relationships'][source]
                        for target in targets:
                            if source in G and target in G:
                                G.add_edge(source, target, weight=2.0, layer=layer_name)
        
        # Add cross-layer relationships
        if hasattr(hbs.consciousness, 'semantic_memory') and 'relationships' in hbs.consciousness.semantic_memory:
            for source in hbs.consciousness.semantic_memory['relationships']:
                targets = hbs.consciousness.semantic_memory['relationships'][source]
                for target in targets:
                    if source in G and target in G and not G.has_edge(source, target):
                        G.add_edge(source, target, weight=1.0, layer='cross')
        
        if len(G.nodes()) == 0:
            print("Warning: No concepts found in memory structures")
            return
            
        # Always add test edges to ensure visibility
        nodes = list(G.nodes())
        for i in range(min(20, len(nodes))):
            for j in range(i+1, min(i+4, len(nodes))):
                G.add_edge(nodes[i], nodes[j], weight=3.0, layer='test-visible')

        # Detect communities for clustering
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # Assign community IDs to nodes
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
        
        # Layer-based positioning - with distinct regions
        layer_positions = {
            'conscious': np.array([-5, 5]),     # Top left
            'subconscious': np.array([0, 0]),   # Center
            'unconscious': np.array([5, -5])    # Bottom right
        }
        
        # Use separate force-directed layout for each layer
        layer_graphs = {layer: nx.Graph() for layer in ['conscious', 'subconscious', 'unconscious']}
        
        # Add nodes and edges to layer-specific graphs
        for node in G.nodes():
            layer = node_layers.get(node, 'conscious')
            if layer in layer_graphs:
                layer_graphs[layer].add_node(node)
                
        for u, v, d in G.edges(data=True):
            u_layer = node_layers.get(u, 'conscious')
            v_layer = node_layers.get(v, 'conscious')
            
            # If both nodes are in the same layer
            if u_layer == v_layer:
                layer_graphs[u_layer].add_edge(u, v)
        
        # Compute layouts for each layer separately
        layer_layouts = {}
        for layer, graph in layer_graphs.items():
            if len(graph.nodes()) > 0:
                layer_layouts[layer] = nx.spring_layout(graph, k=0.3, iterations=50, seed=42)
            else:
                layer_layouts[layer] = {}
        
        # Combine layouts with layer positioning
        pos = {}
        scale_factor = 2.0
        for node in G.nodes():
            layer = node_layers.get(node, 'conscious')
            layer_center = layer_positions[layer]
            
            if node in layer_layouts.get(layer, {}):
                node_offset = layer_layouts[layer][node] * scale_factor
                pos[node] = layer_center + node_offset
            else:
                pos[node] = layer_center + np.random.normal(0, 0.3, 2)
        
        # Start with a fresh figure
        plt.figure(figsize=(36, 24), dpi=100)
        
        # IMPROVED EDGE DRAWING APPROACH - Group by layer and draw in batches
        layer_colors = {
            'conscious': '#0000FF',    # Blue
            'subconscious': '#00FF00', # Green
            'unconscious': '#FF8000',  # Orange
            'cross': '#FF0000',        # Red
            'test-visible': '#FFFF00'  # Yellow
        }
        
        # Group edges by layer
        edge_groups = defaultdict(list)
        for u, v, d in G.edges(data=True):
            layer = d.get('layer', 'cross')
            edge_groups[layer].append((u, v))
        
        # Draw edges by group with proper z-order
        layer_draw_order = ['cross', 'unconscious', 'subconscious', 'conscious', 'test-visible']
        
        # First draw ALL edges as a background
        all_edges = list(G.edges())
        nx.draw_networkx_edges(
            G, pos,
            edgelist=all_edges,
            width=4.0,
            edge_color='#444444',
            style='solid',
            alpha=0.5,
            arrows=False
        )
        
        # Now draw each layer's edges with proper styling
        for layer in layer_draw_order:
            if layer in edge_groups:
                edges = edge_groups[layer]
                color = layer_colors.get(layer, '#000000')
                
                # Style based on layer
                if layer == 'test-visible':
                    width = 6.0
                    alpha = 1.0
                    style = 'solid'
                elif layer in ['conscious', 'subconscious', 'unconscious']:
                    width = 5.0
                    alpha = 0.9
                    style = 'solid'
                else:  # cross-layer connections
                    width = 3.5
                    alpha = 0.8
                    style = 'dashed'
                
                # Draw all edges of this layer at once
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edges,
                    width=width,
                    edge_color=color,
                    style=style,
                    alpha=alpha,
                    arrows=True,
                    arrowsize=20,
                    arrowstyle='-|>'
                )
        
        # Draw smaller nodes to avoid obscuring edges
        for layer_name in ['conscious', 'subconscious', 'unconscious']:
            layer_nodes = [node for node in G.nodes() if node_layers.get(node) == layer_name]
            if not layer_nodes:
                continue
                
            # Smaller node sizes
            sizes = [max(150, node_weights.get(node, 0.1) * 800) for node in layer_nodes]
            
            # Color nodes by community
            colors = [plt.cm.tab20(community_map.get(node, 0) % 20) for node in layer_nodes]
            
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=layer_nodes,
                node_size=sizes,
                node_color=colors,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5
            )
        
        # Draw more visible labels
        sorted_nodes = sorted(G.nodes(), key=lambda x: node_weights.get(x, 0), reverse=True)
        top_nodes = sorted_nodes[:150]  # More labels
        labels = {node: node for node in top_nodes}
        
        nx.draw_networkx_labels(
            G, pos, 
            labels=labels, 
            font_size=10,
            font_weight='bold', 
            font_family='sans-serif',
            bbox=dict(
                facecolor='white', 
                alpha=0.8,
                edgecolor='black', 
                boxstyle='round,pad=0.3'
            )
        )
        
        # Add legends
        layer_legend = [
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='blue', 
                    markersize=15, label='Conscious Concepts'),
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='green', 
                    markersize=15, label='Subconscious Concepts'),
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='orange', 
                    markersize=15, label='Unconscious Concepts')
        ]
        
        edge_legend = [
            plt.Line2D([0], [0], color='blue', lw=4, label='Conscious Connections'),
            plt.Line2D([0], [0], color='green', lw=4, label='Subconscious Connections'),
            plt.Line2D([0], [0], color='orange', lw=4, label='Unconscious Connections'),
            plt.Line2D([0], [0], color='red', lw=3, linestyle='dashed', label='Cross-layer Connections'),
            plt.Line2D([0], [0], color='yellow', lw=6, label='Test Connections')
        ]
        
        # Community legend - fewer items
        community_legend = []
        for i in range(min(5, len(communities))):
            community_legend.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.tab20(i % 20),
                          markersize=10, 
                          label=f'Concept Cluster {i+1}')
            )
        
        # Place legends in different corners
        first_legend = plt.legend(handles=layer_legend, loc='upper left', title="Consciousness Layers")
        plt.gca().add_artist(first_legend)
        
        second_legend = plt.legend(handles=edge_legend, loc='upper right', title="Connection Types")
        plt.gca().add_artist(second_legend)
        
        plt.legend(handles=community_legend, loc='lower right', title="Major Concept Clusters")
        
        plt.title(f'Multi-layered Consciousness Knowledge Network', fontsize=24)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved concept network to: {filename}")
        plt.close('all')
        
    except Exception as e:
        print(f"Error visualizing concept network: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')

def visualize_belief_systems(hbs):
    """Visualize belief systems with optimized performance"""
    try:
        if not hasattr(hbs.consciousness, 'conscious') or not hbs.consciousness.conscious.belief_contexts:
            print("Warning: No belief systems data available")
            return
            
        # Limit data size
        max_beliefs = 1000
        belief_matrices = list(hbs.consciousness.conscious.belief_contexts.values())[:max_beliefs]
        if not belief_matrices:
            print("Warning: Empty belief contexts")
            return
            
        # Pre-allocate fixed size array
        first_matrix = belief_matrices[0]
        max_dims = (min(len(belief_matrices), max_beliefs), *first_matrix.shape)
        belief_data = np.zeros(max_dims)
        
        # Fill array with bounds checking
        for i, matrix in enumerate(belief_matrices[:max_beliefs]):
            belief_data[i] = matrix
            
        # Create single figure and compute once
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Belief Systems Analysis', fontsize=16)
        
        # Compute metrics efficiently
        with np.errstate(invalid='ignore'):
            avg_beliefs = np.nanmean(belief_data, axis=0)
            belief_var = np.nanvar(belief_data, axis=0)
            time_series = np.nanmean(belief_data, axis=(1,2))
        
        # Plot with optimized calls
        sns.heatmap(avg_beliefs, ax=axes[0,0], cmap='viridis')
        axes[0,0].set_title('Average Belief Strength')
        
        sns.heatmap(belief_var, ax=axes[0,1], cmap='viridis')
        axes[0,1].set_title('Belief Variance')
        
        axes[1,0].plot(time_series)
        axes[1,0].set_title('Belief Evolution')
        axes[1,0].grid(True)
        
        sns.histplot(belief_data.ravel(), ax=axes[1,1], bins=50)
        axes[1,1].set_title('Belief Distribution')
        
        plt.tight_layout()
        plt.savefig(f'./plots/belief_systems_{datetime.now():%Y%m%d_%H%M%S}.png')
        plt.close(fig)
        
    except Exception as e:
        print(f"Error visualizing belief systems: {e}")
        plt.close('all')

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
