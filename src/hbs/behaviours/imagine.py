from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import numpy as np
from torch import combinations

class Imagination:
    def __init__(self, consciousness_system):
        self.cs = consciousness_system
        self._init_thresholds()
        self._concept_pattern_cache = {}
        self._belief_strength_cache = {}
        self.dmn_active = False
        self.dmn_thresholds = {
            'pattern_flexibility': 0.5,
            'association_strength': 0.4,
            'creative_connection': 0.3
        }

    def _init_thresholds(self):
        """Initialize configurable thresholds"""
        self.thresholds = {
            'belief': 0.7,
            'pattern': 0.6, 
            'connection': 0.5,
            'cluster': 0.8,
            'indirect': 0.4,
            'opposition': 0.3
        }

    def process_imagination(self, input_stimulus, context=None, emotional_state=None, personality_influence=None, thresholds=None):
        """Main imagination processing pipeline"""
        thresholds = thresholds or (self.dmn_thresholds if self.dmn_active else self.thresholds)
        
        # Process through consciousness layers sequentially
        layer_outputs = {}
        current_output = input_stimulus
        
        for layer_name in ['unconscious', 'subconscious', 'conscious']:
            current_output = self._process_layer(
                current_output,
                layer_name,
                thresholds,
                context=context,
                emotional_state=emotional_state,
                personality_influence=personality_influence
            )
            layer_outputs[layer_name] = current_output
            
        return layer_outputs['conscious']

    def reflect(self, duration=1.0):
        """Handle creative reflection period"""
        try:
            self.dmn_active = True
            active_concepts = self._gather_active_concepts()
            network = self._analyze_network_relationships(active_concepts)
            discoveries = self._process_discoveries(network)
            self._integrate_discoveries(discoveries)
            return discoveries
        finally:
            self.dmn_active = False
            self.clear_caches()

    def _process_discoveries(self, network):
        """Process all types of discoveries from network analysis"""
        discoveries = []
        
        # Process direct connections
        for concept, connections in network['direct_connections'].items():
            for conn in connections:
                discoveries.append({
                    'type': 'direct',
                    'connection': conn['connection']
                })
        
        # Process indirect connections
        for concept, connections in network['indirect_connections'].items():
            for conn in connections:
                discoveries.append({
                    'type': 'indirect',
                    'path': conn['path'],
                    'strength': conn['strength'],
                    'relationship_type': conn['type']
                })
        
        # Process clusters
        for cluster in network['clusters']:
            discoveries.append({
                'type': 'cluster',
                'concepts': list(cluster)
            })
        
        # Process central concepts
        discoveries.append({
            'type': 'central_concepts',
            'concepts': network['central_concepts']
        })
        
        return discoveries

    def _process_layer(self, input_stimulus, layer_name, thresholds, context=None, emotional_state=None, personality_influence=None):
        """Process imagination through a specific consciousness layer"""
        layer = self.cs.layers[layer_name]
        
        # Apply personality and emotional influences
        adjusted_input = self._adjust_for_influences(
            input_stimulus, 
            emotional_state, 
            personality_influence
        )
        
        # Process through layer
        if hasattr(layer, 'process_concept') and context is not None:
            output = layer.process_concept(adjusted_input, context)
        else:
            # Fallback if no context or process_concept not available
            output = adjusted_input
        
        return output

    def _adjust_for_influences(self, stimulus, emotional_state=None, personality_influence=None):
        """Adjust stimulus based on emotional and personality influences"""
        if stimulus is None:
            return stimulus
        
        # Apply influences based on available factors
        # Implementation depends on stimulus type
        return stimulus

    def _gather_active_concepts(self):
        """Gather concepts from semantic memory above threshold"""
        active_concepts = []
        for concept, strength in self.cs.semantic_memory['concepts'].items():
            if strength > self.thresholds['belief']:
                active_concepts.append(concept)
        return active_concepts

    def _analyze_relationship(self, concept_a, concept_b):
        """Analyze relationship between concepts"""
        pattern_sim = self._compare_patterns(concept_a, concept_b)
        semantic_conn = self._check_semantic_connection(concept_a, concept_b)
        
        return {
            'concepts': (concept_a, concept_b),
            'type': self._determine_relationship_type(
                self._get_belief_strength(concept_a),
                self._get_belief_strength(concept_b),
                pattern_sim,
                semantic_conn,
                self._get_current_threshold('connection')
            ),
            'strength': (pattern_sim + semantic_conn) / 2,
            'opposition': self._check_opposition(concept_a, concept_b)
        }

    def _get_belief_strength(self, concept):
        """Get cached belief strength"""
        if concept not in self._belief_strength_cache:
            strengths = []
            for layer in self.cs.layers.values():
                for category in layer.belief_systems.values():
                    for subcategory in category.values():
                        if concept in subcategory:
                            strengths.append(subcategory[concept])
            self._belief_strength_cache[concept] = (
                np.mean(strengths) if strengths else 0.0
            )
        return self._belief_strength_cache[concept]

    def _compare_patterns(self, concept_a, concept_b):
        """Compare concept patterns for similarity"""
        patterns_a = self._get_concept_patterns(concept_a)
        patterns_b = self._get_concept_patterns(concept_b)
        
        if not patterns_a or not patterns_b:
            return 0.0
            
        similarities = []
        for p_a in patterns_a:
            for p_b in patterns_b:
                similarity = np.mean(np.abs(p_a - p_b))
                similarities.append(similarity)
                
        return np.mean(similarities)

    def _check_semantic_connection(self, concept_a, concept_b):
        """Check semantic memory for existing connections"""
        relationships_a = self.cs.semantic_memory['relationships'].get(concept_a, [])
        relationships_b = self.cs.semantic_memory['relationships'].get(concept_b, [])
        
        direct_connection = concept_b in relationships_a or concept_a in relationships_b
        return 1.0 if direct_connection else 0.0

    def _determine_relationship_type(self, belief_a, belief_b, pattern_sim, semantic, connection_threshold):
        """Determine type of relationship between concepts"""
        if pattern_sim > connection_threshold:
            return 'similar'
        elif semantic > 0:
            return 'related'
        elif abs(belief_a - belief_b) > 0.7:
            return 'contrasting'
        return 'weak'

    def _check_opposition(self, concept_a, concept_b):
        """Check if concepts are opposing"""
        belief_a = self._get_belief_strength(concept_a)
        belief_b = self._get_belief_strength(concept_b)
        
        pattern_conflict = self._compare_patterns(concept_a, concept_b) < 0.3
        belief_conflict = abs(belief_a - belief_b) > 0.8
        
        return pattern_conflict and belief_conflict

    def _get_concept_patterns(self, concept):
        """Get cached concept patterns"""
        if concept not in self._concept_pattern_cache:
            patterns = []
            for layer in self.cs.layers.values():
                if hasattr(layer, 'patterns'):
                    for pattern_key, pattern in layer.patterns.items():
                        if concept in str(pattern):
                            patterns.append(pattern)
            self._concept_pattern_cache[concept] = patterns
        return self._concept_pattern_cache[concept]

    def _integrate_discoveries(self, discoveries):
        """Integrate discoveries into consciousness system"""
        for discovery in discoveries:
            if discovery['type'] == 'direct':
                self._integrate_direct_discovery(discovery)
            elif discovery['type'] == 'indirect':
                self._integrate_indirect_discovery(discovery)
            elif discovery['type'] == 'cluster':
                self._integrate_cluster_discovery(discovery)
            elif discovery['type'] == 'central_concepts':
                self._integrate_central_concepts(discovery)

    def _integrate_direct_discovery(self, discovery):
        """Integrate direct discovery into consciousness system"""
        concept_a, concept_b = discovery['connection']['concepts']
        
        # Update semantic memory relationships
        if concept_a not in self.cs.semantic_memory['relationships']:
            self.cs.semantic_memory['relationships'][concept_a] = []
        if concept_b not in self.cs.semantic_memory['relationships'][concept_a]:
            self.cs.semantic_memory['relationships'][concept_a].append(concept_b)
        
        # Strengthen or weaken beliefs based on opposition
        if discovery['opposition']:
            self._strengthen_opposing_beliefs(concept_a, concept_b)
        else:
            self._strengthen_supporting_beliefs(concept_a, concept_b)

    def _integrate_indirect_discovery(self, discovery):
        """Integrate indirect discovery into consciousness system"""
        concept_a, concept_b = discovery['connection']['concepts']
        
        # Update semantic memory relationships
        if concept_a not in self.cs.semantic_memory['relationships']:
            self.cs.semantic_memory['relationships'][concept_a] = []
        if concept_b not in self.cs.semantic_memory['relationships'][concept_a]:
            self.cs.semantic_memory['relationships'][concept_a].append(concept_b)
        
        # Strengthen or weaken beliefs based on opposition
        if discovery['opposition']:
            self._strengthen_opposing_beliefs(concept_a, concept_b)
        else:
            self._strengthen_supporting_beliefs(concept_a, concept_b)

    def _integrate_cluster_discovery(self, discovery):
        """Integrate cluster discovery into consciousness system"""
        concepts = discovery['concepts']
        
        # Update semantic memory relationships
        if concepts[0] not in self.cs.semantic_memory['relationships']:
            self.cs.semantic_memory['relationships'][concepts[0]] = []
        for concept in concepts[1:]:
            if concept not in self.cs.semantic_memory['relationships'][concepts[0]]:
                self.cs.semantic_memory['relationships'][concepts[0]].append(concept)
        
        # Strengthen or weaken beliefs based on opposition
        if discovery['opposition']:
            self._strengthen_opposing_beliefs(concepts[0], concepts[1])
        else:
            self._strengthen_supporting_beliefs(concepts[0], concepts[1])

    def _integrate_central_concepts(self, discovery):
        """Integrate central concepts discovery into consciousness system"""
        concepts = discovery['concepts']
        
        # Update semantic memory relationships
        if concepts[0] not in self.cs.semantic_memory['relationships']:
            self.cs.semantic_memory['relationships'][concepts[0]] = []
        for concept in concepts[1:]:
            if concept not in self.cs.semantic_memory['relationships'][concepts[0]]:
                self.cs.semantic_memory['relationships'][concepts[0]].append(concept)
        
        # Strengthen or weaken beliefs based on opposition
        if discovery['opposition']:
            self._strengthen_opposing_beliefs(concepts[0], concepts[1])
        else:
            self._strengthen_supporting_beliefs(concepts[0], concepts[1])

    def _strengthen_opposing_beliefs(self, concept_a, concept_b):
        """Strengthen beliefs when concepts oppose"""
        for layer in self.cs.layers.values():
            for category in layer.belief_systems.values():
                for subcategory in category.values():
                    if concept_a in subcategory:
                        subcategory[concept_a] = min(1.0, subcategory[concept_a] * 1.1)
                    if concept_b in subcategory:
                        subcategory[concept_b] = min(1.0, subcategory[concept_b] * 1.1)

    def _strengthen_supporting_beliefs(self, concept_a, concept_b):
        """Strengthen beliefs when concepts support each other"""
        for layer in self.cs.layers.values():
            belief_a = self._get_belief_strength(concept_a)
            belief_b = self._get_belief_strength(concept_b)
            
            avg_strength = (belief_a + belief_b) / 2
            for category in layer.belief_systems.values():
                for subcategory in category.values():
                    if concept_a in subcategory:
                        subcategory[concept_a] = min(1.0, subcategory[concept_a] + avg_strength * 0.1)
                    if concept_b in subcategory:
                        subcategory[concept_b] = min(1.0, subcategory[concept_b] + avg_strength * 0.1)

    def _analyze_network_relationships(self, active_concepts):
        """Analyze concept relationships and network structure"""
        return {
            'direct_connections': self._build_connection_graph(active_concepts),
            'indirect_connections': self._find_indirect_connections(active_concepts),
            'clusters': self._identify_concept_clusters(active_concepts),
            'central_concepts': self._identify_central_concepts(active_concepts)
        }

    def _build_connection_graph(self, concepts):
        """Build graph of direct connections more efficiently"""
        connections = defaultdict(list)
        # Use combinations instead of nested loops
        for concept_a, concept_b in combinations(concepts, 2):
            connection = self._analyze_relationship(concept_a, concept_b)
            if connection['strength'] > self.thresholds['connection']:
                connections[concept_a].append({
                    'target': concept_b,
                    'connection': connection
                })
                connections[concept_b].append({
                    'target': concept_a,
                    'connection': connection
                })
        return dict(connections)

    def _find_indirect_connections(self, direct_connections):
        """Find indirect connections using parallel processing"""
        indirect = defaultdict(list)
        
        def process_concept(concept_a):
            results = []
            for direct_conn in direct_connections.get(concept_a, []):
                concept_b = direct_conn['target']
                for secondary_conn in direct_connections.get(concept_b, []):
                    concept_c = secondary_conn['target']
                    if concept_c != concept_a:
                        indirect_strength = (
                            direct_conn['connection']['strength'] * 
                            secondary_conn['connection']['strength']
                        )
                        if indirect_strength > self.thresholds['indirect']:
                            results.append({
                                'source': concept_a,
                                'path': [concept_a, concept_b, concept_c],
                                'strength': indirect_strength,
                                'type': self._determine_indirect_relationship_type(
                                    direct_conn['connection'],
                                    secondary_conn['connection']
                                )
                            })
            return results

        # Process concepts in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_concept, c) 
                      for c in direct_connections.keys()]
            for future in as_completed(futures):
                for result in future.result():
                    indirect[result['source']].append(result)
                    
        return dict(indirect)

    def _determine_indirect_relationship_type(self, connection1, connection2):
        """Determine relationship type for indirect connections"""
        if connection1['type'] == connection2['type']:
            return connection1['type']
        elif connection1['opposition'] != connection2['opposition']:
            return 'conflicting'
        else:
            return 'complex'

    def _identify_concept_clusters(self, connections):
        """Identify clusters using networkx for better performance"""
        G = nx.Graph()
        
        # Build graph
        for source, targets in connections.items():
            for target in targets:
                if target['connection']['strength'] > self.thresholds['cluster']:
                    G.add_edge(source, target['target'])
        
        # Find communities using networkx
        return list(nx.community.greedy_modularity_communities(G))

    def _identify_central_concepts(self, network):
        """Identify central concepts using networkx metrics"""
        G = nx.Graph()
        
        # Build graph including both direct and indirect connections
        for source, targets in network['direct_connections'].items():
            for target in targets:
                G.add_edge(source, target['target'], 
                          weight=target['connection']['strength'])
                
        # Calculate centrality metrics
        degree_cent = nx.degree_centrality(G)
        between_cent = nx.betweenness_centrality(G)
        close_cent = nx.closeness_centrality(G)
        
        # Combine metrics
        concept_scores = {}
        for concept in G.nodes():
            concept_scores[concept] = (
                degree_cent[concept] + 
                between_cent[concept] + 
                close_cent[concept]
            ) / 3
            
        # Return concepts above average centrality
        avg_score = np.mean(list(concept_scores.values()))
        return [c for c, s in concept_scores.items() if s > avg_score]

    def clear_caches(self):
        """Clear cached data"""
        self._concept_pattern_cache.clear()
        self._belief_strength_cache.clear()

    def _get_current_threshold(self, threshold_type):
        """Get appropriate threshold based on DMN state"""
        return (
            self.dmn_thresholds.get(threshold_type, self.thresholds[threshold_type])
            if self.dmn_active
            else self.thresholds[threshold_type]
        )
