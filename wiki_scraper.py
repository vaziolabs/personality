import wikipedia
import nltk
from collections import defaultdict
import os

# Ensure NLTK data path is set
nltk_data_dir = os.path.expanduser('~/nltk_data')
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Now import NLTK modules
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet

class WikiKnowledgeBase:
    def __init__(self):
        self.knowledge_graph = defaultdict(dict)
        self.concept_embeddings = {}
        
    def fetch_topic(self, topic, depth=2):
        try:
            # First try searching for the exact topic
            search_results = wikipedia.search(topic, results=5)
            
            if not search_results:
                print(f"No results found for: {topic}")
                return None
            
            # Try to find best match from search results
            for result in search_results:
                if result.lower() == topic.lower():
                    page = wikipedia.page(result, auto_suggest=False)
                    break
            else:
                # If no exact match, use first result
                try:
                    page = wikipedia.page(search_results[0], auto_suggest=False)
                except:
                    # If first result fails, try others
                    for result in search_results[1:]:
                        try:
                            page = wikipedia.page(result, auto_suggest=False)
                            break
                        except:
                            continue
                    else:
                        print(f"Could not access any pages for: {topic}")
                        return None
            
            # Extract knowledge
            knowledge = {
                'summary': page.summary,
                'content': page.content,
                'links': {link: wikipedia.summary(link, sentences=2) 
                         for link in page.links[:5] if not link.startswith('List of')},
                'categories': page.categories,
                'title': page.title  # Add actual page title
            }
            
            # Process text content
            knowledge['processed'] = self._process_text(knowledge['content'])
            
            print(f"Retrieved article: {page.title}")
            return knowledge
            
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation for {topic}. Trying most relevant option...")
            # Try to find most relevant option based on topic name
            best_match = None
            for option in e.options:
                if topic.lower() in option.lower():
                    best_match = option
                    break
            
            if best_match:
                return self.fetch_topic(best_match)
            else:
                print(f"No relevant disambiguation option found for: {topic}")
                return None
            
        except Exception as e:
            print(f"Error fetching topic {topic}: {str(e)}")
            return None
            
    def _process_text(self, text):
        """Process text content for learning"""
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        # Extract key concepts and relationships
        concepts = []
        relationships = defaultdict(list)
        
        for i, (word, tag) in enumerate(tagged):
            if tag.startswith(('NN', 'VB', 'JJ')):
                concepts.append(word)
                
            if i > 0:  # Build relationships
                prev_word = tagged[i-1][0]
                relationships[prev_word].append(word)
                
        return {
            'concepts': concepts,
            'relationships': relationships
        }
