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
            # Use wikipedia.page() directly
            page = wikipedia.page(topic)
            
            # Extract knowledge
            knowledge = {
                'summary': page.summary,
                'content': page.content,
                'links': {link: wikipedia.summary(link, sentences=2) 
                         for link in page.links[:5]},
                'categories': page.categories
            }
            
            # Process text content
            knowledge['processed'] = self._process_text(knowledge['content'])
            
            return knowledge
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation pages
            print(f"Disambiguation for {topic}. Options: {e.options[:5]}")
            if e.options:
                return self.fetch_topic(e.options[0])  # Try first option
            return None
            
        except wikipedia.exceptions.PageError:
            print(f"Page not found for topic: {topic}")
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
