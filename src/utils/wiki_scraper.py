import wikipedia
import nltk
from collections import defaultdict
import os
import json
from bs4 import BeautifulSoup, GuessedAtParserWarning
import warnings

# Suppress GuessedAtParserWarning
warnings.filterwarnings('ignore', category=GuessedAtParserWarning)

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
        self.knowledge_base_dir = './knowledge-base'
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
    def _get_file_path(self, topic):
        """Generate sanitized file path for topic"""
        safe_name = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        return os.path.join(self.knowledge_base_dir, f"{safe_name}.json")
        
    def fetch_topic(self, topic):
        try:
            # Check cache first
            file_path = self._get_file_path(topic)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)

            # If not in cache, fetch from Wikipedia
            try:
                page = wikipedia.page(topic, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"Disambiguation for {topic}. Trying most relevant option...")
                # Try first option that contains original topic name
                options = [opt for opt in e.options if topic.lower() in opt.lower()]
                if options:
                    page = wikipedia.page(options[0], auto_suggest=False)
                else:
                    page = wikipedia.page(e.options[0], auto_suggest=False)

            # Extract knowledge
            knowledge = {
                'title': page.title,
                'content': page.content,
                'summary': page.summary,
                'links': self._process_links(page.links),
                'url': page.url,
                'references': page.references,
                'categories': page.categories[:10]  # Limit categories
            }
            
            # Cache the result
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge, f, ensure_ascii=False, indent=2)
            
            return knowledge
            
        except Exception as e:
            print(f"Error fetching topic {topic}: {str(e)}")
            return None

    def _process_links(self, links, max_links=10):
        """Process and filter relevant links"""
        processed_links = {}
        count = 0
        
        for link in links:
            if count >= max_links:
                break
            if not any(skip in link.lower() for skip in ['file:', 'help:', 'template:']):
                processed_links[link] = ''
                count += 1
                
        return processed_links
