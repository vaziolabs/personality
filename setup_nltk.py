import os
import sys
import urllib.request
import zipfile

def download_nltk_data():
    """Download NLTK data directly from NLTK servers"""
    # Create data directory with proper home directory expansion
    data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Create subdirectories for different types of data
    corpora_dir = os.path.join(data_dir, 'corpora')
    tokenizers_dir = os.path.join(data_dir, 'tokenizers')
    taggers_dir = os.path.join(data_dir, 'taggers')
    
    for directory in [corpora_dir, tokenizers_dir, taggers_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # NLTK data URLs with their target directories
    resources = {
        'wordnet': ('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip', corpora_dir),
        'punkt': ('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip', tokenizers_dir),
        'punkt_tab': ('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip', tokenizers_dir),
        'averaged_perceptron_tagger': ('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip', taggers_dir),
        'averaged_perceptron_tagger_eng': ('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip', taggers_dir),
        'stopwords': ('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip', corpora_dir)
    }
    
    for resource, (url, target_dir) in resources.items():
        extract_path = os.path.join(target_dir, resource)
        
        # Skip if resource already exists
        if os.path.exists(extract_path):
            continue
            
        zip_path = os.path.join(target_dir, f'{resource}.zip')
        
        try:
            # Download the zip file
            print(f"Downloading {resource}...")
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract the zip file
            print(f"Extracting {resource}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
                
            # Remove the zip file
            os.remove(zip_path)
            
            print(f"Successfully installed {resource} to {extract_path}")
            
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

if __name__ == "__main__":
    download_nltk_data()
