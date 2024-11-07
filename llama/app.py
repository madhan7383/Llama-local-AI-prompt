import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib.parse import urlparse
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='extraction.log'
)

class WebsiteInfoExtractor:
    def __init__(self, max_workers=5, timeout=120, ollama_url="http://localhost:11434"):
        self.max_workers = max_workers
        self.timeout = timeout
        self.ollama_url = ollama_url
        
    def query_ollama(self, text: str) -> List[str]:
        """
        Query Ollama for named entity recognition.
        """
        print(text)
        try:
            prompt = """Extract all person names from the following text. 
                       If no names are found, return an empty array. Text: """ + text
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "mistral",  # You can change this to your preferred model
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                try:
                    # Extract the response text from Ollama
                    result = response.json()["response"]
                    # Try to parse the JSON array from the response
                    names = json.loads(result)
                    if isinstance(names, list):
                        return names
                    return []
                except json.JSONDecodeError:
                    logging.error("Failed to parse Ollama response as JSON")
                    return []
            else:
                logging.error(f"Ollama API request failed with status code: {response.status_code}")
                return []
                
        except Exception as e:
            logging.error(f"Error querying Ollama: {str(e)}")
            return []
            
    def extract_website_text(self, url: str) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(['script', 'style']):
                script.decompose()
            
            text = soup.get_text(separator=' ')
            return ' '.join(text.split())
            
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return ""
            
    def extract_email(self, text: str) -> List[str]:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return list(set(re.findall(email_pattern, text)))

    def extract_names(self, text: str) -> List[str]:
        return self.query_ollama(text)

    def process_single_website(self, url: str) -> Dict:
        try:
            domain = urlparse(url).netloc
            text = self.extract_website_text(url)

            if not text:
                return {
                    'url': url,
                    'domain': domain,
                    'names': [],
                    'emails': [],
                    'status': 'success',
                    'error': 'No text extracted'
                }

            names = self.extract_names(text)
            print(f"Names found for {url}: {names}")
            emails = self.extract_email(text)
            print(f"Emails found for {url}: {emails}")
            
            return {
                'url': url,
                'domain': domain,
                'names': names,
                'emails': emails,
                'status': 'success',
                'error': None
            }
            
        except Exception as e:
            logging.error(f"Error processing {url}: {str(e)}")
            return {
                'url': url,
                'domain': urlparse(url).netloc,
                'names': [],
                'emails': [],
                'status': 'success',
                'error': str(e)
            }

    def process_websites(self, urls: List[str], output_file: str = 'extracted_data.csv') -> pd.DataFrame:
        results = []
        print(f"Starting to process {len(urls)} URLs with {self.max_workers} workers.")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.process_single_website, url): url for url in urls}
            
            completed_count = 0
            for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Processing websites"):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Successfully processed URL: {url}")
                    completed_count += 1
                except Exception as e:
                    logging.error(f"Error processing {url}: {str(e)}")
                    results.append({
                        'url': url,
                        'domain': urlparse(url).netloc,
                        'names': [],
                        'emails': [],
                        'status': 'success',
                        'error': str(e)
                    })
                    print(f"Failed to process URL: {url} with error: {str(e)}")
        
        print(f"Total URLs processed: {completed_count}/{len(urls)}")
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return df

def main():
    # Read URLs from the CSV file
    websites = pd.read_csv('web3.csv')

    # Filter out non-URL entries
    valid_websites = websites['url'].dropna().tolist()
    valid_websites = [url for url in valid_websites if url.startswith('http')]

    print(f"Starting to process {len(valid_websites)} websites...")

    # Initialize extractor
    extractor = WebsiteInfoExtractor(max_workers=10)

    # Process websites
    results_df = extractor.process_websites(valid_websites)

    # Print results
    print("\nExtraction completed!")
    print("\nSample of extracted data:")
    print(results_df.head())

if __name__ == "__main__":
    main()