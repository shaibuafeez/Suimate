import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import requests
import time
import random
from fake_useragent import UserAgent
from io import BytesIO
import base64
from tqdm import tqdm

class NFTEmbeddingGenerator:
    def __init__(self, model_name="google/vit-base-patch16-224", device=None):
        """
        Initialize the NFT embedding generator with a Vision Transformer model.
        
        Args:
            model_name: The name of the pre-trained model to use
            device: The device to run the model on (None for auto-detection)
        """
        # Set device (GPU if available, otherwise CPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the model and processor
        print(f"Loading {model_name} model...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print("Model loaded successfully!")
    
    def infer(self, image_url_or_obj):
        """
        Process an image from a URL or a PIL Image object and return embeddings.

        Args:
            image_url_or_obj: Either a URL string or a PIL Image object
        """
        if isinstance(image_url_or_obj, str):
            # Check if it's a base64 data URL
            if image_url_or_obj.startswith('data:image'):
                try:
                    # Extract the base64 part
                    header, encoded = image_url_or_obj.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                except Exception as e:
                    raise Exception(f"Error decoding base64 image: {e}")
            else:
                # It's a URL, need to download
                try:
                    # Handle IPFS URLs
                    if image_url_or_obj.startswith('ipfs://'):
                        image_url_or_obj = 'https://ipfs.io/ipfs/' + image_url_or_obj[7:]
                    
                    # Download the image from URL with proper headers
                    # Use fake-useragent to generate random user agent
                    ua = UserAgent()
                    headers = {"User-Agent": ua.random}

                    # Add retry mechanism with backoff
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(
                                image_url_or_obj, headers=headers, timeout=10
                            )
                            response.raise_for_status()
                            break
                        except requests.exceptions.RequestException as e:
                            if attempt < max_retries - 1:
                                # Wait before retrying (exponential backoff)
                                time.sleep(2**attempt)
                            else:
                                raise e

                    # Process the image directly from memory instead of saving to disk
                    image = Image.open(BytesIO(response.content)).convert("RGB")

                except Exception as e:
                    raise Exception(f"Error downloading or processing image from URL: {e}")
        else:
            # Assume it's already a PIL Image
            image = image_url_or_obj

        # Process through the model
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get embeddings from the last hidden state (CLS token)
        pooler_output = torch.mean(outputs.last_hidden_state, dim=1)

        return pooler_output.detach().cpu().numpy()

    def process_json_file(self, json_file, output_file, limit=None):
        """
        Process a JSON file containing image URLs and save embeddings
        
        Args:
            json_file: Path to the JSON file containing image URLs
            output_file: Path to save the embeddings
            limit: Optional limit on the number of NFTs to process
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Read the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        if limit:
            data = data[:limit]
        
        print(f"Processing {len(data)} NFTs from {json_file}...")
        
        # Create lists to store embeddings and metadata
        embeddings = []
        identifiers = []
        names = []
        
        # Process each NFT one at a time
        for i, item in enumerate(tqdm(data, desc=f"Processing {os.path.basename(json_file)}")):
            try:
                # Check if image_url exists
                if 'image_url' not in item:
                    print(f"Skipping item without image_url: {item}")
                    continue
                
                # Get embedding for the image
                embedding = self.infer(item['image_url'])
                
                # Save embedding and metadata
                embeddings.append(embedding[0])  # Get the first (and only) embedding
                
                # Handle different JSON formats
                if 'object_id' in item:
                    identifiers.append(item['object_id'])
                else:
                    # Use index as identifier if object_id is not available
                    identifiers.append(f"id_{i}")
                
                if 'name' in item:
                    names.append(item['name'])
                else:
                    # Use collection name from file if name is not available
                    collection_name = os.path.splitext(os.path.basename(json_file))[0].split('_')[0]
                    names.append(f"{collection_name}_{i}")
                    
                # Sleep a bit to avoid rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")
        
        if not embeddings:
            print(f"No valid embeddings extracted from {json_file}")
            return None
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Save embeddings and metadata
        np.savez(
            output_file,
            embeddings=embeddings_array,
            identifiers=np.array(identifiers),
            names=np.array(names)
        )
        
        print(f"Saved {len(embeddings)} embeddings to {output_file}")
        return embeddings_array.shape

def main():
    """
    Main function to generate embeddings for NFT collections
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate embeddings for NFT images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing JSON files with NFT data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save embeddings')
    parser.add_argument('--limit', type=int, help='Limit the number of NFTs to process per collection')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the embedding generator
    generator = NFTEmbeddingGenerator()
    
    # Find all JSON files in the input directory
    json_files = [f for f in os.listdir(args.input_dir) if f.endswith('_links.json')]
    
    if not json_files:
        print(f"No JSON files found with the pattern *_links.json in {args.input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for file in json_files:
        print(f"  - {file}")
    
    # Process each JSON file
    results = {}
    for json_file in json_files:
        collection_name = os.path.splitext(json_file)[0].split('_')[0]
        input_file = os.path.join(args.input_dir, json_file)
        output_file = os.path.join(args.output_dir, f"{collection_name}_embeddings.npz")
        
        print(f"\nProcessing {collection_name} collection...")
        shape = generator.process_json_file(input_file, output_file, args.limit)
        if shape:
            results[collection_name] = shape
    
    # Print summary
    if results:
        print("\nEmbedding Generation Summary:")
        print("-----------------------------")
        for collection, shape in results.items():
            print(f"{collection}: {shape[0]} embeddings of dimension {shape[1]}")
    else:
        print("\nNo embeddings were generated.")

if __name__ == "__main__":
    main()
