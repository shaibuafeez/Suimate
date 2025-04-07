import requests
import csv
import os
import time
from typing import List, Dict, Any

# API configuration
API_KEY = "2uulalvqIxowwmCkEMGozKfUmrW"  # Using the working API key
BASE_URL = "https://api.blockvision.org/v2/sui"
OBJECT_TYPE = "0x8f74a7d632191e29956df3843404f22d27bd84d92cca1b1abde621d033098769::rootlet::Rootlet"

def fetch_nft_batch(cursor=None, limit=20) -> Dict[str, Any]:
    """
    Fetch a batch of NFTs from the BlockVision API
    """
    url = f"{BASE_URL}/nft/list"
    
    params = {
        "objectType": OBJECT_TYPE,
        "limit": limit
    }
    
    if cursor:
        params["cursor"] = cursor
    
    headers = {
        "accept": "application/json",
        "x-api-key": API_KEY
    }
    
    print(f"Fetching NFTs with cursor: {cursor or 'initial'}")
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    return response.json()

def extract_image_links(nft_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract image links and names from the NFT data
    """
    image_data = []
    
    if "result" in nft_data and "data" in nft_data["result"]:
        nfts = nft_data["result"]["data"]
        
        for nft in nfts:
            name = nft.get("name", "Unknown")
            object_id = nft.get("objectId", "")
            
            if "imageURL" in nft and nft["imageURL"]:
                # Handle different URL formats
                image_url = nft["imageURL"]
                if image_url.startswith("ipfs://"):
                    image_url = "https://ipfs.io/ipfs/" + image_url[7:]
                
                image_data.append({
                    "name": name,
                    "object_id": object_id,
                    "image_url": image_url
                })
    
    return image_data

def save_to_csv(image_data: List[Dict[str, str]], filename="rootlet_1000_links.csv"):
    """
    Save the image links to a CSV file
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['name', 'object_id', 'image_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in image_data:
            writer.writerow(data)
    
    print(f"Saved {len(image_data)} links to {filename}")

def main():
    print("Extracting 1000 Rootlet NFT image links...")
    
    # Create a list to store all image data
    all_image_data = []
    
    # Fetch the first batch
    batch_data = fetch_nft_batch()
    
    if not batch_data:
        print("Failed to fetch NFTs")
        return
    
    # Extract image links from the first batch
    image_data = extract_image_links(batch_data)
    all_image_data.extend(image_data)
    
    print(f"Extracted {len(image_data)} links from first batch")
    
    # Get the cursor for the next batch
    cursor = None
    if "result" in batch_data and "cursor" in batch_data["result"]:
        cursor = batch_data["result"]["cursor"]
    
    # Fetch more batches to get 1000 NFTs (50 batches of 20)
    num_batches = 50  # This will fetch a total of 1000 NFTs
    
    for i in range(1, num_batches):
        if not cursor:
            print("No cursor found, stopping")
            break
        
        # Add a delay between API calls to avoid rate limits
        # Use a longer delay every 10 batches to avoid API throttling
        if i % 10 == 0:
            delay = 5
        else:
            delay = 2
        print(f"Waiting {delay} seconds before next API call... ({i}/{num_batches-1} batches completed)")
        time.sleep(delay)
        
        # Fetch the next batch
        batch_data = fetch_nft_batch(cursor)
        
        if not batch_data:
            print(f"Failed to fetch batch {i+1}")
            break
        
        # Extract image links from this batch
        image_data = extract_image_links(batch_data)
        all_image_data.extend(image_data)
        
        print(f"Extracted {len(image_data)} links from batch {i+1}")
        
        # Get the cursor for the next batch
        if "result" in batch_data and "cursor" in batch_data["result"]:
            cursor = batch_data["result"]["cursor"]
        else:
            print("No cursor found, stopping")
            break
    
    # Save all image links to CSV
    if all_image_data:
        save_to_csv(all_image_data)
        print(f"Total links extracted: {len(all_image_data)}")
    else:
        print("No image links were extracted")

if __name__ == "__main__":
    main()
