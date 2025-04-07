import requests
import csv
import os
import time
from typing import List, Dict, Any

# API configuration
API_KEY = "2uulalvqIxowwmCkEMGozKfUmrW"  # Using the working API key
BASE_URL = "https://api.blockvision.org/v2/sui"
OBJECT_TYPE = "0xc4f793bda2ce1db8a0626b5d3e189680bf7b17559bfe8389cd9db10d4e4d61dc::nft::KillaClubNFT"

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

def extract_image_links(nft_data: Dict[str, Any]) -> List[str]:
    """
    Extract just the image links from the NFT data
    """
    image_links = []
    
    if "result" in nft_data and "data" in nft_data["result"]:
        nfts = nft_data["result"]["data"]
        
        for nft in nfts:
            if "imageURL" in nft and nft["imageURL"]:
                image_links.append(nft["imageURL"])
    
    return image_links

def save_to_csv(image_links: List[str], filename="killaclub_1000_links.csv"):
    """
    Save the image links to a CSV file
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_url"])  # Header
        
        for link in image_links:
            writer.writerow([link])
    
    print(f"Saved {len(image_links)} links to {filename}")

def main():
    print("Extracting 1000 Killa Club NFT image links...")
    
    # Create a list to store all image links
    all_image_links = []
    
    # Fetch the first batch
    batch_data = fetch_nft_batch()
    
    if not batch_data:
        print("Failed to fetch NFTs")
        return
    
    # Extract image links from the first batch
    image_links = extract_image_links(batch_data)
    all_image_links.extend(image_links)
    
    print(f"Extracted {len(image_links)} links from first batch")
    
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
        image_links = extract_image_links(batch_data)
        all_image_links.extend(image_links)
        
        print(f"Extracted {len(image_links)} links from batch {i+1}")
        
        # Get the cursor for the next batch
        if "result" in batch_data and "cursor" in batch_data["result"]:
            cursor = batch_data["result"]["cursor"]
        else:
            print("No cursor found, stopping")
            break
    
    # Save all image links to CSV
    if all_image_links:
        save_to_csv(all_image_links)
        print(f"Total links extracted: {len(all_image_links)}")
    else:
        print("No image links were extracted")

if __name__ == "__main__":
    main()
