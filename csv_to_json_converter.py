import os
import csv
import json
import argparse
from tqdm import tqdm

def convert_csv_to_json(csv_file, json_file):
    """
    Convert a CSV file containing NFT data to JSON format
    
    Args:
        csv_file: Path to the input CSV file
        json_file: Path to the output JSON file
    """
    # Read CSV file
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    # Write JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Converted {len(data)} records from {csv_file} to {json_file}")
    return len(data)

def main():
    """
    Main function to convert CSV files to JSON
    """
    parser = argparse.ArgumentParser(description='Convert CSV files to JSON format')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save JSON files')
    parser.add_argument('--pattern', type=str, default='_links.csv', help='Pattern to match CSV files')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all CSV files in the input directory
    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith(args.pattern)]
    
    if not csv_files:
        print(f"No CSV files found with the pattern {args.pattern} in {args.input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to convert:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Convert each CSV file to JSON
    total_records = 0
    for csv_file in tqdm(csv_files, desc="Converting files"):
        input_file = os.path.join(args.input_dir, csv_file)
        output_file = os.path.join(args.output_dir, csv_file.replace('.csv', '.json'))
        records = convert_csv_to_json(input_file, output_file)
        total_records += records
    
    print(f"Conversion complete. Converted {total_records} records from {len(csv_files)} files.")

if __name__ == "__main__":
    main()
