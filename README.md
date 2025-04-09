# 3Mate NFT Tools

This repository contains tools for working with NFT collections on the Sui blockchain, including data extraction scripts and an image classification toolkit.

## NFT Collections

The following NFT collections are included:

1. **Killa Club** - Object type: `0x5fb75feb424b7fd3c7fd0d3cd3c5c9d9f03c8675a115c5c4e42af2c3a50b2f4e::killa_club::KillaClub`
   - Uses IPFS URLs for images
   - CSV file: `killaclub_links.csv`

2. **Kumo** - Object type: `0x5d0ced2b9a10153a68b0ba8d3db8bb347d64d70b0c3f6865c4b5d2d8b1d5a7a5::kumo::Kumo`
   - Uses IPFS URLs for images
   - CSV file: `kumo_1000_links.csv`

3. **Prime Machine** - Object type: `0x034c162f6b594cb5a1805264dd01ca5d80ce3eca6522e6ee37fd9ebfb9d3ddca::factory::PrimeMachin`
   - Uses SM.xyz URLs for images (`https://img.sm.xyz/{object_id}/`)
   - CSV file: `primemachine_1000_links.csv`

4. **Rootlet** - Object type: `0x8f74a7d632191e29956df3843404f22d27bd84d92cca1b1abde621d033098769::rootlet::Rootlet`
   - Uses base64-encoded data URLs for images
   - CSV file: `rootlet_1000_links.csv`

5. **Aeon** - Object type: `0x141d8a2333f9369452fe075331924bb98d2abf0ee98de941db85aaf809c4ef54::aeon::Aeon`
   - Uses base64-encoded WebP images
   - CSV file: `aeon_1000_links.csv`

6. **DoubleUp Citizen** - Object type: `0x862810efecf0296db2e9df3e075a7af8034ba374e73ff1098e88cc4bb7c15437::doubleup_citizens::DoubleUpCitizen`
   - Uses CloudFront CDN URLs
   - CSV file: `doubleup_1000_links.csv`

## Data Collection Scripts

- `extract_killaclub_links.py` - Extracts Killa Club NFT image links
- `extract_kumo_links.py` - Extracts Kumo NFT image links
- `extract_primemachine_links.py` - Extracts Prime Machine NFT image links
- `extract_rootlet_links.py` - Extracts Rootlet NFT image links

Each script can be run independently to extract NFT image links from its respective collection:

```bash
python extract_killaclub_links.py
python extract_kumo_links.py
python extract_primemachine_links.py
python extract_rootlet_links.py
```

The scripts will save the extracted links to CSV files with columns for:
- NFT name
- Object ID
- Image URL

## NFT Image Classification Toolkit

This toolkit enables generating embeddings from NFT images and training classification models using these embeddings.

### Features

- Generate embeddings for NFT images using Vision Transformer (ViT) models
- Support for various image URL formats:
  - IPFS URLs (`ipfs://...`)
  - Base64-encoded data URLs (`data:image/png;base64,...`)
  - Standard HTTP/HTTPS URLs
- Train classification models to identify NFT collections based on visual features
- Convert CSV files containing NFT data to JSON format

### Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Other dependencies listed in `requirements.txt`

### Classification Pipeline

#### 1. Convert CSV files to JSON (if needed)

If your NFT data is in CSV format, you can convert it to JSON:

```bash
python csv_to_json_converter.py --input_dir /path/to/csv/files --output_dir /path/to/output/json
```

#### 2. Generate Embeddings

Generate embeddings for your NFT collections:

```bash
python nft_embeddings_generator.py --input_dir /path/to/json/files --output_dir /path/to/output/embeddings
```

This will:
- Load each JSON file containing NFT data
- Process each NFT image (one at a time to avoid memory issues)
- Generate embeddings using the Vision Transformer model
- Save the embeddings and metadata to NPZ files

#### 3. Train Classification Model

Train a classification model using the generated embeddings:

```bash
python nft_classification_model.py --embedding_dir /path/to/embeddings --output_dir /path/to/output/model
```

This will:
- Load the embeddings from the NPZ files
- Split the data into training and validation sets
- Train a neural network classifier
- Save the trained model, class mapping, and training history

### Model Architecture

The classification model uses a simple feedforward neural network:
- Input layer: 768 dimensions (ViT embedding size)
- Hidden layer 1: 512 neurons with ReLU activation
- Dropout layer (0.3)
- Hidden layer 2: 256 neurons with ReLU activation
- Output layer: Number of classes (collections) with softmax activation

## License

MIT

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for the Vision Transformer model
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [BlockVision API](https://blockvision.org/) for Sui blockchain data
