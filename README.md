# NFT Image Classification

A toolkit for generating embeddings from NFT images and training classification models using these embeddings. This project is designed to work with NFT collections from the Sui blockchain but can be adapted to other blockchains as well.

## Features

- Generate embeddings for NFT images using Vision Transformer (ViT) models
- Support for various image URL formats:
  - IPFS URLs (`ipfs://...`)
  - Base64-encoded data URLs (`data:image/png;base64,...`)
  - Standard HTTP/HTTPS URLs
- Train classification models to identify NFT collections based on visual features
- Convert CSV files containing NFT data to JSON format

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Other dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/3MateLabs/nft_image_classification.git
cd nft_image_classification

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Convert CSV files to JSON (if needed)

If your NFT data is in CSV format, you can convert it to JSON:

```bash
python csv_to_json_converter.py --input_dir /path/to/csv/files --output_dir /path/to/output/json
```

### 2. Generate Embeddings

Generate embeddings for your NFT collections:

```bash
python nft_embeddings_generator.py --input_dir /path/to/json/files --output_dir /path/to/output/embeddings
```

This will:
- Load each JSON file containing NFT data
- Process each NFT image (one at a time to avoid memory issues)
- Generate embeddings using the Vision Transformer model
- Save the embeddings and metadata to NPZ files

### 3. Train Classification Model

Train a classification model using the generated embeddings:

```bash
python nft_classification_model.py --embedding_dir /path/to/embeddings --output_dir /path/to/output/model
```

This will:
- Load the embeddings from the NPZ files
- Split the data into training and validation sets
- Train a neural network classifier
- Save the trained model, class mapping, and training history

## Model Architecture

The classification model uses a simple feedforward neural network:
- Input layer: 768 dimensions (ViT embedding size)
- Hidden layer 1: 512 neurons with ReLU activation
- Dropout layer (0.3)
- Hidden layer 2: 256 neurons with ReLU activation
- Output layer: Number of classes (collections) with softmax activation

## Supported NFT Collections

This toolkit has been tested with the following Sui blockchain NFT collections:
- Kumo (IPFS URLs)
- Killa Club (IPFS URLs)
- Prime Machine (HTTP URLs from SM.xyz)
- Rootlet (Base64-encoded PNGs)
- Aeon (Base64-encoded WebP)
- DoubleUp Citizen (HTTP URLs from CloudFront CDN)

## License

MIT

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for the Vision Transformer model
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [BlockVision API](https://blockvision.org/) for Sui blockchain data
