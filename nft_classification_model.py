import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class NFTEmbeddingDataset(Dataset):
    """
    Dataset for loading NFT embeddings and labels
    """
    def __init__(self, embedding_files, class_mapping=None):
        """
        Initialize the dataset with a list of embedding files
        
        Args:
            embedding_files: List of paths to NPZ files containing embeddings
            class_mapping: Optional dictionary mapping collection names to class indices
        """
        self.embeddings = []
        self.labels = []
        self.identifiers = []
        self.names = []
        
        # If no class mapping is provided, create one
        if class_mapping is None:
            class_mapping = {}
            current_class = 0
        
        # Load embeddings from each file
        for file_path in embedding_files:
            # Extract collection name from file path
            collection_name = os.path.basename(file_path).split('_')[0]
            
            # Assign class index if not already in mapping
            if collection_name not in class_mapping:
                class_mapping[collection_name] = len(class_mapping)
            
            # Load embeddings
            data = np.load(file_path)
            file_embeddings = data['embeddings']
            file_identifiers = data['identifiers']
            file_names = data['names']
            
            # Create labels (all embeddings from the same file have the same label)
            file_labels = np.full(len(file_embeddings), class_mapping[collection_name])
            
            # Add to dataset
            self.embeddings.append(file_embeddings)
            self.labels.append(file_labels)
            self.identifiers.append(file_identifiers)
            self.names.append(file_names)
        
        # Concatenate all data
        self.embeddings = np.concatenate(self.embeddings)
        self.labels = np.concatenate(self.labels)
        self.identifiers = np.concatenate(self.identifiers)
        self.names = np.concatenate(self.names)
        
        # Convert to torch tensors
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Save class mapping
        self.class_mapping = class_mapping
        self.inverse_mapping = {v: k for k, v in class_mapping.items()}
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'label': self.labels[idx],
            'identifier': self.identifiers[idx],
            'name': self.names[idx]
        }

class NFTClassifier(nn.Module):
    """
    Neural network for classifying NFT embeddings
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        """
        Initialize the classifier
        
        Args:
            input_dim: Dimension of the input embeddings
            hidden_dim: Dimension of the hidden layer
            output_dim: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(NFTClassifier, self).__init__()
        
        # Define the network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=5):
    """
    Train the model with early stopping
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Number of epochs to wait for improvement before early stopping
    
    Returns:
        Dictionary containing training history
    """
    # Move model to device
    model.to(device)
    
    # Initialize variables for training
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            # Get data
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Get data
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item() * embeddings.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history

def plot_training_history(history, output_file):
    """
    Plot the training history
    
    Args:
        history: Dictionary containing training history
        output_file: Path to save the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epoch')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs. Epoch')
    ax2.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    """
    Main function to train the NFT classification model
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NFT classification model')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Directory containing embedding files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find all embedding files
    embedding_files = [os.path.join(args.embedding_dir, f) for f in os.listdir(args.embedding_dir) if f.endswith('_embeddings.npz')]
    
    if not embedding_files:
        print(f"No embedding files found in {args.embedding_dir}")
        return
    
    print(f"Found {len(embedding_files)} embedding files:")
    for file in embedding_files:
        print(f"  - {os.path.basename(file)}")
    
    # Create dataset
    dataset = NFTEmbeddingDataset(embedding_files)
    
    # Split dataset into train and validation sets
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    input_dim = dataset.embeddings.shape[1]  # Dimension of embeddings
    output_dim = len(dataset.class_mapping)  # Number of classes
    model = NFTClassifier(input_dim, args.hidden_dim, output_dim)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    print(f"Training model with {train_size} samples, validating with {val_size} samples")
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=args.num_epochs, patience=args.patience
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, 'nft_classifier.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save class mapping
    mapping_path = os.path.join(args.output_dir, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(dataset.class_mapping, f, indent=2)
    print(f"Class mapping saved to {mapping_path}")
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    print(f"Training history plot saved to {plot_path}")

if __name__ == "__main__":
    main()
