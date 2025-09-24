import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, PNAConv, GATConv, GINConv, RGCNConv, TransformerConv
from torch_geometric.utils import degree
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import wandb
import time
import argparse
import json

# Configuration
DATA_DIR = "./data"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# W&B Configuration
WANDB_CONFIG = {
    "project": "gnn-elliptic-dataset", # Name of the project in your W&B account
    "entity": None,                   # Your W&B username or team name
    "name": f"gcn_run_{int(time.time())}" # A unique name for this specific run
}

# Data Loading Function
def load_elliptic_data_with_splits(data_dir):
    """
    Loads the Elliptic dataset and creates a single PyG Data object
    with fixed training, validation, and test masks.
    """
    print("Loading data files...")
    df_features = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_features.csv'), header=None)
    df_edges = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_edgelist.csv'))
    df_classes = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_classes.csv'))

    all_tx_ids = df_features[0].unique()
    node_mapping = {tx_id: i for i, tx_id in enumerate(all_tx_ids)}

    source_nodes = [node_mapping[tx_id] for tx_id in df_edges['txId1']]
    target_nodes = [node_mapping[tx_id] for tx_id in df_edges['txId2']]
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    sorted_features = df_features.set_index(0).loc[all_tx_ids]
    x = torch.tensor(sorted_features.values, dtype=torch.float)

    y = torch.full((len(all_tx_ids),), -1, dtype=torch.long)
    licit_ids = df_classes[df_classes['class'] == '2']['txId']
    illicit_ids = df_classes[df_classes['class'] == '1']['txId']
    y[[node_mapping[tx_id] for tx_id in licit_ids if tx_id in node_mapping]] = 0 # Licit
    y[[node_mapping[tx_id] for tx_id in illicit_ids if tx_id in node_mapping]] = 1 # Illicit

    # Create masks: Train (1-30), Validation (31-34), Test (35-49)
    timesteps = x[:, 0].long()
    train_mask = (timesteps <= 30) & (y != -1)
    val_mask = (timesteps > 30) & (timesteps <= 34) & (y != -1)
    test_mask = (timesteps > 34) & (y != -1)

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = 2

    print("\nData Preprocessing Complete")
    print(data)
    return data

# GCN Model
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# PNA Model
class PNA(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, deg):
        super().__init__()
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.conv1 = PNAConv(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            aggregators=aggregators, 
            scalers=scalers, 
            deg=deg
        )
        self.conv2 = PNAConv(
            in_channels=hidden_channels, 
            out_channels=out_channels, 
            aggregators=aggregators, 
            scalers=scalers, 
            deg=deg
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# GAT Model
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# GIN Model
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(mlp1)
        mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self.conv2 = GINConv(mlp2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# RGCN Model
class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=1):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, x, edge_index):
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long).to(x.device)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

# HELPER CLASS OF THE GRAPHTRANSFORMER CLASS
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, heads=8, dropout=0.1):
        super().__init__()
        # The TransformerConv layer for self-attention
        self.attn = TransformerConv(in_channels, in_channels // heads, heads=heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(in_channels) # First layer normalization
        
        # The Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.ln2 = nn.LayerNorm(in_channels) # Second layer normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Self-Attention with Residual Connection
        attn_out = self.attn(x, edge_index)
        x = x + self.dropout(attn_out) # Residual Connection
        x = self.ln1(x)               # Add & Norm
        
        # Feed-Forward Network with Residual Connection
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out) # Residual Connection
        x = self.ln2(x)             # Add & Norm
        
        return x

# Graph Transformer Model
class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.1, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_channels, heads, dropout) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x, edge_index)
            
        x = self.output_proj(x)
        return x

# Train and Evaluation Functions
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    metrics = {}
    for mask_name, mask in [('train', data.train_mask), ('validation', data.val_mask), ('test', data.test_mask)]:
        y_true = data.y[mask].cpu()
        y_pred = pred[mask].cpu()
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics[f'f1/{mask_name}'] = f1
        
    return metrics

# Main Execution Block
def main():
    # Command-line argument parser
    parser = argparse.ArgumentParser(description='Run GNN models on the Elliptic dataset.')
    parser.add_argument('--model', type=str, default='gcn',
                        help='Name of the GNN model to run')
    args = parser.parse_args()
    print(f"Running with model: {args.model.upper()}")

    # Load hyperparameters from JSON
    with open('model_settings.json', 'r') as f:
        model_configs = json.load(f)
    
    try:
        config = model_configs[args.model.lower()]['params']
        print(f"Loaded config for {args.model.upper()}: {config}")
    except KeyError:
        raise KeyError(f"Model '{args.model}' not found in model_settings.json")


    WANDB_CONFIG["name"] = f"{args.model}_lr_{config['lr']}_hidden_{config['hidden_channels']}"
    
    # Initialize W&B run
    wandb.init(
        project=WANDB_CONFIG["project"],
        entity=WANDB_CONFIG["entity"],
        name=WANDB_CONFIG["name"],
        config=config  # Pass the loaded config directly to wandb
    )
    
    print(f"Using device: {DEVICE}")
    data = load_elliptic_data_with_splits(DATA_DIR).to(DEVICE)

    # Calculate degree for PNA model
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg = torch.zeros(d.max() + 1, dtype=torch.long).to(DEVICE)
    deg[d] = 1

    model = None
    if args.model.lower() == 'gcn':
        model = GCN(data.num_node_features, config['hidden_channels'], data.num_classes).to(DEVICE)
    elif args.model.lower() == 'pna':
        model = PNA(data.num_node_features, config['hidden_channels'], data.num_classes, deg=deg).to(DEVICE)
    elif args.model.lower() == 'gat':
        if config['hidden_channels'] % config['heads'] != 0:
            raise ValueError("For GAT, hidden_channels must be divisible by heads.")
        model = GAT(
            in_channels=data.num_node_features,
            hidden_channels=config['hidden_channels'],
            out_channels=data.num_classes,
            heads=config['heads'],
            dropout=config['dropout']
        ).to(DEVICE)
    elif args.model.lower() == 'gin':
        model = GIN(data.num_node_features, config['hidden_channels'], data.num_classes).to(DEVICE)
    elif args.model.lower() == 'rgcn':
        model = RGCN(
            in_channels=data.num_node_features,
            hidden_channels=config['hidden_channels'],
            out_channels=data.num_classes,
            num_relations=config['num_relations']
        ).to(DEVICE)
    elif args.model.lower() == 'transformer':
        if config['hidden_channels'] % config['heads'] != 0:
            raise ValueError("For GraphTransformer, hidden_channels must be divisible by heads.")
        model = GraphTransformer(
            in_channels=data.num_node_features,
            hidden_channels=config['hidden_channels'],
            out_channels=data.num_classes,
            heads=config['heads'],
            dropout=config['dropout']
        ).to(DEVICE)
    else:
        raise ValueError(f"Model '{args.model}' is not supported.")
    
    wandb.watch(model, log='all')

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    best_test_f1 = 0

    print("\nStarting Training")
    for epoch in tqdm(range(1, config['epochs'] + 1), desc=f"Training {args.model.upper()}"):
        loss = train(model, data, optimizer, criterion)
        metrics = evaluate(model, data)
        
        # Track the best test F1 score based on the best validation F1 score
        if metrics['f1/validation'] > best_val_f1:
            best_val_f1 = metrics['f1/validation']
            best_test_f1 = metrics['f1/test']
        
        # Log all metrics to W&B
        wandb.log({
            "f1/train": metrics['f1/train'],
            "f1/validation": metrics['f1/validation'],
            "f1/test": metrics['f1/test'],
            "best_test_f1": best_test_f1
        })
    
    print("\nTraining Finished")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Test F1 at Best Validation: {best_test_f1:.4f}")
    
    wandb.finish() # End the W&B run

if __name__ == "__main__":
    main()