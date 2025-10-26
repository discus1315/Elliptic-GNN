# GNN Models for Elliptic Dataset

This project provides a framework to train and evaluate several Graph Neural Network (GNN) models on the [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). The Elliptic dataset is a graph of Bitcoin transactions, where the goal is to classify transactions as 'licit' or 'illicit'.

## Models

The following GNN models are implemented:

*   **GCN** (Graph Convolutional Network)
*   **PNA** (Principal Neighbourhood Aggregation)
*   **GAT** (Graph Attention Network)
*   **GIN** (Graph Isomorphism Network)
*   **RGCN** (Relational Graph Convolutional Network)
*   **Graph Transformer**

## Project Structure

```
Elliptic-GNN/
├── .gitignore
├── main.py                # Main script to train and evaluate models
├── model_settings.json    # Hyperparameters for the models
├── requirements.txt       # Python package requirements
└── data/                  # Data directory (not included in the repo)
    ├── elliptic_txs_classes.csv
    ├── elliptic_txs_edgelist.csv
    └── elliptic_txs_features.csv
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/discus1315/Elliptic-GNN.git
    cd Elliptic-GNN
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Download the dataset:**
    Download the Elliptic dataset from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place the CSV files in a `data/` directory in the project root.

2.  **Configure Weights & Biases (W&B):**
    This project uses W&B for experiment tracking. You will need a W&B account.
    - Log in to your W&B account:
      ```bash
      wandb login
      ```
    - In `main.py`, you can change the `WANDB_CONFIG` dictionary to set your project name and entity.

3.  **Run a model:**
    You can run a specific model using the `--model` command-line argument. For example, to run the GCN model:
    ```bash
    python main.py --model gcn
    ```

    Supported model names: `gcn`, `pna`, `gat`, `gin`, `rgcn`, `transformer`.

4.  **Hyperparameters:**
    The hyperparameters for each model are defined in `model_settings.json`. You can modify this file to experiment with different settings.

## Evaluation

The script will evaluate the model on the training, validation, and test sets and log the F1 scores to W&B. The best test F1 score based on the best validation F1 score will be printed at the end of the training.
