"""
Graph Neural Network for Financial Fraud Detection using Directed Edge Attention.

This module implements a directed edge-aware GNN model with attention mechanisms
for detecting fraudulent transactions in financial networks. The model processes
transaction graphs with node (account) and edge (transaction) features using
specialized directed message passing.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from Model.gnn import DirectedGINeWithAttention


def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries.

    Ensures deterministic behavior in Python's random, NumPy, and PyTorch
    (including CUDA operations) for consistent experimental results.

    Args:
        seed: Random seed value (default: 42).
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch seeds for CPU and all GPU devices
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enable deterministic CUDA operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_timestamp_to_cyclical(txn_time: str):
    """Convert transaction time strings to cyclical sine/cosine features.

    Transforms time-of-day into continuous cyclical features using sine and
    cosine encoding, preserving the circular nature of time (23:59 is close
    to 00:00). This encoding helps neural networks better understand temporal
    patterns.

    Args:
        txn_time: Series of transaction times in "HH:MM:SS" format.

    Returns:
        tuple: (sin_time, cos_time) - Two numpy arrays with cyclical time
            encodings.
                - sin_time: Sine component of the cyclical encoding.
                - cos_time: Cosine component of the cyclical encoding.
    """
    # Parse time strings to datetime objects
    timestamps = pd.to_datetime(txn_time, format="%H:%M:%S")

    # Calculate total minutes elapsed since midnight (with fractional seconds)
    total_minutes = (
        timestamps.dt.hour * 60 + timestamps.dt.minute + timestamps.dt.second / 60
    )
    minutes_in_day = 24 * 60  # Total minutes in a day (1440)

    # Apply cyclical transformation: maps time to a point on unit circle
    sin_time = np.sin(2 * np.pi * total_minutes / minutes_in_day)
    cos_time = np.cos(2 * np.pi * total_minutes / minutes_in_day)

    return sin_time, cos_time


def get_node_and_edge_features(df, account_mapping):
    """Extract and preprocess node and edge features from transaction DataFrame.

    Constructs graph structure from transaction data by:
    1. Creating node features from unique accounts with one-hot encoded types.
    2. Building edge features from transaction properties.
    3. Engineering temporal features (time deltas, cyclical time encoding).
    4. Generating train/test masks based on account mapping.

    Args:
        df: Transaction DataFrame with columns:
            - from_acct, to_acct: Source/destination account IDs.
            - from_acct_type, to_acct_type: Account type categories.
            - txn_amt_to_twd: Transaction amount in TWD.
            - timestamp: Unix timestamp of transaction.
            - txn_time: Time of day in "HH:MM:SS" format.
            - is_self_txn: Boolean indicating self-transfers.
            - currency_type: Currency code.
            - channel_type: Transaction channel.
        account_mapping: Dict mapping account IDs to (split, label, original_id).
            - split: 0/1 for train, 2 for test.
            - label: 0 for normal, 1 for fraudulent.
            - original_id: Original account identifier.

    Returns:
        tuple containing:
            - node_features (Tensor): One-hot encoded account types.
            - edge_index (Tensor): Graph connectivity in COO format.
            - edge_features (Tensor): Transaction features.
            - node_labels (Tensor): Binary fraud labels.
            - train_mask (Tensor): Boolean mask for training nodes.
            - test_mask (Tensor): Boolean mask for test nodes.
    """
    # Collect all unique nodes with their account types
    from_nodes = df[["from_acct", "from_acct_type"]].rename(
        columns={"from_acct": "acct", "from_acct_type": "acct_type"}
    )
    to_nodes = df[["to_acct", "to_acct_type"]].rename(
        columns={"to_acct": "acct", "to_acct_type": "acct_type"}
    )

    # Combine and deduplicate nodes
    node_df = (
        pd.concat([from_nodes, to_nodes], ignore_index=True)
        .drop_duplicates(subset="acct")
        .reset_index(drop=True)
    )

    # Create node features: one-hot encode account types
    node_features = pd.get_dummies(node_df[["acct_type"]], prefix="acct_type")
    node_features.index = node_df["acct"]
    node_features = torch.tensor(node_features.values, dtype=torch.float)

    # --- Prepare edge features ---
    edge_df = df[
        [
            "from_acct",
            "to_acct",
            "txn_amt_to_twd",
            "timestamp",
            "is_self_txn",
            "currency_type",
            "channel_type",
            "txn_time",
        ]
    ].copy()

    # Sort by timestamp for temporal feature engineering
    edge_df = edge_df.sort_values("timestamp", ignore_index=True)

    # Add cyclical time features (captures time-of-day patterns)
    sin, cos = convert_timestamp_to_cyclical(edge_df["txn_time"])
    edge_df["sin_txn_time"] = sin
    edge_df["cos_txn_time"] = cos
    edge_df = edge_df.drop(columns=["txn_time"])
    print(edge_df["sin_txn_time"])

    # Compute temporal features: time since previous transaction per account
    edge_df["time_delta_from"] = edge_df.groupby("from_acct")["timestamp"].diff()
    edge_df["time_delta_to"] = edge_df.groupby("to_acct")["timestamp"].diff()

    # Handle missing values (first transaction for each account)
    num_cols = [
        "txn_amt_to_twd",
        "timestamp",
        "time_delta_from",
        "time_delta_to",
    ]
    edge_df[num_cols] = edge_df[num_cols].fillna(0)

    # Standardize numerical features to zero mean and unit variance
    st_scaler = StandardScaler()
    edge_df[num_cols] = st_scaler.fit_transform(edge_df[num_cols])

    # One-hot encode categorical edge attributes
    edge_cats = pd.get_dummies(
        edge_df[["is_self_txn", "currency_type", "channel_type"]],
        columns=["is_self_txn", "currency_type", "channel_type"],
    )

    # Combine all edge features (numerical + categorical)
    edge_features_df = pd.concat([edge_df[num_cols], edge_cats], axis=1)
    edge_features_df = edge_features_df.select_dtypes(
        include=["number", "bool"]
    ).astype(float)
    edge_features = torch.tensor(edge_features_df.values, dtype=torch.float)

    # Create edge index for PyTorch Geometric (COO format)
    edge_index = torch.tensor(
        edge_df[["from_acct", "to_acct"]].values.T, dtype=torch.long
    )

    # --- Generate node labels and data split masks ---
    num_nodes = len(node_df)
    node_labels = torch.zeros(num_nodes, dtype=int)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Populate labels and masks from account mapping
    for i in range(num_nodes):
        node_labels[i] = account_mapping[str(i)][1]  # 0: normal, 1: fraud
        if account_mapping[str(i)][0] == 0 or account_mapping[str(i)][0] == 1:
            train_mask[i] = True  # Training split
        elif account_mapping[str(i)][0] == 2:
            test_mask[i] = True  # Test split

    return node_features, edge_index, edge_features, node_labels, train_mask, test_mask


def train_model_fullbatch(
    model,
    data,
    train_mask,
    optimizer,
    class_weight,
):
    """Perform one epoch of full-batch training with weighted BCE loss.

    Processes the entire graph in a single forward pass, applying class
    weights to handle imbalanced datasets (more normal accounts than
    fraudulent ones).

    Args:
        model: GNN model to train.
        data: PyTorch Geometric Data object with graph and features.
        train_mask: Boolean tensor for training nodes.
        optimizer: PyTorch optimizer (e.g., Adam).
        class_weight: Tensor [2] with weights for [normal, fraud] classes.

    Returns:
        float: Training loss value for this epoch.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass through the entire graph
    out = model(data.x, data.edge_index, data.edge_attr)

    # Compute weighted binary cross-entropy loss on training nodes only
    classification_loss = F.binary_cross_entropy_with_logits(
        out[train_mask].squeeze(),
        data.y[train_mask].float(),
        pos_weight=torch.tensor([class_weight[1] / class_weight[0]]).to(out.device),
    )
    total_loss = classification_loss

    # Backpropagation and optimization step
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def evaluate_model(model, data, mask):
    """Evaluate model performance on validation or test set.

    Computes multiple evaluation metrics in a single pass:
    - F1 score at 0.5 threshold
    - ROC-AUC score
    - Average Precision (AP) score

    Args:
        model: Trained GNN model.
        data: PyTorch Geometric Data object with graph and labels.
        mask: Boolean tensor indicating which nodes to evaluate.

    Returns:
        tuple containing:
            - best_f1 (float): F1 score using 0.5 threshold.
            - auc (float): Area Under the ROC Curve.
            - ap (float): Average Precision score.
            - y_prob (numpy.ndarray): Predicted probabilities.
    """
    model.eval()
    with torch.no_grad():
        # Get model predictions and convert logits to probabilities
        out = model(data.x, data.edge_index, data.edge_attr)
        prob = torch.sigmoid(out[mask])

        # Convert to numpy for sklearn metrics
        y_true = data.y[mask].cpu().numpy()
        y_prob = prob.squeeze().cpu().numpy()

        # Calculate metrics using fixed 0.5 threshold
        best_f1 = f1_score(y_true, np.where(y_prob >= 0.5, 1, 0))
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

    return (
        best_f1,
        auc,
        ap,
        y_prob,
    )


def predict(models, data, mask):
    """Generate ensemble predictions from multiple trained models.

    Combines predictions from cross-validation folds by averaging their
    probability outputs, then applies a threshold based on the top 5%
    of predicted scores to identify fraudulent accounts.

    Args:
        models: List of trained GNN models (one per CV fold).
        data: PyTorch Geometric Data object containing the graph.
        mask: Boolean tensor indicating which nodes to predict.

    Returns:
        numpy.ndarray: Binary predictions (1=fraud, 0=normal).
    """
    probs = []

    # Collect predictions from all models
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            prob = torch.sigmoid(out[mask]).squeeze().cpu()
            probs.append(prob)
    probs = torch.stack(probs, dim=0)
    mean_prob = probs.mean(dim=0)

    threshold = torch.quantile(mean_prob, 0.95)  # top 5%
    preds = (mean_prob >= threshold).int()

    return preds.numpy()


def main():
    """Execute main training and evaluation pipeline for fraud detection GNN.

    Pipeline Overview:
        1. Load transaction data and account mappings.
        2. Extract graph features (nodes, edges, labels).
        3. Train ensemble using stratified k-fold cross-validation.
        4. Generate predictions on test set using ensemble averaging.
        5. Save results and trained models.

    Cross-Validation Strategy:
        - 5-fold stratified split to preserve fraud/normal ratio.
        - Early stopping based on validation F1 score.
        - Learning rate scheduling with plateau detection.

    The ensemble approach helps:
        - Reduce variance in predictions.
        - Handle class imbalance through multiple perspectives.
        - Improve generalization to unseen accounts.
    """
    # Initialize random seeds for reproducibility
    set_seed(42)

    # Load transaction data and account metadata
    df = pd.read_csv("data/formatted_transaction.csv")
    with open("data/account_mapping.json", "r") as f:
        account_mapping = json.load(f)

    # Extract graph structure and features
    node_features, edge_index, edge_features, node_labels, train_mask, test_mask = (
        get_node_and_edge_features(df, account_mapping)
    )

    # Create PyTorch Geometric data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        y=node_labels,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    # Setup device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # Analyze training data distribution
    y_train = node_labels[train_mask]
    print(f"Training labels distribution: {np.bincount(y_train)}")

    # Compute class weights to handle imbalance (normal >> fraud)
    class_counts = np.bincount(y_train)
    class_weight = torch.tensor(
        [1.0, class_counts[0] / (class_counts[1] + 1e-6)], dtype=torch.float
    ).to(device)
    print("Using class weights:", class_weight.cpu().numpy())

    # Prepare data for stratified k-fold cross-validation
    train_indices = torch.where(train_mask)[0].numpy()
    X_train = node_features[train_indices]
    y_train_array = node_labels[train_indices]

    # Initialize cross-validation and model storage
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []

    # Train one model per fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_array)):
        print(f"Training fold {fold + 1}/5")

        # Map fold indices to global node indices
        global_train_idx = train_indices[train_idx]
        global_val_idx = train_indices[val_idx]

        # Create fold-specific masks
        fold_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
        fold_val_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
        fold_train_mask[global_train_idx] = True
        fold_val_mask[global_val_idx] = True

        # Initialize tracking variables for early stopping
        best_model_state = None
        best_f1 = 0.0
        patience = 20
        patience_counter = 0

        # Initialize model with custom architecture
        model = DirectedGINeWithAttention(
            num_features=node_features.shape[1],
            edge_dim=edge_features.shape[1],
            num_gnn_layers=4,
            dropout=0.0,
            n_hidden=16,
        ).to(device)

        # Setup optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=5e-3,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Maximize F1 score
            factor=0.5,
            patience=5,
        )

        # Training loop
        for epoch in range(2000):
            # Train for one epoch
            avg_loss = train_model_fullbatch(
                model, data, fold_train_mask, optimizer, class_weight
            )

            # Evaluate every 50 epochs
            if (epoch + 1) % 50 == 0:
                val_f1, val_auc, val_ap, _ = evaluate_model(model, data, fold_val_mask)
                scheduler.step(val_f1)
                print(
                    f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
                )

                # Track best model and implement early stopping
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Stop if no improvement for 'patience' evaluations
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print("best val F1:", best_f1)

        # Load best model state and save to ensemble
        model.load_state_dict(best_model_state)
        models.append(model)

    # Generate ensemble predictions on test set
    test_preds = predict(models, data, test_mask)

    # Map node indices back to original account IDs
    acct_ids = {int(k): v[-1] for k, v in account_mapping.items()}
    acct_ids = np.array([v for _, v in sorted(acct_ids.items())])
    test_acct_ids = acct_ids[test_mask.numpy()]

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "acct": test_acct_ids,
            "label": test_preds,
        }
    )

    # Print prediction statistics
    print(test_preds.sum())
    print(test_preds.mean())

    # Save results and models
    exp_name = "edge_specific_norm"
    results_df.to_csv(f"{exp_name}.csv", index=False)

    # Save each fold's model for potential later use or analysis
    os.makedirs("gnn_models", exist_ok=True)
    for i, model in enumerate(models):
        torch.save(
            model.state_dict(),
            f"gnn_models/{exp_name}_fold_{i}.pt",
        )
    print("GNN models saved to gnn_models/ directory")


if __name__ == "__main__":
    main()
