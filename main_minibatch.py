"""
Graph Neural Network for Financial Fraud Detection using Directed Edge Attention.
Mini-Batch Implementation with GPU Memory Supervision.
"""

import gc  # Added for memory cleanup
import json
import os
import random
from tqdm import tqdm, trange

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
from torch_geometric.loader import NeighborLoader

from Model.gnn import DirectedGINeWithAttention


class GPUMonitor:
    """Helper to track and log GPU memory usage throughout the pipeline."""
    
    def __init__(self, device):
        self.device = device
        self.enabled = device.type == 'cuda'

    def log(self, stage: str):
        """Prints current memory stats if running on CUDA."""
        if not self.enabled:
            return

        # Synchronize to ensure accurate readings
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        peak = torch.cuda.max_memory_allocated() / 1024**3   # GB
        
        print(f"== [GPU] {stage:<20} | Alloc: {allocated:5.2f} GB | "
              f"Rsrv: {reserved:5.2f} GB | Peak: {peak:5.2f} GB ==")

    def reset_peak(self):
        """Resets the peak memory tracker."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_timestamp_to_cyclical(txn_time: str):
    timestamps = pd.to_datetime(txn_time, format="%H:%M:%S")
    total_minutes = (
        timestamps.dt.hour * 60 + timestamps.dt.minute + timestamps.dt.second / 60
    )
    minutes_in_day = 24 * 60
    sin_time = np.sin(2 * np.pi * total_minutes / minutes_in_day)
    cos_time = np.cos(2 * np.pi * total_minutes / minutes_in_day)
    return sin_time, cos_time


def get_node_and_edge_features(df, account_mapping):
    from_nodes = df[["from_acct", "from_acct_type"]].rename(
        columns={"from_acct": "acct", "from_acct_type": "acct_type"}
    )
    to_nodes = df[["to_acct", "to_acct_type"]].rename(
        columns={"to_acct": "acct", "to_acct_type": "acct_type"}
    )

    node_df = (
        pd.concat([from_nodes, to_nodes], ignore_index=True)
        .drop_duplicates(subset="acct")
        .reset_index(drop=True)
    )
    # print("Number of nodes:", len(node_df))

    node_features = pd.get_dummies(node_df[["acct_type"]], prefix="acct_type")
    node_features.index = node_df["acct"]
    node_features = torch.tensor(node_features.values, dtype=torch.float)

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

    edge_df = edge_df.sort_values("timestamp", ignore_index=True)
    sin, cos = convert_timestamp_to_cyclical(edge_df["txn_time"])
    edge_df["sin_txn_time"] = sin
    edge_df["cos_txn_time"] = cos
    edge_df = edge_df.drop(columns=["txn_time"])

    edge_df["time_delta_from"] = edge_df.groupby("from_acct")["timestamp"].diff()
    edge_df["time_delta_to"] = edge_df.groupby("to_acct")["timestamp"].diff()

    num_cols = ["txn_amt_to_twd", "timestamp", "time_delta_from", "time_delta_to"]
    edge_df[num_cols] = edge_df[num_cols].fillna(0)
    st_scaler = StandardScaler()
    edge_df[num_cols] = st_scaler.fit_transform(edge_df[num_cols])

    edge_cats = pd.get_dummies(
        edge_df[["is_self_txn", "currency_type", "channel_type"]],
        columns=["is_self_txn", "currency_type", "channel_type"],
    )

    edge_features_df = pd.concat([edge_df[num_cols], edge_cats], axis=1)
    edge_features_df = edge_features_df.select_dtypes(
        include=["number", "bool"]
    ).astype(float)
    edge_features = torch.tensor(edge_features_df.values, dtype=torch.float)

    edge_index = torch.tensor(
        edge_df[["from_acct", "to_acct"]].values.T, dtype=torch.long
    )
    # print("Number of edges:", edge_index.shape[1])

    num_nodes = len(node_df)
    node_labels = torch.zeros(num_nodes, dtype=int)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for i in range(num_nodes):
        node_labels[i] = account_mapping[str(i)][1]
        if account_mapping[str(i)][0] == 0 or account_mapping[str(i)][0] == 1:
            train_mask[i] = True
        elif account_mapping[str(i)][0] == 2:
            test_mask[i] = True

    # Compute Number of edges per train nodes:
    # num_edges_per_node = torch.zeros(len(node_df), dtype=torch.long)
    # for i in range(edge_index.shape[1]):
    #     from_node = edge_index[0, i]
    #     to_node = edge_index[1, i]
    #     num_edges_per_node[from_node] += 1
    #     num_edges_per_node[to_node] += 1
    # print("Avg edges per node:", num_edges_per_node[train_mask].float().mean().item())
    # print("Max edges per node:", num_edges_per_node[train_mask].max().item())
    # print("Min edges per node:", num_edges_per_node[train_mask].min().item())
    # print("Median edges per node:", num_edges_per_node[train_mask].median().item())
    # exit()

    return node_features, edge_index, edge_features, node_labels, train_mask, test_mask


def train_model_minibatch(model, loader, optimizer, device, class_weight, monitor):
    """
    Perform one epoch of mini-batch training with memory logging.
    """
    model.train()
    total_loss = 0
    total_examples = 0

    # Reset peak stats at start of epoch to see max usage per epoch
    monitor.reset_peak()

    for i, batch in enumerate(tqdm(loader, desc="Training Batches")):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        
        batch_size = batch.batch_size
        out_seed = out[:batch_size]
        y_seed = batch.y[:batch_size]

        loss = F.binary_cross_entropy_with_logits(
            out_seed.squeeze(),
            y_seed.float(),
            pos_weight=torch.tensor([class_weight[1] / class_weight[0]]).to(device),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        # Log memory for the first batch to ensure graph fits
        # if i == 0:
        #     monitor.log("End of 1st Batch")

    return total_loss / total_examples


def evaluate_model_minibatch(model, loader, device, monitor):
    monitor.log("Start Evaluation")
    model.eval()
    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            batch_size = batch.batch_size
            prob = torch.sigmoid(out[:batch_size])

            y_true_all.append(batch.y[:batch_size].cpu())
            y_prob_all.append(prob.squeeze().cpu())
            monitor.log(f"After Batch {i}")

    y_true = torch.cat(y_true_all).numpy()
    y_prob = torch.cat(y_prob_all).numpy()

    best_f1 = f1_score(y_true, np.where(y_prob >= 0.5, 1, 0))
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    return best_f1, auc, ap, y_prob


def predict_minibatch(models, data, input_mask, device, monitor):
    monitor.log("Start Inference")
    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1, -1, -1],
        batch_size=8192,
        input_nodes=input_mask,
        shuffle=False,
        num_workers=0
    )

    ensemble_probs = []

    for i, model in enumerate(models):
        model.eval()
        model_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                prob = torch.sigmoid(out[:batch.batch_size]).squeeze().cpu()
                model_probs.append(prob)
        
        full_model_probs = torch.cat(model_probs)
        ensemble_probs.append(full_model_probs)
        
        # Clear cache after each model to prevent buildup
        torch.cuda.empty_cache()
        monitor.log(f"After Model {i} Inf")

    monitor.log("End Inference")
    ensemble_probs = torch.stack(ensemble_probs, dim=0)
    mean_prob = ensemble_probs.mean(dim=0)
    threshold = torch.quantile(mean_prob, 0.95)
    preds = (mean_prob >= threshold).int()

    return preds.numpy()


def main():
    set_seed(42)

    # Initialize GPU Monitor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = GPUMonitor(device)
    monitor.log("Script Start")

    # Load Data
    df = pd.read_csv("data/formatted_transaction.csv")
    with open("data/account_mapping.json", "r") as f:
        account_mapping = json.load(f)

    node_features, edge_index, edge_features, node_labels, train_mask, test_mask = (
        get_node_and_edge_features(df, account_mapping)
    )

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        y=node_labels,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    monitor.log("Data Loaded (CPU)")

    # To visualize the difference between 'Allocated' (used by tensors) 
    # and 'Reserved' (cached by allocator) memory.

    y_train = node_labels[train_mask]
    class_counts = np.bincount(y_train)
    class_weight = torch.tensor(
        [1.0, class_counts[0] / (class_counts[1] + 1e-6)], dtype=torch.float
    )

    train_indices = torch.where(train_mask)[0].numpy()
    X_train_dummy = np.zeros(len(train_indices))
    y_train_array = node_labels[train_indices].numpy()

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []

    for fold, (train_idx_local, val_idx_local) in enumerate(tqdm(skf.split(X_train_dummy, y_train_array), desc="Folds")):
        print(f"\n--- Fold {fold + 1}/5 ---")
        
        # Clean up memory from previous fold
        gc.collect()
        torch.cuda.empty_cache()
        monitor.reset_peak()
        monitor.log("Start Fold")

        global_train_idx = torch.tensor(train_indices[train_idx_local])
        global_val_idx = torch.tensor(train_indices[val_idx_local])

        train_loader = NeighborLoader(
            data,
            num_neighbors=[100, 50, 20, 10],
            batch_size=4096*4*16, #TODO: adjust batch size based on GPU memory and expected training time
            input_nodes=global_train_idx,
            shuffle=True,
            num_workers=0
        )

        val_loader = NeighborLoader(
            data,
            num_neighbors=[-1, -1, -1, -1],
            batch_size=8192,
            input_nodes=global_val_idx,
            shuffle=False,
            num_workers=0
        )

        model = DirectedGINeWithAttention(
            num_features=node_features.shape[1],
            edge_dim=edge_features.shape[1],
            num_gnn_layers=4,
            dropout=0.0,
            n_hidden=16,
        ).to(device)

        # Log model memory footprint
        monitor.log("Model Init")

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        best_model_state = None
        best_f1 = 0.0
        patience = 20
        patience_counter = 0

        for epoch in trange(2000):
            avg_loss = train_model_minibatch(
                model, train_loader, optimizer, device, class_weight, monitor
            )

            if (epoch + 1) % 50 == 0:
                monitor.log(f"End Epoch {epoch+1} Training")
                val_f1, val_auc, _, _ = evaluate_model_minibatch(model, val_loader, device, monitor)
                scheduler.step(val_f1)
                
                print(
                    f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
                )
                
                # Check peak usage occasionally
                if (epoch + 1) % 50 == 0:
                    monitor.log(f"End Epoch {epoch+1}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print("best val F1:", best_f1)
        model.load_state_dict(best_model_state)
        models.append(model)
        
        # Explicitly delete model and optimizer to free GPU memory for next fold
        del model
        del optimizer
        del scheduler
        monitor.log("End Fold Cleanup")

    print("\n--- Starting Inference ---")
    test_preds = predict_minibatch(models, data, test_mask, device, monitor)

    acct_ids = {int(k): v[-1] for k, v in account_mapping.items()}
    acct_ids = np.array([v for _, v in sorted(acct_ids.items())])
    test_acct_ids = acct_ids[test_mask.numpy()]

    results_df = pd.DataFrame({"acct": test_acct_ids, "label": test_preds})
    
    print(f"Pred Sum: {test_preds.sum()}")
    print(f"Pred Mean: {test_preds.mean()}")

    exp_name = "minibatch_ensemble_gpu_monitored"
    results_df.to_csv(f"{exp_name}.csv", index=False)

    os.makedirs("gnn_models", exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f"gnn_models/{exp_name}_fold_{i}.pt")
    print("GNN models saved.")

if __name__ == "__main__":
    main()