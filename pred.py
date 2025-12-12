import json
import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from main_minibatch import GPUMonitor, get_node_and_edge_features, predict_minibatch
from Model.gnn import DirectedGINeWithAttention

EDGE_MAX_NUM = 500
WINDOW = 30  # days
NEIGHBOR_SAMPLE_SIZES = [100, 10, 5, 5]
NON_ALERT_FILTER_RATIO = 0.0
FOLD = 3
EPOCH = 2000
PREDICT_STRATEGY = "5%"

print(f"Non-alert node filtering ratio: {NON_ALERT_FILTER_RATIO}")
print(f"Edge max num for train mask: {EDGE_MAX_NUM}")
print(f"Time window (days): {WINDOW}")
print(f"Neighbor sample sizes: {NEIGHBOR_SAMPLE_SIZES}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# print("determine seed")


# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     os.environ["PYTHONHASHSEED"] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     torch.use_deterministic_algorithms(True)

set_seed(42)

# Initialize GPU Monitor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
monitor = GPUMonitor(device)
# monitor.log("Script Start")

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
# monitor.log("Data Loaded (CPU)")

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

exp_name = f"minibatch_window{WINDOW}_edge{EDGE_MAX_NUM}_3fold_{NEIGHBOR_SAMPLE_SIZES[0]}neighbors"
folds = [f"gnn_models/{exp_name}_fold_{i}.pt" for i in range(FOLD)]
models = []
for path in folds:
    model = DirectedGINeWithAttention(
        num_features=node_features.shape[1],
        edge_dim=edge_features.shape[1],
        num_gnn_layers=4,
        dropout=0.0,
        n_hidden=16,
    ).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    models.append(model)

print("\n--- Starting Inference ---")
test_preds = predict_minibatch(models, data, test_mask, device, monitor)

acct_ids = {int(k): v[-1] for k, v in account_mapping.items()}
acct_ids = np.array([v for _, v in sorted(acct_ids.items())])
test_acct_ids = acct_ids[test_mask.numpy()]

results_df = pd.DataFrame({"acct": test_acct_ids, "label": test_preds})

print(f"Pred Sum: {test_preds.sum()}")
print(f"Pred Mean: {test_preds.mean()}")

results_df.to_csv("工作坊_result.csv", index=False)
