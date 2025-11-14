# Financial Fraud Detection with Graph Neural Networks

This project implements a directed Graph Neural Network (GNN) with attention mechanisms to detect fraudulent accounts in financial transaction networks.

## 📁 Project Structure

```
ai-cup-2025/
├── Model/
│   └── gnn.py
│   └── README.md
├── main.py
├── README.md
```

## Harware Required
GPU with 24GB VRAM

## Environment
```
conda create -n ai-cup-2025 python=3.12 -y
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python main.py
```

**Outputs**:
- `{exp_name}.csv`: Predictions for test accounts
- `gnn_models/{exp_name}_fold_{i}.pt`: Saved model weights for each fold

---

## 📥 Input Data

### **Transaction Graph Structure**

The model processes financial data as a directed graph where:
- **Nodes** = Bank accounts
- **Edges** = Transactions between accounts

### **Node Features** (One-hot encoded)
- `from_acct_type` / `to_acct_type`: Account type categories

### **Edge Features** (Transaction properties)

1. **Numerical Features** (standardized):
   - `txn_amt_to_twd`: Transaction amount in TWD
   - `timestamp`: Unix timestamp of transaction
   - `time_delta_from`: Time since previous transaction from sender account
   - `time_delta_to`: Time since previous transaction to receiver account

2. **Temporal Features** (cyclical encoding):
   - `sin_txn_time`, `cos_txn_time`: Time-of-day encoded as sine/cosine (preserves 24-hour cyclical nature)

3. **Categorical Features** (one-hot encoded):
   - `is_self_txn`: Whether the transaction is a self-transfer
   - `currency_type`: Currency code
   - `channel_type`: Transaction channel

### **Labels**
- Binary classification: `0` = Normal account, `1` = Fraudulent account
- Highly imbalanced dataset (normal accounts >> fraudulent accounts)

### **Data Files**
- `formatted_transaction.csv`: Transaction records
- `account_mapping.json`: Account metadata with train/test splits and labels

---

## 🎓 Training Method

### **Cross-Validation Strategy**
- **5-fold Stratified K-Fold** to preserve fraud/normal ratio across folds
- Ensemble of 5 models (one per fold)
- Final predictions averaged across all models

### **Training Configuration**

1. **Loss Function**: Binary Cross-Entropy with class weighting
   - Class weights computed to handle imbalance: `weight = count(normal) / count(fraud)`
   - Prevents model from predicting all accounts as normal

2. **Optimizer**: Adam
   - Learning rate: 0.005 (5e-3)
   - Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

3. **Training Settings**:
   - Max epochs: 2000 per fold
   - Batch size: Full-batch training (entire graph processed at once)
   - Early stopping: Patience of 20 evaluations (checked every 50 epochs)
   - Early stopping metric: Validation F1 score

4. **Evaluation Metrics**:
   - F1 score (threshold: 0.5)

5. **Hyperparameters**:
   - Hidden dimension: 16
   - Number of layers: 4
   - Attention heads: 4
   - Dropout: 0.0 (GNN layers), 0.5 (attention final layer)
   - Residual connections: Yes
   
### **Ensemble Prediction**
- Averages probability outputs from all 5 fold models
- Final threshold: Top 5% of predicted scores (95th percentile)
- Helps reduce variance and improve generalization

### **Reproducibility**
- Fixed random seed: 42
- Deterministic CUDA operations enabled
