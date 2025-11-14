# Financial Fraud Detection with Graph Neural Networks

This project implements a directed Graph Neural Network (GNN) with attention mechanisms to detect fraudulent accounts in financial transaction networks.

---

## 📊 Neural Network Architecture

### **DirectedGINeWithAttention**

A specialized Graph Isomorphism Network with Edge features (GINe) that processes transaction graphs bidirectionally with attention-based edge weighting.

#### Key Components:

1. **Directed Message Passing**
   - Separate processing for incoming edges (who sends money to this account)
   - Separate processing for outgoing edges (who receives money from this account)
   - Distinguishes between sender and receiver patterns, critical for fraud detection

2. **Multi-Head Attention Mechanism**
   - Learns to weight edge importance based on source, destination, and transaction features
   - 4 attention heads by default
   - Helps identify suspicious transaction patterns

3. **GINe Convolution Layers**
   - 4 graph convolution layers
   - Each layer has a 2-layer MLP: `Linear(16→32) → ReLU → Linear(32→16)`
   - Batch normalization and dropout for regularization

4. **Architecture Flow**
   ```
   Input → Node/Edge Embedding (→16D) 
        → [Attention + Directed GINe Conv + BatchNorm] × 4 layers
        → Dropout (0.5) → Linear (→1D) 
        → Binary Classification (Fraud/Normal)
   ```

#### Hyperparameters:
- Hidden dimension: 16
- Number of layers: 4
- Attention heads: 4
- Dropout: 0.0 (GNN layers), 0.5 (attention final layer)
- Residual connections: Yes

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
   - ROC-AUC
   - Average Precision (AP)

### **Ensemble Prediction**
- Averages probability outputs from all 5 fold models
- Final threshold: Top 5% of predicted scores (95th percentile)
- Helps reduce variance and improve generalization

### **Reproducibility**
- Fixed random seed: 42
- Deterministic CUDA operations enabled

---

## 🚀 Usage

```bash
python Model/gnn.py
```

**Outputs**:
- `{exp_name}.csv`: Predictions for test accounts
- `gnn_models/{exp_name}_fold_{i}.pt`: Saved model weights for each fold

---

## 📁 Project Structure

```
ai-cup-2025/
├── Model/
│   └── gnn.py              # Main GNN implementation
│   └── README.md
```

---

## 🔑 Key Insights

1. **Direction Matters**: Separating incoming/outgoing edges captures asymmetric fraud patterns (e.g., money mule accounts receive from many sources but send to few)

2. **Attention Highlights Anomalies**: The model learns which transactions are most indicative of fraud without manual feature engineering

3. **Temporal Patterns**: Time-based features (cyclical encoding, time deltas) capture fraud timing behaviors

4. **Ensemble Robustness**: Cross-validation ensemble reduces overfitting to specific fraud patterns
