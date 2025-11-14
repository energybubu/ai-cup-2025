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

---

## 🔑 Key Insights

1. **Direction Matters**: Separating incoming/outgoing edges captures asymmetric fraud patterns (e.g., money mule accounts receive from many sources but send to few)

2. **Attention Highlights Anomalies**: The model learns which transactions are most indicative of fraud without manual feature engineering

3. **Temporal Patterns**: Time-based features (cyclical encoding, time deltas) capture fraud timing behaviors

4. **Ensemble Robustness**: Cross-validation ensemble reduces overfitting to specific fraud patterns
