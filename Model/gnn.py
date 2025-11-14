import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GINEConv
from torch_geometric.utils import softmax


class DirectedEdgeAttention(nn.Module):
    """Direction-aware attention mechanism for graph edges.

    Computes attention weights that distinguish between incoming and outgoing
    edges, allowing the model to learn asymmetric importance patterns in
    directed transaction networks. This is crucial for fraud detection where
    money flow direction matters (e.g., receiving from suspicious sources vs.
    sending to them).

    The attention mechanism learns to weight edges based on:
    - Source node features (sender in transactions).
    - Destination node features (receiver in transactions).
    - Edge features (transaction properties).

    Attributes:
        heads: Number of attention heads for multi-head attention.
        n_hidden: Dimension of hidden node/edge representations.
        edge_direction: Direction to compute attention for ('in' or 'out').
        att_src: Linear layer for source node attention.
        att_dst: Linear layer for destination node attention.
        att_edge: Linear layer for edge feature attention.
        bias: Direction-specific learnable bias parameter.
        dropout: Dropout layer for attention weights.
        leaky_relu: Leaky ReLU activation function.
    """

    def __init__(self, n_hidden, heads=4, dropout=0.0, edge_direction="in"):
        """Initialize DirectedEdgeAttention.

        Args:
            n_hidden: Dimension of hidden node/edge representations.
            heads: Number of attention heads (default: 4).
            dropout: Dropout probability for attention weights (default: 0.0).
            edge_direction: Direction for attention ('in' or 'out', default: 'in').
        """
        super().__init__()
        self.heads = heads
        self.n_hidden = n_hidden
        self.edge_direction = edge_direction

        # Separate attention parameters for source and destination
        self.att_src = nn.Linear(n_hidden, heads, bias=False)
        self.att_dst = nn.Linear(n_hidden, heads, bias=False)
        self.att_edge = nn.Linear(n_hidden, heads, bias=False)

        # Direction-specific learnable bias
        self.bias = nn.Parameter(torch.zeros(heads))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        """Compute attention weights for graph edges.

        Args:
            x: Node feature matrix [num_nodes, n_hidden].
            edge_index: Graph connectivity in COO format [2, num_edges].
            edge_attr: Edge feature matrix [num_edges, n_hidden].

        Returns:
            Tensor: Normalized attention weights [num_edges, heads].
        """
        src, dst = edge_index

        # Compute attention components from source, destination, and edge features
        alpha_src = self.att_src(x[src])  # [num_edges, heads]
        alpha_dst = self.att_dst(x[dst])  # [num_edges, heads]
        alpha_edge = self.att_edge(edge_attr)  # [num_edges, heads]

        # Combine attention components with direction-specific bias
        alpha = self.leaky_relu(alpha_src + alpha_dst + alpha_edge + self.bias)

        # Apply dropout before normalization
        alpha = self.dropout(alpha)

        # Normalize attention coefficients separately for each node's neighborhood
        if self.edge_direction == "in":
            alpha = softmax(
                alpha, dst, num_nodes=x.size(0)
            )  # normalize over incoming edges
        else:
            alpha = softmax(
                alpha, src, num_nodes=x.size(0)
            )  # normalize over outgoing edges

        return alpha


class DirectedGINeWithAttention(torch.nn.Module):
    """Directed Graph Isomorphism Network with Edge Features and Attention.

    A specialized GNN architecture for fraud detection in financial
    transaction networks that processes directed edges separately with
    attention-weighted message passing.

    Architecture Overview:
        1. Separate message passing for incoming vs. outgoing edges.
        2. Multi-head attention mechanism to weight edge importance.
        3. Optional edge feature updates for evolving patterns.
        4. Batch normalization and residual connections for stability.

    Key Design Choices for Fraud Detection:
        - Directed processing: Money flow direction is critical.
        - Edge features: Transaction metadata provides crucial signals.
        - Attention: Learns to focus on suspicious patterns.

    Attributes:
        n_hidden: Hidden dimension for embeddings.
        num_gnn_layers: Number of graph convolution layers.
        edge_updates: Whether to update edge features during message passing.
        final_dropout: Dropout rate before final classifier.
        attention_heads: Number of attention heads.
        use_attention: Whether to apply attention mechanism.
        node_emb: Input node embedding layer.
        edge_emb: Input edge embedding layer.
        convs_in: List of incoming edge convolution layers.
        convs_out: List of outgoing edge convolution layers.
        attention_in: List of incoming edge attention layers.
        attention_out: List of outgoing edge attention layers.
        emlps: List of edge update MLPs.
        batch_norms: List of batch normalization layers.
        mlp_node: Final classification layer.
    """

    def __init__(
        self,
        num_features,
        num_gnn_layers=3,
        n_classes=1,
        n_hidden=16,
        edge_updates=False,
        residual=True,
        edge_dim=None,
        dropout=0.0,
        final_dropout=0.5,
        attention_heads=4,
        use_attention=True,
    ):
        """Initialize DirectedGINeWithAttention.

        Args:
            num_features: Dimension of input node features.
            num_gnn_layers: Number of graph convolution layers (default: 3).
            n_classes: Number of output classes (default: 1).
            n_hidden: Hidden dimension for embeddings (default: 16).
            edge_updates: Update edge features during message passing (default: False).
            residual: Use residual connections (default: True).
            edge_dim: Dimension of input edge features (required).
            dropout: Dropout rate for GNN layers (default: 0.0).
            final_dropout: Dropout rate before final classifier (default: 0.5).
            attention_heads: Number of attention heads (default: 4).
            use_attention: Apply attention mechanism (default: True).
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.attention_heads = attention_heads
        self.use_attention = use_attention

        # Input embedding layers
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs_in = nn.ModuleList()
        self.convs_out = nn.ModuleList()
        self.attention_in = nn.ModuleList()
        self.attention_out = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            # GINEConv for incoming edges (who sends money to this account?)
            conv_in = GINEConv(
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden * 2),
                    nn.BatchNorm1d(n_hidden * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(n_hidden * 2, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU(),
                ),
                edge_dim=self.n_hidden,
            )

            # GINEConv for outgoing edges (who receives money from this account?)
            conv_out = GINEConv(
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden * 2),
                    nn.BatchNorm1d(n_hidden * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(n_hidden * 2, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU(),
                ),
                edge_dim=self.n_hidden,
            )

            self.convs_in.append(conv_in)
            self.convs_out.append(conv_out)

            # Attention mechanisms for edge importance weighting
            if self.use_attention:
                self.attention_in.append(
                    DirectedEdgeAttention(
                        n_hidden, attention_heads, dropout, edge_direction="in"
                    )
                )
                self.attention_out.append(
                    DirectedEdgeAttention(
                        n_hidden, attention_heads, dropout, edge_direction="out"
                    )
                )

            # Edge feature update network (optional)
            if self.edge_updates:
                self.emlps.append(
                    nn.Sequential(
                        nn.Linear(3 * self.n_hidden, self.n_hidden),
                        nn.ReLU(),
                        nn.Linear(self.n_hidden, self.n_hidden),
                    )
                )

            self.batch_norms.append(BatchNorm(n_hidden))

        # Final classification layer
        self.mlp_node = nn.Sequential(
            nn.Dropout(final_dropout), nn.Linear(self.n_hidden, n_classes)
        )

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        """Forward pass through directed GNN with optional attention extraction.

        Processes the graph through multiple layers of directed message passing,
        where each layer:
        1. Computes attention weights for incoming and outgoing edges.
        2. Applies attention-weighted message passing for each direction.
        3. Combines directional information with residual connections.
        4. Optionally updates edge features based on node states.

        Args:
            x: Node feature matrix [num_nodes, num_features].
            edge_index: Graph connectivity in COO format [2, num_edges].
            edge_attr: Edge feature matrix [num_edges, edge_dim].
            return_attention: Return attention weights for interpretation
                (default: False).

        Returns:
            out: Node-level predictions [num_nodes, n_classes].
            attention_weights (optional): Dict mapping layer names to attention
                tensors if return_attention=True.
        """
        src, dst = edge_index

        # Project input features to hidden dimension
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        attention_weights = {} if return_attention else None

        # Process through each GNN layer
        for i in range(self.num_gnn_layers):
            if self.use_attention:
                # Compute attention weights for incoming edges
                alpha_in = self.attention_in[i](x, edge_index, edge_attr)

                # Compute attention weights for outgoing edges (reversed direction)
                alpha_out = self.attention_out[i](x, edge_index.flip(0), edge_attr)

                # Store attention weights for interpretation if requested
                if return_attention:
                    attention_weights[f"layer_{i}_in"] = alpha_in.detach()
                    attention_weights[f"layer_{i}_out"] = alpha_out.detach()

                # Average attention across heads for edge feature modulation
                alpha_in_mean = alpha_in.mean(dim=1, keepdim=True)  # [num_edges, 1]
                alpha_out_mean = alpha_out.mean(dim=1, keepdim=True)

                # Modulate edge features by attention weights
                edge_attr_in = edge_attr * alpha_in_mean
                edge_attr_out = edge_attr * alpha_out_mean

                # Apply message passing with attention-weighted edges
                x_in = self.convs_in[i](x, edge_index, edge_attr_in)
                x_out = self.convs_out[i](x, edge_index.flip(0), edge_attr_out)

            else:
                # Standard directed message passing without attention
                x_in = self.convs_in[i](x, edge_index, edge_attr)
                x_out = self.convs_out[i](x, edge_index.flip(0), edge_attr)

            # Combine incoming and outgoing messages with residual connection
            x = (x + F.relu(self.batch_norms[i](x_in + x_out))) / 2

            # Update edge features if enabled
            if self.edge_updates:
                # Edge update: f(source_node, dest_node, current_edge_features)
                edge_attr = (
                    edge_attr
                    + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1))
                ) / 2

        # Final classification layer
        out = self.mlp_node(x)

        if return_attention:
            return out, attention_weights
        return out
