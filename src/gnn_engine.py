"""
Module: Graph Neural Networks (GNN) for Fraud Classification
Implements a 2-layer inductive GraphSAGE / GCN architecture in PyTorch to learn
dense 64-dimensional structural embeddings directly from the transaction graph topology.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import PROCESSED_DATA_DIR, MODELS_DIR, ensure_directories, get_logger

logger = get_logger("GNNEngine")


class GraphSAGELayer(nn.Module):
    """
    Inductive GraphSAGE Convolution Layer with Mean Neighbor Aggregation
    and self-feature concatenation: h_v = sigma( W * [h_v || MEAN(h_u, u in N(v))] ).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Self transformation + Neighbor aggregation transformation
        self.fc_self = nn.Linear(in_features, out_features, bias=bias)
        self.fc_neigh = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_features)
        adj_norm: (N, N) row-normalized adjacency matrix
        """
        h_self = self.fc_self(x)
        h_neigh = self.fc_neigh(torch.matmul(adj_norm, x))
        out = h_self + h_neigh
        out = self.norm(F.relu(out))
        return out


class TransactionGraphGNN(nn.Module):
    """
    2-Layer Graph Neural Network for generating node structural embeddings
    and predicting transaction fraud edge probabilities.
    """

    def __init__(self, in_dim: int = 12, hidden_dim: int = 128, embed_dim: int = 64, dropout: float = 0.2):
        super(TransactionGraphGNN, self).__init__()
        self.layer1 = GraphSAGELayer(in_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Edge Classifier Head: predicts fraud probability on [z_u || z_v || |z_u - z_v| || z_u * z_v]
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def get_embeddings(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN layers returning 64-dim node embeddings."""
        h1 = self.dropout(self.layer1(x, adj_norm))
        z = self.layer2(h1, adj_norm)
        return z

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes node embeddings and edge fraud logits for given source and target node pairs.
        """
        z = self.get_embeddings(x, adj_norm)
        z_src = z[src_idx]
        z_dst = z[dst_idx]

        # Symmetrical interaction terms
        diff = torch.abs(z_src - z_dst)
        prod = z_src * z_dst
        edge_feats = torch.cat([z_src, z_dst, diff, prod], dim=1)

        logits = self.edge_mlp(edge_feats).squeeze(-1)
        return logits, z


def prepare_graph_tensors(
    df_graph: pd.DataFrame,
    df_tx: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares node feature tensor, row-normalized adjacency matrix, and edge indices.
    """
    accounts = sorted(df_graph["Account_ID"].unique().tolist())
    node2idx = {acc: i for i, acc in enumerate(accounts)}
    N = len(accounts)

    # Node base features
    feature_cols = [
        "pagerank", "betweenness_centrality", "in_degree_centrality",
        "out_degree_centrality", "in_degree", "out_degree", "degree_ratio",
        "in_cycle_count", "is_in_cycle", "turnover_balance_ratio",
        "is_mule_heuristic"
    ]

    # Normalize node features
    feats = df_graph.set_index("Account_ID").reindex(accounts)[feature_cols].fillna(0.0).values
    # Add log total volume feature
    vol = df_graph.set_index("Account_ID").reindex(accounts)["total_volume"].fillna(0.0).values
    log_vol = np.log1p(vol).reshape(-1, 1)
    feats = np.hstack([feats, log_vol])

    # Z-score normalization per column
    means = np.mean(feats, axis=0, keepdims=True)
    stds = np.std(feats, axis=0, keepdims=True) + 1e-6
    feats_norm = (feats - means) / stds

    x_tensor = torch.tensor(feats_norm, dtype=torch.float32)

    # Construct Normalized Adjacency Matrix
    adj = np.zeros((N, N), dtype=np.float32)
    for _, row in df_tx.iterrows():
        s = row["Sender_Account"]
        r = row["Receiver_Account"]
        if s in node2idx and r in node2idx:
            u, v = node2idx[s], node2idx[r]
            adj[u, v] += float(np.log1p(row["Amount"]))

    # Add self loops
    np.fill_diagonal(adj, 1.0)
    # Row normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums
    adj_tensor = torch.tensor(adj_norm, dtype=torch.float32)

    # Edge training pairs
    valid_tx = df_tx[df_tx["Sender_Account"].isin(node2idx) & df_tx["Receiver_Account"].isin(node2idx)].copy()
    src_idx = torch.tensor([node2idx[s] for s in valid_tx["Sender_Account"]], dtype=torch.long)
    dst_idx = torch.tensor([node2idx[r] for r in valid_tx["Receiver_Account"]], dtype=torch.long)
    edge_labels = torch.tensor(valid_tx["Is_Fraud"].values, dtype=torch.float32)

    return x_tensor, adj_tensor, node2idx, src_idx, dst_idx, edge_labels


def train_gnn_model(
    df_graph: pd.DataFrame,
    df_tx: pd.DataFrame,
    epochs: int = 40,
    lr: float = 0.008
) -> Tuple[TransactionGraphGNN, Dict[str, np.ndarray]]:
    """
    Trains the TransactionGraphGNN on the financial graph and extracts 64-dim embeddings.
    """
    logger.info("Initializing Graph Neural Network (GraphSAGE / GCN) training...")
    ensure_directories()

    x_tensor, adj_tensor, node2idx, src_idx, dst_idx, edge_labels = prepare_graph_tensors(df_graph, df_tx)
    in_dim = x_tensor.shape[1]

    # Calculate class weight for edge BCE
    pos_count = edge_labels.sum().item()
    neg_count = len(edge_labels) - pos_count
    pos_weight = torch.tensor([neg_count / max(1.0, pos_count)])

    model = TransactionGraphGNN(in_dim=in_dim, hidden_dim=128, embed_dim=64, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logger.info(f"Training GNN for {epochs} epochs over {x_tensor.shape[0]} nodes and {len(edge_labels)} edges...")

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, _ = model(x_tensor, adj_tensor, src_idx, dst_idx)
        loss = criterion(logits, edge_labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == edge_labels).float().mean().item()
            logger.info(f"GNN Epoch {epoch:02d}/{epochs:02d} - Edge Loss: {loss.item():.4f} | Training Acc: {acc:.4f}")

    # Extract learned 64-dimensional node structural embeddings
    model.eval()
    with torch.no_grad():
        node_embeddings = model.get_embeddings(x_tensor, adj_tensor).cpu().numpy()

    embeddings_dict = {}
    for acc, idx in node2idx.items():
        embeddings_dict[acc] = node_embeddings[idx]

    # Persist GNN model weights and embeddings
    torch.save(model.state_dict(), MODELS_DIR / "gnn_fraud_model.pt")
    torch.save(embeddings_dict, PROCESSED_DATA_DIR / "gnn_node_embeddings.pt")
    logger.info(f"Saved GNN structural embeddings for {len(embeddings_dict)} accounts to: {PROCESSED_DATA_DIR / 'gnn_node_embeddings.pt'}")

    return model, embeddings_dict


def load_gnn_embeddings() -> Dict[str, np.ndarray]:
    """Loads precomputed GNN structural embeddings if available."""
    emb_path = PROCESSED_DATA_DIR / "gnn_node_embeddings.pt"
    if emb_path.exists():
        return torch.load(emb_path, map_location="cpu", weights_only=False)
    return {}

