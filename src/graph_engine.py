"""
Module 2: Graph Modeling & Analytics Engine
Constructs directed transaction networks using NetworkX, computes structural and
centrality features (PageRank, Degree, Betweenness, Louvain Community Detection),
and detects topological fraud signatures (cycles and mule hubs).
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

from src.utils import ensure_directories, PROCESSED_DATA_DIR, RAW_DATA_DIR, get_logger

logger = get_logger("GraphEngine")


def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Constructs a weighted directed graph G = (V, E) from transactions.
    Edge attributes include aggregated amount, transaction count, and average amount.
    """
    logger.info("Building directed transaction graph from dataset...")
    G = nx.DiGraph()

    # Aggregate transactions between unique pairs
    edge_aggs = df.groupby(["Sender_Account", "Receiver_Account"]).agg(
        total_amount=("Amount", "sum"),
        tx_count=("Amount", "count"),
        avg_amount=("Amount", "mean"),
        max_amount=("Amount", "max"),
        fraud_tx_count=("Is_Fraud", "sum")
    ).reset_index()

    for _, row in edge_aggs.iterrows():
        sender = str(row["Sender_Account"])
        receiver = str(row["Receiver_Account"])
        G.add_edge(
            sender,
            receiver,
            weight=float(row["total_amount"]),
            tx_count=int(row["tx_count"]),
            avg_amount=float(row["avg_amount"]),
            max_amount=float(row["max_amount"]),
            has_fraud=(int(row["fraud_tx_count"]) > 0)
        )

    logger.info(f"Graph constructed: {G.number_of_nodes()} accounts (nodes), {G.number_of_edges()} directed connections (edges).")
    return G


def detect_cycles_bounded(G: nx.DiGraph, max_length: int = 5, max_cycles: int = 150) -> List[List[str]]:
    """
    Finds elementary directed cycles bounded by max_length using depth-first search.
    Crucial for identifying circular laundering and wash trading (A -> B -> C -> A).
    """
    logger.info(f"Detecting directed cycles up to length {max_length}...")
    cycles = []
    
    # We use a bounded DFS search to find short cycles efficiently
    nodes = list(G.nodes())
    node_set = set(nodes)

    for start_node in nodes:
        if len(cycles) >= max_cycles:
            break
            
        stack = [(start_node, [start_node], {start_node})]
        while stack and len(cycles) < max_cycles:
            current, path, visited = stack.pop()
            
            if len(path) > max_length:
                continue

            for neighbor in G.successors(current):
                if neighbor == start_node and len(path) >= 3:
                    # Found a cycle of length 3 to max_length
                    # Normalize rotation to prevent duplicates
                    canonical = min(path[i:] + path[:i] for i in range(len(path)))
                    if canonical not in cycles:
                        cycles.append(canonical)
                        if len(cycles) >= max_cycles:
                            break
                elif neighbor not in visited and len(path) < max_length:
                    stack.append((neighbor, path + [neighbor], visited | {neighbor}))

    logger.info(f"Discovered {len(cycles)} suspicious circular transaction paths.")
    return cycles


def compute_graph_metrics(G: nx.DiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comprehensive structural and centrality features for each account in the network:
    - In-Degree Centrality, Out-Degree Centrality, Degree Ratio
    - PageRank (structural flow importance)
    - Betweenness Centrality (intermediary / bridge mule score)
    - Community Detection (Louvain / Greedy Modularity Partitioning)
    - Circular flow participation
    - Volume inflow, outflow, and net balance ratio
    """
    logger.info("Computing network topological metrics & centralities...")

    nodes = list(G.nodes())
    if not nodes:
        logger.warning("Empty graph provided.")
        return pd.DataFrame()

    # 1. Centralities
    logger.info("Calculating PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight")

    logger.info("Calculating Degree Centralities...")
    in_degree_cent = nx.in_degree_centrality(G)
    out_degree_cent = nx.out_degree_centrality(G)

    logger.info("Calculating Betweenness Centrality (sampled for efficiency)...")
    # Using k=min(250, len(nodes)) for fast, scalable approximation on large graphs
    sample_k = min(300, len(nodes))
    betweenness = nx.betweenness_centrality(G, k=sample_k, weight="weight", seed=42)

    # 2. Community Detection (using Louvain on undirected representation)
    logger.info("Running Louvain Community Detection...")
    G_undirected = G.to_undirected()
    try:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
    except Exception as e:
        logger.warning(f"Louvain failed ({e}), falling back to greedy modularity.")
        communities = nx.community.greedy_modularity_communities(G_undirected)

    community_map = {}
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            community_map[node] = comm_id

    # 3. Cycle Detection
    cycles = detect_cycles_bounded(G, max_length=5, max_cycles=150)
    cycle_nodes_count = {}
    for cycle in cycles:
        for node in cycle:
            cycle_nodes_count[node] = cycle_nodes_count.get(node, 0) + 1

    # 4. Financial Flow Aggregations from dataframe
    inflow_summary = df.groupby("Receiver_Account").agg(
        total_inflow=("Amount", "sum"),
        in_tx_count=("Amount", "count"),
        unique_senders=("Sender_Account", "nunique")
    ).to_dict("index")

    outflow_summary = df.groupby("Sender_Account").agg(
        total_outflow=("Amount", "sum"),
        out_tx_count=("Amount", "count"),
        unique_receivers=("Receiver_Account", "nunique")
    ).to_dict("index")

    # Assemble feature records
    feature_rows = []
    for node in nodes:
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        tot_deg = in_deg + out_deg
        deg_ratio = float((in_deg + 1) / (out_deg + 1))

        inflow_info = inflow_summary.get(node, {"total_inflow": 0.0, "in_tx_count": 0, "unique_senders": 0})
        outflow_info = outflow_summary.get(node, {"total_outflow": 0.0, "out_tx_count": 0, "unique_receivers": 0})

        tot_in = inflow_info["total_inflow"]
        tot_out = outflow_info["total_outflow"]
        total_volume = tot_in + tot_out
        turnover_balance = abs(tot_in - tot_out) / (total_volume + 1e-5)

        # Money mule heuristic: High inflow from many senders rapidly depleted to few receivers
        is_mule_pattern = (
            inflow_info["unique_senders"] >= 5 and
            outflow_info["unique_receivers"] <= 3 and
            tot_in > 5000 and
            turnover_balance < 0.25
        )

        in_cycle_freq = cycle_nodes_count.get(node, 0)

        feature_rows.append({
            "Account_ID": node,
            "pagerank": float(pagerank.get(node, 0.0)),
            "betweenness_centrality": float(betweenness.get(node, 0.0)),
            "in_degree_centrality": float(in_degree_cent.get(node, 0.0)),
            "out_degree_centrality": float(out_degree_cent.get(node, 0.0)),
            "in_degree": int(in_deg),
            "out_degree": int(out_deg),
            "total_degree": int(tot_deg),
            "degree_ratio": deg_ratio,
            "community_id": int(community_map.get(node, -1)),
            "in_cycle_count": int(in_cycle_freq),
            "is_in_cycle": int(in_cycle_freq > 0),
            "total_inflow": float(tot_in),
            "total_outflow": float(tot_out),
            "total_volume": float(total_volume),
            "turnover_balance_ratio": float(turnover_balance),
            "is_mule_heuristic": int(is_mule_pattern)
        })

    features_df = pd.DataFrame(feature_rows)
    logger.info(f"Graph features successfully extracted for {len(features_df)} accounts.")
    return features_df


def extract_ego_subgraph(G: nx.DiGraph, center_node: str, radius: int = 1) -> nx.DiGraph:
    """
    Extracts an ego network (1-hop or 2-hop) centered on a specific account for forensic inspection.
    Includes all internal edges among neighbor accounts.
    """
    if center_node not in G:
        logger.warning(f"Account {center_node} not in graph.")
        return nx.DiGraph()

    # Use undirected projection to get bidirectional neighborhood
    G_undirected = G.to_undirected()
    ego_nodes = set(nx.ego_graph(G_undirected, center_node, radius=radius).nodes())

    subgraph = G.subgraph(ego_nodes).copy()
    return subgraph


def run_graph_pipeline(
    transactions_csv: Optional[Path] = None,
    output_features_csv: Optional[Path] = None
) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """Runs the complete graph pipeline and exports account-level graph features."""
    ensure_directories()
    if transactions_csv is None:
        transactions_csv = RAW_DATA_DIR / "transactions.csv"
    if output_features_csv is None:
        output_features_csv = PROCESSED_DATA_DIR / "account_graph_features.csv"

    if not transactions_csv.exists():
        logger.info(f"Transactions file not found at {transactions_csv}. Generating now...")
        from src.data_generator import generate_financial_dataset
        df_tx = generate_financial_dataset()
    else:
        df_tx = pd.read_csv(transactions_csv)

    G = build_transaction_graph(df_tx)
    features_df = compute_graph_metrics(G, df_tx)

    features_df.to_csv(output_features_csv, index=False)
    logger.info(f"Saved graph features to: {output_features_csv}")

    return G, features_df


if __name__ == "__main__":
    run_graph_pipeline()

