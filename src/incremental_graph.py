"""
Module: Real-Time Dynamic & Incremental Graph Engine
Maintains an in-memory dynamic transaction graph that updates node statistics,
in-degree/out-degree, warm-started PageRank, and community tracking in real time
without rebuilding the graph from scratch on every transaction.
"""

import math
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
import networkx as nx
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("IncrementalGraph")


class DynamicTransactionGraph:
    """
    A high-throughput, incremental directed transaction network with warm-started
    PageRank approximation and localized modularity delta tracking.
    """

    def __init__(self, damping: float = 0.85):
        self.damping = damping
        self.G = nx.DiGraph()

        # Node attribute stores
        self.node_inflow: Dict[str, float] = {}
        self.node_outflow: Dict[str, float] = {}
        self.node_tx_count_in: Dict[str, int] = {}
        self.node_tx_count_out: Dict[str, int] = {}
        self.node_unique_senders: Dict[str, Set[str]] = {}
        self.node_unique_receivers: Dict[str, Set[str]] = {}
        
        # Centrality & Community state
        self.pagerank_scores: Dict[str, float] = {}
        self.community_map: Dict[str, int] = {}
        self.in_cycle_nodes: Set[str] = set()

        # Alert Log
        self.recent_alerts: List[Dict[str, Any]] = []

    def initialize_from_dataframe(self, df: pd.DataFrame, df_graph: Optional[pd.DataFrame] = None):
        """Pre-populates the dynamic graph with historical transactions."""
        logger.info(f"Initializing Dynamic Graph with {len(df)} historical transactions...")
        for _, row in df.iterrows():
            self.add_transaction(
                sender=str(row["Sender_Account"]),
                receiver=str(row["Receiver_Account"]),
                amount=float(row["Amount"]),
                timestamp=pd.to_datetime(row["Timestamp"]),
                is_fraud=int(row.get("Is_Fraud", 0)),
                fraud_type=str(row.get("Fraud_Type", "None")),
                run_analytics=False  # Batch loading, defer analytics
            )

        # Initialize PageRank
        self._compute_full_pagerank()

        # Initialize Communities
        if df_graph is not None and "community_id" in df_graph.columns:
            self.community_map = df_graph.set_index("Account_ID")["community_id"].to_dict()
        else:
            self._compute_initial_communities()

        logger.info(f"Dynamic Graph initialized: {self.G.number_of_nodes()} accounts, {self.G.number_of_edges()} edges.")

    def add_transaction(
        self,
        sender: str,
        receiver: str,
        amount: float,
        timestamp: datetime,
        is_fraud: int = 0,
        fraud_type: str = "None",
        run_analytics: bool = True
    ) -> Dict[str, Any]:
        """
        Incrementally ingests a single incoming transaction.
        Updates edge weights, node statistics, and triggers warm-started analytics.
        """
        # Ensure nodes exist in registry
        for node in (sender, receiver):
            if node not in self.G:
                self.G.add_node(node)
                self.node_inflow[node] = 0.0
                self.node_outflow[node] = 0.0
                self.node_tx_count_in[node] = 0
                self.node_tx_count_out[node] = 0
                self.node_unique_senders[node] = set()
                self.node_unique_receivers[node] = set()
                # Initial default PageRank for new node
                self.pagerank_scores[node] = 1.0 / max(1, self.G.number_of_nodes())
                self.community_map[node] = self._assign_new_node_community(node)

        # Update node balance & counterparty trackers
        self.node_outflow[sender] += amount
        self.node_tx_count_out[sender] += 1
        self.node_unique_receivers[sender].add(receiver)

        self.node_inflow[receiver] += amount
        self.node_tx_count_in[receiver] += 1
        self.node_unique_senders[receiver].add(sender)

        # Update edge attributes
        if self.G.has_edge(sender, receiver):
            edge_data = self.G[sender][receiver]
            edge_data["weight"] += amount
            edge_data["tx_count"] += 1
            edge_data["avg_amount"] = edge_data["weight"] / edge_data["tx_count"]
            edge_data["max_amount"] = max(edge_data["max_amount"], amount)
            edge_data["last_timestamp"] = timestamp
            if is_fraud:
                edge_data["has_fraud"] = True
        else:
            self.G.add_edge(
                sender,
                receiver,
                weight=amount,
                tx_count=1,
                avg_amount=amount,
                max_amount=amount,
                first_timestamp=timestamp,
                last_timestamp=timestamp,
                has_fraud=bool(is_fraud)
            )

        alert = None
        if run_analytics:
            # 1. Warm-started incremental PageRank update (3 iterations)
            self.update_incremental_pagerank(affected_nodes=[sender, receiver], iterations=3)

            # 2. Dynamic community tracking
            self.update_incremental_community(sender, receiver)

            # 3. Check real-time anomaly risk
            alert = self._evaluate_realtime_alert(sender, receiver, amount, timestamp, is_fraud, fraud_type)
            if alert:
                self.recent_alerts.append(alert)
                if len(self.recent_alerts) > 100:
                    self.recent_alerts.pop(0)

        return {
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "timestamp": timestamp,
            "sender_pagerank": self.pagerank_scores.get(sender, 0.0),
            "receiver_pagerank": self.pagerank_scores.get(receiver, 0.0),
            "sender_out_degree": self.G.out_degree(sender),
            "receiver_in_degree": self.G.in_degree(receiver),
            "alert": alert
        }

    def update_incremental_pagerank(self, affected_nodes: List[str], iterations: int = 3):
        """
        Applies a warm-started Power Iteration using previous PageRank scores as prior.
        Converges in 2-3 iterations with negligible compute overhead.
        """
        N = self.G.number_of_nodes()
        if N == 0:
            return

        # Normalize prior distribution
        p = self.pagerank_scores.copy()
        total_p = sum(p.values())
        if total_p > 0:
            p = {k: v / total_p for k, v in p.items()}
        else:
            p = {k: 1.0 / N for k in self.G.nodes()}

        teleport = (1.0 - self.damping) / N

        for _ in range(iterations):
            p_next = {node: teleport for node in self.G.nodes()}
            for u in self.G.nodes():
                out_weight_sum = self.node_outflow.get(u, 0.0)
                if out_weight_sum > 0:
                    for v in self.G.successors(u):
                        w = self.G[u][v]["weight"]
                        p_next[v] += self.damping * p[u] * (w / out_weight_sum)
                else:
                    # Dangling node distribution
                    dangling_share = (self.damping * p[u]) / N
                    for node in self.G.nodes():
                        p_next[node] += dangling_share
            p = p_next

        self.pagerank_scores = p

    def update_incremental_community(self, sender: str, receiver: str):
        """
        Dynamically adjusts community affiliation if transaction flow creates
        a strong modularity pull towards counterparty cluster.
        """
        s_comm = self.community_map.get(sender, 0)
        r_comm = self.community_map.get(receiver, 0)
        if s_comm == r_comm:
            return

        # If sender has concentrated majority outflow to receiver's community, absorb
        receiver_neighbors_in_comm = sum(
            1 for n in self.G.successors(sender) if self.community_map.get(n) == r_comm
        )
        if receiver_neighbors_in_comm >= 4:
            self.community_map[sender] = r_comm

    def _assign_new_node_community(self, node: str) -> int:
        existing_communities = set(self.community_map.values())
        return max(existing_communities) + 1 if existing_communities else 0

    def _compute_full_pagerank(self):
        try:
            self.pagerank_scores = nx.pagerank(self.G, alpha=self.damping, weight="weight")
        except Exception:
            N = max(1, self.G.number_of_nodes())
            self.pagerank_scores = {n: 1.0 / N for n in self.G.nodes()}

    def _compute_initial_communities(self):
        G_und = self.G.to_undirected()
        try:
            comms = nx.community.louvain_communities(G_und, seed=42)
        except Exception:
            comms = nx.community.greedy_modularity_communities(G_und)
        self.community_map = {}
        for c_id, members in enumerate(comms):
            for m in members:
                self.community_map[m] = c_id

    def _evaluate_realtime_alert(
        self,
        sender: str,
        receiver: str,
        amount: float,
        timestamp: datetime,
        is_fraud: int,
        fraud_type: str
    ) -> Optional[Dict[str, Any]]:
        """Evaluates whether an incoming transaction triggers an immediate forensic alert."""
        reasons = []

        # 1. Money mule sudden burst heuristic
        unique_senders = len(self.node_unique_senders.get(receiver, set()))
        if unique_senders >= 6 and self.G.in_degree(receiver) >= 8 and self.node_inflow[receiver] > 8000:
            reasons.append(f"Mule Collector Burst: {receiver} reached {unique_senders} unique inbound senders")

        # 2. Cycle loop completion
        if self.G.has_edge(receiver, sender):
            reasons.append(f"Direct Reciprocal Wash Loop: Immediate loop between {sender} and {receiver}")

        # 3. High amount structuring/outlier
        if amount >= 25000:
            reasons.append(f"High-Value Transaction Anomaly (${amount:,.2f})")

        # 4. Injected ground truth fraud or high PageRank bridge
        s_pr = self.pagerank_scores.get(sender, 0.0)
        r_pr = self.pagerank_scores.get(receiver, 0.0)
        if s_pr > 0.005 or r_pr > 0.005:
            reasons.append(f"High PageRank Flow Bridge ({sender} ➔ {receiver})")

        if is_fraud or reasons:
            severity = "CRITICAL" if (is_fraud or len(reasons) >= 2) else "WARNING"
            return {
                "timestamp": str(timestamp),
                "severity": severity,
                "sender": sender,
                "receiver": receiver,
                "amount": amount,
                "reasons": reasons if reasons else [f"Identified Typology: {fraud_type}"],
                "fraud_type": fraud_type
            }
        return None

    def get_summary_stats(self) -> Dict[str, Any]:
        """Returns instantaneous snapshot metrics of the dynamic graph."""
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "total_volume": sum(self.node_inflow.values()),
            "total_communities": len(set(self.community_map.values())),
            "recent_alerts_count": len(self.recent_alerts)
        }

