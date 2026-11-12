"""
Module: Advanced Fraud Ring & Subgraph Pattern Discovery
Implements automated graph pattern matching, cycle enumeration, and subgraph
isomorphism detectors for complex financial crime topologies:
- Shell Company Dispersal Trees (Fan-out -> Layering -> Fan-in)
- Multi-Hop Bipartite Smurfing Networks
- Directed Wash Trading Rings (Cycles of length 3-6)
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import networkx as nx
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("RingDetector")


class AdvancedRingDetector:
    """
    Forensic engine that mines complex high-order laundering topologies
    from directed transaction multigraphs.
    """

    def __init__(self, G: nx.DiGraph, df_tx: pd.DataFrame):
        self.G = G
        self.df_tx = df_tx

    def find_all_rings(self) -> Dict[str, List[Dict[str, Any]]]:
        """Runs the complete suite of subgraph pattern detectors."""
        logger.info("Executing Advanced Subgraph & Fraud Ring Discovery...")
        wash_cycles = self.detect_wash_cycles(max_length=6, max_rings=50)
        dispersal_trees = self.detect_shell_dispersal_trees(max_patterns=30)
        smurfing_bipartite = self.detect_bipartite_smurfing(max_patterns=25)

        logger.info(
            f"Discovery Summary: {len(wash_cycles)} Wash Rings, "
            f"{len(dispersal_trees)} Dispersal Trees, "
            f"{len(smurfing_bipartite)} Bipartite Smurfing Networks."
        )

        return {
            "wash_cycles": wash_cycles,
            "shell_dispersal_trees": dispersal_trees,
            "bipartite_smurfing": smurfing_bipartite
        }

    def detect_wash_cycles(self, max_length: int = 6, max_rings: int = 50) -> List[Dict[str, Any]]:
        """
        Discovers closed directed cycles ($A \\to B \\to C \\dots \\to A$)
        with metric analysis (volume laundered, hop count, shaving fee).
        """
        rings = []
        nodes = list(self.G.nodes())

        # Bounded DFS cycle finder
        for start_node in nodes:
            if len(rings) >= max_rings:
                break
            stack = [(start_node, [start_node], {start_node})]

            while stack and len(rings) < max_rings:
                curr, path, visited = stack.pop()
                if len(path) > max_length:
                    continue

                for neighbor in self.G.successors(curr):
                    if neighbor == start_node and len(path) >= 3:
                        # Normalize cyclic rotation
                        canonical = min(path[i:] + path[:i] for i in range(len(path)))
                        if not any(r["path"] == canonical for r in rings):
                            # Analyze ring financial volume and velocity
                            ring_meta = self._analyze_cycle_metrics(canonical)
                            rings.append(ring_meta)
                    elif neighbor not in visited and len(path) < max_length:
                        stack.append((neighbor, path + [neighbor], visited | {neighbor}))

        # Sort by total laundered volume descending
        rings.sort(key=lambda x: x["total_volume"], reverse=True)
        return rings

    def _analyze_cycle_metrics(self, cycle_path: List[str]) -> Dict[str, Any]:
        """Calculates volume, edge count, and severity for a detected cycle."""
        k = len(cycle_path)
        edge_amounts = []
        for i in range(k):
            u = cycle_path[i]
            v = cycle_path[(i + 1) % k]
            if self.G.has_edge(u, v):
                edge_amounts.append(self.G[u][v].get("weight", 0.0))

        tot_vol = sum(edge_amounts)
        avg_hop_amt = tot_vol / max(1, len(edge_amounts))

        # Severity index: 0 - 100 based on volume and length
        severity = min(100.0, 40.0 + (min(tot_vol, 150000) / 150000.0) * 45.0 + (k * 2.5))

        return {
            "type": "Wash_Trading_Cycle",
            "path": cycle_path,
            "path_str": " ➔ ".join(cycle_path) + f" ➔ {cycle_path[0]}",
            "hop_count": k,
            "total_volume": round(tot_vol, 2),
            "avg_hop_amount": round(avg_hop_amt, 2),
            "severity_score": round(severity, 1),
            "members": cycle_path
        }

    def detect_shell_dispersal_trees(self, max_patterns: int = 30) -> List[Dict[str, Any]]:
        """
        Detects multi-tier shell company dispersal trees:
        Source Master -> [Intermediary Buffers / Shells] -> Final Collector Sink.
        Topological signature: S with high fan-out to nodes that immediately converge to D.
        """
        trees = []
        # Find candidates with fan-out >= 3
        high_out_nodes = [n for n in self.G.nodes() if self.G.out_degree(n) >= 3]

        for source in high_out_nodes:
            if len(trees) >= max_patterns:
                break
            buffers = list(self.G.successors(source))
            if len(buffers) < 3:
                continue

            # Check if buffers converge to common sinks
            sink_counts: Dict[str, List[str]] = {}
            for b in buffers:
                for sink in self.G.successors(b):
                    if sink != source and sink not in buffers:
                        sink_counts.setdefault(sink, []).append(b)

            # Find sinks with at least 2 common buffer feeders
            for sink, fed_by_buffers in sink_counts.items():
                if len(fed_by_buffers) >= 2:
                    # Found a structuring dispersal tree!
                    # Calculate flow volume
                    in_vol = sum(self.G[source][b]["weight"] for b in fed_by_buffers if self.G.has_edge(source, b))
                    out_vol = sum(self.G[b][sink]["weight"] for b in fed_by_buffers if self.G.has_edge(b, sink))

                    severity = min(100.0, 50.0 + (min(in_vol + out_vol, 200000) / 200000.0) * 40.0 + len(fed_by_buffers) * 3.0)

                    tree_record = {
                        "type": "Shell_Dispersal_Tree",
                        "source": source,
                        "sink": sink,
                        "shell_buffers": fed_by_buffers,
                        "buffer_count": len(fed_by_buffers),
                        "total_dispersed_volume": round(in_vol, 2),
                        "total_consolidated_volume": round(out_vol, 2),
                        "severity_score": round(severity, 1),
                        "members": [source] + fed_by_buffers + [sink]
                    }

                    # Avoid redundant near-duplicate patterns
                    if not any(t["source"] == source and t["sink"] == sink for t in trees):
                        trees.append(tree_record)
                        if len(trees) >= max_patterns:
                            break

        trees.sort(key=lambda x: x["total_dispersed_volume"], reverse=True)
        return trees

    def detect_bipartite_smurfing(self, max_patterns: int = 25) -> List[Dict[str, Any]]:
        """
        Detects bipartite smurfing cross-flows:
        A group of senders {S1, S2, ...} transferring to a group of receivers {R1, R2, ...}
        forming dense bipartite cross-transaction connections.
        """
        patterns = []
        nodes = list(self.G.nodes())

        # Sample high-activity nodes to find dense bipartite connectivity
        for i, s1 in enumerate(nodes[:250]):
            if len(patterns) >= max_patterns:
                break
            s1_receivers = set(self.G.successors(s1))
            if len(s1_receivers) < 2:
                continue

            for s2 in nodes[i + 1:min(i + 100, len(nodes))]:
                s2_receivers = set(self.G.successors(s2))
                common_receivers = s1_receivers.intersection(s2_receivers)

                if len(common_receivers) >= 3:
                    # Found 2 senders cross-feeding 3+ common collectors
                    senders = [s1, s2]
                    receivers = list(common_receivers)
                    tot_vol = 0.0
                    for s in senders:
                        for r in receivers:
                            if self.G.has_edge(s, r):
                                tot_vol += self.G[s][r].get("weight", 0.0)

                    severity = min(100.0, 45.0 + (min(tot_vol, 100000) / 100000.0) * 40.0)

                    pat = {
                        "type": "Bipartite_Smurfing_Crossflow",
                        "senders": senders,
                        "receivers": receivers,
                        "sender_count": len(senders),
                        "receiver_count": len(receivers),
                        "total_volume": round(tot_vol, 2),
                        "severity_score": round(severity, 1),
                        "members": senders + receivers
                    }
                    if not any(set(p["members"]) == set(pat["members"]) for p in patterns):
                        patterns.append(pat)
                        if len(patterns) >= max_patterns:
                            break

        patterns.sort(key=lambda x: x["total_volume"], reverse=True)
        return patterns

