"""
Enterprise Financial Fraud Network & Transaction Relationship Analytics
Forensic Intelligence Platform with GNN Embeddings, Streaming Dynamic Graph,
Advanced Ring Subgraph Mining, SHAP XAI Attribution & PDF Dossier Generation.
"""

import os
import io
import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit.components.v1 as components
from pyvis.network import Network

from src.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, COLOR_PALETTE
from src.graph_engine import build_transaction_graph, extract_ego_subgraph
from src.ring_detector import AdvancedRingDetector
from src.incremental_graph import DynamicTransactionGraph
from src.streaming_engine import LiveTransactionStreamSimulator
from src.report_generator import generate_forensic_pdf
from src.xai_engine import FraudExplainabilityEngine
from src.gnn_engine import load_gnn_embeddings

# Streamlit Page Config
st.set_page_config(
    page_title="Enterprise Financial Fraud Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom FinTech Dark Theme
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stMetric label { color: #94a3b8 !important; font-size: 0.85rem !important; font-weight: 600; }
    .stMetric [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.4rem !important; }
    .badge-critical {
        background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #f59e0b; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;
    }
    .badge-success {
        background-color: #10b981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;
    }
    .card-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .narrative-box {
        background: #111827;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        font-family: monospace;
        font-size: 0.9rem;
        color: #e2e8f0;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading Transaction Dataset & Network Graph...")
def load_core_data():
    raw_tx_path = RAW_DATA_DIR / "transactions.csv"
    graph_feat_path = PROCESSED_DATA_DIR / "account_graph_features.csv"

    if not raw_tx_path.exists() or not graph_feat_path.exists():
        from src.data_generator import generate_financial_dataset
        from src.graph_engine import run_graph_pipeline
        df_tx = generate_financial_dataset()
        _, df_graph = run_graph_pipeline()
    else:
        df_tx = pd.read_csv(raw_tx_path)
        df_graph = pd.read_csv(graph_feat_path)

    df_tx["Timestamp"] = pd.to_datetime(df_tx["Timestamp"])
    G = build_transaction_graph(df_tx)
    gnn_embeddings = load_gnn_embeddings()
    return df_tx, df_graph, G, gnn_embeddings


@st.cache_resource(show_spinner="Initializing Models & XAI Engine...")
def load_models_and_xai():
    model_path = MODELS_DIR / "fraud_detector_xgb.joblib"
    metrics_path = MODELS_DIR / "model_evaluation_metrics.json"

    xgb_model = joblib.load(model_path) if model_path.exists() else None
    metrics_data = {}
    feature_names = []

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
            feature_names = metrics_data.get("feature_names", [])

    xai_engine = None
    if xgb_model is not None and feature_names:
        xai_engine = FraudExplainabilityEngine(xgb_model, feature_names)

    return xgb_model, metrics_data, xai_engine, feature_names


@st.cache_data(show_spinner="Discovering Complex Fraud Rings & Shell Trees...")
def get_ring_topologies(_G: nx.DiGraph, _df_tx: pd.DataFrame):
    detector = AdvancedRingDetector(_G, _df_tx)
    return detector.find_all_rings()


def init_session_state(df_tx: pd.DataFrame, df_graph: pd.DataFrame):
    """Initializes dynamic streaming graph in session state if not already present."""
    if "dynamic_graph" not in st.session_state:
        dyn_g = DynamicTransactionGraph()
        dyn_g.initialize_from_dataframe(df_tx.head(4000), df_graph)
        st.session_state["dynamic_graph"] = dyn_g
        st.session_state["stream_sim"] = LiveTransactionStreamSimulator(dyn_g, df_graph["Account_ID"].tolist())
        st.session_state["stream_events"] = []


def calculate_account_risk_score(account_id: str, df_graph: pd.DataFrame, df_tx: pd.DataFrame, gnn_embeddings: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
    """Calculates account-level composite fraud risk score (0-100%) with GNN and topology factors."""
    row = df_graph[df_graph["Account_ID"] == account_id]
    if row.empty:
        return 0.0, {}

    row_data = row.iloc[0]

    cycle_factor = min(1.0, row_data.get("in_cycle_count", 0) * 0.4)
    mule_factor = float(row_data.get("is_mule_heuristic", 0))
    pr_factor = min(1.0, row_data.get("pagerank", 0.0) * 250.0)
    between_factor = min(1.0, row_data.get("betweenness_centrality", 0.0) * 150.0)
    turnover_factor = float(1.0 - min(1.0, row_data.get("turnover_balance_ratio", 1.0)))

    # GNN structural embedding anomaly factor
    gnn_factor = 0.0
    if gnn_embeddings and account_id in gnn_embeddings:
        z = gnn_embeddings[account_id]
        # Nodes with high embedding magnitude or unusual divergence
        gnn_factor = min(1.0, float(np.linalg.norm(z)) / 8.0)

    acc_txs = df_tx[(df_tx["Sender_Account"] == account_id) | (df_tx["Receiver_Account"] == account_id)]
    fraud_tx_ratio = acc_txs["Is_Fraud"].mean() if not acc_txs.empty else 0.0

    raw_risk = (
        0.25 * fraud_tx_ratio +
        0.20 * cycle_factor +
        0.20 * mule_factor +
        0.15 * between_factor +
        0.10 * turnover_factor +
        0.10 * gnn_factor
    )
    score_pct = round(float(np.clip(raw_risk * 100.0, 0.0, 99.8)), 1)

    factors = {
        "Direct Fraud History": round(fraud_tx_ratio * 100, 1),
        "Cycle / Ring Risk": round(cycle_factor * 100, 1),
        "Mule Hub Signature": round(mule_factor * 100, 1),
        "Betweenness (Bridge Node)": round(between_factor * 100, 1),
        "Turnover Velocity": round(turnover_factor * 100, 1),
        "GNN Structural Anomaly": round(gnn_factor * 100, 1)
    }
    return score_pct, factors


def render_pyvis_network(
    ego_subgraph: nx.DiGraph,
    center_account: str,
    df_graph: pd.DataFrame,
    suspicious_cycles: List[Dict[str, Any]],
    height: str = "560px"
) -> str:
    """Renders interactive physics-enabled PyVis graph canvas."""
    net = Network(height=height, width="100%", directed=True, bgcolor="#0f172a", font_color="#cbd5e1")
    net.force_atlas_2based(gravity=-60, central_gravity=0.015, spring_length=120, spring_strength=0.08, damping=0.88)

    feat_map = df_graph.set_index("Account_ID").to_dict(orient="index")
    cycle_edge_set = set()
    for ring in suspicious_cycles:
        p = ring.get("path", [])
        for i in range(len(p)):
            cycle_edge_set.add((p[i], p[(i + 1) % len(p)]))

    for node in ego_subgraph.nodes():
        node_info = feat_map.get(node, {})
        in_cycle = node_info.get("is_in_cycle", 0)
        is_mule = node_info.get("is_mule_heuristic", 0)
        pagerank = node_info.get("pagerank", 0.001)

        if node == center_account:
            color = COLOR_PALETTE["selected"]
            size = 28
            border_width = 3
            label = f"★ {node}"
        elif in_cycle:
            color = COLOR_PALETTE["cycle_ring"]
            size = 20
            border_width = 2
            label = f"🔄 {node}"
        elif is_mule:
            color = COLOR_PALETTE["mule_hub"]
            size = 22
            border_width = 2
            label = f"⚡ {node}"
        else:
            color = COLOR_PALETTE["legitimate"]
            size = 14
            border_width = 1
            label = node

        title_hover = (
            f"<b>Account: {node}</b><br/>"
            f"PageRank: {pagerank:.5f}<br/>"
            f"In-Degree: {node_info.get('in_degree', 0)} | Out-Degree: {node_info.get('out_degree', 0)}<br/>"
            f"Inflow: ${node_info.get('total_inflow', 0):,.2f}<br/>"
            f"Outflow: ${node_info.get('total_outflow', 0):,.2f}<br/>"
            f"Cycle Ring: {'YES' if in_cycle else 'No'}<br/>"
            f"Mule Pattern: {'YES' if is_mule else 'No'}"
        )

        net.add_node(
            node,
            label=label,
            title=title_hover,
            color={"background": color, "border": "#ffffff", "highlight": {"background": "#f59e0b", "border": "#ffffff"}},
            size=size,
            borderWidth=border_width
        )

    for u, v, data in ego_subgraph.edges(data=True):
        weight = data.get("weight", 0.0)
        tx_count = data.get("tx_count", 1)
        has_fraud = data.get("has_fraud", False)
        is_cycle_edge = (u, v) in cycle_edge_set

        if is_cycle_edge or has_fraud:
            edge_color = COLOR_PALETTE["edge_fraud"]
            edge_width = 3.5
        else:
            edge_color = COLOR_PALETTE["edge_normal"]
            edge_width = 1.2

        edge_title = f"Flow: {u} → {v}<br/>Total: ${weight:,.2f}<br/>Transactions: {tx_count}<br/>Fraud Flag: {has_fraud}"

        net.add_edge(
            u, v,
            title=edge_title,
            value=max(1, int(np.log1p(weight))),
            color=edge_color,
            width=edge_width,
            arrows="to"
        )

    temp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(temp_dir, f"pyvis_graph_{uuid.uuid4().hex[:8]}.html")
    try:
        net.save_graph(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return html_content


def main():
    df_tx, df_graph, G, gnn_embeddings = load_core_data()
    init_session_state(df_tx, df_graph)
    xgb_model, metrics_data, xai_engine, feature_names = load_models_and_xai()
    ring_topologies = get_ring_topologies(G, df_tx)

    # Sidebar Navigation
    st.sidebar.title("🛡️ Forensic Intelligence")
    st.sidebar.markdown("**Enterprise Financial Fraud & GNN Relationship Analytics**")
    st.sidebar.markdown("---")

    view_mode = st.sidebar.radio(
        "Navigation",
        [
            "🔍 Forensic Account Investigator & XAI",
            "🌐 Enterprise Global Observatory (Louvain)",
            "⚡ Real-Time Streaming & Dynamic Graph",
            "🔄 Advanced Fraud Ring & Shell Tree Hunter",
            "🤖 ML & GNN Model Performance"
        ]
    )

    st.sidebar.markdown("---")

    # Flagged suspects quick select
    mule_accounts = df_graph[df_graph["is_mule_heuristic"] == 1]["Account_ID"].tolist()
    cycle_accounts = df_graph[df_graph["is_in_cycle"] == 1]["Account_ID"].tolist()
    fraud_senders = df_tx[df_tx["Is_Fraud"] == 1]["Sender_Account"].unique().tolist()
    suspect_candidates = sorted(list(set(mule_accounts + cycle_accounts + fraud_senders)))

    # VIEW 1: FORENSIC ACCOUNT INVESTIGATOR & XAI
    if view_mode == "🔍 Forensic Account Investigator & XAI":
        st.subheader("🕵️ Forensic Account Relationship & Explainable AI (XAI) Inspector")

        col_search1, col_search2, col_hops = st.columns([2, 2, 1])
        with col_search1:
            quick_select = st.selectbox(
                "⚡ Quick-Select Flagged Suspect Account:",
                options=["(Custom Search)"] + suspect_candidates[:50],
                index=1 if suspect_candidates else 0
            )

        with col_search2:
            all_accounts = sorted(df_graph["Account_ID"].unique().tolist())
            default_index = 0
            if quick_select != "(Custom Search)":
                default_index = all_accounts.index(quick_select) if quick_select in all_accounts else 0
            selected_account = st.selectbox("Account ID to Inspect:", options=all_accounts, index=default_index)

        with col_hops:
            hops = st.radio("Ego Depth:", [1, 2], horizontal=True, index=0)

        risk_score, risk_factors = calculate_account_risk_score(selected_account, df_graph, df_tx, gnn_embeddings)
        acc_metrics = df_graph[df_graph["Account_ID"] == selected_account].iloc[0].to_dict()

        # Top KPI Metric Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            badge = "CRITICAL RISK" if risk_score >= 70 else ("SUSPICIOUS" if risk_score >= 35 else "BENIGN / LOW")
            delta_color = "inverse" if risk_score >= 70 else ("off" if risk_score >= 35 else "normal")
            st.metric("Fraud Risk Score", f"{risk_score}%", delta=badge, delta_color=delta_color)
        with m2:
            st.metric("Total Inflow", f"${acc_metrics['total_inflow']:,.2f}")
        with m3:
            st.metric("Total Outflow", f"${acc_metrics['total_outflow']:,.2f}")
        with m4:
            st.metric("PageRank Score", f"{acc_metrics['pagerank']:.5f}")
        with m5:
            st.metric("Centrality Ratio", f"In: {int(acc_metrics['in_degree'])} | Out: {int(acc_metrics['out_degree'])}")

        st.markdown("---")

        # Two Columns: Interactive Ego Network vs Risk Factors & Radar
        col_graph, col_profile = st.columns([3, 2])

        with col_graph:
            st.markdown(f"#### 🕸️ Transaction Neighborhood ({hops}-Hop Ego Network)")
            ego_sub = extract_ego_subgraph(G, selected_account, radius=hops)
            local_wash_cycles = [r for r in ring_topologies["wash_cycles"] if any(n in ego_sub.nodes() for n in r["path"])]

            try:
                html_vis = render_pyvis_network(ego_sub, selected_account, df_graph, local_wash_cycles, height="580px")
                components.html(html_vis, height=600, scrolling=False)
            except Exception as e:
                st.warning(f"PyVis rendering notice: {e}")

            st.caption("🟢 Legitimate Node | 🔵 Target Account | 🟠 Wash Ring Member | 🟣 Money Mule Hub | 🔴 Suspicious Path")

        with col_profile:
            st.markdown("#### 🎯 Multi-Vector Risk Radar")
            categories = list(risk_factors.keys())
            values = list(risk_factors.values())
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                fillcolor='rgba(239, 68, 68, 0.35)' if risk_score > 50 else 'rgba(56, 189, 248, 0.35)',
                line=dict(color='#ef4444' if risk_score > 50 else '#38bdf8', width=2),
                name='Risk Profile'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], color="#94a3b8"), bgcolor="#1e293b"),
                paper_bgcolor="#0b0f19",
                font_color="#cbd5e1",
                height=310,
                margin=dict(l=40, r=40, t=15, b=15)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Red Flag Alerts
            st.markdown("##### 📌 Topological Red Flags")
            if acc_metrics.get("is_in_cycle"):
                st.error(f"⚠️ Involved in {int(acc_metrics.get('in_cycle_count', 1))} detected wash trading cycle(s)!")
            if acc_metrics.get("is_mule_heuristic"):
                st.warning("⚠️ Money Mule Hub Signature: Disproportionate fan-in with fast total depletion.")
            if acc_metrics.get("betweenness_centrality", 0) > 0.005:
                st.info("ℹ️ High Betweenness: Account acts as a strategic structural bridge between disconnected sub-networks.")
            if not acc_metrics.get("is_in_cycle") and not acc_metrics.get("is_mule_heuristic") and risk_score < 35:
                st.success("✅ Consistent with standard retail account behavioral norms.")

        # EXPLAINABLE AI (XAI) & AUDIT NARRATIVE PANEL
        st.markdown("---")
        st.markdown("### 🧠 Explainable AI (XAI) & Audit Narrative")

        # Generate Natural Language Audit Narrative
        narrative_text = ""
        if xai_engine:
            narrative_text = xai_engine.generate_audit_narrative(selected_account, risk_score, acc_metrics, risk_factors)
        else:
            narrative_text = f"Account {selected_account} has a composite fraud risk score of {risk_score:.1f}% based on graph topology and transaction velocity."

        col_narrative, col_shap = st.columns([1, 1])

        with col_narrative:
            st.markdown("#### 📜 Court-Admissible Audit Narrative")
            st.markdown(f'<div class="narrative-box">{narrative_text.replace(chr(10), "<br/>")}</div>', unsafe_allow_html=True)

            # Action Buttons: Export PDF and CSV
            st.markdown("<br/>", unsafe_allow_html=True)
            acc_txs = df_tx[(df_tx["Sender_Account"] == selected_account) | (df_tx["Receiver_Account"] == selected_account)].copy()
            acc_txs["Direction"] = acc_txs.apply(lambda r: "OUTBOUND" if r["Sender_Account"] == selected_account else "INBOUND", axis=1)

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("📄 Generate Forensic Dossier (PDF)", key="gen_pdf_btn"):
                    with st.spinner("Compiling court-admissible forensic PDF dossier..."):
                        pdf_path = generate_forensic_pdf(
                            account_id=selected_account,
                            risk_score=risk_score,
                            account_metrics=acc_metrics,
                            risk_factors=risk_factors,
                            audit_narrative=narrative_text,
                            flagged_txs=acc_txs[acc_txs["Is_Fraud"] == 1]
                        )
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        st.download_button(
                            label="📥 Download Forensic Dossier (PDF)",
                            data=pdf_bytes,
                            file_name=f"Forensic_Dossier_{selected_account}.pdf",
                            mime="application/pdf"
                        )
            with btn_col2:
                csv_data = acc_txs.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Ledger (CSV)",
                    data=csv_data,
                    file_name=f"Transactions_{selected_account}.csv",
                    mime="text/csv"
                )

        with col_shap:
            st.markdown("#### ⚖️ Primary Risk Vector Decomposition")
            shap_df = pd.DataFrame(list(risk_factors.items()), columns=["Factor", "Risk Contribution %"]).sort_values("Risk Contribution %", ascending=True)
            fig_bar = px.bar(
                shap_df,
                x="Risk Contribution %",
                y="Factor",
                orientation="h",
                color="Risk Contribution %",
                color_continuous_scale="Reds"
            )
            fig_bar.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#1e293b", font_color="#cbd5e1", height=280, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Transaction Ledger
        st.markdown("---")
        st.markdown(f"#### 📜 Historical Transaction Ledger for `{selected_account}` ({len(acc_txs)} transactions)")
        st.dataframe(
            acc_txs[["Transaction_ID", "Timestamp", "Direction", "Sender_Account", "Receiver_Account", "Amount", "Location", "Device_ID", "Fraud_Type", "Is_Fraud"]],
            use_container_width=True,
            height=260
        )

    # VIEW 2: ENTERPRISE GLOBAL OBSERVATORY (LOUVAIN CLUSTERS)
    elif view_mode == "🌐 Enterprise Global Observatory (Louvain)":
        st.subheader("🌐 Enterprise Global Network Observatory & Louvain Community Clustering")

        total_txs = len(df_tx)
        total_vol = df_tx["Amount"].sum()
        fraud_txs = df_tx[df_tx["Is_Fraud"] == 1]
        fraud_vol = fraud_txs["Amount"].sum()

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("Total Transactions", f"{total_txs:,}")
        with g2:
            st.metric("Total Volume Processed", f"${total_vol:,.2f}")
        with g3:
            st.metric("Fraud Volume Identified", f"${fraud_vol:,.2f}", delta=f"{(fraud_vol/total_vol)*100:.1f}% of total", delta_color="inverse")
        with g4:
            st.metric("Louvain Communities", f"{df_graph['community_id'].nunique():,}")

        st.markdown("---")

        col_comm1, col_comm2 = st.columns([3, 2])

        with col_comm1:
            st.markdown("#### 🧩 2D Louvain Community Cluster Map")
            # Calculate 2D PCA/Scatter projection of network centralities colored by community
            fig_comm = px.scatter(
                df_graph,
                x="in_degree",
                y="pagerank",
                color=df_graph["community_id"].astype(str),
                size="total_volume",
                hover_data=["Account_ID", "is_in_cycle", "is_mule_heuristic", "total_inflow", "total_outflow"],
                labels={"in_degree": "In-Degree Centrality", "pagerank": "PageRank Score", "color": "Louvain Community"},
                title="Account Distribution by Louvain Community Partition"
            )
            fig_comm.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#1e293b", font_color="#cbd5e1", height=500)
            st.plotly_chart(fig_comm, use_container_width=True)

        with col_comm2:
            st.markdown("#### 📊 Community Risk Contamination Index")
            comm_stats = []
            for c_id, group in df_graph.groupby("community_id"):
                c_accounts = set(group["Account_ID"])
                c_txs = df_tx[df_tx["Sender_Account"].isin(c_accounts) | df_tx["Receiver_Account"].isin(c_accounts)]
                f_count = c_txs["Is_Fraud"].sum()
                contamination_pct = (f_count / max(1, len(c_txs))) * 100
                comm_stats.append({
                    "Community ID": f"Cluster #{c_id}",
                    "Member Accounts": len(group),
                    "Fraud Txs": int(f_count),
                    "Contamination %": round(contamination_pct, 2),
                    "Avg Inflow": round(group["total_inflow"].mean(), 2)
                })
            comm_df = pd.DataFrame(comm_stats).sort_values("Contamination %", ascending=False)
            st.dataframe(comm_df, use_container_width=True, height=480)

    # VIEW 3: REAL-TIME STREAMING & DYNAMIC GRAPH
    elif view_mode == "⚡ Real-Time Streaming & Dynamic Graph":
        st.subheader("⚡ Real-Time Transaction Streaming & Dynamic Incremental Graph Engine")
        st.markdown(
            "Simulates high-throughput streaming transactions directly updating node balances, "
            "warm-started PageRank scores, and triggering instant forensic alerts without full graph rebuilds."
        )

        dyn_g: DynamicTransactionGraph = st.session_state["dynamic_graph"]
        sim: LiveTransactionStreamSimulator = st.session_state["stream_sim"]

        # Stream Controls Row
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
        with ctrl1:
            if st.button("▶️ Ingest Next 1 Live Transaction", use_container_width=True):
                events = sim.step_stream(batch_size=1)
                st.session_state["stream_events"].extend(events)
        with ctrl2:
            if st.button("⏩ Ingest Batch (5 Transactions)", use_container_width=True):
                events = sim.step_stream(batch_size=5)
                st.session_state["stream_events"].extend(events)
        with ctrl3:
            if st.button("🚨 Simulate Adversarial Attack Burst (10 Tx)", use_container_width=True):
                events = sim.step_stream(batch_size=10, inject_anomaly_prob=0.8)
                st.session_state["stream_events"].extend(events)
        with ctrl4:
            if st.button("🔄 Reset Stream History", use_container_width=True):
                st.session_state["stream_events"] = []

        # Dynamic Graph Stats
        dyn_stats = dyn_g.get_summary_stats()
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Dynamic Nodes in Memory", f"{dyn_stats['total_nodes']:,}")
        with d2:
            st.metric("Dynamic Directed Edges", f"{dyn_stats['total_edges']:,}")
        with d3:
            st.metric("Dynamic Volume Ingested", f"${dyn_stats['total_volume']:,.2f}")
        with d4:
            st.metric("Active Forensic Alerts", f"{len(dyn_g.recent_alerts):,}", delta="LIVE ALERTS", delta_color="inverse")

        st.markdown("---")

        col_alerts, col_feed = st.columns([1, 1])

        with col_alerts:
            st.markdown("#### 🚨 Real-Time Forensic Alert Ticker")
            if dyn_g.recent_alerts:
                alert_rows = []
                for a in reversed(dyn_g.recent_alerts[-15:]):
                    alert_rows.append({
                        "Severity": a["severity"],
                        "Timestamp": a["timestamp"][-8:],
                        "Flow": f"{a['sender']} ➔ {a['receiver']}",
                        "Amount": f"${a['amount']:,.2f}",
                        "Trigger Reason": "; ".join(a["reasons"])
                    })
                st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, height=400)
            else:
                st.success("✅ No anomalous transaction alerts triggered yet. Click 'Simulate Adversarial Attack Burst' above.")

        with col_feed:
            st.markdown("#### 📡 Streaming Ingestion Feed")
            if st.session_state["stream_events"]:
                feed_rows = []
                for ev in reversed(st.session_state["stream_events"][-15:]):
                    feed_rows.append({
                        "Tx ID": ev["Transaction_ID"],
                        "Sender": ev["Sender_Account"],
                        "Receiver": ev["Receiver_Account"],
                        "Amount": f"${ev['Amount']:,.2f}",
                        "Typology": ev["Fraud_Type"],
                        "Fraud Flag": "YES" if ev["Is_Fraud"] else "No"
                    })
                st.dataframe(pd.DataFrame(feed_rows), use_container_width=True, height=400)
            else:
                st.info("Streaming feed idle. Click one of the ingest buttons above to inject live transactions.")

    # VIEW 4: ADVANCED FRAUD RING & SHELL TREE HUNTER
    elif view_mode == "🔄 Advanced Fraud Ring & Shell Tree Hunter":
        st.subheader("🔄 Advanced Multi-Tier Fraud Ring & Subgraph Discovery")
        st.markdown(
            "Mines high-order topological evasions: multi-hop wash trading cycles, "
            "shell company dispersal trees (Fan-out -> Layering -> Fan-in), and dense bipartite smurfing networks."
        )

        tab_cycles, tab_trees, tab_smurfs = st.tabs([
            f"🔁 Wash Trading Cycles ({len(ring_topologies['wash_cycles'])} found)",
            f"🌳 Shell Dispersal Trees ({len(ring_topologies['shell_dispersal_trees'])} found)",
            f"⚡ Bipartite Smurfing Cross-flows ({len(ring_topologies['bipartite_smurfing'])} found)"
        ])

        with tab_cycles:
            st.markdown("##### Closed Directed Wash Cycles ($A \\to B \\to C \\dots \\to A$)")
            cycle_table = []
            for idx, r in enumerate(ring_topologies["wash_cycles"]):
                cycle_table.append({
                    "Ring ID": f"WASH_RING_{idx+1:02d}",
                    "Hops": r["hop_count"],
                    "Total Volume": f"${r['total_volume']:,.2f}",
                    "Avg Hop Amount": f"${r['avg_hop_amount']:,.2f}",
                    "Severity Score": r["severity_score"],
                    "Cycle Path": r["path_str"]
                })
            st.dataframe(pd.DataFrame(cycle_table), use_container_width=True, height=480)

        with tab_trees:
            st.markdown("##### Multi-Tier Shell Company Structuring Trees ($S \\to \\{B_1 \\dots B_k\\} \\to D$)")
            tree_table = []
            for idx, t in enumerate(ring_topologies["shell_dispersal_trees"]):
                tree_table.append({
                    "Tree ID": f"TREE_{idx+1:02d}",
                    "Master Source": t["source"],
                    "Final Sink": t["sink"],
                    "Shell Buffers": ", ".join(t["shell_buffers"]),
                    "Dispersed Volume": f"${t['total_dispersed_volume']:,.2f}",
                    "Consolidated Volume": f"${t['total_consolidated_volume']:,.2f}",
                    "Severity Score": t["severity_score"]
                })
            st.dataframe(pd.DataFrame(tree_table), use_container_width=True, height=480)

        with tab_smurfs:
            st.markdown("##### Dense Bipartite Smurfing Cross-Flows")
            smurf_table = []
            for idx, s in enumerate(ring_topologies["bipartite_smurfing"]):
                smurf_table.append({
                    "Pattern ID": f"BIPARTITE_{idx+1:02d}",
                    "Sender Accounts": ", ".join(s["senders"]),
                    "Receiver Sinks": ", ".join(s["receivers"]),
                    "Cross-Flow Volume": f"${s['total_volume']:,.2f}",
                    "Severity Score": s["severity_score"]
                })
            st.dataframe(pd.DataFrame(smurf_table), use_container_width=True, height=480)

    # VIEW 5: ML & GNN MODEL PERFORMANCE
    elif view_mode == "🤖 ML & GNN Model Performance":
        st.subheader("🤖 Enterprise Machine Learning & GNN Classifier Evaluation")

        if metrics_data:
            xgb_m = metrics_data.get("xgb_metrics", {})
            rf_m = metrics_data.get("rf_metrics", {})

            col_eval1, col_eval2, col_eval3, col_eval4 = st.columns(4)
            with col_eval1:
                st.metric("Hybrid XGBoost ROC-AUC", f"{xgb_m.get('ROC_AUC', 0):.4f}")
            with col_eval2:
                st.metric("Precision-Recall AUC (PR-AUC)", f"{xgb_m.get('PR_AUC', 0):.4f}")
            with col_eval3:
                st.metric("Precision", f"{xgb_m.get('Precision', 0)*100:.2f}%")
            with col_eval4:
                st.metric("Recall (Sensitivity)", f"{xgb_m.get('Recall', 0)*100:.2f}%")

            st.markdown("---")

            col_fi, col_cm = st.columns([3, 2])
            with col_fi:
                st.markdown("#### 🏆 Top Engineered Graph, GNN & Tabular Features")
                top_feats = metrics_data.get("top_features", {})
                if top_feats:
                    fi_df = pd.DataFrame(list(top_feats.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
                    fig_bar = px.bar(
                        fi_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale="Viridis"
                    )
                    fig_bar.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#1e293b", font_color="#cbd5e1", height=450)
                    st.plotly_chart(fig_bar, use_container_width=True)

            with col_cm:
                st.markdown("#### 🎯 Model Confusion Matrix")
                cm = xgb_m.get("Confusion_Matrix", [[0, 0], [0, 0]])
                cm_df = pd.DataFrame(cm, index=["Actual Legitimate", "Actual Fraud"], columns=["Predicted Legitimate", "Predicted Fraud"])
                fig_heat = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto")
                fig_heat.update_layout(paper_bgcolor="#0b0f19", font_color="#cbd5e1", height=450)
                st.plotly_chart(fig_heat, use_container_width=True)


if __name__ == "__main__":
    main()
