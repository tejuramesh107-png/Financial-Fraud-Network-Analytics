"""
Terminal Summary & Demo Inspector - Enterprise Edition
Displays key findings from the upgraded Financial Fraud Network & GNN Analytics platform.
"""

import json
from pathlib import Path
import pandas as pd
import torch

from src.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

def print_summary():
    print("=" * 80)
    print(" ENTERPRISE FINANCIAL FRAUD NETWORK & GNN ANALYTICS - SYSTEM SUMMARY")
    print("=" * 80)

    tx_path = RAW_DATA_DIR / "transactions.csv"
    graph_path = PROCESSED_DATA_DIR / "account_graph_features.csv"
    gnn_path = PROCESSED_DATA_DIR / "gnn_node_embeddings.pt"
    metrics_path = MODELS_DIR / "model_evaluation_metrics.json"

    if not tx_path.exists():
        print("Dataset not yet generated. Please run `python run_pipeline.py` first.")
        return

    df_tx = pd.read_csv(tx_path)
    df_graph = pd.read_csv(graph_path)

    print(f"\n[1] DATASET & NETWORK TOPOLOGY:")
    print(f"  * Total Transactions      : {len(df_tx):,}")
    print(f"  * Total Monitored Accounts: {len(df_graph):,}")
    print(f"  * Total Transaction Volume: ${df_tx['Amount'].sum():,.2f}")
    print(f"  * Fraudulent Transactions : {df_tx['Is_Fraud'].sum():,} ({df_tx['Is_Fraud'].mean()*100:.2f}%)")
    print(f"  * Typology Distribution:")
    for typ, count in df_tx['Fraud_Type'].value_counts().items():
        print(f"      - {typ:<18}: {count:,} transactions")

    print(f"\n[2] GRAPH NEURAL NETWORKS (GNN) REPRESENTATION:")
    if gnn_path.exists():
        gnn_embs = torch.load(gnn_path, map_location="cpu", weights_only=False)
        sample_key = next(iter(gnn_embs.keys()))
        print(f"  * Architecture           : 2-Layer Inductive GraphSAGE / GCN")
        print(f"  * Node Embeddings Tensor : {len(gnn_embs):,} accounts mapped to {len(gnn_embs[sample_key])}-dimensional vectors")
        print(f"  * Saved Weights & Vector : data/processed/gnn_node_embeddings.pt | models/gnn_fraud_model.pt")
    else:
        print("  * GNN embeddings pending pipeline execution.")

    print(f"\n[3] HIGH-FREQUENCY MONEY MULE HUBS:")
    mules = df_graph[df_graph['is_mule_heuristic'] == 1].sort_values(by="total_inflow", ascending=False).head(5)
    for _, row in mules.iterrows():
        print(f"  * Account {row['Account_ID']}: Inflow=${row['total_inflow']:,.2f} | Outflow=${row['total_outflow']:,.2f} | In-Degree={int(row['in_degree'])} | Out-Degree={int(row['out_degree'])}")

    print(f"\n[4] DETECTED ADVANCED LAUNDERING SUBGRAPHS & WASH RINGS:")
    cycle_accs = df_graph[df_graph['is_in_cycle'] == 1]
    print(f"  * Accounts participating in Wash Cycles: {len(cycle_accs)}")
    print(f"  * Sample wash ring nodes: {', '.join(cycle_accs['Account_ID'].head(8).tolist())}")

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        xgb_m = metrics.get("xgb_metrics", {})
        print(f"\n[5] HYBRID CLASSIFIER EVALUATION (Tabular + Network Topology + GNN):")
        print(f"  * XGBoost ROC-AUC  : {xgb_m.get('ROC_AUC')}")
        print(f"  * XGBoost PR-AUC   : {xgb_m.get('PR_AUC')}")
        print(f"  * XGBoost Precision: {xgb_m.get('Precision') * 100:.2f}%")
        print(f"  * XGBoost Recall   : {xgb_m.get('Recall') * 100:.2f}%")
        print(f"  * XGBoost F1-Score : {xgb_m.get('F1_Score')}")
        print(f"  * Confusion Matrix (Test Set):")
        cm = xgb_m.get("Confusion_Matrix", [[0, 0], [0, 0]])
        print(f"      - True Negatives (Legitimate identified) : {cm[0][0]:,}")
        print(f"      - False Positives (Legitimate flagged)   : {cm[0][1]:,}")
        print(f"      - False Negatives (Fraud missed)        : {cm[1][0]:,}")
        print(f"      - True Positives (Fraud caught)         : {cm[1][1]:,}")
        print(f"  * Explainability Engine: SHAP TreeExplainer persisted to models/shap_explainer.joblib")

    print("\n" + "=" * 80)
    print(" TO LAUNCH THE ENTERPRISE DASHBOARD:")
    print(" Run command:  python start.py   (or double-click run_app.bat)")
    print(" URL:          http://localhost:8501")
    print("=" * 80)

if __name__ == "__main__":
    print_summary()
