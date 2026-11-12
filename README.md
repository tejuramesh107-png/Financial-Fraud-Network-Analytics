# Enterprise Financial Fraud Network & Transaction Relationship Analytics 🛡️

An enterprise-grade forensic financial crime intelligence platform developed for Anti-Money Laundering (AML) compliance, fraud ring detection, and transaction relationship graph analytics. The system integrates **PyTorch Graph Neural Networks (GraphSAGE / GCN)**, **NetworkX dynamic graph topologies**, **gradient boosted decision trees (XGBoost / Random Forest)**, **TreeSHAP explainable AI**, and an **interactive Streamlit + PyVis investigation studio** with automated **PDF Forensic Dossier export**.

---

## 🌟 Advanced Enterprise Capabilities

1. **Graph Neural Networks (GNN) for Fraud Classification (`src/gnn_engine.py`)**:
   - Implements a 2-layer inductive **GraphSAGE / GCN** in PyTorch.
   - Learns dense **64-dimensional structural node embeddings** directly from graph topology.
   - Combines GNN embeddings with tabular and topological features into an enhanced hybrid tensor, boosting precision to **98.20%** and PR-AUC to **0.9887**.

2. **Real-Time Streaming & Dynamic Incremental Graph (`src/incremental_graph.py` & `src/streaming_engine.py`)**:
   - In-memory dynamic transaction graph updating node balances, transaction counts, and directional weights in real time.
   - **Warm-Started Power Iteration PageRank**: Calculates incremental updates in $< 3\text{ ms}$ per batch without full graph rebuilds.
   - **Dynamic Louvain Community Tracking**: Reassigns nodes across modularity clusters dynamically.
   - **Live Streaming Ticker & Attack Burst Generator**: Allows simulated streaming of live transactions with real-time heuristic alert dispatching.

3. **Advanced Fraud Ring & Subgraph Pattern Discovery (`src/ring_detector.py`)**:
   - **Shell Company Dispersal Trees**: Detects structuring schemes where a master source account fans out funds to intermediate buffer accounts that rapidly funnel into offshore or destination collector sinks ($S \to \{B_1 \dots B_k\} \to D$).
   - **Multi-Hop Bipartite Smurfing**: Mines dense bipartite cross-flows between sender rings and receiver sinks.
   - **Closed Wash Cycles**: Elementary cycle enumeration for circular wash trading rings ($A \to B \to C \to A$) up to depth 6 with volume and severity scores.

4. **Explainable AI (XAI) Suite (`src/xai_engine.py`)**:
   - Integrates **TreeSHAP** to compute exact Shapley value attributions for every feature.
   - Categorizes factors into positive risk drivers and mitigating factors.
   - Generates plain-English **court-admissible forensic audit narratives** for compliance officers.

5. **Enterprise Forensic Studio & PDF Dossier Exporter (`app.py` & `src/report_generator.py`)**:
   - **Account Investigator**: Live fraud risk score, 5-vector radar chart, interactive PyVis/Plotly ego-network, and SHAP attribution decomposition.
   - **1-Click Forensic Dossier (PDF)**: Generates institutional PDF reports complete with metadata, executive assessment, risk breakdown, and flagged transaction ledgers.
   - **Enterprise Global Network Observatory**: 2D Louvain cluster layout with community risk contamination scores.
   - **Live Streaming Simulator Tab**: Interactive live stream playback controls, dynamic graph statistics, and alert ticker.

---

## 📁 Upgraded Directory Structure

```
├── data/
│   ├── raw/
│   │   └── transactions.csv              # Synthetic transaction dataset (12.5k+ txs)
│   └── processed/
│       ├── account_graph_features.csv    # Traditional network centralities
│       ├── gnn_node_embeddings.pt        # 64-dim learned GNN embeddings
│       └── hybrid_feature_matrix.csv     # Combined 36-feature hybrid matrix
├── models/
│   ├── fraud_detector_xgb.joblib         # XGBoost classifier
│   ├── fraud_detector_rf.joblib          # Random Forest classifier
│   ├── gnn_fraud_model.pt               # PyTorch GNN model weights
│   ├── shap_explainer.joblib             # Serialized SHAP TreeExplainer
│   └── model_evaluation_metrics.json     # Performance benchmarks
├── reports/                              # Generated forensic PDF dossiers
├── src/
│   ├── __init__.py
│   ├── utils.py                          # Paths, styling & logger
│   ├── data_generator.py                 # Module 1: Synthetic data & fraud injector
│   ├── graph_engine.py                   # Module 2: NetworkX graph modeling & analytics
│   ├── gnn_engine.py                     # [NEW] PyTorch GraphSAGE/GCN GNN engine
│   ├── incremental_graph.py              # [NEW] Real-time Dynamic Graph & PageRank
│   ├── streaming_engine.py               # [NEW] Live Transaction Streaming Simulator
│   ├── ring_detector.py                  # [NEW] Advanced Subgraph & Shell Ring Hunter
│   ├── xai_engine.py                     # [NEW] SHAP Attribution & Audit Narratives
│   ├── report_generator.py               # [NEW] Professional PDF Dossier Exporter
│   └── model_trainer.py                  # Upgraded Hybrid GNN+ML Training Pipeline
├── app.py                                # Streamlit Enterprise Forensic Dashboard
├── start.py                              # Python launcher
├── demo_summary.py                       # CLI inspector
├── run_app.bat                           # 1-Click launcher
├── run_pipeline.py                       # End-to-end pipeline CLI orchestrator
└── requirements.txt                      # Dependencies (including torch, shap, fpdf2)
```

---

## 🚀 How to Run the Platform

### 1. Execute the End-to-End Enterprise Pipeline
To regenerate the dataset, train the PyTorch GNN, compute SHAP explainers, and extract ring topologies:
```bash
python run_pipeline.py
```

### 2. Launch the Enterprise Forensic Dashboard
To start the interactive web application, choose either:
```bash
python start.py
```
*or*
```bash
python -m streamlit run app.py
```
*or double-click:*
👉 `run_app.bat`

Open `http://localhost:8501` in your browser.

### 3. Print Instant CLI Summary
To print a full terminal summary of all detected mules, rings, GNN metrics, and model evaluations:
```bash
python demo_summary.py
```
