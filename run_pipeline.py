"""
Enterprise Pipeline Orchestrator
Executes the upgraded end-to-end Financial Fraud Network & Relationship Analytics pipeline:
1. Module 1: Synthetic Data Generation (with wash cycles, rapid layering, and money mules)
2. Module 2: NetworkX Graph Modeling & Topological Centrality Extraction
3. Module 3: Graph Neural Networks (GNN - GraphSAGE/GCN) 64-dim Node Structural Embeddings
4. Module 4: Advanced Subgraph Pattern & Complex Ring Discovery (Shell trees, bipartite smurfing)
5. Module 5: Hybrid ML Training & SHAP Explainability Engine
"""

import time
import json
from pathlib import Path

from src.utils import ensure_directories, get_logger, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_generator import generate_financial_dataset
from src.graph_engine import run_graph_pipeline
from src.model_trainer import run_training_pipeline
from src.ring_detector import AdvancedRingDetector

logger = get_logger("PipelineOrchestrator")


def main():
    start_time = time.time()
    logger.info("=" * 75)
    logger.info("STARTING ENTERPRISE FINANCIAL FRAUD NETWORK & GNN ANALYTICS PIPELINE")
    logger.info("=" * 75)

    ensure_directories()

    # Step 1: Synthetic Data Generation
    logger.info("\n>>> STEP 1: Synthetic Data Generation & Typology Injection")
    df_tx = generate_financial_dataset(
        num_accounts=1000,
        num_normal_txs=12000,
        num_rings=25,
        num_layering_chains=25,
        num_mules=20,
        seed=42
    )

    # Step 2: Graph Engine & Base Centralities
    logger.info("\n>>> STEP 2: Graph Modeling & Centrality Feature Extraction")
    G, df_graph = run_graph_pipeline()

    # Step 3: Advanced Ring & Subgraph Discovery
    logger.info("\n>>> STEP 3: Advanced Fraud Ring & Shell Dispersal Tree Discovery")
    detector = AdvancedRingDetector(G, df_tx)
    all_rings = detector.find_all_rings()
    logger.info(f"Detected {len(all_rings['wash_cycles'])} wash loops and {len(all_rings['shell_dispersal_trees'])} dispersal trees.")

    # Step 4: GNN Structural Embeddings & Hybrid ML Classifier
    logger.info("\n>>> STEP 4: GNN (GraphSAGE / GCN) Representation Learning & Hybrid ML Training")
    results = run_training_pipeline()

    # Read evaluation summary
    metrics_file = MODELS_DIR / "model_evaluation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        logger.info("\n" + "=" * 75)
        logger.info("ENTERPRISE PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 75)
        logger.info(f"Total Transactions  : {len(df_tx):,}")
        logger.info(f"Total Accounts (Nodes): {len(df_graph):,}")
        logger.info(f"Total Graph Edges   : {G.number_of_edges():,}")
        logger.info(f"XGBoost ROC-AUC     : {metrics_data['xgb_metrics']['ROC_AUC']}")
        logger.info(f"XGBoost PR-AUC      : {metrics_data['xgb_metrics']['PR_AUC']}")
        logger.info(f"XGBoost Precision   : {metrics_data['xgb_metrics']['Precision']}")
        logger.info(f"XGBoost Recall      : {metrics_data['xgb_metrics']['Recall']}")
        logger.info(f"Random Forest ROC-AUC: {metrics_data['rf_metrics']['ROC_AUC']}")
        logger.info("Top 6 Indicative Features (XGBoost):")
        for feat, score in list(metrics_data['top_features'].items())[:6]:
            logger.info(f"  * {feat}: {score:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"\nPipeline successfully completed in {elapsed:.2f} seconds.")
    logger.info("Launch enterprise dashboard via: python start.py  (or run_app.bat)")


if __name__ == "__main__":
    main()
