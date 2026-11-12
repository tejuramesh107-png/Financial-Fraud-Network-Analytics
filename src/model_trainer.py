"""
Module 3: Upgraded Machine Learning & GNN Hybrid Fraud Classifier
Combines transaction tabular features, topological centralities, and 64-dimensional
Graph Neural Network (GraphSAGE / GCN) structural embeddings.
Trains XGBoost and Random Forest, computes SHAP explainability values, and evaluates
metrics (ROC-AUC, PR-AUC, Precision, Recall).
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import xgboost as xgb

from src.utils import ensure_directories, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, get_logger
from src.gnn_engine import train_gnn_model, load_gnn_embeddings

logger = get_logger("ModelTrainer")


def create_hybrid_feature_matrix(
    df_tx: pd.DataFrame,
    df_graph: pd.DataFrame,
    gnn_embeddings: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Merges transaction attributes with sender/receiver graph centralities
    and 64-dimensional GNN structural embeddings.
    """
    logger.info("Assembling hybrid transaction + graph + GNN feature matrix...")

    df = df_tx.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # 1. Transaction-level tabular and temporal features
    df["log_amount"] = np.log1p(df["Amount"].values)
    df["tx_hour"] = df["Timestamp"].dt.hour
    df["tx_dayofweek"] = df["Timestamp"].dt.dayofweek
    df["is_weekend"] = (df["tx_dayofweek"] >= 5).astype(int)

    loc_freq = df["Location"].value_counts(normalize=True).to_dict()
    df["location_rarity"] = df["Location"].map(loc_freq).fillna(0.0)

    # 2. Graph Centrality Features mapping
    graph_cols = [
        "pagerank", "betweenness_centrality", "in_degree_centrality",
        "out_degree_centrality", "in_degree", "out_degree", "total_degree",
        "degree_ratio", "community_id", "in_cycle_count", "is_in_cycle",
        "total_inflow", "total_outflow", "total_volume",
        "turnover_balance_ratio", "is_mule_heuristic"
    ]

    sender_graph = df_graph.set_index("Account_ID")[graph_cols].add_prefix("sender_")
    receiver_graph = df_graph.set_index("Account_ID")[graph_cols].add_prefix("receiver_")

    df = df.merge(sender_graph, left_on="Sender_Account", right_index=True, how="left")
    df = df.merge(receiver_graph, left_on="Receiver_Account", right_index=True, how="left")

    fill_defaults = {col: 0.0 for col in df.columns if col.startswith(("sender_", "receiver_"))}
    df = df.fillna(fill_defaults)

    # 3. Relational & Pairwise Interaction Graph Features
    df["same_community"] = (df["sender_community_id"] == df["receiver_community_id"]).astype(int)
    df["both_in_cycle"] = (df["sender_is_in_cycle"] & df["receiver_is_in_cycle"]).astype(int)
    df["either_mule_flag"] = (df["sender_is_mule_heuristic"] | df["receiver_is_mule_heuristic"]).astype(int)
    df["amount_to_sender_outflow_ratio"] = df["Amount"] / (df["sender_total_outflow"] + 1.0)
    df["amount_to_receiver_inflow_ratio"] = df["Amount"] / (df["receiver_total_inflow"] + 1.0)
    df["pagerank_difference"] = np.abs(df["sender_pagerank"] - df["receiver_pagerank"])
    df["betweenness_sum"] = df["sender_betweenness_centrality"] + df["receiver_betweenness_centrality"]

    # 4. GNN Structural Embedding Features (if provided)
    gnn_cols = []
    if gnn_embeddings:
        logger.info(f"Fusing 64-dim GNN node structural embeddings into feature space...")
        # Fallback vector for unseen nodes
        sample_vec = next(iter(gnn_embeddings.values()))
        dim = len(sample_vec)
        zero_vec = np.zeros(dim, dtype=np.float32)

        cos_sims = []
        eucl_dists = []
        dot_prods = []
        src_norms = []
        dst_norms = []

        for _, row in df.iterrows():
            z_s = gnn_embeddings.get(row["Sender_Account"], zero_vec)
            z_r = gnn_embeddings.get(row["Receiver_Account"], zero_vec)

            norm_s = float(np.linalg.norm(z_s))
            norm_r = float(np.linalg.norm(z_r))
            src_norms.append(norm_s)
            dst_norms.append(norm_r)

            dot = float(np.dot(z_s, z_r))
            dot_prods.append(dot)

            denom = (norm_s * norm_r) + 1e-7
            cos_sims.append(dot / denom)
            eucl_dists.append(float(np.linalg.norm(z_s - z_r)))

        df["gnn_cosine_similarity"] = cos_sims
        df["gnn_euclidean_distance"] = eucl_dists
        df["gnn_dot_product"] = dot_prods
        df["gnn_sender_norm"] = src_norms
        df["gnn_receiver_norm"] = dst_norms

        gnn_cols = [
            "gnn_cosine_similarity",
            "gnn_euclidean_distance",
            "gnn_dot_product",
            "gnn_sender_norm",
            "gnn_receiver_norm"
        ]

    # Assemble complete feature column contract
    feature_cols = [
        "Amount", "log_amount", "tx_hour", "tx_dayofweek", "is_weekend", "location_rarity",
        "sender_pagerank", "sender_betweenness_centrality", "sender_in_degree",
        "sender_out_degree", "sender_degree_ratio", "sender_is_in_cycle",
        "sender_in_cycle_count", "sender_turnover_balance_ratio", "sender_is_mule_heuristic",
        "receiver_pagerank", "receiver_betweenness_centrality", "receiver_in_degree",
        "receiver_out_degree", "receiver_degree_ratio", "receiver_is_in_cycle",
        "receiver_in_cycle_count", "receiver_turnover_balance_ratio", "receiver_is_mule_heuristic",
        "same_community", "both_in_cycle", "either_mule_flag",
        "amount_to_sender_outflow_ratio", "amount_to_receiver_inflow_ratio",
        "pagerank_difference", "betweenness_sum"
    ] + gnn_cols

    X = df[feature_cols].copy()
    y = df["Is_Fraud"].astype(int).copy()

    processed_output = PROCESSED_DATA_DIR / "hybrid_feature_matrix.csv"
    df[feature_cols + ["Transaction_ID", "Sender_Account", "Receiver_Account", "Fraud_Type", "Is_Fraud"]].to_csv(
        processed_output, index=False
    )
    logger.info(f"Assembled hybrid feature matrix: {X.shape[0]} rows, {X.shape[1]} features.")
    return X, y, feature_cols


def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    test_size: float = 0.2,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains XGBoost and Random Forest, computes SHAP explainers, and logs performance.
    """
    logger.info("Splitting dataset into train/test sets (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = float(neg_count / max(1, pos_count))

    results = {}

    # 1. XGBoost
    logger.info("Training Hybrid XGBoost Fraud Classifier...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss"
    )
    xgb_clf.fit(X_train, y_train)

    xgb_preds = xgb_clf.predict(X_test)
    xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]

    xgb_roc_auc = float(roc_auc_score(y_test, xgb_probs))
    xgb_pr_auc = float(average_precision_score(y_test, xgb_probs))
    xgb_precision = float(precision_score(y_test, xgb_preds, zero_division=0))
    xgb_recall = float(recall_score(y_test, xgb_preds, zero_division=0))
    xgb_f1 = float(f1_score(y_test, xgb_preds, zero_division=0))

    results["XGBoost"] = {
        "model": xgb_clf,
        "metrics": {
            "ROC_AUC": round(xgb_roc_auc, 4),
            "PR_AUC": round(xgb_pr_auc, 4),
            "Precision": round(xgb_precision, 4),
            "Recall": round(xgb_recall, 4),
            "F1_Score": round(xgb_f1, 4),
            "Confusion_Matrix": confusion_matrix(y_test, xgb_preds).tolist()
        },
        "feature_importances": dict(zip(feature_names, [float(v) for v in xgb_clf.feature_importances_]))
    }

    # 2. Random Forest
    logger.info("Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)

    rf_preds = rf_clf.predict(X_test)
    rf_probs = rf_clf.predict_proba(X_test)[:, 1]

    results["Random_Forest"] = {
        "model": rf_clf,
        "metrics": {
            "ROC_AUC": round(float(roc_auc_score(y_test, rf_probs)), 4),
            "PR_AUC": round(float(average_precision_score(y_test, rf_probs)), 4),
            "Precision": round(float(precision_score(y_test, rf_preds, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, rf_preds, zero_division=0)), 4),
            "F1_Score": round(float(f1_score(y_test, rf_preds, zero_division=0)), 4),
            "Confusion_Matrix": confusion_matrix(y_test, rf_preds).tolist()
        },
        "feature_importances": dict(zip(feature_names, [float(v) for v in rf_clf.feature_importances_]))
    }

    # Export models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_clf, MODELS_DIR / "fraud_detector_xgb.joblib")
    joblib.dump(rf_clf, MODELS_DIR / "fraud_detector_rf.joblib")

    # Initialize SHAP explainer
    try:
        import shap
        logger.info("Fitting and persisting SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(xgb_clf)
        joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
    except Exception as e:
        logger.warning(f"Could not persist SHAP TreeExplainer: {e}")

    sorted_importances = sorted(results["XGBoost"]["feature_importances"].items(), key=lambda x: x[1], reverse=True)

    metadata = {
        "feature_names": feature_names,
        "xgb_metrics": results["XGBoost"]["metrics"],
        "rf_metrics": results["Random_Forest"]["metrics"],
        "top_features": dict(sorted_importances[:18])
    }

    with open(MODELS_DIR / "model_evaluation_metrics.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Model Training Complete! XGBoost ROC-AUC: {xgb_roc_auc:.4f} | PR-AUC: {xgb_pr_auc:.4f}")
    return results


def run_training_pipeline() -> Dict[str, Any]:
    """Runs complete GNN representation learning followed by hybrid classifier training."""
    ensure_directories()
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

    # 1. Train GNN representation model
    try:
        _, gnn_embeddings = train_gnn_model(df_graph, df_tx, epochs=35)
    except Exception as e:
        logger.warning(f"GNN training encountered an error: {e}. Proceeding with topological features.")
        gnn_embeddings = load_gnn_embeddings()

    # 2. Assemble hybrid feature matrix & train classifiers
    X, y, feature_names = create_hybrid_feature_matrix(df_tx, df_graph, gnn_embeddings)
    results = train_and_evaluate_models(X, y, feature_names)
    return results


if __name__ == "__main__":
    run_training_pipeline()
