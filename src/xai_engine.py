"""
Module: Explainable AI (XAI) & Attribution Engine
Integrates SHAP (TreeExplainer) and transparent multi-factor attribution to explain
exactly why an account or transaction was flagged, generating plain-English compliance narratives.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib

from src.utils import MODELS_DIR, get_logger

logger = get_logger("XAIEngine")


class FraudExplainabilityEngine:
    """
    Computes exact Shapley feature attributions (SHAP) and synthesizes
    plain-English forensic compliance narratives for flagged entities.
    """

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            import shap
            logger.info("Initializing SHAP TreeExplainer on trained model...")
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP TreeExplainer: {e}. Fallback to feature importance weighting.")
            self.explainer = None

    def explain_transaction(self, feature_row: pd.Series) -> Dict[str, Any]:
        """
        Computes SHAP value attribution for a single transaction or account profile vector.
        """
        # Ensure row is formatted as 2D array
        vals = feature_row[self.feature_names].values.reshape(1, -1)

        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(vals)
                # If binary classifier returns 2 classes, pick class 1 (fraud)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    sv = shap_values[1][0]
                elif len(shap_values.shape) == 2:
                    sv = shap_values[0]
                elif len(shap_values.shape) == 3:
                    sv = shap_values[0, :, 1]
                else:
                    sv = shap_values.flatten()
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}. Using gradient/weight fallback.")
                sv = vals.flatten() * getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)))
        else:
            fi = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)))
            sv = vals.flatten() * fi

        # Pair features with SHAP scores
        attribution = dict(zip(self.feature_names, [float(x) for x in sv]))
        sorted_attributions = sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)

        top_positive = [(k, v) for k, v in sorted_attributions if v > 0][:5]
        top_negative = [(k, v) for k, v in sorted_attributions if v < 0][:3]

        return {
            "all_attributions": attribution,
            "top_risk_drivers": top_positive,
            "mitigating_factors": top_negative
        }

    def generate_audit_narrative(
        self,
        account_id: str,
        risk_score: float,
        account_metrics: Dict[str, Any],
        risk_factors: Dict[str, float]
    ) -> str:
        """
        Generates an institutional audit narrative for compliance and regulatory reporting.
        """
        threat_level = "CRITICAL" if risk_score >= 70 else ("ELEVATED" if risk_score >= 35 else "BENIGN")
        
        narrative_lines = [
            f"Account {account_id} has been designated as a {threat_level} THREAT with an estimated fraud probability of {risk_score:.1f}%."
        ]

        driver_points = []

        # 1. Direct fraud history
        if risk_factors.get("Direct Fraud History", 0) > 30:
            driver_points.append(
                f"Historical transaction contamination with {risk_factors['Direct Fraud History']:.1f}% of associated counterparties labeled as fraudulent"
            )

        # 2. Cycle ring
        if account_metrics.get("is_in_cycle", 0) or risk_factors.get("Cycle / Ring Risk", 0) > 20:
            driver_points.append(
                f"Topological participation in {int(account_metrics.get('in_cycle_count', 1))} closed circular wash-trading loops (A -> B -> C -> A) designed to obscure beneficiary identity"
            )

        # 3. Mule signature
        if account_metrics.get("is_mule_heuristic", 0) or risk_factors.get("Mule Hub Signature", 0) > 40:
            driver_points.append(
                f"Rapid transit hub behavior: high in-degree fan-in ({int(account_metrics.get('in_degree', 0))} senders) rapidly depleted to external sinks (turnover ratio: {account_metrics.get('turnover_balance_ratio', 0.0):.2f})"
            )

        # 4. Betweenness centrality
        if account_metrics.get("betweenness_centrality", 0.0) > 0.005:
            driver_points.append(
                f"Abnormally high betweenness centrality ({account_metrics['betweenness_centrality']:.5f}), establishing the account as a critical strategic conduit between isolated financial clusters"
            )

        # 5. Volume scale
        tot_vol = account_metrics.get("total_inflow", 0.0) + account_metrics.get("total_outflow", 0.0)
        if tot_vol > 50000:
            driver_points.append(f"Aggregate financial throughput exceeding ${tot_vol:,.2f}")

        if driver_points:
            narrative_lines.append("Primary topological and behavioral risk drivers include:")
            for i, p in enumerate(driver_points[:4], 1):
                narrative_lines.append(f"  {i}. {p}.")
        else:
            narrative_lines.append("Transaction patterns, counterparty velocity, and graph centralities align with standard commercial baseline behavior.")

        return "\n".join(narrative_lines)

