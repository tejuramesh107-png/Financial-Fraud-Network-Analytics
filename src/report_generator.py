"""
Module: Forensic Investigation Report Generator (PDF & CSV)
Generates audit-ready forensic PDF dossiers using fpdf2 for compliance teams,
financial crime investigators, and regulatory reporting.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from fpdf import FPDF

from src.utils import REPORTS_DIR, get_logger

logger = get_logger("ReportGenerator")


class ForensicReportPDF(FPDF):
    """Custom styled PDF for financial crime forensic intelligence."""

    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(15, 23, 42)  # Dark slate
        self.cell(0, 10, "FINANCIAL CRIME FORENSIC INVESTIGATION DOSSIER", ln=True, align="C")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "CONFIDENTIAL // FOR AUTHORIZED COMPLIANCE & LAW ENFORCEMENT AUDIT ONLY", ln=True, align="C")
        self.line(10, 26, 200, 26)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Generated via Financial Fraud Network & Relationship Analytics Platform", align="C")


def generate_forensic_pdf(
    account_id: str,
    risk_score: float,
    account_metrics: Dict[str, Any],
    risk_factors: Dict[str, float],
    audit_narrative: str,
    flagged_txs: pd.DataFrame,
    output_path: Optional[Path] = None
) -> str:
    """
    Assembles a comprehensive, court-admissible forensic PDF dossier for an account.
    """
    logger.info(f"Generating Forensic PDF Dossier for {account_id}...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"Forensic_Dossier_{account_id}_{timestamp_str}.pdf"

    pdf = ForensicReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Metadata Block
    pdf.set_fill_color(241, 245, 249)  # Light slate background
    pdf.rect(10, 30, 190, 24, "F")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.set_xy(12, 32)
    pdf.cell(90, 5, f"TARGET ACCOUNT: {account_id}", ln=False)
    pdf.cell(90, 5, f"INVESTIGATION DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.set_x(12)
    pdf.cell(90, 5, f"DOSSIER ID: INV-{account_id}-{int(datetime.now().timestamp())}", ln=False)
    pdf.cell(90, 5, "ANALYTICS ENGINE: GNN + NetworkX + TreeSHAP", ln=True)
    pdf.set_x(12)
    severity_label = "CRITICAL THREAT" if risk_score >= 70 else ("ELEVATED RISK" if risk_score >= 35 else "LOW / BENIGN")
    pdf.cell(90, 5, f"THREAT CLASSIFICATION: {severity_label}", ln=False)
    pdf.cell(90, 5, f"COMPOSITE FRAUD RISK SCORE: {risk_score:.1f}%", ln=True)

    pdf.ln(12)

    # 2. Executive Assessment & Risk Meter
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "1. EXECUTIVE RISK ASSESSMENT", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 5, audit_narrative)

    pdf.ln(4)

    # 3. Account Topological & Financial Metrics
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "2. GRAPH TOPOLOGY & CASH FLOW ATTRIBUTES", ln=True)

    metrics_table = [
        ("Total Inflow Volume", f"${account_metrics.get('total_inflow', 0.0):,.2f}", "Total Outflow Volume", f"${account_metrics.get('total_outflow', 0.0):,.2f}"),
        ("PageRank Centrality", f"{account_metrics.get('pagerank', 0.0):.6f}", "Betweenness Centrality", f"{account_metrics.get('betweenness_centrality', 0.0):.6f}"),
        ("In-Degree (Senders)", str(int(account_metrics.get('in_degree', 0))), "Out-Degree (Receivers)", str(int(account_metrics.get('out_degree', 0)))),
        ("Louvain Community ID", str(int(account_metrics.get('community_id', -1))), "Mule Hub Signature", "DETECTED" if account_metrics.get('is_mule_heuristic') else "None"),
        ("Cycle Ring Membership", "YES (Wash Loop)" if account_metrics.get('is_in_cycle') else "None", "Cycle Count", str(int(account_metrics.get('in_cycle_count', 0))))
    ]

    col_w = 47.5
    for row in metrics_table:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(col_w, 6, row[0], border=1, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(col_w, 6, row[1], border=1)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(col_w, 6, row[2], border=1, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(col_w, 6, row[3], border=1)
        pdf.ln()

    pdf.ln(6)

    # 4. Multi-Factor Risk Vector Attribution (SHAP / Explanations)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. RISK FACTOR DECOMPOSITION & ATTRIBUTION", ln=True)
    pdf.set_font("Helvetica", "", 9)

    for factor_name, factor_val in risk_factors.items():
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(70, 5, factor_name, ln=False)
        pdf.set_font("Helvetica", "", 9)
        # Visual text bar
        bar_len = int(factor_val / 5)
        bar_str = "[" + ("=" * bar_len) + (" " * (20 - bar_len)) + "]"
        pdf.cell(50, 5, bar_str, ln=False)
        pdf.cell(30, 5, f"{factor_val:.1f} / 100", ln=True)

    pdf.ln(6)

    # 5. Flagged Transaction Ledger
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. FLAGGED TRANSACTION LEDGER (AUDIT TRAIL)", ln=True)

    if flagged_txs is not None and not flagged_txs.empty:
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(28, 6, "Transaction ID", border=1, fill=True)
        pdf.cell(32, 6, "Timestamp", border=1, fill=True)
        pdf.cell(24, 6, "Counterparty", border=1, fill=True)
        pdf.cell(22, 6, "Direction", border=1, fill=True)
        pdf.cell(24, 6, "Amount", border=1, fill=True)
        pdf.cell(35, 6, "Typology", border=1, fill=True)
        pdf.cell(25, 6, "Fraud Flag", border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for _, tx in flagged_txs.head(15).iterrows():
            tx_id = str(tx.get("Transaction_ID", ""))[:14]
            t_stamp = str(tx.get("Timestamp", ""))[:19]
            is_out = (tx.get("Sender_Account") == account_id)
            cparty = str(tx.get("Receiver_Account") if is_out else tx.get("Sender_Account"))
            direction = "OUTBOUND" if is_out else "INBOUND"
            amt = f"${float(tx.get('Amount', 0.0)):,.2f}"
            typ = str(tx.get("Fraud_Type", "None"))[:20]
            flag = "FRAUD" if tx.get("Is_Fraud") else "BENIGN"

            pdf.cell(28, 5, tx_id, border=1)
            pdf.cell(32, 5, t_stamp, border=1)
            pdf.cell(24, 5, cparty, border=1)
            pdf.cell(22, 5, direction, border=1)
            pdf.cell(24, 5, amt, border=1)
            pdf.cell(35, 5, typ, border=1)
            pdf.cell(25, 5, flag, border=1)
            pdf.ln()
    else:
        pdf.cell(0, 6, "No flagged transactions recorded for this account.", ln=True)

    pdf.ln(8)

    # 6. Compliance Sign-Off Box
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 4, "CERTIFICATION: This document constitutes an automated forensic intelligence extraction.", ln=True)
    pdf.cell(0, 4, "Signature: __________________________   Investigator ID: AGY-AML-7701", ln=True)

    pdf.output(str(output_path))
    logger.info(f"Forensic PDF report successfully saved to: {output_path}")
    return str(output_path)

