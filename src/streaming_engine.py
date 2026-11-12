"""
Module: Real-Time Transaction Streaming Engine
Simulates live streaming ingestion of financial transactions (both benign retail flows
and adversarial laundering bursts) into the Dynamic Incremental Graph Engine.
"""

import time
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Generator, Optional, Any
import numpy as np
import pandas as pd

from src.incremental_graph import DynamicTransactionGraph
from src.utils import get_logger

logger = get_logger("StreamingEngine")


class LiveTransactionStreamSimulator:
    """
    Simulates a live event stream of banking transactions for real-time fraud monitoring.
    Supports stepping through incoming batches, replaying historical transactions,
    or synthesizing dynamic attack bursts.
    """

    def __init__(self, dynamic_graph: DynamicTransactionGraph, accounts: Optional[List[str]] = None):
        self.graph = dynamic_graph
        self.accounts = accounts or [f"ACC_{i:04d}" for i in range(1, 1001)]
        self.current_sim_time = datetime.now()
        self.stream_history: List[Dict[str, Any]] = []

    def generate_single_live_transaction(self, inject_anomaly_prob: float = 0.12) -> Dict[str, Any]:
        """Generates a realistic live incoming transaction."""
        self.current_sim_time += timedelta(seconds=random.randint(2, 45))
        is_adversarial = (random.random() < inject_anomaly_prob)

        if is_adversarial:
            typology = random.choice(["Wash_Cycle_Step", "Mule_Burst", "Structuring_Smurf", "High_Value_Outlier"])
            if typology == "Wash_Cycle_Step":
                # Create or continue an immediate cycle loop
                sender = random.choice(self.accounts[:100])
                receiver = random.choice(self.accounts[:100])
                amount = float(round(random.uniform(18000, 48000), 2))
                fraud_flag = 1
            elif typology == "Mule_Burst":
                # Rapid deposit to a known hub account
                sender = random.choice(self.accounts)
                receiver = "ACC_0042" # Designated hot mule hub
                amount = float(round(random.uniform(3000, 8500), 2))
                fraud_flag = 1
            elif typology == "Structuring_Smurf":
                sender = random.choice(self.accounts)
                receiver = random.choice(self.accounts)
                amount = float(round(random.uniform(9200, 9950), 2)) # Just under $10,000 threshold
                fraud_flag = 1
            else:
                sender = random.choice(self.accounts)
                receiver = random.choice(self.accounts)
                amount = float(round(random.uniform(65000, 150000), 2))
                fraud_flag = 1
        else:
            typology = "None"
            fraud_flag = 0
            sender = random.choice(self.accounts)
            receiver = random.choice(self.accounts)
            while receiver == sender:
                receiver = random.choice(self.accounts)
            amount = float(round(np.random.lognormal(mean=4.2, sigma=1.0) + 5.0, 2))

        tx = {
            "Transaction_ID": f"LIVE_{uuid.uuid4().hex[:8].upper()}",
            "Timestamp": self.current_sim_time,
            "Sender_Account": sender,
            "Receiver_Account": receiver,
            "Amount": amount,
            "Fraud_Type": typology,
            "Is_Fraud": fraud_flag
        }
        return tx

    def step_stream(self, batch_size: int = 5, inject_anomaly_prob: float = 0.15) -> List[Dict[str, Any]]:
        """
        Executes one live streaming step: generates batch_size transactions,
        ingests them into the dynamic incremental graph, and returns processed events.
        """
        processed_events = []
        for _ in range(batch_size):
            raw_tx = self.generate_single_live_transaction(inject_anomaly_prob=inject_anomaly_prob)
            result = self.graph.add_transaction(
                sender=raw_tx["Sender_Account"],
                receiver=raw_tx["Receiver_Account"],
                amount=raw_tx["Amount"],
                timestamp=raw_tx["Timestamp"],
                is_fraud=raw_tx["Is_Fraud"],
                fraud_type=raw_tx["Fraud_Type"],
                run_analytics=True
            )
            event_record = {**raw_tx, **result}
            self.stream_history.append(event_record)
            processed_events.append(event_record)

        if len(self.stream_history) > 300:
            self.stream_history = self.stream_history[-300:]

        return processed_events

