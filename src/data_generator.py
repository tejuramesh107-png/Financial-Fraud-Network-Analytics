"""
Module 1: Data Generation & Synthetic Fraud Pipeline
Generates synthetic financial transaction datasets with injected typologies:
- Circular Money Flows (wash trading / ring laundering: A -> B -> C -> A)
- Rapid Layering Chains (smurfing / structuring)
- High-Frequency Hub Accounts (Money Mules)
"""

import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path

from src.utils import ensure_directories, RAW_DATA_DIR, get_logger

logger = get_logger("DataGenerator")

CITIES = [
    "New York, USA", "London, UK", "Singapore, SG", "Zurich, CH",
    "Tokyo, JP", "Mumbai, IN", "Toronto, CA", "Sydney, AU",
    "Frankfurt, DE", "Dubai, AE", "Hong Kong, HK", "Paris, FR"
]

def generate_account_registry(num_accounts: int = 1000) -> Dict[str, Dict]:
    """Generates an initial registry of accounts with primary devices and base locations."""
    registry = {}
    for i in range(1, num_accounts + 1):
        acc_id = f"ACC_{i:04d}"
        registry[acc_id] = {
            "primary_device": f"DEV_{random.randint(1000, 9999)}",
            "secondary_device": f"DEV_{random.randint(1000, 9999)}",
            "home_location": random.choice(CITIES),
            "created_at": datetime(2026, 1, 1) + timedelta(days=random.randint(0, 180))
        }
    return registry

def generate_normal_transactions(
    registry: Dict[str, Dict],
    num_transactions: int = 12000,
    start_date: datetime = datetime(2026, 9, 1),
    duration_days: int = 30
) -> List[Dict]:
    """Generates benign baseline financial transactions following empirical log-normal distributions."""
    accounts = list(registry.keys())
    transactions = []
    
    # Establish a preferential attachment structure (power-law merchants/peers)
    popular_receivers = random.sample(accounts, k=max(10, int(len(accounts) * 0.05)))
    
    for _ in range(num_transactions):
        sender = random.choice(accounts)
        
        # 60% chance to pay a popular receiver/merchant, 40% random peer
        if random.random() < 0.6:
            receiver = random.choice(popular_receivers)
        else:
            receiver = random.choice(accounts)
            
        while receiver == sender:
            receiver = random.choice(accounts)
            
        # Log-normal distribution for typical retail/commercial banking amounts
        amount = float(np.round(np.random.lognormal(mean=4.2, sigma=1.1) + 5.0, 2))
        amount = min(amount, 25000.0) # Cap normal outlier
        
        # Timestamp
        tx_time = start_date + timedelta(
            seconds=random.randint(0, duration_days * 86400)
        )
        
        # Location & Device logic: mostly home location and primary device
        if random.random() < 0.85:
            location = registry[sender]["home_location"]
            device = registry[sender]["primary_device"]
        elif random.random() < 0.95:
            location = registry[sender]["home_location"]
            device = registry[sender]["secondary_device"]
        else:
            location = random.choice(CITIES)
            device = f"DEV_{random.randint(1000, 9999)}"
            
        transactions.append({
            "Transaction_ID": f"TX_{uuid.uuid4().hex[:10].upper()}",
            "Sender_Account": sender,
            "Receiver_Account": receiver,
            "Timestamp": tx_time,
            "Amount": amount,
            "Location": location,
            "Device_ID": device,
            "Fraud_Type": "None",
            "Is_Fraud": 0
        })
        
    return transactions

def inject_circular_flows(
    registry: Dict[str, Dict],
    num_rings: int = 25,
    start_date: datetime = datetime(2026, 9, 1),
    duration_days: int = 30
) -> List[Dict]:
    """
    Simulates Circular Money Laundering / Round-Tripping (e.g. A -> B -> C -> A).
    Features tight time intervals, high amounts, and slight step-wise fee subtractions.
    """
    accounts = list(registry.keys())
    fraud_txs = []
    
    for ring_idx in range(num_rings):
        ring_length = random.choice([3, 4, 5])
        ring_accounts = random.sample(accounts, k=ring_length)
        
        # Base ring amount: between $15,000 and $65,000
        current_amount = float(round(random.uniform(15000, 65000), 2))
        
        ring_start_time = start_date + timedelta(
            days=random.randint(2, duration_days - 2),
            hours=random.randint(0, 20),
            minutes=random.randint(0, 50)
        )
        
        # Laundering rings frequently share a proxy device or suspicious single IP location
        ring_device = f"DEV_FRAUD_RING_{ring_idx:02d}"
        ring_location = random.choice(CITIES)
        
        for step in range(ring_length):
            sender = ring_accounts[step]
            receiver = ring_accounts[(step + 1) % ring_length]
            
            # Step time: 10 to 45 minutes after previous transfer
            step_time = ring_start_time + timedelta(minutes=step * random.randint(12, 40))
            
            fraud_txs.append({
                "Transaction_ID": f"TX_CIR_{uuid.uuid4().hex[:8].upper()}",
                "Sender_Account": sender,
                "Receiver_Account": receiver,
                "Timestamp": step_time,
                "Amount": round(current_amount, 2),
                "Location": ring_location if random.random() < 0.7 else random.choice(CITIES),
                "Device_ID": ring_device if random.random() < 0.6 else registry[sender]["primary_device"],
                "Fraud_Type": "Circular_Flow",
                "Is_Fraud": 1
            })
            
            # 1.5% to 3.5% commission/laundering cut shaved at each hop
            current_amount *= (1.0 - random.uniform(0.015, 0.035))
            
    return fraud_txs

def inject_rapid_layering(
    registry: Dict[str, Dict],
    num_chains: int = 25,
    start_date: datetime = datetime(2026, 9, 1),
    duration_days: int = 30
) -> List[Dict]:
    """
    Simulates Rapid Layering & Structuring (Smurfing):
    Master Sender -> Multiple Smurfs -> Final Destination within a narrow window.
    """
    accounts = list(registry.keys())
    fraud_txs = []
    
    for chain_idx in range(num_chains):
        master_sender = random.choice(accounts)
        final_receiver = random.choice(accounts)
        while final_receiver == master_sender:
            final_receiver = random.choice(accounts)
            
        num_smurfs = random.randint(3, 6)
        smurfs = [acc for acc in random.sample(accounts, k=num_smurfs + 2) if acc not in (master_sender, final_receiver)][:num_smurfs]
        
        chain_start_time = start_date + timedelta(
            days=random.randint(1, duration_days - 2),
            hours=random.randint(1, 22)
        )
        
        # Large initial fund being dispersed
        total_fund = random.uniform(25000, 90000)
        split_amounts = np.random.dirichlet(np.ones(num_smurfs)) * total_fund
        
        layering_loc = random.choice(CITIES)
        
        # Hop 1: Master -> Smurfs (fan-out within 1-2 hours)
        for i, smurf in enumerate(smurfs):
            hop1_time = chain_start_time + timedelta(minutes=random.randint(5, 75))
            smurf_amt = float(round(split_amounts[i], 2))
            
            fraud_txs.append({
                "Transaction_ID": f"TX_LAY1_{uuid.uuid4().hex[:8].upper()}",
                "Sender_Account": master_sender,
                "Receiver_Account": smurf,
                "Timestamp": hop1_time,
                "Amount": smurf_amt,
                "Location": layering_loc,
                "Device_ID": registry[master_sender]["primary_device"],
                "Fraud_Type": "Rapid_Layering",
                "Is_Fraud": 1
            })
            
            # Hop 2: Smurfs -> Final Destination (fan-in shortly after, e.g., 30-120 mins)
            hop2_time = hop1_time + timedelta(minutes=random.randint(30, 150))
            consolidated_amt = float(round(smurf_amt * random.uniform(0.96, 0.99), 2))
            
            fraud_txs.append({
                "Transaction_ID": f"TX_LAY2_{uuid.uuid4().hex[:8].upper()}",
                "Sender_Account": smurf,
                "Receiver_Account": final_receiver,
                "Timestamp": hop2_time,
                "Amount": consolidated_amt,
                "Location": layering_loc,
                "Device_ID": f"DEV_LAY_PROXY_{chain_idx:02d}",
                "Fraud_Type": "Rapid_Layering",
                "Is_Fraud": 1
            })
            
    return fraud_txs

def inject_money_mules(
    registry: Dict[str, Dict],
    num_mules: int = 18,
    start_date: datetime = datetime(2026, 9, 1),
    duration_days: int = 30
) -> List[Dict]:
    """
    Simulates High-Frequency Hub Accounts (Money Mules):
    High in-degree bursts (many small victims/feeders) rapidly followed by
    high out-degree consolidation transfers to external accounts.
    """
    accounts = list(registry.keys())
    mule_accounts = random.sample(accounts, k=num_mules)
    fraud_txs = []
    
    for mule_idx, mule in enumerate(mule_accounts):
        mule_window_start = start_date + timedelta(
            days=random.randint(1, duration_days - 4),
            hours=random.randint(0, 18)
        )
        
        num_feeders = random.randint(7, 16)
        feeders = [acc for acc in random.sample(accounts, k=num_feeders + 1) if acc != mule][:num_feeders]
        
        destination = random.choice([acc for acc in accounts if acc != mule and acc not in feeders])
        
        total_inflow = 0.0
        
        # High in-degree burst
        for i, feeder in enumerate(feeders):
            feed_time = mule_window_start + timedelta(
                hours=random.randint(0, 36),
                minutes=random.randint(0, 59)
            )
            # Smurfed / phished amount
            feed_amt = float(round(random.uniform(1500, 7500), 2))
            total_inflow += feed_amt
            
            fraud_txs.append({
                "Transaction_ID": f"TX_MULE_IN_{uuid.uuid4().hex[:8].upper()}",
                "Sender_Account": feeder,
                "Receiver_Account": mule,
                "Timestamp": feed_time,
                "Amount": feed_amt,
                "Location": registry[feeder]["home_location"],
                "Device_ID": registry[feeder]["primary_device"],
                "Fraud_Type": "Money_Mule",
                "Is_Fraud": 1
            })
            
        # Mule flushes out ~92% - 98% of total inflow to destination in 1 or 2 chunks
        drain_time = mule_window_start + timedelta(hours=random.randint(38, 48))
        outflow_amt = float(round(total_inflow * random.uniform(0.92, 0.98), 2))
        
        fraud_txs.append({
            "Transaction_ID": f"TX_MULE_OUT_{uuid.uuid4().hex[:8].upper()}",
            "Sender_Account": mule,
            "Receiver_Account": destination,
            "Timestamp": drain_time,
            "Amount": outflow_amt,
            "Location": random.choice(CITIES),
            "Device_ID": f"DEV_MULE_PROXY_{mule_idx:02d}",
            "Fraud_Type": "Money_Mule",
            "Is_Fraud": 1
        })
        
    return fraud_txs

def generate_financial_dataset(
    num_accounts: int = 1000,
    num_normal_txs: int = 12000,
    num_rings: int = 25,
    num_layering_chains: int = 25,
    num_mules: int = 20,
    seed: int = 42
) -> pd.DataFrame:
    """
    Orchestrates end-to-end synthetic dataset generation combining legitimate
    transactions with multiple adversarial fraud typologies.
    """
    logger.info("Initializing synthetic transaction generation pipeline...")
    random.seed(seed)
    np.random.seed(seed)
    ensure_directories()
    
    registry = generate_account_registry(num_accounts)
    
    # 1. Normal baseline transactions
    logger.info(f"Generating {num_normal_txs} baseline transactions across {num_accounts} accounts...")
    normal_txs = generate_normal_transactions(registry, num_transactions=num_normal_txs)
    
    # 2. Injected fraud typologies
    logger.info("Injecting Circular Money Flow patterns (wash rings)...")
    ring_txs = inject_circular_flows(registry, num_rings=num_rings)
    
    logger.info("Injecting Rapid Layering & Structuring patterns...")
    layering_txs = inject_rapid_layering(registry, num_chains=num_layering_chains)
    
    logger.info("Injecting High-Frequency Money Mule hubs...")
    mule_txs = inject_money_mules(registry, num_mules=num_mules)
    
    all_txs = normal_txs + ring_txs + layering_txs + mule_txs
    df = pd.DataFrame(all_txs)
    
    # Sort chronologically
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(by="Timestamp").reset_index(drop=True)
    
    # Summary metrics
    total_txs = len(df)
    fraud_txs_count = df["Is_Fraud"].sum()
    fraud_pct = (fraud_txs_count / total_txs) * 100
    
    logger.info(f"Generated {total_txs} total transactions ({fraud_txs_count} fraudulent, {fraud_pct:.2f}% fraud rate).")
    logger.info(f"Breakdown by Fraud_Type:\n{df['Fraud_Type'].value_counts().to_string()}")
    
    output_path = RAW_DATA_DIR / "transactions.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved raw transaction dataset to: {output_path}")
    
    return df

if __name__ == "__main__":
    generate_financial_dataset()

