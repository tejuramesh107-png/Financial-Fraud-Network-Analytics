import os
import logging
from pathlib import Path

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
def ensure_directories():
    """Ensure that all necessary project directories exist."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Logger setup
def get_logger(name: str = "FraudAnalytics") -> logging.Logger:
    """Setup a standardized logger for the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# Visual color palette for fraud detection
COLOR_PALETTE = {
    "legitimate": "#2ecc71",     # Emerald Green
    "suspicious": "#f39c12",     # Amber Orange
    "fraud": "#e74c3c",          # Alizarin Red
    "mule_hub": "#9b59b6",       # Amethyst Purple
    "cycle_ring": "#e67e22",     # Carrot Orange
    "layering": "#d35400",       # Dark Orange
    "selected": "#3498db",       # Peter River Blue
    "background": "#0f172a",     # Slate 900
    "edge_normal": "#94a3b8",    # Slate 400
    "edge_fraud": "#ef4444"      # Red 500
}

