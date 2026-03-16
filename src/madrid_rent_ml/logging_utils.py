import os
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        os.makedirs("artifacts", exist_ok=True)
        fh = logging.FileHandler("artifacts/pipeline.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger