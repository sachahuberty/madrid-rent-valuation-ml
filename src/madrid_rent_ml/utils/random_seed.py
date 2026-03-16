import numpy as np
import random
from src.madrid_rent_ml.logging_utils import get_logger

logger = get_logger(__name__)

def set_seed(seed: int = 42):
    logger.info(f"Setting random seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)