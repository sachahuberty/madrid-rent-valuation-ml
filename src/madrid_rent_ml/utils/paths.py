import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()

def get_root_path() -> Path:
    return ROOT_DIR