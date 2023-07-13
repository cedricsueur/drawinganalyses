import os
from pathlib import Path

LOCAL_DATA_DIR = Path(os.environ.get("MADE_DATA_DIR"))
assert LOCAL_DATA_DIR.exists()