import sys
from pathlib import Path

# tests/ has no __init__.py, so pytest's default import mode wouldn't
# otherwise put the repo root (where the scripts under test live) on
# sys.path.
sys.path.insert(0, str(Path(__file__).parent))
