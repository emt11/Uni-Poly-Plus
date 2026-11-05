"""Final CPU checks for the PH cycle. This module is not a license to run early."""
import ast
from pathlib import Path

def test_sources_parse():
    root=Path(__file__).resolve().parents[1]
    for name in ('src/dataset/ua3d_ph.py','src/modules/ua3d_ph.py','src/training/ua3d_ph.py','scripts/run_ua3d_ph.py','scripts/prepare_ua3d_ph.py'):
        ast.parse((root/name).read_text())
