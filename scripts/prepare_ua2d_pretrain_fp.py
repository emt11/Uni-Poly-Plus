#!/usr/bin/env python3
"""Single authorized training-label cache construction, after baseline freeze."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.ua2d_pretrain_fp import prepare_cache
if __name__=='__main__':prepare_cache()
