#!/usr/bin/env python3
"""PH campaign controller. Scientific jobs are launched only after the recorded STABILITY gate."""
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/ua3d_ph_20261028/r1'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command', choices=['status','prepare','diagnostic','ft','test','cpu'])
    args=parser.parse_args()
    manifest=json.loads((OUT/'execution_manifest.json').read_text())
    if manifest.get('status')=='STOPPED':
        raise SystemExit('PH dispatch is stopped by the recorded ordering incident; continuation must preserve that record')
    if args.command=='status':
        print(json.dumps(dict(status=manifest.get('status'),phase=manifest.get('phase')),ensure_ascii=False))
        return
    raise SystemExit('command is registered but its implementation gate is not open')

if __name__=='__main__':
    main()
