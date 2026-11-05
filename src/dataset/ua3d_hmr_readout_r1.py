"""Role-gated cache reader. No geometry, BASE encoder or fitted-statistics path."""
from pathlib import Path
import torch
from scripts.bootstrap_ua3d_hmr_readout_r1 import P2,P3,OUT,read
from src.dataset.ua3d_hmr_r2 import load_surface as existing_surface

def require_frozen(root=OUT):
    p=Path(root)/'selection/final_frozen.json'
    if not p.exists() or not read(p)['frozen']:raise ValueError('TEST access before READOUT freeze')
def cache_path(task,fold,formal,root=OUT):
    if formal:require_frozen(root)
    return (P3 if formal else P2)/'feature_cache'/('test' if formal else 'trainval')/task/f'fold{fold}.pt'
def cache(task,fold,formal=False):return torch.load(cache_path(task,fold,formal),map_location='cpu',weights_only=False)
def stats(task,fold):return read(P2/'scalers'/task/f'fold{fold}.json')
def surface(task):return existing_surface(P2,task)
def old_output(arm,task,fold):
    require_frozen()
    if arm not in ('C_ANCHOR','C_FIXED'):raise ValueError('old output role')
    return P3/'formal'/('C_ANCHOR' if arm=='C_ANCHOR' else 'S_FIXED')/task/f'fold{fold}'/'test_predictions.npz'
def canonical(arm,task,fold,root=OUT):
    entries=read(Path(root)/'registry/canonical.json');e=entries['entries'][f'{arm}/{task}/{fold}']
    kind='base_output_alias' if arm=='C_ANCHOR' else 'fixed_r3_science_payload' if arm=='C_FIXED' else 'readout_new'
    if e['source_kind']!=kind or any(e[k]!=v for k,v in dict(arm=arm,task=task,fold=fold,seed=42+fold).items()):raise ValueError('canonical role identity')
    return e

def fixed_payload(task,fold,name='best'):
    e=canonical('C_FIXED',task,fold);raw=torch.load(Path(e['folder'])/f'{name}.pt',map_location='cpu',weights_only=False)
    if raw.get('schema')!='ua3d-hmr-execution-r3' or raw.get('source_kind')!='r3_new_ft' or raw.get('science_revision')!='r2' or raw.get('execution_revision')!='r3':raise ValueError('C_FIXED execution schema')
    p=raw['science_payload']
    from scripts.bootstrap_ua3d_hmr_readout_r1 import ROOT
    if p['schema']!='ua3d-hmr-residual-r2' or any(p[k]!=v for k,v in dict(arm='S_FIXED',task=task,fold=fold,seed=42+fold).items()) or p['config']!=read(ROOT/'configs/ua3d_hmr_r2_gpu2.json'):raise ValueError('C_FIXED scientific schema/config')
    return p
