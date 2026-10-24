"""One campaign identity shared by controller, workers, PT, freeze and reports."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def campaign_paths(campaign,cycle_root=None,log_root=None):
    if campaign not in ('r1','r1_retest1'): raise ValueError('unsupported UA2D campaign')
    expected_out=ROOT/'results/ua2d_sota_20261018'/campaign
    expected_log=ROOT/'logs/ua2d_sota_20261018'/campaign
    out=Path(cycle_root).absolute() if cycle_root else expected_out
    log=Path(log_root).absolute() if log_root else expected_log
    if out!=expected_out or log!=expected_log: raise ValueError('campaign/output/log identity mismatch')
    if any(p.is_symlink() for root in (out,log) for p in (root,*root.parents)):
        raise ValueError('campaign roots may not use symbolic links')
    return out,log

def configure(campaign,cycle_root=None,log_root=None):
    global CAMPAIGN,OUT,LOG
    out,log=campaign_paths(campaign,cycle_root,log_root)
    CAMPAIGN,OUT,LOG=campaign,out,log
    os.environ.update(UA2D_SOTA_CAMPAIGN=campaign,UA2D_SOTA_CYCLE_ROOT=str(out),UA2D_SOTA_LOG_ROOT=str(log))
    return out,log

CAMPAIGN=os.environ.get('UA2D_SOTA_CAMPAIGN','r1')
OUT,LOG=campaign_paths(CAMPAIGN,os.environ.get('UA2D_SOTA_CYCLE_ROOT'),os.environ.get('UA2D_SOTA_LOG_ROOT'))
