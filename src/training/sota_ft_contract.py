"""New A0-A3-only scope; historical scopes are read-only."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/sota_ft_20261005';LOG=ROOT/'logs/sota_ft_20261005'
PLAN='SOTA-FT-20261005-01';REV='r1'
PARENT=ROOT/'results/mcl_ph_r4_20261002/r3_glt_ref_fast_ra_01/deploy_05000.pt'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLITS=ROOT/'data/splits/mips_paper5_outer'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def reference(task,fold):return ROOT/f'results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/glt_ref/{task}/fold{fold}'
def key(stage,task,fold,arm):return f'A/{stage}/{task}/fold{fold}/{arm}/seed{42+fold}'
def recipe(stage,task,fold,arm):
 if stage not in ('smoke','development') or task not in ('eps','nc') or fold not in (0,1) or arm not in ('F0','SP'):raise ValueError('unit outside A0-A3')
 if stage=='smoke' and (task,fold)!=('eps',0):raise ValueError('smoke scope')
 return dict(plan=PLAN,revision=REV,task=task,fold=fold,arm=arm,seed=42+fold,stage=stage,head_epochs=1 if stage=='smoke' else 10,joint_epochs=2 if stage=='smoke' else 60,batch_size=24,precision='fp32',workers=0,drop_last=False,target_transform='standard',scaler_fit='train',alpha=0. if arm=='F0' else .001,head_lr=.0003,backbone_lr=.0001,joint_head_lr=.001,weight_decay=.02,head_dropout=.1,encoder_dropout=0.,clip=5.,head_scheduler=dict(T_max=10,eta_min=.00003),joint_scheduler=dict(T_0=1 if stage=='smoke' else 10,T_mult=1,eta_min=.000001),K=5,ensemble='last min(5,complete joint cycles); raw equal mean',outer_test='NOT_RUN',source=str(DOWN),parent=str(PARENT),head_initial=str(reference(task,fold)/'head_initial.pt'),split_path=str(SPLITS/f'{task}.json'),historical_fp_step0_exact='未核实',smoke_only=stage=='smoke')
def check_scope(scope,stage,task,fold,arm,output):
 if scope.get('plan')=='MCL-PH-OPT-20261006-01':
  k=key(stage,task,fold,arm)
  if stage!='development' or task!='nc' or fold not in (2,3,4) or arm!='SP' or scope.get('status')!='AUTHORIZED_R1':raise ValueError('unit outside authorized R1')
  r=recipe('development','nc',0,'SP')
  r.update(plan=scope['plan'],revision='r2',fold=fold,seed=42+fold,head_initial=str(reference(task,fold)/'head_initial.pt'),K=0,ensemble='disabled; single best only')
  if k not in scope['units'] or scope['units'][k]['recipe']!=r:raise ValueError('R1 recipe mismatch')
  root=ROOT/'results/mcl_ph_opt_20261006/R1'
  if Path(output).resolve()!=root/k:raise ValueError('output outside unique R1 scope')
  return scope['units'][k]
 k=key(stage,task,fold,arm)
 if scope.get('plan')!=PLAN or scope.get('revision')!=REV or scope.get('status')!='FROZEN_A0_A3' or scope.get('allowed_stages')!=['smoke','development']:raise ValueError('scope not frozen or authorized')
 if k not in scope['units'] or scope['units'][k]['recipe']!=recipe(stage,task,fold,arm):raise ValueError('scope recipe mismatch')
 if Path(output).resolve()!=OUT/k:raise ValueError('output outside unique scope')
 return scope['units'][k]
