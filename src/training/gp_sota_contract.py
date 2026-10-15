"""Explicit fixed-scope GP experiment values and per-unit budgets."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/gp_sota_20261004';LOG=ROOT/'logs/gp_sota_20261004'
PLAN='GP-SOTA-20261004-01';REV='r2'
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLITS=ROOT/'data/splits/mips_paper5_outer'
INDEX=ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json'
STAT=ROOT/'data/processed/mcl_ph_r4_20261001/statistics_train_full.npz'
CACHE=ROOT/'data/processed/mcl_ph_r4_20261001/trajectory_train_5040000'
COMMON=OUT/'initialization/common_step0_205_v2.pt'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
def read(p):return json.loads(Path(p).read_text())
def epochs(task):return 60 if task=='egc' else 70

def pretrain_recipe(route):
    if route not in ('GP','GU','GP_A'):raise ValueError('GP route scope differs')
    return dict(plan=PLAN,revision=REV,route=route,seed=42,microbatch=84,global_batch=1008,world=4,accumulation=3,
       max_optimizer_steps=5000,lr=2e-4,weight_decay=0.,amp_dtype='bf16',warmup_steps=2000,schedule_total_steps=20000,end_lr=1e-9,
       atom_mask_ratio=.3,noise_sigma=.03,dropout=.1,third_task='none',loss_weights=[1.,1.,0.],balance_weight=0. if route=='GU' else .001,
       alignment_weight=.1 if route=='GP_A' else 0.,align_temperature=.1,router_dense_updates=500,residual_scale=.1,
       cutoff=[2.,3.,4.],common_step0=str(COMMON),pretraining_objectives=['atom','geometry']+(['balance'] if route!='GU' else [])+(['align'] if route=='GP_A' else []))

def ft_recipe(task,fold,route,recipe):
    if task not in TASKS or fold not in range(5) or route not in ('GP','GU','GP_A') or recipe not in ('F0','F1'):raise ValueError('GP FT scope differs')
    if recipe=='F1' and task not in ('ei','eps','nc','xc'):raise ValueError('F1 task scope differs')
    lr=2e-4 if task=='xc' else 1e-4
    return dict(plan=PLAN,revision=REV,task=task,fold=fold,route=route,recipe=recipe,seed=42+fold,transform='standard',scaler_fit='train',precision='fp32',batch_size=24,
       head_epochs=10,joint_epochs=epochs(task)-10,head_initial_lr=3e-4,head_joint_lr=1e-3,backbone_joint_lr=lr,public_joint_lr=lr*(.3 if recipe=='F1' else 1.),
       weight_decay=0. if task=='egc' else 1e-4 if task=='xc' else .02,head_dropout=0. if task=='egc' else .3 if task=='xc' else .1,
       head_scheduler=dict(T_max=10,eta_min=3e-5),joint_scheduler=dict(T_0=10,T_mult=1,eta_min=1e-6),source_root=str(DOWN),split=read(SPLITS/f'{task}.json'))

def validate_ft_scope(stage,task,fold,route,recipe,scope):
    ft_recipe(task,fold,route,recipe)
    key=f'{recipe}/{route}/{task}/fold{fold}/seed{42+fold}'
    if stage in ('development','formal') and scope.get('revision')=='r3':
        if scope.get('plan')!=PLAN or scope.get('continuation')!='NC_R3' or scope.get('configuration_revision')!=REV:
            raise ValueError('r3 requires the explicit Nc continuation and unchanged r2 recipe')
        if stage=='development':
            expected={
                'R3_Nc_GU':{f'F0/GU/nc/fold{f}/seed{42+f}' for f in (0,1)},
                'R3_Nc_GP':{f'F0/GP/nc/fold{f}/seed{42+f}' for f in (2,3,4)},
            }.get(scope.get('name'),set())
        else:
            if scope.get('status')!='FROZEN':raise ValueError('r3 formal selection is not frozen')
            expected={f'F0/GP/nc/fold{f}/seed{42+f}' for f in range(5)}
        allowed=scope.get('allowed_units',[])
        if not expected or set(allowed)!=expected or len(allowed)!=len(expected) or key not in expected:
            raise ValueError('unit outside bounded r3 Nc scope')
        return key
    if stage=='smoke':
        if (task,fold,route,recipe)!=('xc',0,'GP','F0'):raise ValueError('only authorized FT smoke')
    elif stage=='development':
        if scope.get('plan')!=PLAN or scope.get('revision')!=REV or key not in scope.get('allowed_units',[]):raise ValueError('unit absent from GP scope')
    elif stage=='formal':
        if route=='GU' or scope.get('plan')!=PLAN or scope.get('revision')!=REV or scope.get('status')!='FROZEN' or key not in scope.get('allowed_units',[]):raise ValueError('formal test requires matching frozen contract')
    else:raise ValueError('unknown GP FT stage')
    return key

def validate_r3_output(stage,task,fold,route,recipe,scope,output):
    """One immutable output per authorized r3 unit; alternate retry folders fail."""
    if scope.get('revision')!='r3':return
    validate_ft_scope(stage,task,fold,route,recipe,scope)
    if stage=='development':expected=OUT/'development'/scope['name']/route/task/f'fold{fold}'
    else:expected=OUT/'formal_r3'/route/recipe/task/f'fold{fold}'
    if Path(output).resolve()!=expected.resolve():raise ValueError('r3 output differs from its single authorized unit folder')
