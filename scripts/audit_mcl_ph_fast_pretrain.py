import sys,json,math
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,str(Path.cwd()))
from src.training.glt_dual_runtime import open_source,IndexedFrozenDualSource,load_sample_index_artifact,scheduled_lr
from src.training.mcl_ph_direct_contract import assert_actual_equal,direct_binding
from scripts.pretrain_mcl_ph import open_trajectory_cache
from src.modules.mcl_ph import deployment_package
from src.dataset.mcl_ph_view import PH_SEMANTICS,ROUTER_RADII,DESCRIPTOR_COLUMNS
name=sys.argv[1];glt=name.startswith('r3_glt_ref');log=Path('logs/mcl_ph_r4_20261002')/name;out=Path('results/mcl_ph_r4_20261002')/name
cfg=json.loads((log/'resolved_config.json').read_text());state=torch.load(out/'resume_05000.pt',map_location='cpu',weights_only=False);package=torch.load(out/'deploy_05000.pt',map_location='cpu',weights_only=False)
assert log.joinpath('exit.code').read_text().strip()=='0'
assert state['step']==package['step']==5000 and state['next_position']==5040000 and state['scheduler']['step']==5000
assert state['identity']['config']==cfg and state['identity']['world_size']==4
assert state['scheduler']['lr']==scheduled_lr(4999,**{k:cfg[k] for k in ('lr','warmup_steps','schedule_total_steps','end_lr')})
assert len(state['rng'])==4
root=state['direct_binding']['source_root'];base,_=open_source(root,root,dual_static_root=root);idx=load_sample_index_artifact(cfg['sample_index_artifact'],'train');source=IndexedFrozenDualSource(base,idx['indices'])
binding=direct_binding(source,common_init=cfg['common_init_artifact'],seed=42,total_positions=5040000,**({} if glt else dict(shared_init='results/mcl_ph_r4_20261001/shared_new_init_v2.pt',statistics=cfg['statistics_artifact'])))
assert_actual_equal(state['direct_binding'],binding);assert state['ordered_keys']==[k.hex() for k,_ in source.samples]
expected={k[len('encoder.'):]:v for k,v in state['model'].items() if k.startswith('encoder.') and (not glt or not k.startswith('encoder.predictor.'))}
assert_actual_equal(package['state_dict'],expected)
def finite(x):
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all()
 elif isinstance(x,np.ndarray) and x.dtype.kind in 'fc':assert np.isfinite(x).all()
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(list,tuple)):
  for v in x:finite(v)
 elif isinstance(x,float):assert math.isfinite(x)
finite(state);finite(package)
if glt:assert package['fusion_mode']=='concat' and cfg['third_task']=='fp'
else:
 cache,block=open_trajectory_cache(None,config=cfg,batch_size=1008,sample_index=idx,source=source)
 for k,v in block.items():assert_actual_equal(state['identity']['trajectory_cache'][k],v,k)
 assert package['ph_semantics']==PH_SEMANTICS and package['ph_definition']==dict(radii=[float(v) for v in ROUTER_RADII],columns=DESCRIPTOR_COLUMNS,max_edge_length=4.5,field='F2',max_dimension=1)
 assert package['source']['resume_checkpoint']=='resume_05000.pt' and Path(package['source']['cohort_root']).resolve()==base.root.resolve()
 assert Path(package['source']['statistics']).resolve()==Path(cfg['statistics_artifact']).resolve();cache.close()
report=dict(conclusion='PASS',step=5000,next_position=5040000,source_sampling_initialization='actual values exact',export='complete encoder tensors exact to checkpoint',model_optimizer_rng='finite, four stored rank states',scheduler='exact truncated schedule',trajectory_statistics_ph='actual binding and PH semantics checked' if not glt else 'GLT original concat+FP',extra_updates=0)
(log/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);source.close()
