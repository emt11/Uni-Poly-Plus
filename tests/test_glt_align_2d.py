"""Only U1 final CPU correctness; actual four-rank evidence comes from PT logs."""
import copy,json
import pytest
import torch
from torch import nn
from scripts import run_glt_align_2d_unit as unit
from scripts import select_glt_align_2d as selection
from scripts import run_glt_align_2d as controller
from scripts import finetune_mcl_ph as ft
from scripts.pretrain_glt_dual import alignment_weight_at
from src.modules.glt_dual_pretrain import DualPretrainer,alignment_loss,alignment_local_anchor_count,deployment_package,u1_alignment_options
from src.modules.glt_dual import build_dual_glt_model
from src.dataset.glt_dual import build_glt_2d_sample,glt_2d_collate
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample,pretrain_collate
from tests.test_dual_glt import sample


def trainer(align):
 return DualPretrainer('token',fusion_revision='glt_fusion_r1',geometry_head_norm=True,atom_ce_input='fused',opt_mode='GLT_FUSION_R1',
  **unit.GEOMETRY,align_enabled=align,**(dict(align_revision='glt_align_2d_u1',align_init_seed=314161) if align else {}))


def test_public_initialization_and_projection_rng_independence():
 torch.manual_seed(42);old=trainer(False);old_state=old.state_dict();rng=torch.get_rng_state().clone()
 torch.manual_seed(42);new=trainer(True)
 assert torch.equal(rng,torch.get_rng_state())
 extra=set(new.state_dict())-set(old_state)
 assert sum(new.state_dict()[k].numel() for k in extra)==328448
 assert all(k.startswith(('align_proj2.','align_proj3.')) for k in extra)
 for k,v in old_state.items():assert torch.equal(v,new.state_dict()[k])
 common=torch.load(unit.COMMON,map_location='cpu',weights_only=False)['model']
 saved=torch.load(unit.OUT/'initialization/P_ALIGN_step0.pt',map_location='cpu',weights_only=False)
 for k,v in common.items():assert torch.equal(v,saved['model'][k])
 for k in extra:assert torch.equal(new.state_dict()[k],saved['model'][k])
 assert saved['config']['align_init_seed']==314161 and saved['config']['expected_world_size']==4


def test_raw_align_inputs_validity_gradients_and_multi_positive():
 records=[]
 for i,text in enumerate(('*COC*','*CCN*')):
  top,tri,_=sample(text)
  records.append(prepare_pretrain_sample(top,tri,text,seed=42,key='synthetic',position=i,opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident',sigma=0))
 data,labels=pretrain_collate(records);m=trainer(True).eval();captured={};inputs={}
 original=m.encoder.encode
 def encode(*args,**kwargs):
  value=original(*args,**kwargs);captured.update(value);return value
 m.encoder.encode=encode
 handles=[m.align_proj2.register_forward_pre_hook(lambda mod,args:inputs.update(two=args[0])),m.align_proj3.register_forward_pre_hook(lambda mod,args:inputs.update(three=args[0]))]
 result=m(data,labels)
 for h in handles:h.remove()
 assert inputs['two'] is captured['graph_2d'] and inputs['three'] is captured['graph_3d']
 assert result['extra_counts'][0]==2
 result['extra_sums'][0].backward()
 for mod in (m.encoder.o8,m.encoder.glt,m.align_proj2,m.align_proj3):assert sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)>0
 x=torch.nn.functional.normalize(torch.randn(4,128),dim=-1).requires_grad_();y=torch.nn.functional.normalize(torch.randn(4,128),dim=-1).requires_grad_()
 ids=['same','same','other','invalid'];valid=torch.tensor([True,True,True,False])
 loss,count=alignment_loss(x,y,ids,valid,temperature=.1);assert count==3
 score=x[:3]@y[:3].T/.1
 positive=torch.tensor([[True,True,False],[True,True,False],[False,False,True]])
 def direction(s):return (torch.logsumexp(s,dim=1)-(s*positive).sum(1)/positive.sum(1)).sum()
 torch.testing.assert_close(loss,.5*(direction(score)+direction(score.T)))
 loss.backward();assert x.grad[:3].abs().sum()>0 and y.grad[:3].abs().sum()>0
 assert x.grad[3].abs().sum()==0 and y.grad[3].abs().sum()==0
 zero,n=alignment_loss(x,y,ids,torch.zeros(4,dtype=torch.bool));assert zero==0 and n==0
 assert alignment_local_anchor_count(['same']*4,torch.ones(4,dtype=torch.bool))==0
 cfg=unit.read(unit.OUT/'initialization/P_ALIGN.json')
 assert [alignment_weight_at(cfg,s) for s in (1,500,1000,5000)]==pytest.approx([.0001,.05,.1,.1])


def test_2d_builder_collation_isolation_and_no_geometry_forward():
 tops=[sample(t)[0] for t in ('*COC*','*CCN*')]
 items=[build_glt_2d_sample(t,text) for t,text in zip(tops,('*COC*','*CCN*'))]
 before={k:v.clone() for k,v in tops[0].items() if torch.is_tensor(v)}
 data=glt_2d_collate(items)
 forbidden=('bond_distance','line_angle','incidence_index','geometry_valid','md200','trimer_pos','bond_z_a')
 assert all(k not in data for k in forbidden)
 n=items[0].mips_x.size(0);r=items[0].lga_spd.numel()
 torch.testing.assert_close(data.lga_edge_index[:,r:],items[1].lga_edge_index+n)
 path=items[1].lga_path_index;torch.testing.assert_close(data.lga_path_index[r:],torch.where(path>=0,path+n,path))
 assert torch.equal(data.canonical_graph_index[data.lga_edge_index[0]],data.canonical_graph_index[data.lga_edge_index[1]])
 full=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',dropout=0,**unit.GEOMETRY)
 model=unit.Pure2D(full).eval();assert set(dict(model.named_children()))=={'o8','norm2','head'}
 expected=model(data);changed=data.clone();changed.bond_distance=torch.tensor([float('nan')]);changed.geometry_valid=torch.tensor([False,False])
 torch.testing.assert_close(model(changed),expected,rtol=0,atol=0)
 changed=data.clone();changed.mips_x[n:]+=1
 torch.testing.assert_close(model(changed)[0],expected[0],rtol=0,atol=0)
 for k,v in before.items():assert torch.equal(v,tops[0][k])
 ft.periodic_tdl_trainability(model,head_only=True);assert all(not p.requires_grad for p in model.o8.parameters())
 ft.periodic_tdl_trainability(model,head_only=False);model(data).sum().backward()
 assert sum(float(p.grad.abs().sum()) for p in model.o8.parameters() if p.grad is not None)>0


@pytest.mark.parametrize('arm',unit.ARMS)
def test_strict_parent_best_modes_and_shared_heads(arm):
 cfg=unit.read(unit.OUT/'initialization'/f'{arm}.json');package=torch.load(unit.parent(arm),map_location='cpu',weights_only=False)
 model=unit.build_arm(package,arm,0,cfg)
 if arm.endswith('_2D'):
  assert not hasattr(model,'model') and not hasattr(model,'glt')
  assert model.head[0].in_features==512
 else:assert model.model.predictor[0].in_features==1024 and model.model.token_fusion.attention_mode=='global'
 best=torch.load(unit.ft_folder(arm,'eea',0)/'best.pt',map_location='cpu',weights_only=False);unit.strict_restore(model,best['state_dict'])
 with pytest.raises(ValueError):unit.strict_restore(model,{})
 wrong=dict(cfg,input_dim=1024 if arm.endswith('_2D') else 512)
 with pytest.raises(ValueError):unit.build_arm(package,arm,0,wrong)
 for key in ('prediction_path','input_dim','ft_profile','parent_align_revision','parent','architecture'):
  wrong=dict(cfg);wrong.pop(key)
  with pytest.raises(ValueError):unit.validate_config(wrong,arm)
 if arm.startswith('UA_'):
  for key in u1_alignment_options(package):
   broken=dict(package);broken.pop(key)
   with pytest.raises(ValueError):unit.build_arm(broken,arm,0,cfg)
 head=torch.load(unit.ft_folder(arm,'eea',0)/'head_initial.pt',map_location='cpu',weights_only=False)
 counterpart=('UA_' if arm.startswith('U0_') else 'U0_')+arm.split('_',1)[1]
 other=torch.load(unit.ft_folder(counterpart,'eea',0)/'head_initial.pt',map_location='cpu',weights_only=False)
 for k,v in head.items():assert torch.equal(v,other[k])
 assert head['0.weight'].shape==(512,cfg['input_dim'])


def test_uniform_seven_task_real_profile_and_actual_budget():
 # Actual execution evidence answers epoch/profile questions without rerunning training.
 consumption=unit.read(unit.RECOVERY/'consumption.json')
 assert (consumption['pt_starts'],consumption['pt_completed_updates'],consumption['pt_validation_completed'],consumption['pt_validation_presentations'])==(1,5000,3,3072)
 assert (consumption['ft_starts'],consumption['ft_epochs'],consumption['ft_completed_updates'],consumption['formal_completed'])==(140,9800,224000,140)
 assert consumption['ft_head_epochs']==1400 and consumption['ft_joint_epochs']==8400
 for arm in unit.ARMS:
  for task in unit.TASKS:
   for fold in range(5):
    history=unit.read(unit.ft_folder(arm,task,fold)/'progress.json')['history'];assert len(history)==70
    assert sum(h['stage']=='head' for h in history)==10 and sum(h['stage']=='joint' for h in history)==60
    for h in history:
     assert h['ft_profile']=='uniform_u1' and h['encoder_dropout']==0 and h['head_dropout']==.1 and h['grad_clip']==5
     for g in h['optimizer_groups']:
      assert g['weight_decay']==.02
      assert g['lr']==(1e-4 if g['name']=='backbone' else (3e-4 if h['stage']=='head' else 1e-3))
 # Task-specific dictionary entries do not affect the explicit profile.
 small=nn.Module();small.head=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Dropout(.3),nn.Linear(512,1));small.encoder=nn.Sequential(nn.Linear(512,512),nn.Dropout(.4))
 for task in unit.TASKS:
  ft.configure_periodic_tdl_dropout(small,task,profile='uniform_u1');assert small.head[2].p==.1 and small.encoder[1].p==0
  for head_only in (True,False):
   ft.periodic_tdl_trainability(small,head_only=head_only);opt,evidence=ft.periodic_tdl_optimizer(small,task,head_only=head_only,profile='uniform_u1')
   assert all(g['weight_decay']==.02 for g in evidence)
   if not head_only:assert evidence[0]['lr']==1e-4
 with pytest.raises(ValueError):ft.periodic_tdl_optimizer(small,'xc',head_only=False,backbone_lr=2e-4,profile='uniform_u1')


def test_complete_freeze_and_partial_matrix_rejection(tmp_path,monkeypatch):
 out=tmp_path/'out';parents={a:tmp_path/(a+'.pt') for a in unit.ARMS}
 monkeypatch.setattr(selection,'ROOT',tmp_path);monkeypatch.setattr(selection,'OUT',out)
 monkeypatch.setattr(selection,'parent',lambda a:parents[a]);monkeypatch.setattr(unit,'parent',lambda a:parents[a])
 monkeypatch.setattr(selection,'ft_folder',lambda a,t,f:out/'finetune'/a/t/f'fold{f}')
 configs={a:dict(unit.read(unit.OUT/'initialization'/f'{a}.json'),parent=str(parents[a])) for a in unit.ARMS}
 scaler=dict(mean=[1.],scale=[2.],fit='train',transform='standard')
 for a,cfg in configs.items():unit.write(out/'initialization'/f'{a}.json',cfg)
 for task in unit.TASKS:
  unit.write(tmp_path/'data/splits/mips_paper5_outer'/f'{task}.json',dict(folds=[dict(train_indices=[0],test_indices=[3,2]) for _ in range(5)]))
  for a in unit.ARMS:
   for fold in range(5):
    folder=selection.ft_folder(a,task,fold)
    unit.write(folder/'metrics.json',dict(status='PASS',outer_test='NOT_RUN',epochs=70,updates=dict(attempted=70,completed=70)))
    unit.write(folder/'scaler.json',scaler)
    torch.save(dict(config=configs[a],parent=str(parents[a]),task=task,fold=fold,seed=42+fold,scaler_mean=[1.],scaler_scale=[2.],best_epoch=7,
     state_dict={'synthetic':torch.zeros(1)},prediction_path=configs[a]['prediction_path'],input_dim=configs[a]['input_dim'],head_initial=str(selection.head_path(a,task,fold))),folder/'best.pt')
 missing=selection.ft_folder('UA_FUSED','xc',4)/'metrics.json';saved=missing.read_text();missing.unlink()
 with pytest.raises(FileNotFoundError):selection.freeze()
 assert not (out/'evaluation_before_test.json').exists()
 missing.write_text(saved);selection.freeze();frozen=unit.read(out/'evaluation_before_test.json');assert len(frozen['units'])==140
 for u in frozen['units']:
  assert u['test_indices']==[2,3] and u['scaler']==scaler and u['parent']==str(parents[u['arm']])
  assert u['prediction_path']==configs[u['arm']]['prediction_path'] and u['input_dim']==configs[u['arm']]['input_dim']
 with pytest.raises(FileExistsError):selection.freeze()


def test_path_specific_effects_and_macro():
 result=selection.conditional_effects(.70,.71,.74,.72,.76)
 assert result['UA_2D_minus_U0_2D']==pytest.approx(.03)
 assert result['UA_FUSED_minus_U0_FUSED']==pytest.approx(.04)
 assert result['ALIGN_path_difference']==pytest.approx(.01)
 assert selection.macro_r2({t:[.8]*5 for t in unit.TASKS})==pytest.approx(.8)


def test_missing_head_parent_and_existing_head_preservation(tmp_path):
 head=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1))
 path=tmp_path/'initialization/heads_2d/eea/fold0.pt'
 assert unit.save_head_initial(head,path,resume=True)
 original=path.read_bytes();mtime=path.stat().st_mtime_ns
 assert not unit.save_head_initial(head,path,resume=True)
 assert path.read_bytes()==original and path.stat().st_mtime_ns==mtime
 with pytest.raises(FileExistsError):unit.save_head_initial(head,path)
 with torch.no_grad():head[0].weight[0,0]+=1
 with pytest.raises(ValueError):unit.save_head_initial(head,path,resume=True)
 assert path.read_bytes()==original and path.stat().st_mtime_ns==mtime
