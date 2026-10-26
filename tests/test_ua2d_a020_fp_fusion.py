"""Single final synthetic CPU suite for the authorized A020 fusion cycle."""
import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.modules import ua2d_a020_fp_fusion as m
from src.modules.ua2d_p10k_nofp_align import arm_config,attention_record
from src.training import ua2d_a020_fp_fusion as t
from src.dataset import ua2d_a020_fp_fusion as d
from scripts import run_ua2d_a020_fp_fusion as c

torch.set_num_threads(2)


def test_a020_parent_identity_and_no_fp_semantics():
    parent=dict(schema='ua2d-p10k-nofp-align-pretrain-r1',step=10000,plan='UA2D-P10K-NOFP-ALIGN-20261024-01',
        revision='r1.1-GPU23',arm='D_A020',config=asdict(arm_config('D_A020')),eligible_rows=910756,
        execution=dict(plan='UA2D-P10K-NOFP-ALIGN-20261024-01',world_size=1,microbatch=168,accumulation=6,global_batch=1008),
        attention=attention_record(arm_config('D_A020')),stream_positions=dict(two_d_position=60000,three_d_position=60000,
        two_d_base=86400011,three_d_base=86400013),state_dict={'o8.synthetic':torch.ones(1)})
    assert m.validate_parent(parent) is parent and parent['config']['align_lambda']==.2
    for key,value in (('arm','D_A010'),('step',5000),('schema','wrong')):
        bad=copy.deepcopy(parent);bad[key]=value
        with pytest.raises(ValueError):m.validate_parent(bad)
    bad=copy.deepcopy(parent);bad['config']['fp_enabled']=True
    with pytest.raises(ValueError):m.validate_parent(bad)
    bad=copy.deepcopy(parent);bad['state_dict']['fp_head.weight']=torch.zeros(1)
    with pytest.raises(ValueError):m.validate_parent(bad)
    bad=copy.deepcopy(parent);bad['stream_positions']['policy']='disjoint_counter_v2'
    with pytest.raises(ValueError):m.validate_parent(bad)


def test_cache_join_uses_row_key_and_smiles():
    graph=dict(row=7,key='k');fp=dict(row=7,key='k',smiles='*CO*')
    d.validate_join(graph,fp,7,'k','*CO*')
    for bad in (dict(fp,row=8),dict(fp,key='other'),dict(fp,smiles='other')):
        with pytest.raises(ValueError):d.validate_join(graph,bad,7,'k','*CO*')
    bits=np.zeros(2048,dtype=np.uint8);bits[5]=1;d.validate_bits(bits)
    for bad in (np.zeros(2048,dtype=np.uint8),np.ones(2048,dtype=np.uint8),bits.astype(np.float32),bits[:256]):
        with pytest.raises(ValueError):d.validate_bits(bad)
    left=dict(y_true=np.array([1.,2.]),y_pred=np.array([3.,4.]),indices=np.array([7,8]),sample_keys=np.array(['a','b']))
    right=dict(y_true=np.array([2.,1.]),y_pred=np.array([40.,30.]),indices=np.array([8,7]),sample_keys=np.array(['b','a']))
    assert t.align_predictions(left,right).tolist()==[30.,40.]
    assert t.blend(left,right,.25)['y_pred'].tolist()==[23.25,31.]
    right['sample_keys']=np.array(['bad','a'])
    with pytest.raises(ValueError):t.align_predictions(left,right)


@pytest.fixture(scope='module')
def common():return m.common_package()


def build(arm,common):
    torch.manual_seed(42)
    return m.FPModel(common) if arm=='FP_MLP' else m.FusionModel(arm,'eea',common)


def test_isolated_common_encoder_initialization_and_pair_identity(common):
    torch.manual_seed(987);before=torch.random.get_rng_state().clone();another=m.common_package()
    assert torch.equal(before,torch.random.get_rng_state())
    for name,value in common['state_dict'].items():assert torch.equal(value,another['state_dict'][name])
    reference=None
    for left,right in (('UNI_ZERO','UNI_FP'),('TMOL_ZERO','TMOL_FP')):
        a,b=build(left,common),build(right,common)
        assert a.state_dict().keys()==b.state_dict().keys()
        for name,value in a.state_dict().items():assert torch.equal(value,b.state_dict()[name]),name
        assert torch.count_nonzero(a.residual.weight)==0
        current=a.fp_encoder.state_dict()
        if reference is not None:
            for key,value in reference.items():assert torch.equal(value,current[key])
        reference=current
        assert all(layer.p==0. for layer in a.adapter.modules() if isinstance(layer,torch.nn.Dropout))
        assert a.head[2].p==.1
    fp=build('FP_MLP',common)
    for key,value in reference.items():assert torch.equal(value,fp.fp_encoder.state_dict()[key])
    assert not hasattr(fp,'o8')


@pytest.mark.parametrize('family',('UNI','TMOL'))
def test_zero_residual_preserves_graph_and_later_adapter_gradients(family,common):
    model=build(family+'_FP',common);zero=build(family+'_ZERO',common)
    model.eval();zero.eval();torch.manual_seed(23)
    h=torch.randn(3,512);fp=torch.randint(0,2,(3,2048)).float()
    with torch.no_grad():
        original=model.head(h).view(-1)
        assert torch.equal(model.predict_features(h,fp),original)
        assert torch.equal(zero.predict_features(h,fp),original)
    captured=[]
    handle=zero.fp_encoder.register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach().clone()))
    zero.predict_features(h,fp);handle.remove()
    assert torch.count_nonzero(captured[0])==0
    with pytest.raises(ValueError):zero.predict_features(h,torch.zeros_like(fp))
    opt,_=m.optimizer(model,'head')
    loss=(model.predict_features(h,fp)-torch.tensor([1.,2.,3.])).square().mean();loss.backward()
    assert model.residual.weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.fp_encoder.parameters())
    opt.step();opt.zero_grad(set_to_none=True)
    model.predict_features(h,fp).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.fp_encoder.parameters())
    if family=='TMOL':
        _,weights=zero.adapter(torch.randn(3,128),torch.randn(3,128),0)
        assert torch.equal(weights,torch.full((3,2),.5))


def test_scaler_uses_only_train_and_original_units():
    y=np.array([1.,3.,100.,1000.]);s=dict(mean=[2.],scale=[1.],fit='train',transform='standard')
    assert t.scaler_validate(y,[0,1],s)==s
    changed=y.copy();changed[2:]=-2000
    assert t.scaler_validate(changed,[0,1],s)==s
    assert np.array_equal(t.transform(t.transform(y,s),s,True),y)
    with pytest.raises(ValueError):t.scaler_validate(y,[0,1],dict(s,mean=[250.]))


@pytest.mark.parametrize('arm',m.ARMS)
def test_stage_optimizer_coverage_lr_and_freeze(arm,common):
    model=build(arm,common)
    for stage,rate in (('head',3e-4),('joint',1e-3)):
        opt,evidence=m.optimizer(model,stage)
        identities=[id(p) for group in opt.param_groups for p in group['params']]
        assert len(identities)==len(set(identities))
        assert set(identities)=={id(p) for p in model.parameters() if p.requires_grad}
        assert opt.defaults['weight_decay']==.02 and evidence[-1]['lr']==rate
        if arm!='FP_MLP':
            model.train();assert model.o8.training==(stage=='joint')
            assert all(p.requires_grad==(stage=='joint') for p in model.o8.parameters())
            if stage=='joint':assert evidence[0]['lr']==1e-4
        else:assert len(evidence)==1 and all(p.requires_grad for p in model.parameters())


def test_global_selection_ties_and_baseline_self_comparison():
    validation={a:dict(macro=.5) for a in (*t.ARMS,'B_A020')};late={w:dict(macro=.5) for w in t.WEIGHTS}
    w,final,values=t.nominate(validation,late)
    assert w==1. and final=='B_A020'
    late[.25]['macro']=.6
    w,final,_=t.nominate(validation,late);assert w==.25 and final=='NEURAL_LATE'
    validation['UNI_FP']['macro']=.7;validation['TMOL_FP']['macro']=.7
    assert t.nominate(validation,late)[1]=='TMOL_FP'
    formal={a:t.summary({task:[.1+i*.01]*5 for task in t.TASKS}) for i,a in enumerate(t.ALL_POLICIES)}
    comparisons=t.contrasts(formal,'B_A020');zero=comparisons['FINAL_minus_B_A020']
    assert zero['delta']==0. and zero['improved_folds']==0
    for v in zero['tasks'].values():assert v['mean']==v['std']==0. and v['folds']==[0.]*5
    assert 'UNI_FP_minus_UNI_ZERO' in comparisons and 'TMOL_FP_minus_TMOL_ZERO' in comparisons
    assert t.contrasts({},'UNI_FP')['FINAL_minus_B_A020']['status']=='未完成／不可比较'


def test_freeze_boundary_and_no_overwrite(tmp_path,monkeypatch):
    with pytest.raises(ValueError):t.frozen_unit(dict(frozen=False),'UNI_FP','eea',0)
    selected=dict(config=t.config('UNI_FP','eea',0),best=str(t.unit_path('UNI_FP','eea',0)/'best.pt'))
    frozen=dict(plan=t.PLAN,revision=t.REVISION,frozen=True,units={'UNI_FP/eea/0':selected})
    assert t.frozen_unit(frozen,'UNI_FP','eea',0)==selected
    bad=copy.deepcopy(frozen);bad['units']['UNI_FP/eea/0']['config']['arm']='UNI_ZERO'
    with pytest.raises(ValueError):t.frozen_unit(bad,'UNI_FP','eea',0)
    path=tmp_path/'once.json';t.exclusive_json(path,{'frozen':True})
    with pytest.raises(FileExistsError):t.exclusive_json(path,{'frozen':False})
    monkeypatch.setattr(t,'OUT',tmp_path);monkeypatch.setattr(t,'check_prepared',lambda:None)
    t.exclusive_json(tmp_path/'selection/final_frozen.json',{'frozen':True})
    with pytest.raises(FileExistsError):t.freeze()


def test_budget_fixed_matrix_and_partial_not_refunded():
    expected=dict.fromkeys(t.LIMITS,0);jobs=[]
    for kind in ('ft','test','late'):
        arms=t.ARMS if kind!='late' else ('NEURAL_LATE',)
        for arm in arms:
            for task in t.TASKS:
                for fold in range(5):
                    r=c.reservation(kind,task,fold)
                    for key,value in r.items():expected[key]+=value
                    jobs.append(dict(kind=kind,reservation=r,exit=1,consumption=None))
    for kind in ('prepare','cpu'):
        r=c.reservation(kind);jobs.append(dict(kind=kind,reservation=r,exit=1));expected.update({k:expected[k]+v for k,v in r.items()})
    assert expected==t.LIMITS
    b=c.ledger(jobs);assert b['charged_or_reserved']==t.LIMITS and not b['budget_violations']
    assert b['formal_policy_charged_units']==210 and b['formal_policy_charged_presentations']==35250
    assert b['known_actual']['attempted_updates']==0 and b['zero_cost']==t.ZERO_COST
    jobs.append(dict(kind='cpu',reservation={'cpu_suites':1},exit=1))
    assert c.ledger(jobs)['budget_violations']==['cpu_suites']


@pytest.mark.parametrize('kind',('ft','test'))
def test_external_occupancy_busy_without_dispatch_or_budget(kind,tmp_path,monkeypatch):
    controller=c.Controller.__new__(c.Controller);controller.jobs=[]
    monkeypatch.setattr(t,'OUT',tmp_path)
    monkeypatch.setattr(controller,'deadline',lambda:None)
    def forbidden(*args,**kwargs):raise AssertionError('busy must not consume budget or launch')
    monkeypatch.setattr(controller,'refresh',forbidden)
    monkeypatch.setattr(c,'gpu_snapshot',lambda:{900})
    assert controller.launch(kind,'UNI_FP','eea',0) is None
    monkeypatch.setattr(c,'gpu_snapshot',lambda:None)
    assert controller.launch(kind,'UNI_FP','eea',0) is None
    path=tmp_path/'worker.json';t.write(path,dict(process=dict(pid=10,start_ticks=1),workers=[{'pid':11}]))
    controller.jobs=[dict(kind=kind,exit=None,consumption=str(path))]
    monkeypatch.setattr(c,'pid_aliases',lambda identity:{10,900})
    monkeypatch.setattr(c,'gpu_snapshot',lambda:{900})
    assert controller.gpu_ready()
    monkeypatch.setattr(c,'gpu_snapshot',lambda:{900,11})
    assert not controller.gpu_ready()  # CPU worker is not a known GPU main process
    monkeypatch.setattr(c,'pid_aliases',lambda identity:set())
    assert not controller.gpu_ready()
    controller.jobs*=4
    assert not controller.gpu_ready()


def test_mock_gpu_query_and_pid_namespace_errors(monkeypatch):
    answers=iter(['2, GPU-2\n3, GPU-3\n','GPU-2, 22\nGPU-3, 33\nGPU-3, 33\n'])
    monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**kw:next(answers))
    assert c.gpu_snapshot()=={33}
    def failed(*a,**kw):raise c.subprocess.CalledProcessError(1,a[0])
    monkeypatch.setattr(c.subprocess,'check_output',failed);assert c.gpu_snapshot() is None
    monkeypatch.setattr(t,'process_alive',lambda identity:True)
    monkeypatch.setattr(Path,'read_text',lambda self:'NSpid:\t900 10\n')
    assert c.pid_aliases(dict(pid=10,start_ticks=1))=={900,10}
    monkeypatch.setattr(Path,'read_text',lambda self:'NSpid:\t900 11\n')
    assert not c.pid_aliases(dict(pid=10,start_ticks=1))


def exit_job(tmp_path):
    return dict(name='synthetic_worker',attempt='repair-r1-retrain',log=str(tmp_path),exit=None)


def test_atomic_exit_publication_true_nonzero_and_no_overwrite(tmp_path,monkeypatch):
    job=exit_job(tmp_path);path=tmp_path/'exit.txt'
    monkeypatch.setattr(c,'wrapper_status',lambda job:(True,True))
    assert c.poll_exit(job) is None
    actual=c.os.replace;seen=[]
    def replace(source,dest):
        assert source.parent==dest.parent and not dest.exists()
        assert source.read_text()=='1\n'
        assert c.poll_exit(job) is None  # unpublished temporary is not a terminal
        seen.append(source);actual(source,dest)
    monkeypatch.setattr(c.os,'replace',replace)
    c.atomic_exit(path,1)
    assert seen and c.poll_exit(job)==1 and path.read_text()=='1\n'
    with pytest.raises(FileExistsError):c.atomic_exit(path,0)
    assert path.read_text()=='1\n'


@pytest.mark.parametrize('content',(None,''))
def test_missing_empty_terminal_has_ten_second_bound(content,tmp_path,monkeypatch):
    job=exit_job(tmp_path)
    if content is not None:(tmp_path/'exit.txt').write_text(content)
    monkeypatch.setattr(c,'wrapper_status',lambda job:(True,False))
    assert c.poll_exit(job,clock=lambda:0.) is None
    assert c.poll_exit(job,clock=lambda:9.999) is None
    with pytest.raises(c.ExitProtocolError,match='exceeded10seconds') as error:c.poll_exit(job,clock=lambda:10.)
    assert job['name'] in str(error.value) and 'repair-r1-retrain' in str(error.value) and str(tmp_path/'exit.txt') in str(error.value)


def test_terminal_wait_resets_during_real_work(tmp_path,monkeypatch):
    job=exit_job(tmp_path);job['terminal_wait_started']=0.
    monkeypatch.setattr(c,'wrapper_status',lambda job:(True,True))
    assert c.poll_exit(job,clock=lambda:100.) is None and 'terminal_wait_started' not in job
    monkeypatch.setattr(c,'wrapper_status',lambda job:(True,False))
    assert c.poll_exit(job,clock=lambda:100.) is None
    assert c.poll_exit(job,clock=lambda:109.) is None
    with pytest.raises(c.ExitProtocolError):c.poll_exit(job,clock=lambda:110.)


@pytest.mark.parametrize('content',('bad','NaN','1.0','0\n1'))
def test_bad_terminal_never_means_success(content,tmp_path):
    job=exit_job(tmp_path);(tmp_path/'exit.txt').write_text(content)
    with pytest.raises(c.ExitProtocolError,match='invalid terminal content') as error:c.poll_exit(job)
    assert 'process' in str(error.value) and job['exit'] is None


def test_dead_wrapper_and_publication_death_boundary(tmp_path,monkeypatch):
    job=exit_job(tmp_path);monkeypatch.setattr(c,'wrapper_status',lambda job:(False,False))
    with pytest.raises(c.ExitProtocolError,match='dead wrapper'):c.poll_exit(job)
    replies=iter([(None,None),(0,'0\n')]);monkeypatch.setattr(c,'read_terminal',lambda job:next(replies))
    assert c.poll_exit(job)==0


def test_unreadable_terminal_reports_path_and_identity(tmp_path,monkeypatch):
    job=exit_job(tmp_path);original=Path.read_text
    def unreadable(path,*a,**kw):
        if path==tmp_path/'exit.txt':raise PermissionError('synthetic read error')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',unreadable)
    with pytest.raises(c.ExitProtocolError,match='exit read failed') as error:c.poll_exit(job)
    assert str(tmp_path/'exit.txt') in str(error.value) and 'process' in str(error.value)


def test_repeated_signals_do_not_interrupt_first_failure_cleanup(monkeypatch):
    handlers={}
    monkeypatch.setattr(c.signal,'signal',lambda sig,handler:handlers.__setitem__(sig,handler))
    first=c.install_termination_handler()
    with pytest.raises(RuntimeError,match='controlled termination'):first(c.signal.SIGTERM,None)
    assert handlers[c.signal.SIGTERM]==c.signal.SIG_IGN and handlers[c.signal.SIGINT]==c.signal.SIG_IGN
    # A second signal has no raising handler, so error/consumption writes can complete.
    terminal=dict(status='FAILED',original_failure='first signal')
    t.ignore_termination_signals()
    assert terminal['status']=='FAILED' and handlers[c.signal.SIGTERM]==c.signal.SIG_IGN


def test_owned_group_requires_matching_pid_start_ticks(tmp_path,monkeypatch):
    job=exit_job(tmp_path);identity=dict(pid=101,start_ticks=20)
    t.write(tmp_path/'process.json',dict(wrapper=dict(pid=201,start_ticks=40),timeout_identity=identity))
    monkeypatch.setattr(t,'process_alive',lambda item:item==identity)
    monkeypatch.setattr(c.os,'getpgid',lambda pid:101)
    assert c.owned_group(job)==101
    monkeypatch.setattr(t,'process_alive',lambda item:False)
    assert c.owned_group(job) is None
    monkeypatch.setattr(t,'process_alive',lambda item:item==identity)
    monkeypatch.setattr(c.os,'getpgid',lambda pid:999)
    assert c.owned_group(job) is None


def test_cleanup_ignores_refresh_errors_and_is_bounded_ownership_only(tmp_path,monkeypatch):
    controller=c.Controller.__new__(c.Controller);controller.repair=False;controller.parent_budget=None;controller.jobs=[]
    monkeypatch.setattr(t,'OUT',tmp_path);monkeypatch.setattr(t,'REPAIR_ROOT',None)
    monkeypatch.setattr(t,'ignore_termination_signals',lambda:None)
    for i in (101,102,999):
        path=tmp_path/str(i);path.mkdir()
        controller.jobs.append(dict(name=str(i),kind='ft',exit=None,log=str(path),reservation={},consumption=None))
    def broken(*a,**kw):raise AssertionError('cleanup must not call ordinary refresh')
    monkeypatch.setattr(controller,'refresh',broken)
    def read_error(job):raise c.ExitProtocolError('synthetic exit reader failure')
    monkeypatch.setattr(c,'poll_exit',read_error)
    killed=set();sent=[];clock=[0.]
    monkeypatch.setattr(c,'owned_group',lambda job:None if int(job['name'])==999 or int(job['name']) in killed else int(job['name']))
    def signal(group,sig):
        sent.append((group,sig))
        if sig==c.signal.SIGKILL:killed.add(group)
    monkeypatch.setattr(c.os,'killpg',signal)
    monkeypatch.setattr(c.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(c.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    controller.stop_owned()
    assert clock[0]==30. and all(group in (101,102) for group,sig in sent)
    for group in (101,102):assert sent.count((group,c.signal.SIGTERM))==sent.count((group,c.signal.SIGKILL))==1
    assert t.read(tmp_path/'cleanup.json')['errors']


@pytest.fixture
def synthetic_attempt_mapping(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'OUT',tmp_path/'parent');monkeypatch.setattr(t,'REPAIR_ROOT',None);monkeypatch.setattr(t,'ATTEMPT_MAP',None)
    root=t.OUT/'repairs/controller_repair1';root.mkdir(parents=True)
    keys=[(a,task,f) for a in t.ARMS for task in t.TASKS for f in range(5)]
    jobs=[]
    for i,(a,task,f) in enumerate(keys[:48]):
        folder=t.OUT/'finetune'/a/task/f'fold{f}';folder.mkdir(parents=True)
        exit=0 if i<46 else 1
        if exit==0:
            t.write(folder/'metrics.json',dict(status='COMPLETE',config=t.config(a,task,f)))
            t.write(folder/'consumption.json',dict(status='COMPLETE',completed_epochs=70))
        else:(folder/'best.pt').write_text('failed BEST is never selected')
        jobs.append(dict(kind='ft',arm=a,task=task,fold=f,exit=exit))
    mapping=c.register_attempt_map(root,dict(jobs=jobs))
    return root,mapping,jobs


def test_fixed_attempt_map_routes_old_new_freeze_and_test(synthetic_attempt_mapping,monkeypatch):
    root,mapping,jobs=synthetic_attempt_mapping
    assert len(mapping['units'])==175
    assert sum(e['source']=='parent_complete' for e in mapping['units'].values())==46
    assert sum(e['source']=='repair' for e in mapping['units'].values())==129
    assert {key for key,e in mapping['units'].items() if e['attempt']=='repair-r1-retrain'}==t.RETRAIN_KEYS
    with pytest.raises(FileExistsError):c.register_attempt_map(root,dict(jobs=jobs))
    t.enable_repair(root)
    for a,task,f in (('UNI_ZERO','eea',0),('UNI_FP','egc',1),('TMOL_FP','xc',4)):
        source=mapping['units'][f'{a}/{task}/{f}']
        assert t.unit_path(a,task,f)==Path(source['ft_dir'])
        assert t.unit_path(a,task,f,True)==root/'formal'/a/task/f'fold{f}'
        selected=dict(config=t.config(a,task,f),best=str(t.unit_path(a,task,f)/'best.pt'),attempt_source=source)
        frozen=dict(plan=t.PLAN,revision=t.REVISION,frozen=True,units={f'{a}/{task}/{f}':selected})
        assert t.frozen_unit(frozen,a,task,f)==selected
        bad=copy.deepcopy(frozen);bad['units'][f'{a}/{task}/{f}']['attempt_source']['attempt']='other'
        with pytest.raises(ValueError):t.frozen_unit(bad,a,task,f)
    bad=copy.deepcopy(mapping);bad['units']['UNI_FP/egc/1'].update(source='parent_complete',attempt='r1',ft_dir=str(t.OUT/'finetune/UNI_FP/egc/fold1'))
    with pytest.raises(ValueError):t.validate_attempt_map(bad,root,t.OUT)
    monkeypatch.setattr(t,'check_prepared',lambda:None)
    t.exclusive_json(root/'selection/final_frozen.json',dict(frozen=True))
    with pytest.raises(FileExistsError):t.freeze()
    with pytest.raises(RuntimeError,match='no new preparation'):t.prepare()


def test_parent_failures_are_historical_and_cumulative_budget_is177(synthetic_attempt_mapping):
    root,mapping,jobs=synthetic_attempt_mapping
    parent_charged=dict.fromkeys(t.LIMITS,0)
    parent_charged.update(prepare_starts=1,ft_starts=48,epochs_started=3360,attempted_updates=83860,
        train_graphs=1977220,validation_started=3360,validation_graphs=495740)
    parent=dict(limits=t.LIMITS,charged_or_reserved=parent_charged,known_actual=dict(parent_charged),jobs=jobs)
    parent['known_actual'].update(epochs_started=3356,attempted_updates=83344,train_graphs=1964962,validation_started=3354,validation_graphs=492494)
    new=[]
    for key,entry in mapping['units'].items():
        if entry['source']=='repair':
            a,task,f=key.split('/');new.append(dict(kind='ft',name=key,exit=0,reservation=c.reservation('ft',task,int(f)),consumption=None))
    b=c.cumulative_ledger(new,parent)
    assert b['new_charged_or_reserved']['ft_starts']==129 and b['charged_or_reserved']['ft_starts']==177
    for key in ('epochs_started','attempted_updates','train_graphs','validation_started','validation_graphs'):
        assert b['charged_or_reserved'][key]==c.CUMULATIVE_LIMITS[key]
        assert b['new_charged_or_reserved'][key]==c.NEW_LIMITS[key]
    assert not b['budget_violations'] and t.LIMITS['ft_starts']==175
    assert b['known_actual']['attempted_updates']==83344  # no invented/refunded historical actual
    assert len(b['history_jobs'])==48
    controller=c.Controller.__new__(c.Controller);controller.jobs=new
    controller.check_failures()  # parent failures cannot poison the new active queue
    controller.jobs=[dict(name='new_failure',exit=1)]
    with pytest.raises(RuntimeError):controller.check_failures()
    extra=new+[dict(kind='ft',name='extra',exit=0,reservation=c.reservation('ft','egc',1),consumption=None)]
    assert 'ft_starts' in c.cumulative_ledger(extra,parent)['budget_violations']
