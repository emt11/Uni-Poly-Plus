"""One final WAVG CPU suite; no scientific model training or baseline replay."""
import copy
from pathlib import Path
import time
import pytest
import torch
from torch import nn
from src.training import ua2d_wavg as t

torch.set_num_threads(2)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.o8=nn.Linear(2,2);self.norm2=nn.LayerNorm(2);self.head=nn.Linear(2,1)
        self.register_buffer('floating',torch.ones(2));self.register_buffer('counter',torch.tensor(7,dtype=torch.long))
    def forward(self,x):return self.head(self.norm2(self.o8(x)))


def assert_equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:assert_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    else:assert a==b


def test_fixed_averages_preserve_student_gradient_optimizer_and_rng():
    torch.manual_seed(9);model=Tiny();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    model(torch.ones(2,2)).square().sum().backward();opt.step()
    avg=t.FixedAverages(model);optimizer_before=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
    assert set(dict(model.named_parameters())).issubset(avg.swa)
    assert set(avg.swa)=={k for k,v in model.state_dict().items() if v.is_floating_point()}
    for k in avg.swa:
        assert avg.swa[k].dtype==torch.float32 and not avg.swa[k].requires_grad
        assert avg.swa[k].data_ptr()!=avg.ema[k].data_ptr()
        assert avg.swa[k].data_ptr()!=model.state_dict()[k].data_ptr()
    for epoch in range(1,71):
        with torch.no_grad():
            for p in model.parameters():p.fill_(epoch)
            model.floating.fill_(epoch)
        before=copy.deepcopy(model.state_dict());grads={n:p.grad.clone() for n,p in model.named_parameters()}
        avg.observe(model,epoch)
        assert_equal(before,model.state_dict());assert_equal(optimizer_before,opt.state_dict())
        assert torch.equal(rng,torch.get_rng_state())
        for n,p in model.named_parameters():assert torch.equal(grads[n],p.grad)
        if epoch<=10:assert avg.ema_epochs==[] and avg.z==0
    swa=avg.final_state('SWA4');ema=avg.final_state('EMA60')
    z=0.;expected=0.
    for epoch in range(11,71):expected=.9*expected+.1*epoch;z=.9*z+.1
    assert avg.swa_epochs==[40,50,60,70] and len(avg.ema_records)==60
    for k in avg.swa:
        assert torch.allclose(swa[k],torch.full_like(swa[k],55.))
        assert torch.allclose(ema[k],torch.full_like(ema[k],expected/z),atol=1e-5)
    assert swa['counter'].item()==7 and ema['counter'].item()==7
    acc=avg.accumulator();assert acc['count']==60 and acc['source_epochs']==list(range(11,71))
    assert acc['z']==pytest.approx(1-.9**60) and acc['z']!=1
    assert torch.equal(rng,torch.get_rng_state())


def test_average_sources_counts_and_nonfloating_consistency():
    model=Tiny();avg=t.FixedAverages(model)
    with pytest.raises(ValueError):avg.observe(model,2)
    avg.observe(model,1)
    with pytest.raises(ValueError):avg.final_state('SWA4')
    with pytest.raises(ValueError):avg.observe(model,1)
    model.counter.fill_(8)
    for epoch in range(2,71):avg.observe(model,epoch)
    with pytest.raises(ValueError):avg.final_state('SWA4')
    with pytest.raises(ValueError):avg.observe(model,71)


@pytest.mark.parametrize('kind',t.KINDS)
def test_complete_strict_restore_and_provenance(kind,tmp_path):
    model=t.MeanModel();cfg=t.config('eea',0);info=t.checkpoint_info(kind,cfg,7,'head')
    package=dict(schema=t.SCHEMA,architecture=t.ARCHITECTURE,checkpoint_kind=kind,checkpoint_metadata=info,config=cfg,state_dict=model.state_dict())
    if kind=='BEST':package.update(best_epoch=7,best_stage='head')
    p=tmp_path/'checkpoint.pt';torch.save(package,p);saved=torch.load(p,map_location='cpu',weights_only=False)
    rng=torch.get_rng_state().clone();rebuilt=t.restore(saved,cfg,kind)
    assert torch.equal(rng,torch.get_rng_state());assert_equal(model.state_dict(),rebuilt.state_dict())
    bad=dict(saved,checkpoint_metadata=dict(info,count=999))
    with pytest.raises(ValueError):t.restore(bad,cfg,kind)
    with pytest.raises(ValueError):t.restore(dict(saved,config=dict(cfg,fold=1)),cfg,kind)
    if kind!='BEST':
        assert 'best_epoch' not in info and 'best_stage' not in info
        with pytest.raises(ValueError):t.restore(dict(saved,best_epoch=70),cfg,kind)
    broken=dict(saved,state_dict=dict(saved['state_dict']));broken['state_dict'].pop(next(iter(broken['state_dict'])))
    with pytest.raises(ValueError):t.restore(broken,cfg,kind)


def test_single_trajectory_training_and_endpoint_accounting(tmp_path,monkeypatch):
    from scripts import run_ua2d_wavg as script
    monkeypatch.setattr(script,'OUT',tmp_path)
    controller=script.Controller.__new__(script.Controller)
    controller.jobs=[];controller.start=time.monotonic();controller.started_unix=1.;controller.cached_state=None;controller.cached_budget=None;controller.evidence_only=False
    root=tmp_path/'trajectory';root.mkdir();log=tmp_path/'ft_log';log.mkdir()
    state=dict(status='COMPLETE',epochs_started=70,attempted_updates=1600,validation_started=72,
        training_validation_started=70,endpoint_validation_started=2,test_graphs=0)
    t.write(root/'consumption.json',state)
    controller.jobs.append(dict(name='only_ft',kind='ft',exit=0,log=str(log),consumption=str(root/'consumption.json'),reservation=dict(epochs_started=70,attempted_updates=1600,validation_started=72)))
    for i in range(3):
        d=tmp_path/f'test{i}';d.mkdir();l=tmp_path/f'log{i}';l.mkdir();t.write(d/'consumption.json',dict(status='COMPLETE',test_started=1,test_graphs=4))
        controller.jobs.append(dict(name=f'estimate{i}',kind='test',exit=0,log=str(l),consumption=str(d/'consumption.json'),reservation=dict(test_graphs=4)))
    b=controller.refresh()
    assert b['used']['ft_starts']==1 and b['used']['epochs_started']==70 and b['used']['attempted_updates']==1600
    assert b['used']['validation_started']==72 and b['training_validation_started']==70 and b['endpoint_validation_started']==2
    assert b['used']['test_starts']==3 and b['used']['test_graphs']==12
    assert b['used']['prepare_starts']==b['used']['initialization_starts']==b['used']['pt_starts']==0
    count=b['time'];assert controller.refresh()['time']==count
    assert t.LIMITS['validation_started']==35*72
    interrupted=dict(epochs_started=70,validation_started=69,attempted_updates=1590,
        epoch_reservation=dict(open=True,before_updates=1590,updates=10,before_validation=69,validations=1))
    charged=t.conservative_ft_counts(interrupted)
    assert charged['charged']['attempted_updates']==1600 and charged['charged']['validation_started']==70


def test_freeze_all105_sources_and_nominate_only_averages(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'OUT',tmp_path);monkeypatch.setattr(t,'SPLIT',tmp_path/'splits')
    (tmp_path/'splits').mkdir()
    for task in t.TASKS:t.write(tmp_path/'splits'/f'{task}.json',dict(folds=[dict(test_indices=[f]) for f in range(5)]))
    validation={}
    for kind in t.KINDS:
        units={}
        for task in t.TASKS:
            for fold in range(5):
                cfg=t.config(task,fold);info=t.checkpoint_info(kind,cfg,8,'head');checkpoint=t.unit_path(kind,task,fold)/'checkpoint.pt';checkpoint.parent.mkdir(parents=True);checkpoint.touch()
                if kind=='SWA4':
                    for name in info['source_snapshots']:Path(name).parent.mkdir(parents=True,exist_ok=True);Path(name).touch()
                if kind=='EMA60':
                    Path(info['accumulator']).parent.mkdir(parents=True,exist_ok=True);Path(info['accumulator']).touch();(Path(cfg['trajectory'])/'ema_updates.json').write_text('[]')
                units[f'{kind}/{task}/{fold}']=dict(config=cfg,checkpoint=str(checkpoint),checkpoint_metadata=info,scaler=dict(mean=[0],scale=[1]),test_indices=[fold])
        validation[kind]=dict(macro=.99 if kind=='BEST' else .1,units=units)
    assert t.nominate(validation)=='SWA4'
    bad=copy.deepcopy(validation);bad['SWA4']['units']['SWA4/eea/0']['checkpoint_metadata']['source_epochs']=[39,50,60,70]
    with pytest.raises(ValueError):t.freeze_final(bad)
    frozen=t.freeze_final(validation);assert frozen['FINAL']=='SWA4' and frozen['models']==list(t.KINDS) and len(frozen['units'])==105
    assert frozen['units']['EMA60/eea/0']['checkpoint_metadata']['count']==60
    with pytest.raises(FileExistsError):t.freeze_final(validation)


def test_performance_requires_both_unrounded_references():
    assert t.performance(.010,.010)['performance_pass']
    assert not t.performance(.02,.009999999)['performance_pass']
    assert not t.performance(.009999999,.02)['performance_pass']
    assert t.performance(None,.02)['performance_pass'] is None
