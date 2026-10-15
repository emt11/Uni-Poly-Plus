import numpy as np
import json
import pytest
from scripts.run_mcl_ph_r4_fast import route_gate,choose_routes,DEV_TASKS,epochs,parse_rank_progress


def test_route_gates_reject_large_task_regression_and_incomplete_evidence():
    assert route_gate([.011,.011,.011],d2=True)['passed']
    assert not route_gate([.04,.04,-.021],d2=True)['passed']
    assert route_gate([[.015,.015],[.015,.015],[0.,0.]])['passed']
    assert not route_gate([[.008,.008],[.008,.008],[.008,.008]])['passed']
    with pytest.raises(ValueError): route_gate([.02,.02])
    with pytest.raises(ValueError): route_gate([.02,.02,np.nan],d2=True)


def test_selection_returns_reference_when_combined_gate_fails():
    values={arm:{task:[1.,1.] for task in DEV_TASKS} for arm in ('glt_ref','o8_only','m_xattn')}
    values['o8_only']['ei']=[.98,.98]
    values['m_xattn']['nc']=[.995,.995]
    selected,gate,_=choose_routes(values,'m_xattn')
    assert not gate['passed']
    assert selected==dict.fromkeys(DEV_TASKS,'glt_ref')
    values['m_xattn']['nc']=[.98,.98]
    selected,gate,_=choose_routes(values,'m_xattn')
    assert gate['passed'] and selected['ei']=='o8_only' and selected['nc']=='m_xattn'


def test_exact_unique_development_and_max_formal_budget():
    units={(a,t,0) for a in ('m_o8_only','m_xattn') for t in ('ei','nc')}
    units|={(a,t,0) for a in ('m_xattn','m_xattn_scale') for t in DEV_TASKS}
    units|={(a,t,f) for a in ('glt_ref','o8_only','m_xattn_scale') for t in DEV_TASKS for f in (0,1)}
    assert len(units)==23 and sum(epochs(t) for _,t,_ in units)==1610
    tasks=('eea','egb','egc','ei','eps','nc','xc')
    formal={(a,t,f) for t in tasks for a in (('glt_ref','m_xattn') if t in ('eea',*DEV_TASKS) else ('glt_ref',)) for f in range(5)}
    assert len(formal)==55 and sum(epochs(t) for _,t,_ in formal)==3800


def test_progress_handles_concatenated_four_rank_json_and_nonfinite(tmp_path):
    path=tmp_path/'stdout.log'
    path.write_text(''.join(json.dumps(dict(step=10,rank=i,losses=[1.],
        cpu_observed_timing=dict(step_seconds=.5))) for i in range(4)))
    progress=parse_rank_progress(path)
    assert progress['all_rank_updates']==10 and progress['recent_step_seconds']==.5
    path.write_text('{"step":11,"rank":0,"losses":[NaN]}')
    with pytest.raises(ValueError):parse_rank_progress(path)
