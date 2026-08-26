#!/usr/bin/env python3
"""Paired-occurrence, step-zero parity, and first-gradient sanity for AC/RA."""
from __future__ import annotations
import importlib.util, json, os, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.periodic_line_glt import derive_relation_source_distance_observations
from src.modules.mts_glt_v2 import MTSGraphLineModelV2

OUT=ROOT/'results/mts_glt_v2/joint_radial_angular_screen_v1'
CKPT=ROOT/'results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth'

def batch():
    path=ROOT/'tests/test_mts_periodic_line_glt.py'; spec=importlib.util.spec_from_file_location('joint_fixture',path)
    f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    values=[]
    for smi in ('*CCO*','*CCCC*'):
        sample=f._sample(smi); record=f.build_periodic_line_sample(f.sample_key_from_smiles(smi),sample,sample)
        item=f._attach(sample,record)
        token_names=('token_atom_a','token_atom_b','token_shift','token_endpoint_z_a','token_endpoint_z_b','token_bond_type','token_label','token_observation_distances','token_observation_count','token_valid')
        relation_names=('relation_source','relation_target','relation_center_atom','relation_multiplicity','relation_observation_angles','relation_observation_count','relation_valid','relation_is_fallback')
        tokens={name:getattr(item,'glt_'+name).numpy() for name in token_names}
        relations={name:getattr(item,'glt_'+name).numpy() for name in relation_names}
        paired=derive_relation_source_distance_observations(tokens,relations)
        item.glt_relation_source_distances=torch.tensor(paired['source_distances'])
        item.glt_relation_source_distance_valid=torch.tensor(paired['observation_valid'])
        item.glt_relation_source_distance_slot=torch.tensor(paired['source_slots']).long()
        item.glt_relation_source_cross_ru=torch.tensor(paired['source_cross_ru'])
        values.append(item)
    return mips_trimer_collate(values)

def model(mode):
    m=MTSGraphLineModelV2(glt_layers=6,joint_basis_mode=mode)
    payload=torch.load(CKPT,map_location='cpu',weights_only=False)
    source={k[6:]:v for k,v in payload['state_dict'].items() if k.startswith('model.')}
    state=m.state_dict(); extra=[k for k in state if k.startswith('glt.joint_basis_bias.')]
    if sorted(set(state)-set(source)) != sorted(extra) or set(source)-set(state): raise RuntimeError('checkpoint namespace mismatch')
    state.update(source); m.load_state_dict(state,strict=True)
    m.downstream_mode={'control':'o8_glt_atom_sbf_angle_control','radial':'o8_glt_atom_sbf_radial_angle'}[mode]
    return m

def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def main():
    torch.manual_seed(42); data=batch()
    baseline=MTSGraphLineModelV2(glt_layers=6).eval(); payload=torch.load(CKPT,map_location='cpu',weights_only=False)
    source={k[6:]:v for k,v in payload['state_dict'].items() if k.startswith('model.')}; baseline.load_state_dict(source,strict=True); baseline.downstream_mode='o8_glt_atom'
    ac=model('control').eval(); ra=model('radial').eval()
    with torch.no_grad():
        pb,pa,pr=baseline(data),ac(data),ra(data)
        torch.testing.assert_close(pb,pa,rtol=0,atol=0); torch.testing.assert_close(pb,pr,rtol=0,atol=0)
        _,_,bb,_=baseline.glt._relations(data,torch.float32)
        _,_,ab,av=ac.glt._relations(data,torch.float32,return_joint_basis=True)
        _,_,rb,rv=ra.glt._relations(data,torch.float32,return_joint_basis=True)
        torch.testing.assert_close(bb,ab,rtol=0,atol=0); torch.testing.assert_close(bb,rb,rtol=0,atol=0)
        d=torch.tensor([1.2,1.6]); t=torch.tensor([0.8,0.8]); t2=torch.tensor([0.9,0.9])
        ac_d=ac.glt.joint_basis_bias.observation_features(d,t); ac_d2=ac.glt.joint_basis_bias.observation_features(d+0.1,t)
        ra_d=ra.glt.joint_basis_bias.observation_features(d,t); ra_d2=ra.glt.joint_basis_bias.observation_features(d+0.1,t)
        ac_t=ac.glt.joint_basis_bias.observation_features(d,t2); ra_t=ra.glt.joint_basis_bias.observation_features(d,t2)
    gradients={}
    for label,m in [('AC',ac),('RA',ra)]:
        m.train(); m.zero_grad(set_to_none=True); m(data).square().mean().backward()
        vals={n:float(p.grad.abs().sum()) if p.grad is not None else 0.0 for n,p in m.glt.joint_basis_bias.named_parameters() if 'projection' in n}
        gradients[label]={'finite':all(torch.isfinite(p.grad).all() for n,p in m.glt.joint_basis_bias.named_parameters() if 'projection' in n and p.grad is not None),'absolute_sums':vals,'nonzero':any(x>0 for x in vals.values())}
    pair={'schema':'mts-glt-v2-joint-ra-pair-sanity-v1','occurrence_slots_valid':bool(data.glt_relation_source_distance_valid.any()),'step0_prediction_parity':True,'existing_relation_bias_unchanged':True,'ac_distance_invariant':bool(torch.equal(ac_d,ac_d2)),'ra_distance_sensitive':bool(not torch.equal(ra_d,ra_d2)),'ac_angle_sensitive':bool(not torch.equal(ac_d,ac_t)),'ra_angle_sensitive':bool(not torch.equal(ra_d,ra_t)),'parameter_count_ac':sum(p.numel() for p in ac.glt.joint_basis_bias.parameters()),'parameter_count_ra':sum(p.numel() for p in ra.glt.joint_basis_bias.parameters())}
    grad={'schema':'mts-glt-v2-joint-ra-gradient-sanity-v1','first_backward':gradients,'pass':all(v['finite'] and v['nonzero'] for v in gradients.values())}
    if not all([pair['ac_distance_invariant'],pair['ra_distance_sensitive'],pair['ac_angle_sensitive'],pair['ra_angle_sensitive'],grad['pass']]): raise RuntimeError('joint basis sanity failed')
    atomic(OUT/'paired_occurrence_sanity.json',pair); atomic(OUT/'gradient_sanity.json',grad)
    print(json.dumps({'pair':pair,'gradient':grad},sort_keys=True))
if __name__=='__main__': main()
