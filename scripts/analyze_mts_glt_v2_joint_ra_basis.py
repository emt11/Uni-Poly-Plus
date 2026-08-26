#!/usr/bin/env python3
"""Fixed, descriptive SBF audit on deterministic frozen-sidecar observations."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.dataset.periodic_line_glt import PeriodicLineGLTSidecar, derive_relation_source_distance_observations
from src.modules.periodic_line_glt_v2 import JointRadialAngularRelationBiasV2

def metrics(x):
    x=x.double(); x=x-x.mean(0); singular=torch.linalg.svdvals(x)
    eigen=singular.square(); prob=eigen/eigen.sum().clamp_min(1e-30)
    return {'effective_rank':float(torch.exp(-(prob*prob.clamp_min(1e-30).log()).sum())),'participation_ratio':float(eigen.sum().square()/eigen.square().sum().clamp_min(1e-30)),'near_constant_channels':int((x.std(0,unbiased=False)<1e-4).sum())}

def main():
    side=PeriodicLineGLTSidecar(ROOT/'data/processed/mips_trimer_scage/periodic_line_glt_v1/PI1M_v2')
    ids=np.linspace(0,len(side)-1,num=min(10000,len(side)),dtype=np.int64); ds=[]; ts=[]
    for i in ids:
        row=side.model_row(int(i)); paired=derive_relation_source_distance_observations(row['tokens'],row['relations'])
        mask=paired['observation_valid']; angle=np.asarray(row['relations']['relation_observation_angles'])
        ds.append(paired['source_distances'][mask]); ts.append(angle[mask])
        if sum(x.size for x in ds)>=200000: break
    d=torch.from_numpy(np.concatenate(ds)[:200000]).float(); t=torch.from_numpy(np.concatenate(ts)[:200000]).float()
    ac=JointRadialAngularRelationBiasV2(mode='control'); ra=JointRadialAngularRelationBiasV2(mode='radial')
    chunks=lambda m: torch.cat([m.observation_features(d[i:i+10000],t[i:i+10000]).detach() for i in range(0,len(d),10000)])
    a=chunks(ac); r=chunks(ra); delta=(r-a).norm(dim=1)
    payload={'schema':'mts-glt-v2-joint-ra-basis-sanity-v1','seed':42,'observation_count':int(len(d)),'feature_dimension':42,'finite_fraction_ac':float(torch.isfinite(a).float().mean()),'finite_fraction_ra':float(torch.isfinite(r).float().mean()),'AC':metrics(a),'RA':metrics(r),'mean_ra_minus_ac_norm':float(delta.mean()),'median_ra_minus_ac_norm':float(delta.median()),'implementation':'torch_geometric.nn.models.dimenet.SphericalBasisLayer','num_spherical':7,'num_radial':6,'cutoff':3.75,'envelope_exponent':5,'reference_distance':1.407050}
    path=ROOT/'results/mts_glt_v2/joint_radial_angular_screen_v1/basis_sanity.json'; path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path); print(json.dumps(payload,sort_keys=True))
if __name__=='__main__': main()
