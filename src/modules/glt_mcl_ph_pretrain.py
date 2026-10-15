"""GP atom/geometry objectives, with balance and optional alignment; no FP."""
from pathlib import Path
import torch
from torch import nn
from .glt_dual_pretrain import DualPretrainer,alignment_loss
from .glt_mcl_ph import GPEncoder
from .mcl_ph import balance_term

class GPPretrainer(DualPretrainer):
    def __init__(self,*,route='GP',common_path=None):
        super().__init__('concat',geometry_head_norm=True,third_task='none')
        original=self.encoder.state_dict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            new=GPEncoder(route=route);new.predictor=nn.Identity()
        # Explicit whole-state copy: all common keys plus independently initialized new keys.
        merged=new.state_dict();assert set(original)<=set(merged)
        merged.update(original);new.load_state_dict(merged,strict=True);self.encoder=new
        self.route=route
        if route=='GP_A':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                self.align_proj2=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,128))
                self.align_proj3=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,128))
        if common_path:
            p=torch.load(common_path,map_location='cpu',weights_only=False)
            if p['schema'] not in ('gp-sota-common-init-v2','gp-common-init-no-fp-v1'):raise ValueError('GP common schema differs')
            # Legacy common states are initialization only, never optimizer resumes.
            # The four deleted FP tensors do not belong to the current model.
            current=self.state_dict();common={k:v for k,v in p['common_state_dict'].items() if not k.startswith('fp_head.')}
            allowed={k for k in current if k.startswith(('encoder.experts.','encoder.router.','encoder.adaptor.','align_proj2.','align_proj3.'))}
            if set(current)-allowed!=set(common):raise ValueError('GP common mapping coverage differs')
            for k,v in common.items():
                if current[k].shape!=v.shape or current[k].dtype!=v.dtype or not torch.isfinite(v).all():raise ValueError('GP common tensor incompatible: '+k)
                current[k]=v
            self.load_state_dict(current,strict=True)
    def forward(self,data,labels,*,collect_diagnostics=False):
        out=super().forward(data,labels,collect_diagnostics=collect_diagnostics)
        enc=self.encoder.last_encoded
        balance,empty=balance_term(enc['router_dense'],enc['mcl_valid'])
        out['balance']=balance if self.route!='GU' else balance*0
        if self.route=='GP_A':
            z2=nn.functional.normalize(self.align_proj2(enc['graph_2d']).float(),dim=-1)
            z3=nn.functional.normalize(self.align_proj3(enc['graph_3d']).float(),dim=-1)
            out['align_sum'],out['align_count']=alignment_loss(z2,z3,labels['align_identity'],enc['geometry_valid'],temperature=.1)
        else:out['align_sum'],out['align_count']=out['sums'].sum()*0,out['counts'].new_tensor(0)
        out['router_weights']=enc['router_weights'].detach();out['router_valid']=enc['mcl_valid'].detach()
        return out

def deployment_package(model,step):
    return dict(schema='gp-sota-deploy-v2',route=model.route,step=int(step),architecture=model.encoder.architecture_name,
                state_dict={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()},
                pretraining_objectives=['atom','geometry','balance']+(['align'] if model.route=='GP_A' else []))
