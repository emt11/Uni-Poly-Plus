"""r4: shared physical-atom spatial residual inside the native Galformer."""
import torch
from torch import nn
from torch.nn import functional as F
from src.modules import ua3m_paperfuse as native
from src.modules.ua2d_galformer_pretrain import MaskedGalformer3D
from src.modules.glt_dual import element_index, triplet_type, mean_pool, strict_module_state
from src.modules.glt_dual_opt import incident_pool

ARMS=('BASE','C_CONST','S_RAD','S_RADANG')
TASKS=native.TASKS
PT_SCHEMA='ua3d-spatial-pretrain-r4'
FT_SCHEMA='ua3d-spatial-best-r4'
PROBE_SCHEMA='ua3d-spatial-probe-r4'
NATIVE=native.NATIVE
CONFIG=dict(native=native.CONFIG,spatial=dict(radius=4.5,min_distance=1e-8,exclude_hops=2,
    rbf_count=32,rbf_width=4.5/31,constant_distance=3.,angles=['mean_cos','mean_P2'],
    hidden=32,mlp=[34,64,32],layers_shared=6,residual_scale=1/6,
    added_parameters=37632,seed=20261025,physical_instances=True,neighbor_cap=None,
    coordinate_noise=.03,initial_up='zero',torsion=False))
property_head=native.property_head
optimizer_ft=native.optimizer_ft


class SpatialResidual(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS[1:]:raise ValueError('spatial arm required')
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(20261025)
            self.down=nn.Linear(512,32)
            self.geometry=nn.Sequential(nn.Linear(34,64),nn.GELU(),nn.Linear(64,32))
            self.up=nn.Linear(32,512)
            nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        if sum(p.numel() for p in self.parameters())!=37632:raise ValueError('spatial parameter count')

    def forward(self,states,data):
        endpoints=data.sp_bond_atoms.long();edges=data.sp_edges.long();count=data.sp_degree.numel()
        if endpoints.shape!=(2,len(states)):raise ValueError('physical token incidence mismatch')
        if edges.numel()==0:return states*0.
        atoms=mean_pool(torch.cat((states,states)),endpoints.reshape(-1),count)
        distance=data.sp_distance.float()
        with torch.autocast(states.device.type,enabled=False):
            if self.arm=='C_CONST':distance=torch.full_like(distance,3.)
            centers=torch.linspace(0.,4.5,32,device=states.device,dtype=torch.float32)
            radial=torch.exp(-.5*((distance[:,None]-centers)/(4.5/31))**2)
            angle=data.sp_angle.float() if self.arm=='S_RADANG' else torch.zeros_like(data.sp_angle,dtype=torch.float32)
            geometry=torch.cat((radial,angle[edges[1]]),-1)
        gate=torch.sigmoid(self.geometry(geometry))
        messages=gate*self.down(atoms)[edges[0]]
        aggregate=mean_pool(messages,edges[1],count)
        increments=self.up(aggregate)*(data.sp_degree>0)[:,None].to(states)
        return (increments[endpoints[0]]+increments[endpoints[1]])/12.


class SpatialGalformer(MaskedGalformer3D):
    """Native forward with one shared residual after each of the six blocks."""
    def forward(self,data,labels=None):
        za,zb=element_index(data.bond_z_a),element_index(data.bond_z_b)
        distance=data.bond_distance.float()
        if not torch.isfinite(distance).all() or (distance<=0).any():raise ValueError('invalid input distance')
        endpoints=self.endpoint(F.one_hot(za,101).float())+self.endpoint(F.one_hot(zb,101).float())
        radial=self.distance_projection(self.distance_basis(distance,torch.stack((za,zb),-1)))
        states=self.triplet(torch.cat((endpoints,radial),-1))
        types=triplet_type(data.bond_z_a,data.bond_z_b,data.bond_type)
        if labels is not None:
            action,donor=labels['token_action'],labels['token_donor']
            states=torch.where((action==2)[:,None],states[donor],states)
            types=torch.where(action==2,types[donor],types)
            states=torch.where((action==1)[:,None],self.mask_token[None].to(states),states)
            types=torch.where(action==1,torch.full_like(types,25857),types)
        bias=mean_pool(self.angle_bias(types,data.line_path,data.line_angle,data.line_mask),data.line_path_group,data.line_source.numel())
        real=(data.line_mask.sum(-1)==1)&~data.line_is_self
        angle=self.angle_bias.gaussian(data.line_angle[real,0],types[data.line_path[real,:2]])
        states=states+mean_pool(self.angle_value(angle),data.line_target[data.line_path_group[real]],len(states))
        for layer in self.layers:
            states=layer(states,data.line_source,data.line_target,bias)
            states=states+self.spatial(states,data)
        graph,available=incident_pool(states,data.incidence_index,data.canonical_graph_index,data.geometry_valid.numel())
        return states,graph,available&data.geometry_valid.bool()


class SpatialFuse(native.PaperFuse):
    def __init__(self,arm,pretraining=True,install=True):
        if arm not in ARMS:raise ValueError('unknown spatial arm')
        super().__init__('TMOL3M',pretraining)
        self.spatial_arm=arm
        if install:self.install_spatial()

    def install_spatial(self):
        if self.spatial_arm!='BASE':
            if hasattr(self.encoder3d,'spatial'):raise ValueError('already installed')
            # Change only behavior; preserve every native parameter and its RNG history.
            self.encoder3d.__class__=SpatialGalformer
            self.encoder3d.spatial=SpatialResidual(self.spatial_arm)


def load_initial(model,common,heads,modal):
    if hasattr(model.encoder3d,'spatial'):raise ValueError('load native strictly before spatial installation')
    result=native.load_initial(model,common,heads,modal)
    model.install_spatial()
    result.update(spatial_arm=model.spatial_arm,added_parameters=0 if model.spatial_arm=='BASE' else 37632,
                  spatial_seed=20261025,common_parameters_strict=True)
    return result


def validate_package(package):
    if (package.get('plan')!='UA3D-SPATIAL-20261025-01' or package.get('revision')!='r4'
        or package.get('schema')!=PT_SCHEMA or package.get('step')!=10000
        or package.get('config')!=CONFIG or package.get('arm') not in ARMS or package.get('fp_prediction') is not False):
        raise ValueError('spatial endpoint identity mismatch')
    if package.get('stream',{}).get('counters')!=[60000]*3:raise ValueError('incomplete modal RNG streams')


def initialize_ft(package,head):
    validate_package(package);model=SpatialFuse(package['arm'],False)
    wanted={k:v for k,v in package['state_dict'].items() if not k.startswith(('atom_head.','line_head.'))}
    expected={k for k in model.state_dict() if not k.startswith(('fusion.','head.'))}
    if set(wanted)!=expected:raise ValueError('explicit spatial PT/FT state set mismatch')
    for name in (*NATIVE,*native.modal_modules('TMOL3M').keys()):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in wanted.items() if k.startswith(prefix)})
    strict_module_state(model.head,head);return model


class Frozen3D(nn.Module):
    def __init__(self,package):
        super().__init__();validate_package(package)
        # Only instantiate the PT-native 3D path; no 2D, FP, gate or FULL state.
        with torch.random.fork_rng(devices=[]):
            self.encoder3d=MaskedGalformer3D(.1,attention_scale_denominator=512)
            if package['arm']!='BASE':
                self.encoder3d.__class__=SpatialGalformer
                self.encoder3d.spatial=SpatialResidual(package['arm'])
            self.norm3=nn.LayerNorm(512);self.A3=nn.Linear(512,256)
        for name in ('encoder3d','norm3','A3'):
            prefix=name+'.';strict_module_state(getattr(self,name),{k[len(prefix):]:v for k,v in package['state_dict'].items() if k.startswith(prefix)})
        self.requires_grad_(False);self.eval()

    def forward(self,data,rows,count):
        z=self.A3.weight.new_zeros((count,256))
        if data is not None:
            _,h,valid=self.encoder3d(data)
            if not valid.all():raise ValueError('invalid frozen 3D subbatch')
            z=z.index_copy(0,rows,self.A3(self.norm3(h)))
        return z
