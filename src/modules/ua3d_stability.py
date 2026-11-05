"""Strict numerical execution and exact evidence comparison; no scientific operator replacements."""
import copy
import importlib.metadata
import os
import platform
import random
import subprocess
import numpy as np
import torch
from src.modules.ua3d_base_opt_r6 import (BaseOptFuse,CONFIG,DEFAULT_FT,NATIVE,TASKS,native,
    initialize_ft,optimizer_ft,strict_module_state,pt_config,validate_package)

DET_ENV=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='20261028')

def numerical_state():
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
        cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        env={k:os.environ.get(k) for k in (*DET_ENV,'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','WORLD_SIZE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})

def check_det():
    s=numerical_state()
    expected=dict(deterministic=True,warn_only=False,cudnn_benchmark=False,cudnn_deterministic=True,matmul_tf32=False,cudnn_tf32=False,matmul_precision='highest',fill_uninitialized_memory=True,cpu_threads=2,interop_threads=1)
    if any(s[k]!=v for k,v in expected.items()) or any(os.environ.get(k)!=v for k,v in DET_ENV.items()):raise ValueError('strict numerical policy mismatch')
    return s

def configure(policy):
    if torch.cuda.is_initialized():raise RuntimeError('numerical policy must precede CUDA initialization')
    torch.set_num_threads(2)
    before=numerical_state()
    if policy=='DET':
        if any(os.environ.get(k)!=v for k,v in DET_ENV.items()):raise ValueError('DET environment must precede Python startup')
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.utils.deterministic.fill_uninitialized_memory=True
        check_det()
    elif policy!='DEFAULT':raise ValueError('unregistered numerical policy')
    return dict(policy=policy,configured_before_cuda=True,inherited=before,actual=numerical_state())

def versions():
    packages={}
    for package in ('torch-geometric','torch-scatter','numpy'):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    return dict(python=platform.python_version(),pytorch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),packages=packages,
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,driver_version','--format=csv,noheader'],text=True,timeout=10))

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])

def clone_tree(value):
    if torch.is_tensor(value):
        if value.is_floating_point() and not torch.isfinite(value).all():raise FloatingPointError('nonfinite recorded tensor')
        return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:clone_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [clone_tree(x) for x in value]
    if isinstance(value,tuple):return tuple(clone_tree(x) for x in value)
    return copy.deepcopy(value)

def exact_compare(left,right):
    """No hashing/allclose. Report the first mismatch and raw maximum absolute difference."""
    result=dict(equal=True,first_mismatch=None,max_abs_diff=0.,compared_leaves=0)
    def mismatch(path,reason,diff=None):
        result['equal']=False
        if result['first_mismatch'] is None:result['first_mismatch']=dict(path=path,reason=reason,max_abs_diff=diff)
        if diff is not None:result['max_abs_diff']=max(result['max_abs_diff'],float(diff))
    def walk(a,b,path):
        if torch.is_tensor(a) or torch.is_tensor(b):
            result['compared_leaves']+=1
            if not(torch.is_tensor(a) and torch.is_tensor(b)):mismatch(path,'tensor type mismatch');return
            if a.shape!=b.shape or a.dtype!=b.dtype:mismatch(path,'tensor shape/dtype mismatch');return
            if not torch.equal(a,b):
                diff=float((a.to(torch.float64)-b.to(torch.float64)).abs().max()) if a.numel() else 0.
                mismatch(path,'torch.equal false',diff)
        elif isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
            result['compared_leaves']+=1
            if not(isinstance(a,np.ndarray) and isinstance(b,np.ndarray)) or a.shape!=b.shape or a.dtype!=b.dtype:mismatch(path,'array shape/dtype mismatch');return
            if not np.array_equal(a,b):
                diff=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.dtype.kind in 'biuf' and a.size else None
                mismatch(path,'array_equal false',diff)
        elif isinstance(a,dict) or isinstance(b,dict):
            if not(isinstance(a,dict) and isinstance(b,dict)) or set(a)!=set(b):mismatch(path,'dictionary keys mismatch');return
            for k in a:walk(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(tuple,list)) or isinstance(b,(tuple,list)):
            if type(a)!=type(b) or len(a)!=len(b):mismatch(path,'sequence type/length mismatch');return
            for i,(x,y) in enumerate(zip(a,b)):
                label=str(i)+(':'+x['module'] if isinstance(x,dict) and isinstance(x.get('module'),str) else '')
                walk(x,y,path+'/'+label)
        else:
            result['compared_leaves']+=1
            if type(a)!=type(b) or a!=b:
                diff=abs(float(a)-float(b)) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None
                mismatch(path,'scalar mismatch',diff)
    walk(left,right,'root')
    return result

class Hooks:
    """Capture only outputs of already budgeted forwards; never call a module."""
    def __init__(self,model):
        self.handles=[];self.records=[];self.enabled=True
        names=('o8.layers.0.attention','encoder3d.layers.0.attention','norm2','norm3','A2','A3','fp_encoder','fusion','head')
        lookup=dict(model.named_modules())
        for name in names:
            if name not in lookup:raise ValueError('registered diagnostic hook missing '+name)
            def hook(module,args,out,name=name):
                if self.enabled:self.records.append(dict(module=name,output=clone_tree(out)))
            self.handles.append(lookup[name].register_forward_hook(hook))
    def begin(self):self.records=[]
    def take(self):return self.records
    def close(self):
        for h in self.handles:h.remove()
