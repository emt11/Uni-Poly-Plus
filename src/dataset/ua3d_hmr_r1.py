"""Label-free finite-trimer surfaces and compact, precomputed spectral moments."""
import math
import signal
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.linalg import eigh
import torch

class Unavailable(Exception):pass
REASONS={'EMPTY_SURFACE','GRID_CAP','MESH_CAP','NONMANIFOLD','DISCONNECTED','NO_CENTRAL_AREA','EIGEN_NOT_CONVERGED','TIMEOUT'}

def finite(*arrays):
    if any(not np.isfinite(np.asarray(a)).all() for a in arrays):raise FloatingPointError('nonfinite geometry/spectrum/moments')
def sentinel(reason):
    if reason not in REASONS|{'SOURCE_MISSING_3D'}:raise ValueError('unregistered unavailable reason')
    return dict(available=False,reason=reason,lam=np.zeros(64),C=np.zeros((64,16)),a=np.zeros(64),A=1.,Ac=1.,stat=np.zeros(16),mode_mask=np.zeros(64,dtype=bool))
def grid_spec(pos,radii):
    lo=np.min(pos-radii[:,None],axis=0);hi=np.max(pos+radii[:,None],axis=0)
    for step in range(11):
        h=.5+.05*step;origin=lo-2*h;shape=np.ceil((hi-lo)/h).astype(int)+5
        if int(np.prod(shape))<=2000000:return h,origin,shape
    raise Unavailable('GRID_CAP')
def mesh_surface(pos,radii):
    from skimage.measure import marching_cubes
    h,origin,shape=grid_spec(pos,radii);field=np.full(tuple(shape),2*h,dtype=np.float64)
    for p,r in zip(pos,radii):
        low=np.maximum(0,np.floor((p-r-2*h-origin)/h).astype(int));high=np.minimum(shape,np.ceil((p+r+2*h-origin)/h).astype(int)+1)
        xyz=[origin[j]+np.arange(low[j],high[j])*h-p[j] for j in range(3)]
        d=np.sqrt(xyz[0][:,None,None]**2+xyz[1][None,:,None]**2+xyz[2][None,None,:]**2)-r
        sl=tuple(slice(a,b) for a,b in zip(low,high));field[sl]=np.minimum(field[sl],d)
    finite(field)
    if field.min()>=0 or field.max()<=0:raise Unavailable('EMPTY_SURFACE')
    v,f,_,_=marching_cubes(field,level=0,spacing=(h,h,h),step_size=1,allow_degenerate=False,method='lewiner');v=v.astype(np.float64)+origin;f=f.astype(np.int64)
    twice=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1);f=f[twice>2e-10]
    if not len(f):raise Unavailable('EMPTY_SURFACE')
    used=np.unique(f);remap=np.full(len(v),-1,dtype=int);remap[used]=np.arange(len(used));v=v[used];f=remap[f]
    if len(v)>8192:raise Unavailable('MESH_CAP')
    vol=np.sum(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])))/6
    if vol<0:f=f[:,[0,2,1]];vol=-vol
    return v,f,dict(h=h,origin=origin.tolist(),grid_shape=shape.tolist(),grid_points=int(np.prod(shape)),volume=float(vol))
def operators(v,f):
    finite(v)
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);ue,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    if np.any(counts!=2):raise Unavailable('NONMANIFOLD')
    n=len(v);adj=sp.coo_matrix((np.ones(len(ue)*2),(np.r_[ue[:,0],ue[:,1]],np.r_[ue[:,1],ue[:,0]])),shape=(n,n)).tocsr()
    if connected_components(adj,directed=False,return_labels=False)!=1:raise Unavailable('DISCONNECTED')
    # Closed 2-manifold vertex links: each link must be one cycle, not two joined fans.
    link_edges=[[] for _ in range(n)]
    for x,y,z in f:
        link_edges[x].append((y,z));link_edges[y].append((z,x));link_edges[z].append((x,y))
    for links in link_edges:
        nb={}
        for a,b in links:nb.setdefault(a,[]).append(b);nb.setdefault(b,[]).append(a)
        if any(len(q)!=2 for q in nb.values()):raise Unavailable('NONMANIFOLD')
        seen=set();stack=[next(iter(nb))]
        while stack:
            a=stack.pop()
            if a not in seen:seen.add(a);stack.extend(nb[a])
        if len(seen)!=len(nb):raise Unavailable('NONMANIFOLD')
    p=v[f];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area2=np.linalg.norm(cross,axis=1);areas=area2/2
    mass=np.zeros(n);normal=np.zeros((n,3));angles=np.zeros(n);ii=[];jj=[];weights=[]
    for i in range(3):
        a=p[:,(i+1)%3]-p[:,i];b=p[:,(i+2)%3]-p[:,i];dot=np.einsum('ij,ij->i',a,b);cot=dot/area2
        angle=np.arctan2(area2,dot);np.add.at(angles,f[:,i],angle);np.add.at(mass,f[:,i],areas/3);np.add.at(normal,f[:,i],cross)
        ii.extend([f[:,(i+1)%3],f[:,(i+2)%3]]);jj.extend([f[:,(i+2)%3],f[:,(i+1)%3]]);weights.extend([cot/2,cot/2])
    W=sp.coo_matrix((np.concatenate(weights),(np.concatenate(ii),np.concatenate(jj))),shape=(n,n)).tocsr();L=sp.diags(np.asarray(W.sum(1)).ravel())-W
    if np.any(mass<=0):raise Unavailable('NONMANIFOLD')
    norm=np.linalg.norm(normal,axis=1)
    if np.any(norm==0):raise Unavailable('NONMANIFOLD')
    normal/=norm[:,None];H=np.einsum('ij,ij->i',L@v,normal)/(2*mass);K=(2*np.pi-angles)/mass
    finite(mass,normal,H,K,L.data)
    return L.tocsr(),mass,normal,H,K

def spectrum(L,mass):
    n=len(mass)
    try:
        if n<=64:lam,phi=eigh(L.toarray(),np.diag(mass))
        else:lam,phi=eigsh(L,k=64,M=sp.diags(mass),sigma=-1e-5,which='LM',tol=1e-6,maxiter=5000,v0=np.linspace(1,2,n))
    except ArpackNoConvergence:raise Unavailable('EIGEN_NOT_CONVERGED')
    order=np.argsort(lam);lam=lam[order];phi=phi[:,order];finite(lam,phi)
    residual=L@phi-mass[:,None]*phi*lam[None,:]
    denom=(np.max(np.asarray(abs(L).sum(1)))+np.abs(lam)*mass.max())*np.linalg.norm(phi,axis=0)
    err=float(np.max(np.linalg.norm(residual,axis=0)/np.maximum(denom,1e-30)));orth=float(np.max(np.abs(phi.T@(mass[:,None]*phi)-np.eye(len(lam)))))
    if lam.min() < -1e-8 or err>1e-5 or orth>1e-5:raise Unavailable('EIGEN_NOT_CONVERGED')
    lam=np.maximum(lam,0);groups=[];start=0
    for k in range(1,len(lam)):
        if abs(lam[k]-lam[k-1])>1e-5*max(abs(lam[k]),abs(lam[k-1]),1e-12):groups.append((start,k));start=k
    groups.append((start,len(lam)));keep=groups[-1][0]
    for a,b in groups[:-1]:lam[a:b]=np.mean(lam[a:b])
    if not keep:raise Unavailable('EIGEN_NOT_CONVERGED')
    return lam[:keep],phi[:,:keep],dict(residual=err,mass_orthogonality=orth,computed_modes=len(lam),retained_modes=keep,groups=groups,normalization='norm residual / ((L_inf + abs(lambda)*M_inf)*norm(phi)); group-average eigenvalues')

def moments(lam,phi,mass,F,w):
    return phi.T@(mass[:,None]*F),phi.T@(mass*w)
def fixed_descriptor(s):
    lam=np.asarray(s['lam']);g=np.exp(-(lam[None,:]/4)**2-np.array([.25,1,4,16])[:,None]*lam[None,:])*s['mode_mask'][None,:]
    u=np.einsum('k,jk,kc->jc',s['a'],g,s['C'])/s['Ac'];q=np.einsum('jk,kc->jc',g,s['C']**2)/s['A']
    return np.concatenate([u,q],axis=1).ravel()
def surface(pos,z,center,radius_table):
    pos=np.asarray(pos,dtype=np.float64);z=np.asarray(z,dtype=int);center=np.asarray(center,dtype=int);finite(pos)
    if pos.shape!=(len(z),3) or not len(z) or len(center)==0 or len(set(center))!=len(center) or min(center)<0 or max(center)>=len(z):raise ValueError('geometry shape/center identity')
    radii=np.array([radius_table[str(int(k))] for k in z]);v,f,audit=mesh_surface(pos,radii);L,mass,normals,H,K=operators(v,f);lam,phi,a=spectrum(L,mass);audit.update(a)
    # Work in blocks so vertex-owner assignment cannot allocate Nvertices*Nheavy*3 globally.
    owners=np.concatenate([np.argmin(np.linalg.norm(v[i:i+128,None,:]-pos[None,:,:],axis=-1)-radii[None,:],axis=1) for i in range(0,len(v),128)])
    w=np.isin(owners,center).astype(float);A=mass.sum();Ac=np.dot(mass,w)
    if Ac<=0:raise Unavailable('NO_CENTRAL_AREA')
    element=np.full(len(v),7,dtype=int)
    for i,items in enumerate(([6],[7],[8],[16],[15],[9],[17,35,53])):element[np.isin(z[owners],items)]=i
    onehot=np.eye(8)[element];F=np.c_[onehot,radii[owners],np.tanh(H),np.tanh(K),w]
    heat=np.exp(-lam[:,None]*np.array([.25,1,4,16])[None,:]);hks=phi**2@heat;hks=hks*A/heat.sum(0)[None,:];F=np.c_[F,np.log1p(hks)]
    stat=[np.log1p(A),np.log1p(Ac),np.log1p(abs(audit['volume'])),Ac/A]
    for field in (np.abs(H),K):
        mean=np.dot(mass,field)/A;stat.extend([mean,np.sqrt(np.dot(mass,(field-mean)**2)/A)])
    stat=np.r_[stat,((mass*w)[:,None]*onehot).sum(0)/Ac];C,a=moments(lam,phi,mass,F,w);count=len(lam)
    result=dict(available=True,reason=None,lam=np.pad(lam,(0,64-count)),C=np.pad(C,((0,64-count),(0,0))),a=np.pad(a,(0,64-count)),A=float(A),Ac=float(Ac),stat=stat,mode_mask=np.arange(64)<count,vertices=v,faces=f,mass=mass,phi=phi,normals=normals,H=H,K=K,owner=owners,audit=audit)
    finite(result['lam'],C,a,stat,fixed_descriptor(result));return result

def surface_worker(item,out,radius_table):
    import resource
    from scripts.bootstrap_ua3d_hmr_r1 import write,identity
    torch.set_num_threads(1)
    started=time.time();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(Unavailable('TIMEOUT')));signal.alarm(180)
    path=Path(out)/item['id'];path.mkdir(parents=True,exist_ok=False)
    write(path/'process.json',dict(**identity(),started_unix=started,identity={k:v for k,v in item.items() if k not in ('pos','z','center')},threads=1,cwd=str(Path.cwd()),argv=__import__('sys').argv,CUDA_VISIBLE_DEVICES=__import__('os').environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        try:result=surface(item['pos'],item['z'],item['center'],radius_table)
        except Unavailable as e:result=sentinel(str(e))
        signal.alarm(0)
        if result['available']:
            np.savez_compressed(path/'audit.npz',**{k:result.pop(k) for k in ('vertices','faces','mass','phi','normals','H','K','owner')})
        np.savez(path/'moments.npz',**{k:result[k] for k in ('lam','C','a','A','Ac','stat','mode_mask')})
        summary=dict(available=result['available'],reason=result['reason'],audit=result.get('audit'),elapsed_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_identity={k:v for k,v in item.items() if k not in ('pos','z','center')})
        write(path/'status.json',summary);write(path/'end.json',dict(exit=0,started_unix=started,ended_unix=time.time()));return summary
    except BaseException:
        signal.alarm(0);write(path/'end.json',dict(exit=1,started_unix=started,ended_unix=time.time(),error=__import__('traceback').format_exc()));raise

def load_surface(root,task):
    with np.load(Path(root)/'surface/compact'/f'{task}.npz',allow_pickle=False) as x:return {k:torch.from_numpy(x[k].copy()).float() for k in x.files}
