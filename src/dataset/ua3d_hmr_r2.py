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
_AUDIT={}
REASONS={'EMPTY_SURFACE','GRID_CAP','MESH_CAP','NONMANIFOLD','NO_CENTRAL_AREA','EIGEN_NOT_CONVERGED','TIMEOUT','SPECTRAL_CAP'}

def finite(*arrays):
    if any(not np.isfinite(np.asarray(a)).all() for a in arrays):raise FloatingPointError('nonfinite geometry/spectrum/moments')
def sentinel(reason):
    if reason not in REASONS|{'SOURCE_MISSING_3D'}:raise ValueError('unregistered unavailable reason')
    return dict(available=False,reason=reason,lam=np.zeros(64),C=np.zeros((64,16)),a=np.zeros(64),A=1.,Ac=1.,stat=np.zeros(16),mode_mask=np.zeros(64,dtype=bool))
TAU=.15
RHO0=1.000001
TAIL_EPS=1e-10

def validate_center(center,z,ru,unmatched,canonical_z=None,base_id=None,canonical_base=None):
    raw=np.asarray(center)
    if raw.ndim!=1 or not np.issubdtype(raw.dtype,np.integer):raise ValueError('center mapping must be integer canonical vector')
    if np.any(raw < -1) or int(np.sum(raw==-1))!=unmatched:raise ValueError('center negative sentinel/unmatched count')
    valid=raw[raw>=0]
    if len(np.unique(valid))!=len(valid) or (len(valid) and (valid.max()>=len(z) or np.any(np.asarray(ru)[valid]!=0))):raise ValueError('center heavy representatives uniqueness/range/RU')
    if canonical_z is not None:
        cz=np.asarray(canonical_z)
        if len(cz)!=len(raw) or np.any(cz[raw<0]!=1) or not np.array_equal(np.asarray(z)[valid],cz[raw>=0]):raise ValueError('center element or isotope-H identity')
    if canonical_base is not None and (len(canonical_base)!=len(raw) or not np.array_equal(np.asarray(base_id)[valid],np.asarray(canonical_base)[raw>=0])):raise ValueError('center base-id identity')
    return valid.copy()

def grid_spec(pos,radii):
    tail=TAU*np.log(len(pos)/TAIL_EPS);extent=radii+tail;lo=np.min(pos-extent[:,None],axis=0);hi=np.max(pos+extent[:,None],axis=0)
    for step in range(11):
        h=.5+.05*step;origin=lo-2*h;shape=np.ceil((hi-lo)/h).astype(int)+5
        if int(np.prod(shape))<=2000000:return h,origin,shape
    raise Unavailable('GRID_CAP')

def density(points,pos,radii,gradient=False):
    points=np.asarray(points);out=np.zeros((len(points),3) if gradient else len(points));D=TAU*np.log(len(pos)/TAIL_EPS)
    for i in range(0,len(points),128):
        diff=points[i:i+128,None,:]-pos[None,:,:];dist=np.linalg.norm(diff,axis=-1);d=dist-radii[None,:];weight=np.where(d<=D,np.exp(-d/TAU),0.)
        if gradient:out[i:i+128]=np.sum(weight[:,:,None]*diff/np.maximum(dist[:,:,None],1e-30)/TAU,axis=1) # -grad rho
        else:out[i:i+128]=weight.sum(1)
    finite(out);return out

def component_labels(n,f):
    edge=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));adj=sp.coo_matrix((np.ones(len(edge)*2),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(n,n)).tocsr();return connected_components(adj,directed=False)

def mesh_surface(pos,radii):
    from skimage.measure import marching_cubes
    h,origin,shape=grid_spec(pos,radii);field=np.zeros(tuple(shape),dtype=np.float64);D=TAU*np.log(len(pos)/TAIL_EPS)
    for p,r in zip(pos,radii):
        low=np.maximum(0,np.floor((p-r-D-origin)/h).astype(int));high=np.minimum(shape,np.ceil((p+r+D-origin)/h).astype(int)+1)
        # Local atom boxes, sliced into x blocks: never voxels x all-atoms.
        xyz=[origin[j]+np.arange(low[j],high[j])*h-p[j] for j in range(3)]
        for start in range(0,len(xyz[0]),16):
            d=np.sqrt(xyz[0][start:start+16,None,None]**2+xyz[1][None,:,None]**2+xyz[2][None,None,:]**2)-r
            contribution=np.where(d<=D,np.exp(-d/TAU),0.);sl=(slice(low[0]+start,min(low[0]+start+16,high[0])),slice(low[1],high[1]),slice(low[2],high[2]));field[sl]+=contribution
    finite(field)
    if field.min()>=RHO0 or field.max()<=RHO0:raise Unavailable('EMPTY_SURFACE')
    v,f,_,_=marching_cubes(field,level=RHO0,spacing=(h,h,h),step_size=1,allow_degenerate=False,method='lewiner');v=v.astype(np.float64)+origin;f=f.astype(np.int64)
    input_faces=len(f);_,keep=np.unique(np.sort(f,axis=1),axis=0,return_index=True);f=f[np.sort(keep)];area2=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1);zero=int(np.sum(area2==0));f=f[area2>0]
    if not len(f):raise Unavailable('EMPTY_SURFACE')
    used=np.unique(f);remap=np.full(len(v),-1,dtype=int);remap[used]=np.arange(len(used));v=v[used];f=remap[f]
    _AUDIT.update(vertices=len(v),faces=len(f),grid_points=int(np.prod(shape)),h=h,tau=TAU,rho0=RHO0,tail_distance=D)
    if len(v)>16384:raise Unavailable('MESH_CAP')
    count,labels=component_labels(len(v),f);volumes=[];orientation=[]
    for component in range(count):
        ids=np.flatnonzero(labels[f[:,0]]==component);tri=v[f[ids]];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);outward=density(tri.mean(1),pos,radii,True);dot=float(np.einsum('ij,ij->',cross,outward));finite(dot)
        if dot==0:raise Unavailable('NONMANIFOLD')
        if dot<0:f[ids]=f[ids][:,[0,2,1]];tri=v[f[ids]];cross=-cross
        volumes.append(float(np.einsum('ij,ij->',tri[:,0],np.cross(tri[:,1],tri[:,2]))/6));orientation.append(abs(dot))
    return v,f,dict(tau=TAU,rho0=RHO0,tail_epsilon=TAIL_EPS,tail_distance=D,h=h,origin=origin.tolist(),grid_shape=shape.tolist(),grid_points=int(np.prod(shape)),vertices=len(v),faces=len(f),removed_duplicate_or_zero_faces=input_faces-len(f),strict_zero_faces=zero,components=count,component_signed_volumes=volumes,orientation_dot=orientation,volume=abs(float(sum(volumes))))
def operators(v,f):
    finite(v)
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);ue,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    _AUDIT.update(edges=len(ue),boundary_edges=int(np.sum(counts==1)),non_two_face_edges=int(np.sum(counts!=2)))
    if np.any(counts!=2):raise Unavailable('NONMANIFOLD')
    n=len(v);adj=sp.coo_matrix((np.ones(len(ue)*2),(np.r_[ue[:,0],ue[:,1]],np.r_[ue[:,1],ue[:,0]])),shape=(n,n)).tocsr()
    # All closed components are retained. No connectivity-based rejection.
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

def groups(lam):
    result=[];start=0
    for k in range(1,len(lam)):
        if abs(lam[k]-lam[k-1])>1e-5*max(abs(lam[k]),abs(lam[k-1]),1e-12):result.append((start,k));start=k
    if len(lam):result.append((start,len(lam)))
    return result

def spectrum(L,mass,labels=None):
    if labels is None:
        adj=L.copy();adj.setdiag(0);adj.eliminate_zeros();_,labels=connected_components(adj,directed=False)
    labels=np.asarray(labels);components=np.unique(labels)
    if len(components)>=64:raise Unavailable('SPECTRAL_CAP')
    n=len(mass);zero=[];positive=[];component_audit=[];maxerr=0.;maxorth=0.
    for component in components:
        ids=np.flatnonzero(labels==component);lc=L[ids][:,ids];mc=mass[ids];area=float(mc.sum());z=np.zeros(n);z[ids]=1/np.sqrt(area);zero.append(z)
        try:
            if len(ids)<=64:lam,phi=eigh(lc.toarray(),np.diag(mc));complete=True
            else:lam,phi=eigsh(lc,k=min(65,len(ids)-1),M=sp.diags(mc),sigma=-1e-5,which='LM',tol=1e-6,maxiter=5000,v0=np.linspace(1,2,len(ids)));complete=False
        except ArpackNoConvergence:raise Unavailable('EIGEN_NOT_CONVERGED')
        order=np.argsort(lam);lam=lam[order];phi=phi[:,order];finite(lam,phi)
        residual=lc@phi-mc[:,None]*phi*lam[None,:];denom=(float(np.max(np.asarray(abs(lc).sum(1))))+np.abs(lam)*mc.max())*np.linalg.norm(phi,axis=0);err=float(np.max(np.linalg.norm(residual,axis=0)/np.maximum(denom,1e-30)));orth=float(np.max(np.abs(phi.T@(mc[:,None]*phi)-np.eye(len(lam)))))
        if lam.min() < -1e-8 or err>1e-5 or orth>1e-5:raise Unavailable('EIGEN_NOT_CONVERGED')
        maxerr=max(maxerr,err);maxorth=max(maxorth,orth)
        if abs(lam[0])>1e-7:raise Unavailable('EIGEN_NOT_CONVERGED')
        # Solver's approximate nullvector is discarded, replaced by exact component constant.
        lam=np.maximum(lam[1:],0);phi=phi[:,1:];localgroups=groups(lam);safe=len(lam) if complete else localgroups[-1][0] if localgroups else 0
        for a,b in localgroups:
            if b<=safe:lam[a:b]=np.mean(lam[a:b])
        for k in range(safe):
            v=phi[:,k].copy();v-=np.dot(mc,v)/area;v/=np.sqrt(np.dot(mc,v*v));full=np.zeros(n);full[ids]=v;positive.append((float(lam[k]),full))
        component_audit.append(dict(component=int(component),vertices=len(ids),area=area,computed_modes=len(lam)+1,safe_positive_modes=safe,residual=err,mass_orthogonality=orth,zero_mode_explicit=True))
    positive.sort(key=lambda x:x[0]);plam=np.array([x[0] for x in positive]);capacity=64-len(zero);keep=0
    for a,b in groups(plam):
        if b>capacity:break
        plam[a:b]=np.mean(plam[a:b]);keep=b
    lam=np.r_[np.zeros(len(zero)),plam[:keep]];phi=np.stack(zero+[x[1] for x in positive[:keep]],axis=1);finite(lam,phi)
    orth=float(np.max(np.abs(phi.T@(mass[:,None]*phi)-np.eye(len(lam)))))
    residual=L@phi-mass[:,None]*phi*lam[None,:];denom=(float(np.max(np.asarray(abs(L).sum(1))))+np.abs(lam)*mass.max())*np.linalg.norm(phi,axis=0);err=float(np.max(np.linalg.norm(residual,axis=0)/np.maximum(denom,1e-30)))
    if orth>1e-5 or err>1e-5:raise Unavailable('EIGEN_NOT_CONVERGED')
    return lam,phi,dict(residual=err,mass_orthogonality=orth,retained_modes=len(lam),zero_modes=len(zero),components=len(zero),component_spectra=component_audit,component_areas=[float(mass[labels==i].sum()) for i in components],normalization='explicit mass-normalized component constants; positive eigenspaces and complete boundary groups')

def moments(lam,phi,mass,F,w):
    return phi.T@(mass[:,None]*F),phi.T@(mass*w)
def fixed_descriptor(s):
    lam=np.asarray(s['lam']);g=np.exp(-(lam[None,:]/4)**2-np.array([.25,1,4,16])[:,None]*lam[None,:])*s['mode_mask'][None,:]
    u=np.einsum('k,jk,kc->jc',s['a'],g,s['C'])/s['Ac'];q=np.einsum('jk,kc->jc',g,s['C']**2)/s['A']
    return np.concatenate([u,q],axis=1).ravel()
def surface(pos,z,center,radius_table):
    pos=np.asarray(pos,dtype=np.float64);z=np.asarray(z,dtype=int);center=np.asarray(center,dtype=int);finite(pos)
    if pos.shape!=(len(z),3) or np.any(center < -1):raise ValueError('geometry shape/center identity')
    center=center[center>=0]
    if not len(center):raise Unavailable('NO_CENTRAL_AREA')
    if len(np.unique(center))!=len(center) or center.max()>=len(z):raise ValueError('center heavy representatives')
    radii=np.array([radius_table[str(int(k))] for k in z]);v,f,audit=mesh_surface(pos,radii);L,mass,normals,H,K=operators(v,f);_,labels=component_labels(len(v),f);lam,phi,a=spectrum(L,mass,labels);audit.update(a);_AUDIT.update(audit)
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
    result=dict(available=True,reason=None,lam=np.pad(lam,(0,64-count)),C=np.pad(C,((0,64-count),(0,0))),a=np.pad(a,(0,64-count)),A=float(A),Ac=float(Ac),stat=stat,mode_mask=np.arange(64)<count,vertices=v,faces=f,mass=mass,phi=phi,normals=normals,H=H,K=K,owner=owners,component=labels,audit=audit)
    finite(result['lam'],C,a,stat,fixed_descriptor(result));return result

def surface_worker(item,out,radius_table):
    import resource
    from scripts.bootstrap_ua3d_hmr_r2 import write,identity
    torch.set_num_threads(1)
    _AUDIT.clear();started=time.time();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(Unavailable('TIMEOUT')));signal.alarm(180)
    path=Path(out)/item['id'];path.mkdir(parents=True,exist_ok=False)
    write(path/'process.json',dict(**identity(),started_unix=started,identity={k:v for k,v in item.items() if k not in ('pos','z','center')},threads=1,cwd=str(Path.cwd()),argv=__import__('sys').argv,CUDA_VISIBLE_DEVICES=__import__('os').environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        try:result=surface(item['pos'],item['z'],item['center'],radius_table)
        except Unavailable as e:result=sentinel(str(e));result['audit']=dict(_AUDIT)
        signal.alarm(0)
        if result['available']:
            np.savez_compressed(path/'audit.npz',**{k:result.pop(k) for k in ('vertices','faces','mass','phi','normals','H','K','owner','component')})
        np.savez(path/'moments.npz',**{k:result[k] for k in ('lam','C','a','A','Ac','stat','mode_mask')})
        summary=dict(available=result['available'],reason=result['reason'],audit=result.get('audit'),elapsed_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_identity={k:v for k,v in item.items() if k not in ('pos','z','center')})
        write(path/'status.json',summary);write(path/'end.json',dict(exit=0,started_unix=started,ended_unix=time.time()));return summary
    except BaseException:
        signal.alarm(0);write(path/'end.json',dict(exit=1,started_unix=started,ended_unix=time.time(),error=__import__('traceback').format_exc()));raise

def load_surface(root,task):
    with np.load(Path(root)/'surface/compact'/f'{task}.npz',allow_pickle=False) as x:return {k:torch.from_numpy(x[k].copy()).float() for k in x.files}
