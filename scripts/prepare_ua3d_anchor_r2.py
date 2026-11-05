#!/usr/bin/env python3
"""One registered frozen BASE feature-cache unit; trainval or post-freeze TEST."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def cache_unit(task,fold,formal,setup):
    import os
    import numpy as np
    import torch
    from src.training import ua3d_anchor_r2 as t
    from src.dataset import ua3d_anchor_r2 as d
    t.require_tmux(); t.guard(preparation=not formal); physical=t.gpu_contract()
    if formal and not t.read(t.OUT/'selection/final_frozen.json')['frozen']:
        raise ValueError('TEST teacher cache before freeze')
    kind='cache_test' if formal else 'cache_trainval'; prefix=kind
    folder=t.job_path(kind,task=task,fold=fold); folder.mkdir(parents=True,exist_ok=False)
    u=t.Consumption(folder,kind); u.add(prefix+'_units'); u.record(); ds=None
    try:
        t.seed_all(42+fold); model,source_state=t.load_base(task,fold); model.to('cuda:0'); model.eval()
        if any(p.requires_grad for p in model.parameters()) or any(x.training for x in model.modules()):
            raise ValueError('teacher BASE not frozen/eval')
        source=t.anchor(task,fold); scaler=source['scaler']
        ds=t.original_data.Data(task,fold,formal,t.DOWN,t.SPLIT,t.INPUT)
        if not formal:
            t.scaler_validate(ds.raw_targets,ds.train,scaler)
        ds.targets=t.transform(ds.raw_targets,scaler)
        t.write(folder/'environment.json',dict(versions=t.numerical.versions(),setup=setup,numerical=t.numerical.check_det(),physical=physical))
        rows=[]; predictions={}; partition_indices={}; byte_count=0
        def count_U(module,args):
            u.add('encoder_U_batches'); u.add('encoder_U_batches',completed=True)
        hook=model.A2.register_forward_pre_hook(count_U)
        partitions=[('test',ds.validation)] if formal else [('train',ds.train),('validation',ds.validation)]
        for partition,ids in partitions:
            truth=[]; pred=[]; indices=[]; keys=[]
            batches=t.original_data.loader(ds,ids)
            for batch_no,batch in enumerate(batches):
                t.guard(preparation=not formal)
                batch_ids=ids[batch_no*24:(batch_no+1)*24]
                batch=batch.to('cuda:0'); size=len(batch.y)
                u.add(prefix+'_batches'); u.add(prefix+'_rows',size); u.add('full_encoder_batches'); u.record()
                # Exactly one encoder call; H uses these same representations.
                with torch.inference_mode():
                    z,atoms,bonds=model.encode_modalities(batch)
                    u.add('frozen_H_batches'); fused,_=model.fuse(z,batch.availability); p=model.head(fused).view(-1)
                    if not torch.isfinite(p).all():
                        raise FloatingPointError('teacher prediction')
                    cpu_z=[x.cpu() for x in z]; cpu_atoms=atoms.cpu(); cpu_bonds=bonds.cpu() if bonds is not None else None
                    cpu_y=batch.y.cpu(); cpu_p=p.cpu(); cpu_available=batch.availability.cpu()
                    atom_graph=batch.two_d.canonical_graph_index.cpu()
                    three_rows=batch.three_d_rows.cpu().tolist()
                    bond_graph=batch.three_d.bond_batch.cpu() if batch.three_d is not None else None
                original_y=t.transform(cpu_y.numpy(),scaler,True); original_p=t.transform(cpu_p.numpy(),scaler,True)
                for local,i in enumerate(batch_ids):
                    row=ds.indices[i]; key=str(ds.keys[i]); sub=three_rows.index(local) if local in three_rows else None
                    # Clone outside inference_mode: on-disk features are ordinary FP32 tensors.
                    record=dict(schema=d.SCHEMA,task=task,fold=fold,row=row,key=key,partition=partition,
                        z=[x[local].detach().clone() for x in cpu_z],atoms=cpu_atoms[atom_graph==local].detach().clone(),
                        bonds=cpu_bonds[bond_graph==sub].detach().clone() if sub is not None else None,
                        availability=cpu_available[local].clone(),y=cpu_y[local].view(()).detach().clone(),y_base=cpu_p[local].view(()).detach().clone(),
                        y_base_original=float(original_p[local]),raw_target=float(ds.raw_targets[i]),source_best=source['best'])
                    path=folder/f'row{row}.pt'
                    with path.open('xb') as stream:
                        torch.save(record,stream)
                    byte_count+=path.stat().st_size
                    rows.append(dict(row=row,key=key,partition=partition,available=bool(cpu_available[local,1])))
                    indices.append(row); keys.append(key)
                truth.extend(original_y.tolist()); pred.extend(original_p.tolist())
                for k,amount in ((prefix+'_batches',1),(prefix+'_rows',size),('full_encoder_batches',1),('frozen_H_batches',1)):
                    u.add(k,amount,True)
                u.value['attempted']['cache_bytes']=byte_count; u.record()
            values=dict(indices=np.asarray(indices),sample_keys=np.asarray(keys),y_true=np.asarray(truth),y_pred=np.asarray(pred))
            np.savez(folder/f'{partition}_predictions.npz',**values)
            predictions[partition]=values; partition_indices[partition]=indices
        hook.remove()
        for k,v in model.state_dict().items():
            if not torch.equal(v.detach().cpu(),source_state[k]):
                raise ValueError('teacher frozen BASE changed')
        gate=None
        if not formal:
            current=predictions['validation']
            with np.load(source['validation_predictions'],allow_pickle=False) as ref:
                for key in ('indices','sample_keys','y_true'):
                    if not np.array_equal(current[key],ref[key]):
                        raise ValueError('BASE validation identity mismatch: '+key)
                if not np.allclose(current['y_pred'],ref['y_pred'],atol=t.SPEC['prediction_atol'],rtol=t.SPEC['prediction_rtol']):
                    raise ValueError('BASE validation predictions exceed fixed tolerance')
                new=t.metrics(current['y_true'],current['y_pred']); old=t.metrics(ref['y_true'],ref['y_pred'])
                gate=dict(identity_equal=True,prediction_allclose=True,r2_delta=new['r2']-old['r2'],new_metrics=new,reference_metrics=old,
                          max_abs_diff=float(np.max(np.abs(current['y_pred']-ref['y_pred']))),scaler=scaler)
                if abs(gate['r2_delta'])>t.SPEC['r2_tolerance']:
                    raise ValueError('BASE validation R2 consistency gate')
            t.write(folder/'validation_gate.json',gate)
        t.write(folder/'metadata.json',dict(schema=d.SCHEMA,status='COMPLETE',task=task,fold=fold,formal=formal,rows=rows,
            partitions=partition_indices,scaler=scaler,source_best=source['best'],source_schema=source['schema'],source_config=source['source_config'],
            source_best_epoch=source['best_epoch'],source_best_stage=source['best_stage'],encoder_passes=u.value['attempted']['full_encoder_batches'],
            features='ordinary detached finite FP32',validation_gate=gate))
        byte_count=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        u.value['attempted']['cache_bytes']=byte_count; u.value['completed']['cache_bytes']=byte_count
        u.add(prefix+'_units',completed=True); u.finish()
    except BaseException:
        u.fail(); raise
    finally:
        if ds is not None:
            ds.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--task',required=True); parser.add_argument('--fold',required=True,type=int); parser.add_argument('--formal',action='store_true')
    args=parser.parse_args()
    from src.modules.ua3d_stability import configure
    setup=configure('DET'); cache_unit(args.task,args.fold,args.formal,setup)
