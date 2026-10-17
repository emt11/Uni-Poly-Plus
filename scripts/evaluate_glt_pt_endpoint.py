#!/usr/bin/env python3
"""Read-only post-hoc PT step5000 evaluation; never resume optimization."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLD = ROOT / 'results/glt_fusion_20261007/r1'
OUT = ROOT / 'results/glt_ft_generalization_20261008/r1'
PI = ROOT / 'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
ARMS = ('B0','T_RAW','T0','T0_A','K0','K_MD','K_MD_A','TK_MD','TK_MD_A','TK_MD_L','G_NORM','G_FGR')


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def restore_endpoint(model, checkpoint, config):
    if checkpoint.get('step') != 5000 or checkpoint.get('next_position') != 5000 * config['global_batch']:
        raise ValueError('endpoint checkpoint step/position mismatch')
    if checkpoint.get('identity', {}).get('config') != config:
        raise ValueError('endpoint checkpoint config mismatch')
    # Full PT state includes atom/geometry and enabled auxiliary heads.
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    model.requires_grad_(False)


def component_means(sums, counts):
    import torch
    if sums.shape != counts.shape or not torch.isfinite(sums).all() or not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError('invalid endpoint sums/counts')
    if bool(((counts == 0) & (sums != 0)).any()):
        raise ValueError('nonzero loss without valid graphs')
    return [float(s / c) if c else None for s, c in zip(sums.tolist(), counts.tolist())]


def endpoint_positions(start, rank):
    if start == 1008:
        return list(range(1008 + rank * 4, 1008 + (rank + 1) * 4))
    return list(range(start + rank * 84, start + (rank + 1) * 84))


def load_endpoint_subset(path):
    from src.training.glt_dual_runtime import load_sample_index_artifact
    subset = load_sample_index_artifact(path, 'fixed_validation')
    if not subset['direct_contract'] or len(subset['indices']) != 1024:
        raise ValueError('fixed validation must be original 1024 direct rows')
    return subset


def evaluate_arm(arm):
    import torch
    import torch.distributed as dist
    from src.modules.glt_dual_pretrain import DualPretrainer
    from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
    from src.training.glt_dual_runtime import open_source, load_sample_index_artifact, pretrain_source_fields, prepare_source_sample, move_labels, require_tmux
    require_tmux()
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('endpoint evaluation requires original four-rank pool')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    folder = OUT / 'pt_endpoint' / arm
    source = None
    presented = 0
    try:
        if rank == 0:
            folder.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        config = json.loads((OLD / 'initialization' / f'{arm}.json').read_text())
        checkpoint_path = OLD / 'pretrain' / arm / 'resume_05000.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        kwargs = {key: config[key] for key in ('fusion_mode','geometry_head_norm','third_task','opt_mode','fusion_revision','use_md200','atom_ce_input','align_enabled','local_enabled')}
        kwargs.update(geometry_scales=config.get('geometry_scales'), fgr_mu=config.get('fgr_mu',0.), fgr_sigma=config.get('fgr_sigma',1.))
        model = DualPretrainer(**kwargs)
        restore_endpoint(model, checkpoint, config)
        del checkpoint
        model.to(device)
        subset = load_endpoint_subset(config['sample_index_artifact'])
        source, _ = open_source(PI, PI, selected_indices=subset['indices'].tolist(), verify_key_digest=False, md200_root=config.get('md200_root'))
        if len(source) != 1024 or any(row['pretrain_role'] != 'validation' for row in source.entries):
            raise ValueError('endpoint source count/validation role differs')
        if rank == 0:
            write(folder / 'evaluation_spec.json', dict(arm=arm, checkpoint=str(checkpoint_path), config=config,
                disclosure='原训练过程中未执行PT validation，本次为事后补评；原时序未满足事实保留',
                source_indices=subset['indices'].tolist(), sample_keys=[k.hex() for k,_ in source.samples],
                seed=42, position='fixed list index 0..1023', mask_ratio=.3, noise_sigma=.03,
                rng='original arithmetic mask/noise; separate MD seed20261007+2000003*position and FGR RNG',
                md_view='original deterministic 0.5 PT channel mask retained', model_mode='eval/no_grad',
                amp='bf16 as parent PT', world=4, microbatch=84, pool_sizes=[336,336,336,16], padding=0,
                reduction='global sum/count of original per-graph losses; angle raw separately records valid-angle graphs',
                training_starts=0, optimizer_updates=0, checkpoint_selection=False))
        sums = torch.zeros(8, dtype=torch.float64, device=device)
        counts = torch.zeros_like(sums)
        targets = torch.zeros(4, dtype=torch.float64, device=device)
        fallback = 0
        with torch.no_grad():
            for start in range(0, 1024, 336):
                positions = endpoint_positions(start, rank)
                records = []
                for position in positions:
                    records.append(prepare_source_sample(source, position,seed=42,key=source.samples[position][0].hex(),position=position,
                        sigma=config['noise_sigma'],ratio=config['atom_mask_ratio'],
                        third_task=config['third_task'],fgr_mu=config.get('fgr_mu',0.),fgr_sigma=config.get('fgr_sigma',1.),
                        fgr_max_pairs=config.get('fgr_max_pairs',32),opt_mode=config['opt_mode']))
                data, labels = pretrain_collate(records)
                data = data.to(device)
                labels = move_labels(labels, device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model(data, labels)
                values = torch.cat([result['sums'],result['extra_sums'],result['raw_geometry_sums'],result['raw_geometry_sums'][1:2]])
                denominators = torch.cat([result['counts'],result['extra_counts'],result['counts'][1].repeat(2),result['angle_graphs'].reshape(1)])
                if not torch.isfinite(values).all():
                    raise FloatingPointError('nonfinite endpoint losses')
                sums += values.double()
                counts += denominators.double()
                targets += result['targets'].double()
                fallback += labels['fallback_count']
                presented += len(positions)
                write(folder / f'progress_rank{rank}.json', dict(presented=presented, optimizer_updates=0))
        for value in (sums, counts, targets):
            dist.all_reduce(value)
        stats = torch.tensor([presented, fallback], device=device, dtype=torch.long)
        dist.all_reduce(stats)
        if int(stats[0]) != 1024:
            raise ValueError('endpoint presentation budget mismatch')
        means = component_means(sums, counts)
        if rank == 0:
            names = ['atom_ce','geometry_objective','fgr','align','local','length_raw_mse','angle_raw_training_denominator','angle_raw_valid_angle_graphs']
            metrics = {name:dict(sum=float(sums[i]),count=int(counts[i]),mean=means[i]) for i,name in enumerate(names)}
            weights = [1.,1.,.1 if config['third_task']=='fgr' else 0.,.1 if config['align_enabled'] else 0.,.05 if config['local_enabled'] else 0.]
            write(folder/'metrics.json',dict(status='PASS',arm=arm,presented=1024,post_hoc=True,components=metrics,
                target_counts=dict(zip(['masked_atoms','center_bonds','angles','fgr_pairs'],targets.long().tolist())),
                fallback_graphs=int(stats[1]),endpoint_total_loss=sum(w*(m or 0.) for w,m in zip(weights,means)),
                total_loss_not_for_cross_recipe_ranking=True,training_starts=0,optimizer_updates=0))
    except BaseException as error:
        if folder.exists():
            write(folder/f'failure_rank{rank}.json',dict(error=repr(error),traceback=traceback.format_exc(),presented=presented,optimizer_updates=0))
        raise
    finally:
        if source is not None:
            source.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    evaluate_arm(parser.parse_args().arm)


def process_validation_positions(start,rank,world):
    if world not in (2,4) or rank not in range(world):raise ValueError('invalid process validation rank')
    count=min(84,(1024-start)//world)
    if count<=0:raise ValueError('invalid process validation position')
    return range(start+rank*count,start+(rank+1)*count)


def t3_validation_world(config):
    from scripts.run_fce_tune_t3_unit import validate_pt_config
    validate_pt_config(config)
    return 2


def t3b_validation_world(config):
    from scripts.run_fce_tune_t3b_unit import validate_pt_config
    validate_pt_config(config)
    return 2


def evaluate_process_validation(model, config, step, device, rank, world):
    """Exactly 1024 fixed presentations; leave model/RNG/training stream alone."""
    import torch
    import torch.distributed as dist
    from src.training.glt_dual_runtime import rng_state, restore_rng, open_source, pretrain_source_fields, prepare_source_sample, move_labels
    from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
    expected_world=2 if config.get('torsion_revision') else 4
    if config.get('ua2d_pt_revision'):
        from src.training.ua2d_pt20k import validate_config as validate_ua2d_pt
        validate_ua2d_pt(config)
        expected_world = 2
    if config.get('t3_revision'):expected_world=t3_validation_world(config)
    if config.get('t3b_revision'):expected_world=t3b_validation_world(config)
    if config.get('r10_revision'):
        from src.modules.glt_r10 import validate_pt_config
        validate_pt_config(config);expected_world=2
    if config.get('reference_revision'):
        from scripts.run_glt_pt_matched_reference_unit import validate_config
        validate_config(config)
        expected_world=2
    if config.get('torsion_revision') and not config.get('r10_revision'):
        from src.modules.glt_dual import s4_resource_options
        s4_resource_options(config)
    if world != expected_world or step not in (1000,2500,5000):raise ValueError('process validation budget mismatch')
    saved = rng_state(); training = model.training; source = None
    folder = Path(config['cycle_root'])/'pt_validation'/config['arm']/f'step{step:05d}'
    presented = 0
    try:
        if rank == 0:folder.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        subset=load_endpoint_subset(config['sample_index_artifact'])
        source,_=open_source(PI,PI,selected_indices=subset['indices'].tolist(),verify_key_digest=False,readonly_readahead=True)
        from src.training.glt_dual_runtime import apply_geometry_quality
        source=apply_geometry_quality(source,config.get('quality_root'))
        if config.get('qc_weight_root'):
            from src.dataset.glt_r10 import attach_qc_source
            source=attach_qc_source(source,config['qc_weight_root'])
        if len(source)!=1024 or any(row['pretrain_role']!='validation' for row in source.entries):
            raise ValueError('process validation source role mismatch')
        model.eval()
        sums=torch.zeros(5,dtype=torch.float64,device=device);counts=torch.zeros_like(sums)
        targets=torch.zeros(7,dtype=torch.float64,device=device)
        coverage=torch.zeros(6,dtype=torch.long,device=device)
        with torch.no_grad():
            for start in range(0,1024,84*world):
                records=[]
                for position in process_validation_positions(start,rank,world):
                    records.append(prepare_source_sample(source,position,seed=42,key=source.samples[position][0].hex(),position=position,
                        sigma=config['noise_sigma'],ratio=config['atom_mask_ratio'],
                        opt_mode=config['opt_mode'],geometry_mode=config['geometry_mode'],
                        torsion_mode=config.get('torsion_mode') if config.get('torsion_modules') else None,torsion_revision=config.get('torsion_revision'),nonbonded_revision=config.get('nonbonded_revision')))
                data,labels=pretrain_collate(records);data=data.to(device);labels=move_labels(labels,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):result=model(data,labels)
                values=torch.cat((result['sums'][:2],result['raw_geometry_sums'],result['extra_sums'][:1]))
                den=torch.stack((result['counts'][0],result['counts'][1],result['counts'][1],result['angle_graphs'],result['extra_counts'][0]))
                if not torch.isfinite(values).all():raise FloatingPointError('nonfinite process validation')
                sums+=values.double();counts+=den.double()
                targets+=torch.stack((result['targets'][0],result['targets'][1],result['targets'][2],
                    data.bond_length_weight.sum(),data.geometry_valid.sum(),
                    torch.tensor(data.periodic_keys.size(0) if config['geometry_mode']=='periodic_template' else 0,device=device),
                    torch.tensor(data.bond_distance.numel(),device=device))).double()
                if config.get('torsion_revision'):coverage+=data.torsion_counts.sum(0)
                presented+=len(records)
                write(folder/f'progress_rank{rank}.json',dict(presented=presented,training_updates=0))
        for value in (sums,counts,targets,coverage):dist.all_reduce(value)
        total=torch.tensor(presented,device=device);dist.all_reduce(total)
        if presented!=1024//world:raise ValueError('validation per-rank presentation mismatch')
        if int(total)!=1024:raise ValueError('validation presentation mismatch')
        if rank==0:
            write(folder/'metrics.json',dict(status='PASS',step=step,arm=config['arm'],presented=int(total),
                components={name:dict(sum=float(sums[i]),count=float(counts[i]) if config.get('qc_weight_root') else int(counts[i]),mean=component_means(sums,counts)[i]) for i,name in enumerate(('atom_ce','geometry','length_mse','angle_mse','align'))},
                targets=dict(zip(('masked_atoms','physical_length_targets','physical_angles','length_weight_denominator','valid_graphs','periodic_bonds','physical_bonds'),targets.tolist())),
                torsion_counts=coverage.tolist(),checkpoint_selection=False,padding=0,config=config))
    except BaseException as error:
        if folder.exists():write(folder/f'failure_rank{rank}.json',dict(error=repr(error),traceback=traceback.format_exc(),presented=presented))
        raise
    finally:
        if source is not None:source.close()
        model.train(training);restore_rng(saved)
