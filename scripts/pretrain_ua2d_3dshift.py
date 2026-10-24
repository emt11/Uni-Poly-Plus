"""One independent single-GPU UA2D-3DSHIFT pretraining branch (F_G10K or P_G)."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import signal
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import time
import traceback

import torch

from src.training.glt_dual_runtime import require_tmux, rng_state, save_checkpoint
from src.training.ua2d_3dshift import (ARMS, EXECUTION_ARMS, STEPS, CACHE, INITIALIZATION, OUT, PLAN, PT_SCHEMA, PSCALE,
                                       REVISION, SIDECAR, BASE, cache_references, now, pt_attention,process_identity,speed_reference,
                                       pt_learning_rate, pt_values, read, shift_objective,
                                       summarize_timing, write, save_recovery_checkpoint, CHECKPOINT_INTERVAL, atomic_recovery_checkpoint)
from src.dataset.ua2d_3dshift import PreparedShiftedMicrobatches, decode_batch, worker_init
from src.modules.ua2d_3dshift import (NEW_PREFIXES, UA2D3DShiftPretrainer, architecture_record,
                                      shift_deployment_package, shift_record,
                                      strict_load_shift_initialization)
from src.modules.ua2d_galformer_pretrain import PretrainConfig, assert_attention_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=ARMS)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise RuntimeError('the 3dshift branch must run on physical GPU2 only')
    if os.environ.get('WORLD_SIZE') != '1':
        raise RuntimeError('the 3dshift branch is single-process/world1')
    require_tmux()
    def interrupted(signum,frame):raise KeyboardInterrupt('PT stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    arm = args.arm
    if arm not in EXECUTION_ARMS:raise PermissionError('current user authorization permits F_G10K/P_G only')
    total_updates = STEPS[arm]
    output = OUT / 'pretrain' / arm
    output.mkdir(parents=True, exist_ok=False)
    context = mp.get_context('spawn')
    prepared_counts = context.Array('q', 4)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    started = time.monotonic()
    state = dict(plan=PLAN, revision=REVISION, arm=arm, status='RUNNING', pid=os.getpid(),
                 process=process_identity(os.getpid()), workers=[], attempted_updates=0, completed_updates=0,
                 consumed_presentations=0, sample_presentations=0, prefetched_not_consumed=0, elapsed_seconds=0.)
    iterator = None

    def consumption(status):
        state['status'] = status
        with prepared_counts.get_lock():state['sample_presentations'] = int(sum(prepared_counts[:]))
        state['prefetched_not_consumed'] = state['sample_presentations'] - state['consumed_presentations']
        state['elapsed_seconds'] = time.monotonic() - started
        state['time'] = now()
        write(output / 'consumption.json', state)
        return dict(state)

    try:
        references = cache_references()
        values = pt_values()
        config = PretrainConfig(**values)
        if (config.strategy != 'mask_align' or not config.fp_enabled
                or config.attention_scale_denominator != 512):
            raise ValueError('unexpected objective recipe')
        torch.manual_seed(42)
        model = UA2D3DShiftPretrainer(config, variant=arm)
        model.completed_updates = 0
        initialization = strict_load_shift_initialization(model, INITIALIZATION)
        assert_attention_scale(model)
        model.to(device)
        metadata = dict(plan=PLAN, revision=REVISION, arm=arm, variant=arm,
                        config=values, attention=pt_attention(), shift=shift_record(arm),
                        architecture=architecture_record(arm),
                        started=now(), pid=os.getpid(), argv=os.sys.argv, cwd=str(os.getcwd()),
                        seed=42, world_size=1, physical_gpus=[2], device='cuda:0',
                        microbatch=168, accumulation=6, global_batch=1008,
                        contrast_candidates_max=168, total_updates=total_updates,
                        initialization=str(INITIALIZATION),
                        common_step0=str(INITIALIZATION / 'common_step0.pt'),
                        mask_head=str(INITIALIZATION / 'task_heads/V1.pt'),
                        fp_seed=20261019, eligible_rows=910756, excluded_rows=635,
                        source=references['cache']['source'], index=references['cache']['index'],
                        cache=references['cache'], mapping=references['mapping'],
                        transfer=references['transfer'], sidecar=references['sidecar'],
                        t_init=initialization,
                        precision=dict(backbone='bf16_autocast', losses='fp32',
                                       optimizer_step='fp32_master_weights'),
                        objective_normalization=dict(world_size=1, accumulation=6,
                                                     window_counts='six_microbatch_global_counts',
                                                     second_acc6_division=False),
                        lr=dict(schedule='two_5k_warmup_cosine_cycles', warmup=500,
                                cycle_updates=5000, end_update=total_updates,
                                position='completed_updates % 5000', reset_optimizer=False,
                                peak=2e-4, end=1e-6, optimizer='AdamW', weight_decay=0.,
                                betas=[.9, .999], eps=1e-8, clip=1.),
                        objective_policy=config.objective_policy,
                        task_weights=model.weights, loss_weight_warmup_updates=config.weight_warmup,
                        align_direction_reduction='mean', align_multiplier=config.alignment_multiplier,
                        fp_schedule='constant 1 from update1; no warmup or decay',
                        align_schedule='constant 1 from update1; no warmup or decay', temperature=.1,
                        mask_ratio=.4, noise_sigma=.03, dropout=.1,
                        attention_denominator=512,
                        prediction_paths=['pure_2d'], outer_test='NOT_RUN',
                        engineering=dict(workers=4, prefetch=2, persistent=True,checkpoint_interval=1000,checkpoint_keep=2,
                                         timeout_seconds=21600, wallclock_limit_seconds=129600,
                                         resume='restart only; no automatic continuation'),
                        references=dict(FPDECAY_BASE=str(references['references']['BASE']),
                                        PTSCALE_P10K=str(references['references']['P10K']),
                                        FAST_R4_B_FP=str(references['references']['B_FP'])))
        write(output / 'config.json', metadata)
        from src.training.glt_dual_runtime import restore_rng
        restore_rng(torch.load(BASE.parent/'initial_training_rng.pt',map_location='cpu',weights_only=False))
        save_checkpoint(output / 'initial_training_rng.pt', rng_state())
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0., betas=(.9, .999),
                                      eps=1e-8)
        dataset = PreparedShiftedMicrobatches(CACHE, SIDECAR, PSCALE, config, variant=arm,
                                              prepared_counts=prepared_counts)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=4, multiprocessing_context=context,
            persistent_workers=True, prefetch_factor=2, worker_init_fn=worker_init,
            generator=torch.Generator().manual_seed(42))
        iterator = iter(loader)
        state['process'] = process_identity(os.getpid())
        state['workers'] = [process_identity(w.pid) for w in iterator._workers]
        consumption('RUNNING')
        attempted = completed = consumed = presented = 0
        all_timings, window_timings, windows = [], [], []
        reference_median = speed_reference()
        slow_windows = 0
        with (output / 'sampling.jsonl').open('x') as sampling, (output / 'step_timing.jsonl').open('x') as timing_log:
            for update in range(total_updates):
                tick = time.monotonic()
                model.completed_updates = update
                weights = model.weights
                attempted += 1
                state['attempted_updates']=attempted;consumption('RUNNING')
                loader_wait = decode_seconds = 0.
                batches, totals = [], {key: torch.zeros(()) for key in weights}
                for micro in range(6):
                    io_tick = time.monotonic()
                    payload = next(iterator)
                    loader_wait += time.monotonic() - io_tick
                    io_tick = time.monotonic()
                    data, labels, trace = decode_batch(payload)
                    decode_seconds += time.monotonic() - io_tick
                    if (trace['micro_index'] != update * 6 + micro
                            or trace['start_position'] != (update * 6 + micro) * 168):
                        raise ValueError('ordered global position mismatch')
                    if trace.get('variant') != arm:
                        raise ValueError('microbatch variant mismatch')
                    consumed += 168
                    state['consumed_presentations']=consumed
                    sampling.write(json.dumps(dict(step=update+1,micro=micro,**trace))+'\n')
                    counts = model.target_counts(data, labels)
                    for key in totals:
                        totals[key] = totals[key] + counts[key]
                    data = data.pin_memory()
                    labels = {k: v.pin_memory() if torch.is_tensor(v) else v for k, v in labels.items()}
                    batches.append((data, labels))
                sampling.flush()
                presented += 1008
                state.update(attempted_updates=attempted, consumed_presentations=consumed)
                consumption('RUNNING')
                names = list(totals)
                packed = torch.stack([totals[k] for k in names]).to(device, non_blocking=True)
                totals = {k: packed[i] for i, k in enumerate(names)}
                data_seconds = time.monotonic() - tick
                rate = pt_learning_rate(update)
                for group in optimizer.param_groups:
                    group['lr'] = rate
                optimizer.zero_grad(set_to_none=True)
                losses = {k: torch.zeros((), device=device) for k in names}
                diagnostics = []
                for data, labels in batches:
                    data = data.to(device, non_blocking=True)
                    labels = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                              for k, v in labels.items()}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model(data, labels)
                    with torch.autocast('cuda', enabled=False):
                        loss = shift_objective(result, weights, totals)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite pretraining objective')
                    loss.backward()
                    for key in names:
                        losses[key] = losses[key] + result['sums'][key].detach().float()
                    diagnostics.append(result['diagnostics'])
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                completed += 1
                raw = (torch.stack([losses[k] for k in names]) / packed.clamp_min(1)).tolist()
                elapsed = time.monotonic() - started
                seconds = time.monotonic() - tick
                measured = dict(step=completed, step_seconds=seconds, data_seconds=data_seconds,
                                loader_wait_seconds=loader_wait, decode_seconds=decode_seconds,
                                graphs_per_second=1008 / seconds, elapsed_seconds=elapsed,
                                compute_seconds=seconds - data_seconds,
                                projected_total_seconds=elapsed / completed * total_updates,
                                remaining_seconds=elapsed / completed * (total_updates - completed),
                                gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
                timing_log.write(json.dumps(measured) + '\n')
                timing_log.flush()
                all_timings.append(measured)
                window_timings.append(measured)
                write(output / 'timing.json', measured)
                state.update(completed_updates=completed)
                consumption('RUNNING')
                if completed % 500 == 0:
                    summary = summarize_timing(window_timings)
                    windows.append(dict(window=[completed - 499, completed],reference_median=reference_median, **summary))
                    with (output / 'windows.jsonl').open('a') as handle:
                        handle.write(json.dumps(windows[-1]) + '\n')
                    write(output / 'timing_summary.json',
                          dict(full=summarize_timing(all_timings), window=summary,
                               data_seconds_definition='includes target counts and pinning; not pure I/O; '
                                                       'no added CUDA synchronization'))
                    window_timings = []
                    if completed > 500:
                        slow = summary['step_seconds']['median'] > 1.5 * reference_median
                        slow_windows = slow_windows + 1 if slow else 0
                        if slow_windows >= 2:
                            raise RuntimeError('two consecutive 500-update windows exceed 1.5x the reference '
                                               'median; stop the whole cycle')
                raw_loss = dict(zip(names, raw))
                print(json.dumps(dict(step=completed, arm=arm, raw_losses=raw_loss,
                                      weighted_losses={k: v * weights[k] for k, v in raw_loss.items()},
                                      weights=weights, global_counts=dict(zip(names, packed.tolist())),
                                      lr=rate, grad_norm=float(norm), timing=measured,
                                      diagnostics=diagnostics)), flush=True)
                if completed % CHECKPOINT_INTERVAL == 0:
                    event=save_recovery_checkpoint(output,model,optimizer,loader,state,metadata)
                    state['last_checkpoint_step']=completed
                    state['checkpoint_seconds']=state.get('checkpoint_seconds',0.)+event['seconds']
                    state['checkpoint_retention']=event
                    consumption('RUNNING')
        if next(iterator, None) is not None:
            raise ValueError('unexpected extra microbatch after step 10000')
        iterator._shutdown_workers()
        iterator = None
        consumption('SAVING')
        save_checkpoint(output / 'rng.pt', rng_state())
        package = shift_deployment_package(model, total_updates)
        package.update(schema=PT_SCHEMA, plan=PLAN, revision=REVISION, arm=arm,
                       eligible_rows=910756, execution=metadata, consumption=state,
                       initialization_source=str(INITIALIZATION), trained_parent=False)
        atomic_recovery_checkpoint(output / 'deploy.pt', package)
        atomic_recovery_checkpoint(output / 'resume.pt',
                        dict(schema=PT_SCHEMA, plan=PLAN, revision=REVISION, arm=arm, step=total_updates,
                             model=model.state_dict(), optimizer=optimizer.state_dict(),
                             completed_updates=completed, attempted_updates=attempted,
                             schedule_position=completed, next_global_position=completed * 1008,
                             next_micro_index=completed * 6, loader_rng=loader.generator.get_state(),
                             rng=str(output / 'rng.pt'), lr_cycle_position=completed%5000,loss_warmup_position=completed,execution=metadata, consumption=state,
                             automatic_resume=False))
        consumption('COMPLETE')
        write(output / 'metadata.json',
              dict(plan=PLAN, revision=REVISION, arm=arm, variant=arm, schema=PT_SCHEMA,
                   status='COMPLETE', step=total_updates, world_size=1, microbatch=168, global_batch=1008,
                   deploy=str(output / 'deploy.pt'), resume=str(output / 'resume.pt'),
                   initialization_source=str(INITIALIZATION),
                   common_mask_head=str(INITIALIZATION / 'task_heads/V1.pt'),
                   fp_seed=20261019, seed=42, workers=state['workers'], consumption=state,
                   timing=dict(full=summarize_timing(all_timings), windows=windows,
                               total_wallclock_seconds=time.monotonic() - started,checkpoint_seconds=state.get('checkpoint_seconds',0.)),
                   time=now()))
        write(output / 'final_timing.json',
              dict(full=summarize_timing(all_timings), windows=windows,
                   total_wallclock_seconds=time.monotonic() - started,
                   checkpoint_seconds=state.get('checkpoint_seconds',0.),includes_initialization_checkpoint_and_worker_shutdown=True))
    except BaseException:
        if iterator is not None:
            iterator._shutdown_workers()
            iterator = None
        consumption('ABNORMAL_STOP')
        write(output / 'failure.json', dict(error=traceback.format_exc(), retry=False, time=now()))
        raise
    finally:
        if iterator is not None:
            iterator._shutdown_workers()


if __name__ == '__main__':
    main()
