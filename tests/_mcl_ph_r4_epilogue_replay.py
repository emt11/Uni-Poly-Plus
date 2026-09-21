#!/usr/bin/env python3
"""r4 run B: the r3 export epilogue on four GPUs, with zero training updates.

The r3 cat arm completed two optimizer updates, wrote ``resume_00002.pt`` and
then stalled before ``deploy_00002.pt``: rank 0 sat in a host wait while ranks
1-3 spun in a collective.  Run A showed the export *code* completes offline on
CPU, which leaves the distributed, on-device and worker-related parts of that
window untested -- this script replays exactly those, from the audited
checkpoint, with no forward, no backward and no optimizer step.

Replayed (production functions, not a parallel "happy path"): the device choice
and process group of the runner, the model built at the checkpoint's declared
settings, the DDP wrap with ``find_unused_parameters=True``, ``rng_state()`` plus
``all_gather_object``, rank 0's ``deployment_package`` over CUDA tensors (the
device-to-host copies that run A could not exercise), ``save_checkpoint`` and the
closing ``barrier``.

Not reproduced, and therefore not exonerated by this run: the two training
steps' optimizer and reducer state, the RNG stream consumed by training, the
DataLoader and its twelve preparation workers per rank, and the autocast/AMP
state.  A run that completes here shows the epilogue does not deadlock *in this
zero-update configuration*; it does not show the r3 hang is fixed.

Forensic settings that are deliberately not production's: a finite process-group
timeout, a stage stall watchdog, and an external ``timeout`` around the process.
They make a stall visible instead of endless; none of them is a fix.
"""
import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402

from scripts.pretrain_mcl_ph import StageLogger, save_checkpoint  # noqa: E402
from src.modules.mcl_ph import deployment_package, load_deployment  # noqa: E402
from src.modules.mcl_ph_pretrain import MCLPHPretrainer  # noqa: E402
from src.training.glt_dual_runtime import rng_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'results/mcl_ph_20260921/p1/pretrain_r3b/cat/resume_00002.pt'
REFERENCE = ROOT / 'results/mcl_ph_20260921/p1/r4_recovery/deploy_00002_recovery_candidate.pt'
OUTPUT_DIR = ROOT / 'results/mcl_ph_20260921/p1/r4_recovery'
REPLAY = OUTPUT_DIR / 'deploy_00002_epilogue_replay.pt'
ARM = 'cat'
PG_TIMEOUT_SECONDS = 90


def main():
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    stages = StageLogger(OUTPUT_DIR, rank, stall_seconds=60.0)
    stages.mark('epilogue_replay', 'enter', world=world)
    problems = []
    if world != 4:
        raise SystemExit(f'the replay is declared for four ranks, got {world}')

    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    stages.mark('process_group', 'enter', backend='nccl', timeout_seconds=PG_TIMEOUT_SECONDS)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=PG_TIMEOUT_SECONDS))
    stages.mark('process_group', 'complete')

    blob = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    identity = blob['identity']
    model = MCLPHPretrainer(ARM, dropout=0.1, cutoffs=tuple(identity['cutoffs']),
                            router_dense_updates=int(identity['router_dense_updates']))
    model.to(device)
    model.load_state_dict(blob['model'], strict=True)
    stages.mark('state_loaded', 'complete', device=str(device))
    wrapped = DistributedDataParallel(model, device_ids=[device.index],
                                      find_unused_parameters=True)
    stages.mark('ddp_wrapped', 'complete')

    stages.mark('rng_gather', 'enter')
    states = [None] * world
    dist.all_gather_object(states, rng_state())
    stages.mark('rng_gather', 'complete', entries=len(states))

    if rank == 0:
        stages.mark('deployment_package', 'enter')
        package = deployment_package(
            wrapped.module.encoder, int(blob['step']), cutoffs=tuple(identity['cutoffs']),
            router_dense_updates=int(identity['router_dense_updates']),
            router_top_k=int(identity['router_top_k']),
            source={'checkpoint': str(CHECKPOINT.relative_to(ROOT)),
                    'role': 'epilogue replay, zero updates, r4 run B'},
            progress=stages.tensor_progress)
        stages.mark('deployment_package', 'complete', tensors=len(package['state_dict']))
        stages.mark('deploy_save', 'enter')
        save_checkpoint(REPLAY, package)
        stages.mark('deploy_save', 'complete')
    stages.mark('barrier', 'enter')
    dist.barrier()
    stages.mark('barrier', 'complete')

    if rank == 0 and REFERENCE.is_file():
        reference = torch.load(REFERENCE, map_location='cpu', weights_only=False)
        replay = torch.load(REPLAY, map_location='cpu', weights_only=False)
        same_keys = set(reference['state_dict']) == set(replay['state_dict'])
        differing = (sorted(name for name in replay['state_dict']
                            if not torch.equal(replay['state_dict'][name],
                                               reference['state_dict'][name]))
                     if same_keys else ['<key sets differ>'])
        reloaded = MCLPHPretrainer(ARM, dropout=0.1, cutoffs=tuple(identity['cutoffs']),
                                   router_dense_updates=int(identity['router_dense_updates']))
        load_deployment(reloaded.encoder, replay, expected_step=int(blob['step']),
                        expected_fusion=ARM)
        if differing:
            problems.append(f'{len(differing)} tensors differ from the CPU export')
        stages.mark('replay_verified', 'complete', tensors=len(replay['state_dict']),
                    identical_to_cpu_export=len(replay['state_dict']) - len(differing))

    stages.mark('process_group', 'enter_destroy')
    dist.destroy_process_group()
    stages.mark('epilogue_replay', 'complete')
    stages.close()
    record = {'rank': rank, 'world_size': world, 'status': 'PASS' if not problems else 'FAILED',
              'problems': problems, 'replayed': ['process group', 'state load', 'DDP wrap',
                                                 'rng gather', 'deployment package',
                                                 'checkpoint save', 'barrier'],
              'not_replayed': ['two training steps', 'optimizer and reducer state',
                               'training RNG stream', 'dataloader and prep workers',
                               'autocast state']}
    (OUTPUT_DIR / f'epilogue_replay_rank{rank}.json').write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if not problems else 4


if __name__ == '__main__':
    raise SystemExit(main())
