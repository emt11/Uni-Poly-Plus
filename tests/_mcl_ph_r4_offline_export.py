#!/usr/bin/env python3
"""r4 run A: the export path on CPU, from the audited r3 checkpoint.

The r3 cat arm hung between ``resume_00002.pt`` and ``deploy_00002.pt``.  This
script reproduces *only that code path*, on CPU, with no process group, no
data loader and no training: it constructs the cat model, loads the encoder
state stored in the audited checkpoint, calls the production
``deployment_package`` (with the tensor-progress hook) and the production
``save_checkpoint``, then strict-loads the result with ``load_deployment`` and
compares every tensor against the checkpoint it came from.

Scope limits, stated here so the result is not over-read:

* a pass shows the export code itself completes offline on CPU; it does **not**
  show the distributed hang is fixed, and it cannot exercise the device-to-host
  copies that only exist when the tensors live on a GPU;
* the output is a **recovery export candidate**, not acceptance material for the
  r3 run, and it must not be used to start fine-tuning;
* the frozen r3 artifacts are read only; nothing next to them is written.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from scripts.pretrain_mcl_ph import StageLogger, save_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'results/mcl_ph_20260921/p1/pretrain_r3b/cat/resume_00002.pt'
OUTPUT_DIR = ROOT / 'results/mcl_ph_20260921/p1/r4_recovery'
OUTPUT = OUTPUT_DIR / 'deploy_00002_recovery_candidate.pt'
LOG = ROOT / 'logs/mcl_ph_20260921' / 'r4_offline_export.json'
ARM = 'cat'
STEP = 2


def main():
    started = time.perf_counter()
    problems = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stages = StageLogger(OUTPUT_DIR, 0, stall_seconds=60.0)
    stages.mark('offline_export', 'enter', checkpoint=str(CHECKPOINT.relative_to(ROOT)))

    blob = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    identity = blob['identity']
    if identity['fusion_mode'] != ARM:
        problems.append(f"the checkpoint belongs to {identity['fusion_mode']!r}")
    stages.mark('offline_export', 'checkpoint_loaded', step=int(blob['step']))

    from src.modules.mcl_ph import deployment_package, load_deployment
    from src.modules.mcl_ph_pretrain import MCLPHPretrainer
    model = MCLPHPretrainer(ARM, dropout=0.1, cutoffs=tuple(identity['cutoffs']),
                            router_dense_updates=int(identity['router_dense_updates']))
    missing, unexpected = model.load_state_dict(blob['model'], strict=False)
    if missing or unexpected:
        problems.append(f'checkpoint/model mismatch: missing={list(missing)[:3]} '
                        f'unexpected={list(unexpected)[:3]}')
    model.eval()
    stages.mark('offline_export', 'state_loaded')
    checkpoint_encoder = {name: value.detach().clone()
                          for name, value in model.encoder.state_dict().items()}

    source = {'checkpoint': str(CHECKPOINT.relative_to(ROOT)),
              'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              'step': int(blob['step']),
              'note': 'recovery export candidate produced by tests/_mcl_ph_r4_offline_export.py; '
                      'the r3 distributed run never completed its own export'}
    stages.mark('deployment_package', 'enter')
    package = deployment_package(model.encoder, int(blob['step']),
                                 cutoffs=tuple(identity['cutoffs']),
                                 router_dense_updates=int(identity['router_dense_updates']),
                                 router_top_k=int(identity['router_top_k']), source=source,
                                 progress=stages.tensor_progress)
    stages.mark('deployment_package', 'complete', tensors=len(package['state_dict']))
    stages.mark('deploy_save', 'enter', path=str(OUTPUT.relative_to(ROOT)))
    save_checkpoint(OUTPUT, package)
    stages.mark('deploy_save', 'complete')

    # Strict load through the production reader, then a tensor-by-tensor compare
    # against the encoder state the export started from.
    reloaded = MCLPHPretrainer(ARM, dropout=0.1, cutoffs=tuple(identity['cutoffs']),
                               router_dense_updates=int(identity['router_dense_updates']))
    loaded = torch.load(OUTPUT, map_location='cpu', weights_only=False)
    load_deployment(reloaded.encoder, loaded, expected_step=int(blob['step']),
                    expected_fusion=ARM)
    stages.mark('strict_load', 'complete',
                router_mode_after_load=reloaded.encoder.branch.router.mode,
                inference_mode=loaded['router']['inference_mode'])
    exported = loaded['state_dict']
    compared = sum(1 for name, value in checkpoint_encoder.items()
                   if torch.equal(exported[name], value))
    differing = sorted(name for name, value in checkpoint_encoder.items()
                       if not torch.equal(exported[name], value))
    if differing:
        problems.append(f'{len(differing)} exported tensors differ from the checkpoint: '
                        f'{differing[:3]}')
    if tuple(reloaded.encoder.state_dict()[name].shape for name in exported) != \
            tuple(exported[name].shape for name in exported):
        problems.append('a reloaded tensor changed shape')
    stages.mark('offline_export', 'complete', compared=compared)
    stages.close()

    payload = dict(
        status='PASS' if not problems else 'FAILED', problems=problems,
        role='recovery export candidate (NOT r3 acceptance material)',
        arm=ARM, step=int(blob['step']),
        checkpoint=source['checkpoint'], checkpoint_sha256=source['checkpoint_sha256'],
        output=str(OUTPUT.relative_to(ROOT)),
        output_bytes=OUTPUT.stat().st_size,
        output_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        exported_tensors=len(exported),
        tensors_identical_to_checkpoint=compared,
        strict_load={'expected_step': int(blob['step']), 'expected_fusion': ARM,
                     'inference_mode': loaded['router']['inference_mode'],
                     'router_mode_after_load': reloaded.encoder.branch.router.mode},
        wall_seconds=time.perf_counter() - started,
        not_verified=['that the distributed epilogue completes',
                      'device-to-host copies from CUDA tensors',
                      'that the r3 run may be treated as complete'],
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not problems else 4


if __name__ == '__main__':
    raise SystemExit(main())
