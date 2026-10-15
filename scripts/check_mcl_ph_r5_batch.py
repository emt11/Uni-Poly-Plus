"""Exactly one Xc train batch forward/backward, no optimizer or test access."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import finetune_mcl_ph as ft
from scripts.audit_mcl_ph_r5 import OUT,SOURCE,SPLITS,PACKS,CONFIGS,tensors_equal,R4,read
from scripts.run_mcl_ph_r4_fast import atomic_json,timestamp
from src.training.glt_dual_runtime import require_tmux,open_source
from src.utils import scale_targets,set_global_seed

def main():
    require_tmux();assert read(OUT/'a0/report.json')['status']=='PASS'
    manifest=read(SPLITS/'xc.json');train=manifest['folds'][0]['train_indices']
    source,frame=open_source(SOURCE,SOURCE,task='xc',selected_indices=sorted(train),
        dual_static_root=SOURCE,expected_task_rows=432,record_index_path=SOURCE/'xc_offsets.npy',split_manifest=manifest)
    try:
        set_global_seed(42);package=torch.load(PACKS['xc_standard'],map_location='cpu',weights_only=False)
        model=ft.GLTReferenceArm(torsion=bool(package.get('torsion_modules',False)))
        ft.load_dual_deployment(model.model,package,5000)
        tensors_equal(ft.periodic_tdl_head(model).state_dict(),torch.load(R4/'paper5_outer/glt_ref/xc/fold0/head_initial.pt',weights_only=False))
        ft.configure_periodic_tdl_dropout(model,'xc');model.to('cuda');model.train()
        labels=frame['label'].to_numpy(dtype=np.float64);dataset=ft.CleanDualDataset(source,labels)
        scaler=scale_targets(dataset,'xc',train_indices=list(range(len(dataset))),transform_mode='standard')
        indices=list(range(24));batch=ft.dual_glt_collate([dataset[i] for i in indices]).to('cuda')
        prediction,_=ft.AdaptationAdapter(model)(batch);loss=torch.nn.functional.mse_loss(prediction.reshape(-1),batch.y.reshape(-1))
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        np.testing.assert_allclose(scaler.inverse_transform(dataset.targets).reshape(-1),labels,rtol=1e-12,atol=1e-12)
        output=OUT/'batch_check';output.mkdir(parents=True,exist_ok=False)
        atomic_json(output/'report.json',dict(status='PASS',checked_at=timestamp(),task='xc',fold=0,
            train_source_indices=[sorted(train)[i] for i in indices],graphs=24,forwards=1,backwards=1,
            optimizer_updates=0,test_access=False,target_transform='standard',loss=float(loss.detach().cpu()),
            initial_head_exact=True,scaler_fit_rows=len(train),all_gradients_finite=True))
        print('PASS: one Xc train batch, zero updates, finite gradients',flush=True)
    finally: source.close()

if __name__=='__main__':main()
