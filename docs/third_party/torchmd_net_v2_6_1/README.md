# TorchMD-Net v2.6.1 source attribution

Source: https://github.com/torchmd/torchmd-net/tree/v2.6.1

- `torchmd_et.py.txt`: original `torchmdnet/models/torchmd_et.py` at tag v2.6.1.
- `utils.py.txt`: original `torchmdnet/models/utils.py` at tag v2.6.1.
- `LICENSE`: the upstream MIT license, retained with the local adaptation.

`src/modules/ua2d_et_single.py` adapts the scalar/vector edge operators with local PyTorch operations. No TorchMD-Net package or compiled extension is installed or imported. Project-specific changes are the explicit uncapped 4 Å/bond/self union; a single shared chemical/self cutoff gate in scalar, vector, and neighbor paths; self messages in neighbor embedding; 102 corrupted element categories; fixed Gaussian RBF; and the project pretraining decoder/readout. These changes are not an unchanged upstream model or a claim of numerical identity to the original implementation.
