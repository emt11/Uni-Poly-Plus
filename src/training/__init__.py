"""Small, explicitly scoped training helpers.

The package intentionally does not provide a trainer abstraction.  Pretrain
and finetune keep their own loops and only share mechanical checkpoint/RNG
and runtime helpers.
"""
