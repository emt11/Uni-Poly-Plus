"""Continue the fixed independent PT/FT queue after initial P/A/T0 execution."""
from ua2d_sota_dispatch import *
from src.training.glt_dual_runtime import require_tmux
require_tmux()
try:
    after_initial()
    if not (OUT/'development/PT_blocked.json').exists(): integration_and_kernel()
except BaseException as error:
    write(OUT/'development/continuation_failure.json',dict(status='阻断',error=repr(error),traceback=__import__('traceback').format_exc(),time=now(),retry=False))
    raise
