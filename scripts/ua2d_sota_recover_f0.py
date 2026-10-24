"""One user-authorized F0 recovery; pause dispatch, drain FT, preserve all budgets."""
from ua2d_sota_dispatch import *
import signal
from src.training.glt_dual_runtime import require_tmux
require_tmux()
pid=125308
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
if 'python scripts/ua2d_sota_dispatch.py P_BLR P_HLR P_WD P_COS A T0' not in cmd: raise ValueError('controller identity changed')
birth=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
write(OUT/'recovery_authorization.json',dict(authorized=True,source='User explicit async reply: 授权 F0 恢复一次，保持原总预算',time=now(),
    retries=1,feature_limits=dict(units=70,graphs=47000),controller_pid=pid,controller_birth=birth))
os.kill(pid,signal.SIGSTOP)
write(OUT/'development/F0_recovery_dispatch.json',dict(status='DRAINING_FT',time=now(),controller_paused=pid))
try:
    while any(not (p.parent/'exit.txt').exists() for p in LOG.glob('ft_*/command.json')):
        refresh_budget();time.sleep(10)
    write(OUT/'development/F0_recovery_dispatch.json',dict(status='RUNNING',time=now(),controller_paused=pid))
    success=wait_exit(launch('kernel_F0_recovery1',['--unit','F0_recovery1'],2))
    write(OUT/'development/F0_recovery_dispatch.json',dict(status='PASS' if success else '阻断',time=now(),retry_remaining=0))
finally:
    current=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    if current!=birth: raise RuntimeError('controller PID reused; do not resume unknown process')
    os.kill(pid,signal.SIGCONT)
