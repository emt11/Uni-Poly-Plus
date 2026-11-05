"""Read-only scientific r3 operators, explicit r3 input/r4 BEST schema adapter."""
from src.modules import ua3d_cond_spectral_r3 as source
Model=source.Model
Surface=source.Surface
Gate4=source.Gate4
ARMS=source.ARMS
TASKS=source.TASKS
NATIVE=source.NATIVE
PLAN=source.PLAN
CONFIG=source.CONFIG  # Input architecture provenance stays r3, science_recipe r2.
FT_SCHEMA='ua3d-cond-spectral-best-r4'
INPUT_PT_SCHEMA='ua3d-cond-spectral-pretrain-r3'
INPUT_HEAD_SCHEMA='ua3d-cond-spectral-head-r3'
strict_module_state=source.strict_module_state
optimizer_ft=source.optimizer_ft

def initialize_ft(package,head,arm,fold):
    # Parent validator checks exact r3 schema/revision/science/config/10K stream.
    source.validate_package(package,10000)
    if package.get('fp_prediction') is not False:raise ValueError('FP prediction source mismatch')
    if package['schema']!=INPUT_PT_SCHEMA or head.get('schema')!=INPUT_HEAD_SCHEMA or head.get('revision')!='r3' or head.get('science_recipe')!='r2':raise ValueError('only registered current-campaign r3 PT/head accepted')
    return source.initialize_ft(package,head,arm,fold)
