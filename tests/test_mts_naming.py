import json
from pathlib import Path

from src.dataset.mips_trimer_contract import (
    CONFIG_SCHEMA,
    CHECKPOINT_SCHEMA,
    ROUTE_NAME,
    ROUTE_SHORT_NAME,
    STAGE1_ID,
    STAGE2_ID,
    normalize_route,
    normalize_stage,
)


def test_mts_public_route_and_stage_names():
    assert ROUTE_NAME == "MIPS-Trimer-SCAGE"
    assert ROUTE_SHORT_NAME == "MTS"
    assert normalize_route("MTS") == "mips_trimer_scage"
    assert normalize_route("MIPS-Trimer-SCAGE") == "mips_trimer_scage"
    assert normalize_stage("mts_topology") == STAGE1_ID
    assert normalize_stage("stage2_geometry_adapt") == STAGE2_ID


def test_mts_config_uses_new_identity_but_old_cache_schemas(tmp_path):
    config = json.loads(
        Path("configs/mts/default.json").read_text(encoding="utf-8")
    )
    assert config["schema_version"] == CONFIG_SCHEMA
    assert config["checkpoint_schema"] == CHECKPOINT_SCHEMA
    assert config["route"] == ROUTE_NAME
    assert config["route_short_name"] == ROUTE_SHORT_NAME
    assert config["stage1"]["id"] == STAGE1_ID
    assert config["stage2"]["id"] == STAGE2_ID
    assert config["feature_schema"].startswith("mips-trimer-scage-")
