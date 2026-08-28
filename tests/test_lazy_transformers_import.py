import ast
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_project_transformers_imports_are_not_module_level():
    for relative in ("src/dataset/dataset.py", "src/modules/uni_encoder.py"):
        path = Path(PROJECT_ROOT) / relative
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name.startswith("transformers") for name in names)


def test_graph_only_imports_do_not_construct_transformers_objects():
    code = """
import transformers

def fail(*args, **kwargs):
    raise AssertionError('graph-only import constructed a Transformers object')

transformers.AutoTokenizer.from_pretrained = fail
transformers.RobertaModel.from_pretrained = fail
import src.dataset
import src.modules
import src.training.pretrain.engine
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
