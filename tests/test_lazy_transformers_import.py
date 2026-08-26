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


def test_smiles_helpers_load_transformers_on_demand(monkeypatch):
    import transformers

    from src.dataset.dataset import _load_auto_tokenizer
    from src.modules.uni_encoder import _load_roberta_encoder

    tokenizer_sentinel = object()
    encoder_sentinel = object()
    tokenizer_calls = []
    encoder_calls = []

    def fake_tokenizer(name):
        tokenizer_calls.append(name)
        return tokenizer_sentinel

    def fake_encoder(name):
        encoder_calls.append(name)
        return encoder_sentinel

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake_tokenizer
    )
    monkeypatch.setattr(
        transformers.RobertaModel, "from_pretrained", fake_encoder
    )
    monkeypatch.setenv("UNIPOLY_MODEL_LOAD_LOG", "verbose")

    assert _load_auto_tokenizer("tokenizer-test") is tokenizer_sentinel
    assert _load_roberta_encoder("encoder-test") is encoder_sentinel
    assert tokenizer_calls == ["tokenizer-test"]
    assert encoder_calls == ["encoder-test"]
