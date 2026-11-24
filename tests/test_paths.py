import os
from pathlib import Path

import pytest

import hybrik


@pytest.fixture(autouse=True)
def _reset_model_dir():
    hybrik.set_model_dir(None)
    yield
    hybrik.set_model_dir(None)


def test_packaged_config_resolves():
    path = hybrik.config_path("smplx/256x192_hrnet_smplx_kid.yaml")
    assert os.path.isfile(path)
    assert path.startswith(str(hybrik.config_dir()))


def test_model_dir_env_override(tmp_path, monkeypatch):
    (tmp_path / "smplx").mkdir()
    (tmp_path / "J_regressor_h36m.npy").write_bytes(b"")
    monkeypatch.setenv("HYBRIK_MODEL_DIR", str(tmp_path))

    assert hybrik.model_dir() == tmp_path.resolve()
    assert hybrik.model_path("J_regressor_h36m.npy") == str(
        tmp_path / "J_regressor_h36m.npy"
    )
    assert hybrik.model_path("model_files/smplx") == str(tmp_path / "smplx")


def test_model_dir_override_clears(monkeypatch, tmp_path):
    monkeypatch.delenv("HYBRIK_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    hybrik.set_model_dir(tmp_path)
    assert hybrik.model_dir() == tmp_path.resolve()

    hybrik.set_model_dir(None)
    source_model_dir = Path(hybrik.__file__).resolve().parent.parent / "model_files"
    if source_model_dir.is_dir():
        pytest.skip("a model_files directory exists next to the package")
    with pytest.raises(FileNotFoundError):
        hybrik.model_dir()


def test_models_import_is_lazy():
    from hybrik.models import builder

    assert builder.build_sppe.__name__ == "build_sppe"
