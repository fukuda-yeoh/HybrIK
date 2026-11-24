"""Locate HybrIK data files.

When HybrIK is used from its own checkout, the data directories live next to
the package (``<repo>/model_files`` and ``<repo>/hybrik/configs``). When it is
installed as a dependency (wheel or git), the licensed model files are not
packaged and must be provided by the embedding project.

Resolution order for ``model_files``:

1. the ``HYBRIK_MODEL_DIR`` environment variable, or :func:`set_model_dir`
2. ``./model_files`` in the current working directory (repo scripts)
3. ``<package parent>/model_files`` (editable/source installs)

Only the configuration files are packaged; they live in ``hybrik/configs``.
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR_ENV = "HYBRIK_MODEL_DIR"
CONFIG_DIR_ENV = "HYBRIK_CONFIG_DIR"

_PACKAGE_DIR = Path(__file__).resolve().parent
_model_dir_override: Path | None = None
_config_dir_override: Path | None = None


def _as_dir(path: Path, kind: str, env_var: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(
            f"HybrIK {kind} directory not found: {path}\n"
            f"Set {env_var} to the directory that contains the {kind} data."
        )
    return path


def set_model_dir(path: str | os.PathLike[str] | None) -> None:
    """Point HybrIK at a model directory for this process.

    Equivalent to setting ``HYBRIK_MODEL_DIR``. Pass ``None`` to clear the
    override and fall back to automatic discovery.
    """
    global _model_dir_override
    _model_dir_override = (
        None if path is None else Path(path).expanduser().resolve()
    )


def set_config_dir(path: str | os.PathLike[str] | None) -> None:
    """Override the packaged configuration directory for this process."""
    global _config_dir_override
    _config_dir_override = (
        None if path is None else Path(path).expanduser().resolve()
    )


def model_dir() -> Path:
    """Return the directory containing ``J_regressor_h36m.npy``, ``smplx/``, etc."""
    if _model_dir_override is not None:
        return _as_dir(_model_dir_override, "model", MODEL_DIR_ENV)

    env = os.environ.get(MODEL_DIR_ENV)
    if env:
        return _as_dir(
            Path(env).expanduser().resolve(), "model", MODEL_DIR_ENV
        )

    for candidate in (Path.cwd() / "model_files", _PACKAGE_DIR.parent / "model_files"):
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "HybrIK model directory not found.\n"
        f"Set {MODEL_DIR_ENV} to the directory containing model_files "
        "(J_regressor_h36m.npy, basicModel_neutral_lbs_10_207_0_v1.0.0.pkl, "
        "h36m_mean_beta.npy, smplx/, ...)."
    )


def model_path(*parts: str | os.PathLike[str]) -> str:
    """Resolve a path inside :func:`model_dir`.

    Accepts either component names or a single path such as
    ``"model_files/smplx/SMPLX_NEUTRAL.npz"``.
    """
    if len(parts) == 1:
        path = Path(parts[0])
        if path.parts and path.parts[0] == "model_files":
            parts = path.parts[1:]
        else:
            parts = (path,)
    return str(model_dir().joinpath(*parts))


def config_dir() -> Path:
    """Return the directory containing the packaged HybrIK YAML configs."""
    if _config_dir_override is not None:
        return _as_dir(_config_dir_override, "config", CONFIG_DIR_ENV)

    env = os.environ.get(CONFIG_DIR_ENV)
    if env:
        return _as_dir(
            Path(env).expanduser().resolve(), "config", CONFIG_DIR_ENV
        )

    return _as_dir(_PACKAGE_DIR / "configs", "config", CONFIG_DIR_ENV)


def config_path(name: str | os.PathLike[str]) -> str:
    """Resolve a packaged config name such as ``"256x192_..._3dhp.yaml"``.

    Absolute paths and names prefixed with ``configs/`` are also accepted.
    """
    path = Path(name)
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(f"HybrIK config not found: {path}")
        return str(path)

    if path.parts and path.parts[0] == "configs":
        path = Path(*path.parts[1:])

    resolved = config_dir() / path
    if not resolved.is_file():
        raise FileNotFoundError(
            f"HybrIK config not found: {resolved}\n"
            f"Available configs are under {config_dir()}."
        )
    return str(resolved)
