"""HybrIK: Hybrid Analytical-Neural Inverse Kinematics.

The package is intentionally cheap to import: heavyweight modules (``torch``,
``pytorch3d``) are only loaded when the corresponding subpackage is used.

Data locations are resolved through :mod:`hybrik.paths`, so the package works
from any working directory and when installed as a dependency.
"""

from .paths import (
    config_dir,
    config_path,
    model_dir,
    model_path,
    set_config_dir,
    set_model_dir,
)

try:
    from importlib.metadata import version as _distribution_version

    __version__ = _distribution_version("hybrik")
except Exception:  # pragma: no cover - source checkout without installed metadata
    from .version import __version__

__all__ = [
    "__version__",
    "config_dir",
    "config_path",
    "model_dir",
    "model_path",
    "set_config_dir",
    "set_model_dir",
]
