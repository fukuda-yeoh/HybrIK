"""Datasets exported lazily.

Dataset modules import PyTorch3D and, for SMPL-X, load licensed body models.
Importing :mod:`hybrik.datasets` alone should not do either.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    'H36mSMPL': '.h36m_smpl',
    'HP3D': '.hp3d',
    'PW3D': '.pw3d',
    'MixDataset': '.mix_dataset',
    'AGORAX': '.agora_smplx',
    'MixDatasetCam': '.mix_dataset_cam',
    'MixDataset2Cam': '.mix_dataset2_cam',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(_import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
