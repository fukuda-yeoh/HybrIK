"""Presets exported lazily.

The SMPL-X preset imports PyTorch3D, so importing this package alone must not
load it. Only the requested preset module is imported.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    'SimpleTransform': '.simple_transform',
    'SimpleTransform3DSMPL': '.simple_transform_3d_smpl',
    'SimpleTransform3DSMPLCam': '.simple_transform_3d_smpl_cam',
    'SimpleTransformCam': '.simple_transform_cam',
    'SimpleTransform3DSMPLX': '.simple_transform_3d_smplx',
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
