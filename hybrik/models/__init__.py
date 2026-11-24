"""Model classes exported lazily.

Importing :mod:`hybrik.models` should not pull in every model (and therefore
PyTorch3D). Only the requested module is imported, so
``from hybrik.models import builder`` works with just PyTorch installed.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    'Simple3DPoseBaseSMPL': '.simple3dposeBaseSMPL',
    'Simple3DPoseBaseSMPL24': '.simple3dposeBaseSMPL24',
    'Simple3DPoseBaseSMPLCam': '.simple3dposeSMPLWithCam',
    'Simple3DPoseBaseSMPLCamReg': '.simple3dposeSMPLWithCamReg',
    'HRNetSMPLCam': '.HRNetWithCam',
    'HRNetSMPLCamReg': '.HRNetWithCamReg',
    'HRNetSMPLXCamKid': '.HRNetSMPLXCamKid',
    'HRNetSMPLXCamKidReg': '.HRNetSMPLXCamKidReg',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(_import_module(module_name, __name__), name)
        globals()[name] = value
        return value

    # Expose submodules such as `builder` and `criterion` as attributes,
    # matching normal package behaviour without importing them eagerly.
    try:
        module = _import_module(f'.{name}', __name__)
    except ModuleNotFoundError as exc:
        if exc.name != f'{__name__}.{name}':
            raise
    else:
        globals()[name] = module
        return module

    # Preserve the old `from .criterion import *` behaviour: expose names
    # defined in `criterion` when an unknown attribute is requested.
    criterion = _import_module('.criterion', __name__)
    if hasattr(criterion, name):
        value = getattr(criterion, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
