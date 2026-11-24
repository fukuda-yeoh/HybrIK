from importlib import import_module

from torch import nn

from hybrik.utils import Registry, build_from_cfg


SPPE = Registry('sppe')
LOSS = Registry('loss')

_PACKAGE = 'hybrik.models'


def _register_module_from_name(name, registry):
    """Import the module that defines ``name`` so it self-registers.

    Model modules are imported lazily, so building a model straight from a
    config (without importing its class first) has to resolve ``cfg.TYPE`` to
    the module that owns it.
    """
    if registry.get(name) is not None:
        return

    if registry is LOSS:
        # Every loss lives in a single module.
        import_module(f'{_PACKAGE}.criterion')
        return

    # ``hybrik.models`` already knows which module exports each public class.
    from hybrik.models import _EXPORTS

    module_name = _EXPORTS.get(name)
    if module_name is not None:
        import_module(module_name, _PACKAGE)


def _ensure_registered(cfg, registry):
    if not isinstance(cfg, dict):
        return
    obj_type = cfg.get('TYPE')
    if isinstance(obj_type, str):
        _register_module_from_name(obj_type, registry)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = []
        for cfg_ in cfg:
            _ensure_registered(cfg_, registry)
            modules.append(build_from_cfg(cfg_, registry, default_args))
        return nn.Sequential(*modules)
    else:
        _ensure_registered(cfg, registry)
        return build_from_cfg(cfg, registry, default_args)


def build_sppe(cfg):
    return build(cfg, SPPE)


def build_loss(cfg, **kwargs):
    default_args = dict()

    for key, value in kwargs.items():
        default_args[key] = value

    return build(cfg, LOSS, default_args=default_args)
