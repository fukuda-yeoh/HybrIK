"""Read SMPL-family ``.pkl`` files without the ``chumpy`` package.

The original SMPL/SMPL-X ``.pkl`` releases store a few arrays as
``chumpy.ch.Ch`` objects, which are thin ``numpy.ndarray`` subclasses carrying
optimisation bookkeeping.  ``chumpy`` is unmaintained and does not install on
current Python versions, but the array payload is just ``state['x']``.

This module provides a :class:`pickle.Unpickler` that maps those references
onto a local ``ndarray`` subclass, so the model files can be read with only
NumPy and SciPy.
"""

import pickle

import numpy as np

__all__ = ['load_smpl_pickle']


class _ChumpyArray(np.ndarray):
    """Minimal stand-in for ``chumpy.ch.Ch``.

    Only the array payload is preserved; the chumpy dependency graph is
    dropped because the model files are read-only.
    """

    def __new__(cls, *args, **kwargs):
        # ``Ch()`` is instantiated with no arguments during unpickling and then
        # filled in through ``__setstate__``.
        if not args and not kwargs:
            return super().__new__(cls, (0,), dtype=np.float64)
        return super().__new__(cls, *args, **kwargs)

    def __array_finalize__(self, obj):
        pass

    def __setstate__(self, state):
        if not isinstance(state, dict):
            np.ndarray.__setstate__(self, state)
            return

        value = state.get('x')
        if value is None:
            value = np.zeros((0,), dtype=np.float64)
        array = np.ascontiguousarray(np.asarray(value))
        np.ndarray.__setstate__(
            self, (1, array.shape, array.dtype, False, array.tobytes()))


class _SmplUnpickler(pickle.Unpickler):
    """Translate Python 2 / chumpy references into current equivalents."""

    def find_class(self, module, name):
        if module == 'chumpy.ch' and name == 'Ch':
            return _ChumpyArray

        # Model files were written by Python 2.
        if module == '__builtin__':
            module = 'builtins'

        # ``numpy.core`` was renamed to ``numpy._core`` in NumPy 2.0.
        if module == 'numpy.core' or module.startswith('numpy.core.'):
            module = module.replace('numpy.core', 'numpy._core', 1)

        # ``scipy.sparse.csc`` is a deprecated alias of ``scipy.sparse``.
        if module == 'scipy.sparse.csc':
            module = 'scipy.sparse'

        return super().find_class(module, name)


def load_smpl_pickle(path):
    """Load a SMPL/SMPL-X ``.pkl`` file and return its raw dictionary."""
    with open(path, 'rb') as handle:
        return _SmplUnpickler(handle, encoding='latin1').load()
