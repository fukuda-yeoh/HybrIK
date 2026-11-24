import contextlib
import pickle
import sys
import types

import numpy as np

from hybrik.utils.smpl_pickle import load_smpl_pickle


@contextlib.contextmanager
def _legacy_chumpy():
    """Expose a fake ``chumpy.ch.Ch`` so pickles can reference it."""
    package = types.ModuleType('chumpy')
    module = types.ModuleType('chumpy.ch')

    class Ch:
        def __init__(self, x=None):
            self.x = x

        def __reduce__(self):
            # Mirrors chumpy: no constructor args, payload in __setstate__.
            return (
                Ch,
                (),
                {
                    'x': self.x,
                    '_dirty_vars': set(),
                    '_itr': None,
                    '_depends_on_deps': {},
                },
            )

    Ch.__module__ = 'chumpy.ch'
    Ch.__qualname__ = 'Ch'
    module.Ch = Ch
    sys.modules['chumpy'] = package
    sys.modules['chumpy.ch'] = module
    try:
        yield Ch
    finally:
        sys.modules.pop('chumpy.ch', None)
        sys.modules.pop('chumpy', None)


def test_chumpy_array_payload_is_preserved(tmp_path):
    shapedirs = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    path = tmp_path / 'model.pkl'

    with _legacy_chumpy() as Ch:
        with open(path, 'wb') as handle:
            pickle.dump({'shapedirs': Ch(shapedirs), 'f': np.zeros((1, 3))}, handle)

    # ``chumpy`` is gone again: the loader has to resolve it itself.
    data = load_smpl_pickle(path)

    assert isinstance(data['shapedirs'], np.ndarray)
    assert data['shapedirs'].shape == (2, 3, 4)
    assert data['shapedirs'].dtype == np.float64
    np.testing.assert_array_equal(data['shapedirs'], shapedirs)


def test_plain_pickle_still_loads(tmp_path):
    path = tmp_path / 'plain.pkl'
    with open(path, 'wb') as handle:
        pickle.dump({'v_template': np.ones((4, 3))}, handle)

    data = load_smpl_pickle(path)
    np.testing.assert_array_equal(data['v_template'], np.ones((4, 3)))
