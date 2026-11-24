"""Lazily build the SMPL-X body model layers.

The layers read licensed model files, so they are created on first use instead
of at import time. This keeps ``import hybrik.datasets`` usable without
``model_files`` and without paying the load cost up front.
"""

from functools import lru_cache

from hybrik.paths import model_path

from .body_models import SMPLXLayer


def _layer(model_name: str, age: str) -> SMPLXLayer:
    return SMPLXLayer(
        model_path=model_path('smplx', model_name),
        num_betas=10,
        use_pca=False,
        age=age,
        kid_template_path=model_path('smplx_kid_template.npy'),
    )


@lru_cache(maxsize=1)
def load_models():
    """Return the SMPL-X layers as ``(neutral, male, female, *kid)``."""
    return (
        _layer('SMPLX_NEUTRAL.npz', 'adult'),
        _layer('SMPLX_MALE.npz', 'adult'),
        _layer('SMPLX_FEMALE.npz', 'adult'),
        _layer('SMPLX_NEUTRAL.npz', 'kid'),
        _layer('SMPLX_MALE.npz', 'kid'),
        _layer('SMPLX_FEMALE.npz', 'kid'),
    )
