"""Compatibility entry: the production objective is in gib4cl.learning.objectives."""
from gib4cl.learning.objectives import compute_gib_loss


def __getattr__(name):
    if name in ('compute_legacy_fim_loss','compute_crlb'):
        import warnings
        warnings.warn('Historical FIM implementation; not a production objective',DeprecationWarning,stacklevel=2)
        from legacy import fim_objective
        return getattr(fim_objective,name)
    raise AttributeError(name)
