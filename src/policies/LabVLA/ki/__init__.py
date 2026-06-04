import importlib.util as _importlib_util

from .fast_tokenizer import FastTokenizerWrapper

__all__ = ["FastTokenizerWrapper"]

# Optional DiscreteActionHead import. Check module presence first so a real
# ImportError *inside* an existing ki_head.py (bad import, renamed symbol,
# broken dependency) propagates instead of being swallowed as "module absent",
# which would silently drop DiscreteActionHead and mask a main-path failure.
if _importlib_util.find_spec(".ki_head", __name__) is not None:
    from .ki_head import DiscreteActionHead
    __all__.append("DiscreteActionHead")
