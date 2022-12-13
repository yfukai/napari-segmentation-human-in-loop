try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._sample_data import make_sample_data
from ._widget import wizard_widget

__all__ = (
    "make_sample_data",
    "wizard_widget",
)
