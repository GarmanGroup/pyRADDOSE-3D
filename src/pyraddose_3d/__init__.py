"""A differentiable PyTorch port of RADDOSE-3D."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyRADDOSE-3D")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"
