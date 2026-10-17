"""A Python port of RADDOSE-3D, differentiable with PyTorch where installed."""

try:
    # Written by hatch-vcs when the package is built or installed.
    from pyraddose_3d._version import __version__
except ImportError:  # a source tree that was never installed
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"
