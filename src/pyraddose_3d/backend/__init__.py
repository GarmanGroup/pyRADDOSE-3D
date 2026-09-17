"""Numerical core.

Nothing in this subpackage imports pydantic, or anything else from
:mod:`pyraddose_3d.models`. Every function here takes tensors and plain
Python values and returns tensors.

That separation is deliberate and load-bearing:

- the dose calculation must be callable with tensors that are autograd
  leaves, without a configuration object in the way;
- it keeps the numerical core directly testable and benchmarkable;
- it leaves ``torch.compile`` viable.

Configuration objects reach this layer only through
``Manager.make_backend_kwargs() -> dict[str, Any]``.
"""
