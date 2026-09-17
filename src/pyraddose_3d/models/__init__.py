"""Pydantic models describing a RADDOSE-3D experiment.

Configuration is YAML in, validated models out. Following Leopard-EM, YAML
files hold *scalars only* -- grids and other tensors are derived on demand by
``@property`` -- so configurations stay small, diffable and reviewable.
"""
