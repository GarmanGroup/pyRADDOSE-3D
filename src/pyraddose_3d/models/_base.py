"""Base model shared by every configuration object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import os

__all__ = ["RD3DModel"]


class RD3DModel(BaseModel):
    """Base class for RADDOSE-3D configuration models.

    ``extra="forbid"`` is deliberate: a mistyped key in a YAML file is a hard
    validation error rather than a silently ignored setting. Dose calculations
    are often run unattended, and a silently dropped parameter is far worse
    than a loud failure at load time.
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> Self:
        """Load and validate a configuration from a YAML file.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Path to the YAML file.

        Returns
        -------
        Self
            The validated model.
        """
        with open(path, encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle)
        return cls(**data)

    def to_yaml(self, path: str | os.PathLike[str]) -> None:
        """Write this configuration to a YAML file.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Destination path.
        """
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)
