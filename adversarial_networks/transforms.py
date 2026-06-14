"""Declarative parameter constraints (bijectors) for structural models.

Instead of hand-rolling a reparameterisation (``beta_cap*tanh(raw)``,
``exp(log_sigma_sq)``) *and* its inverse for initialisation — plus the
``clamp(1e-6, 1-1e-6)`` overflow footgun — a model declares each parameter's
admissible space:

```python
class MyGame(NetworkGameGenerator):
    beta     = Interval(-1, 1)   # bijection R <-> (-1, 1)
    sigma_sq = Positive()        # bijection R <-> (0, inf)
    alpha    = Real()            # identity
```

Each :class:`Transform` provides ``forward`` (unconstrained → constrained) and
``inverse`` (constrained → unconstrained, for initialising from a desired value).
The base :class:`~adversarial_networks.generators.NetworkGameGenerator` wires a
learnable unconstrained ``nn.Parameter`` per declared field and assembles
``constrained_params()`` automatically. This is the GPyTorch /
``torch.nn.utils.parametrize`` / TFP-bijector pattern, kept to a small
dependency-light helper. The two built-in games keep their own (hand-written)
reparameterisations for bit-stable numerics; new games should prefer this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

_EPS = 1e-6


class Transform(ABC):
    """A smooth bijection between the unconstrained line and an admissible set."""

    @abstractmethod
    def forward(self, raw: Tensor) -> Tensor:
        """Map an unconstrained tensor to its constrained (admissible) value."""

    @abstractmethod
    def inverse(self, value: float | Tensor) -> Tensor:
        """Map a desired constrained value back to its unconstrained float32 leaf."""

    @abstractmethod
    def default_constrained(self) -> float:
        """The constrained value at ``raw = 0`` (the default initialisation)."""

    @staticmethod
    def _as_tensor(value: float | Tensor) -> Tensor:
        return torch.as_tensor(float(value) if not isinstance(value, Tensor) else value, dtype=torch.float32)


class Real(Transform):
    """Identity bijection ``R <-> R`` (an unconstrained parameter)."""

    def forward(self, raw: Tensor) -> Tensor:
        return raw

    def inverse(self, value: float | Tensor) -> Tensor:
        return self._as_tensor(value).clone()

    def default_constrained(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "Real()"


class Positive(Transform):
    """Bijection ``R <-> (0, inf)`` via ``exp`` / ``log`` (matches ``sigma_sq = exp``)."""

    def forward(self, raw: Tensor) -> Tensor:
        return torch.exp(raw)

    def inverse(self, value: float | Tensor) -> Tensor:
        v = self._as_tensor(value)
        if torch.any(v <= 0.0):
            raise ValueError(f"Positive() requires a strictly positive value, got {float(v.min())}.")
        return torch.log(v.clamp_min(1e-30))

    def default_constrained(self) -> float:
        return 1.0

    def __repr__(self) -> str:
        return "Positive()"


class Interval(Transform):
    """Bijection ``R <-> (low, high)`` via a (shifted/scaled) ``tanh``.

    ``forward(raw) = center + half_width * tanh(raw)`` with
    ``center = (low + high) / 2`` and ``half_width = (high - low) / 2``. For a
    symmetric ``Interval(-c, c)`` this is exactly ``c * tanh(raw)`` — the
    reparameterisation the linear-in-means peer effect uses — so the constrained
    value is *always* strictly inside ``(low, high)`` and no optimiser step can
    leave the admissible set. ``inverse`` clamps to ``(-1+eps, 1-eps)`` before
    ``atanh`` so initialising at (or past) a boundary cannot overflow.
    """

    def __init__(self, low: float, high: float) -> None:
        if not (low < high):
            raise ValueError(f"Interval requires low < high, got low={low}, high={high}.")
        self.low = float(low)
        self.high = float(high)
        self.center = 0.5 * (self.low + self.high)
        self.half_width = 0.5 * (self.high - self.low)

    def forward(self, raw: Tensor) -> Tensor:
        return self.center + self.half_width * torch.tanh(raw)

    def inverse(self, value: float | Tensor) -> Tensor:
        v = self._as_tensor(value)
        ratio = ((v - self.center) / self.half_width).clamp(-1.0 + _EPS, 1.0 - _EPS)
        return torch.atanh(ratio)

    def default_constrained(self) -> float:
        return self.center

    def __repr__(self) -> str:
        return f"Interval(low={self.low}, high={self.high})"


__all__ = ["Transform", "Real", "Positive", "Interval"]
