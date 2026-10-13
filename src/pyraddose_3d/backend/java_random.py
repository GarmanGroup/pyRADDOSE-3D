"""``java.util.Random``, draw for draw.

RADDOSE-3D's escape modelling takes its random numbers from one
``java.util.Random`` seeded once per run. Its algorithm is part of the Java
specification -- a 48-bit linear congruential generator, with ``nextInt``
rejecting draws that would bias a bound that is not a power of two -- so
reproducing it reproduces the Java's draws exactly, given the same seed and
the same sequence of calls. That is what lets the port match RADDOSE-3D's
escape results rather than only their statistics.
"""

from __future__ import annotations

import numpy as np

__all__ = ["JavaRandom"]

_MULTIPLIER = 0x5DEECE66D
_ADDEND = 0xB
_MASK = (1 << 48) - 1
_DOUBLE_UNIT = 1.0 / (1 << 53)
_LOW24 = (1 << 24) - 1
#: Draws generated one at a time before the rest are reached by jumping.
_SEED_BLOCK = 512


def _mulmod48(x: np.ndarray, factor: int) -> np.ndarray:
    """``x * factor mod 2**48`` for uint64 ``x < 2**48``, without overflowing.

    Both operands are split into 24-bit halves, so no partial product needs
    more than 49 bits.
    """
    x_low, x_high = x & _LOW24, x >> 24
    f_low, f_high = np.uint64(factor & _LOW24), np.uint64(factor >> 24)
    cross = (x_high * f_low + x_low * f_high) & np.uint64(_LOW24)
    return (x_low * f_low + (cross << np.uint64(24))) & np.uint64(_MASK)


def _int32(value: int) -> int:
    """Wrap to a signed 32-bit integer, as a Java ``int`` would."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


class JavaRandom:
    """A ``java.util.Random``.

    Parameters
    ----------
    seed : int
        As passed to ``new Random(seed)``.
    """

    def __init__(self, seed: int) -> None:
        self._seed = (seed ^ _MULTIPLIER) & _MASK

    def _next(self, bits: int) -> int:
        self._seed = (self._seed * _MULTIPLIER + _ADDEND) & _MASK
        return _int32(self._seed >> (48 - bits))

    def next_int(self, bound: int) -> int:
        """``nextInt(bound)``: uniform on ``[0, bound)``.

        Parameters
        ----------
        bound : int
            Exclusive upper bound; positive.

        Returns
        -------
        int
            The draw.
        """
        if bound <= 0:
            raise ValueError("bound must be positive")
        r = self._next(31)
        m = bound - 1
        if bound & m == 0:
            return _int32((bound * r) >> 31)
        u = r
        while True:
            r = u % bound
            if _int32(u - r + m) >= 0:
                return r
            u = self._next(31)

    def _states(self, count: int) -> np.ndarray:
        """The next ``count`` generator states, advancing past them.

        The first few are stepped one at a time; the block is then doubled
        repeatedly by jumping ahead, ``s[n + k] = a^k s[n] + c(a^k - 1)/(a - 1)``.
        """
        first = min(count, _SEED_BLOCK)
        states = np.empty(first, dtype=np.uint64)
        state = self._seed
        for i in range(first):
            state = (state * _MULTIPLIER + _ADDEND) & _MASK
            states[i] = state
        jump_a, jump_c = _MULTIPLIER, _ADDEND
        for _ in range(first - 1):
            jump_c = (jump_c * _MULTIPLIER + _ADDEND) & _MASK
            jump_a = (jump_a * _MULTIPLIER) & _MASK
        # (jump_a, jump_c) now advance a state by `first` steps.
        while len(states) < count:
            ahead = (_mulmod48(states, jump_a) + np.uint64(jump_c)) & np.uint64(_MASK)
            states = np.concatenate([states, ahead])
            jump_c = (jump_c * (1 + jump_a)) & _MASK
            jump_a = (jump_a * jump_a) & _MASK
        states = states[:count]
        self._seed = int(states[-1])
        return states

    def next_ints(self, bounds: np.ndarray) -> np.ndarray:
        """``nextInt`` once for each bound, in order, as successive calls would.

        Vectorised; the result is exactly that of a loop over
        :meth:`next_int`, including the rare draw ``nextInt`` rejects and
        replaces, which shifts every later draw along by one.

        Parameters
        ----------
        bounds : np.ndarray
            Positive bounds, one per draw.

        Returns
        -------
        np.ndarray
            ``int64`` draws.
        """
        bounds = np.asarray(bounds, dtype=np.int64)
        if np.any(bounds <= 0):
            raise ValueError("bounds must be positive")
        out = np.empty(len(bounds), dtype=np.int64)
        done = 0
        while done < len(bounds):
            pending = bounds[done:]
            states = self._states(len(pending))
            raw = (states >> np.uint64(17)).astype(np.int64)
            power_of_two = (pending & (pending - 1)) == 0
            values = np.where(power_of_two, (pending * raw) >> 31, raw % pending)
            rejected = ~power_of_two & (raw - values + (pending - 1) >= 1 << 31)
            if not rejected.any():
                out[done:] = values
                return out
            first = int(np.argmax(rejected))
            out[done : done + first] = values[:first]
            # Rewind to just after the rejected draw and redraw it in order.
            self._seed = int(states[first])
            bound = int(pending[first])
            while True:
                u = self._next(31)
                r = u % bound
                if _int32(u - r + bound - 1) >= 0:
                    break
            out[done + first] = r
            done += first + 1
        return out

    def next_double(self) -> float:
        """``nextDouble()``: uniform on ``[0, 1)``."""
        return ((self._next(26) << 27) + self._next(27)) * _DOUBLE_UNIT
