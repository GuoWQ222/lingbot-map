"""Step-aware mixture sampler for Manip + external 3D datasets.

Designed to be used with ``torch.utils.data.ConcatDataset`` whose layout is::

    [ Manip | External_0 | External_1 | ... | External_{N-1} ]

At every ``__iter__`` (i.e. once per epoch in the main process), the sampler
reads its internal ``global_step``, computes the current Manip probability via
a piecewise-linear schedule, then for each of ``epoch_length`` samples it:

  1. Flips a Bernoulli(``p_manip``) coin to choose Manip vs. External.
  2. If External, uniformly picks one of the N enabled external datasets.
  3. Returns a global ConcatDataset index inside that dataset.

The schedule::

    p_manip(step) = p_start                                    if step <= warmup_start
                  = p_start + r * (p_end - p_start)            if warmup_start < step < warmup_end
                                 (r = (step - warmup_start) / (warmup_end - warmup_start))
                  = p_end                                      if step >= warmup_end

Per-external is uniform (each enabled external gets ``(1 - p_manip) / N``).

Notes
-----
*  Works with ``persistent_workers=True``: the sampler runs in the main
   process and only forwards integer indices to workers, so updating
   ``global_step`` in the main loop takes effect at the next epoch start.
*  The schedule is re-evaluated **once per epoch**, not per sample. With
   ``LIMIT_TRAIN_BATCHES = 1000`` and ``max_steps = 100000``, that gives 100
   weight updates over the run — plenty of resolution.
*  All randomness is driven by a numpy ``Generator`` seeded from
   ``(seed XOR global_step)`` so different epochs produce different draws.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


class CurriculumMixtureSampler(Sampler[int]):
    """Mixture sampler with a piecewise-linear Manip-vs-External schedule.

    Parameters
    ----------
    manip_size:
        Number of samples in the Manip portion of the ConcatDataset.
        Must be > 0.
    external_sizes:
        Sizes of each external sub-dataset, **in ConcatDataset order**
        (i.e. the order they were concat'd in after Manip).
    external_names:
        Human-readable names for each external (e.g. "dl3dv", "scannetpp"),
        same length as ``external_sizes``. Used for tensorboard logging.
    epoch_length:
        Number of indices yielded per ``__iter__``. Should equal the per-epoch
        batch budget (``LIMIT_TRAIN_BATCHES`` with batch_size=1).
    p_manip_start, p_manip_end:
        Manip probability at the start / end of the schedule.
        Must each be in [0, 1].
    warmup_start, warmup_end:
        Step range over which p_manip linearly ramps from start to end.
        Before ``warmup_start`` we hold at ``p_manip_start``; after
        ``warmup_end`` we hold at ``p_manip_end``. ``warmup_end`` is clamped
        to be >= ``warmup_start``.
    seed:
        Base seed; mixed with ``global_step`` for per-epoch RNG.
    """

    def __init__(
        self,
        *,
        manip_size: int,
        external_sizes: Sequence[int],
        external_names: Sequence[str],
        epoch_length: int,
        p_manip_start: float = 0.30,
        p_manip_end: float = 0.90,
        warmup_start: int = 2000,
        warmup_end: int = 70000,
        seed: int = 0,
    ) -> None:
        if manip_size <= 0:
            raise ValueError(f"manip_size must be > 0, got {manip_size}")
        if len(external_sizes) != len(external_names):
            raise ValueError(
                f"external_sizes (len={len(external_sizes)}) and external_names "
                f"(len={len(external_names)}) must have the same length"
            )
        for i, s in enumerate(external_sizes):
            if s <= 0:
                raise ValueError(
                    f"external_sizes[{i}] ({external_names[i]}) must be > 0, got {s}"
                )
        for label, p in (("p_manip_start", p_manip_start), ("p_manip_end", p_manip_end)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {p}")
        if epoch_length <= 0:
            raise ValueError(f"epoch_length must be > 0, got {epoch_length}")

        self.manip_size = int(manip_size)
        self.external_sizes = [int(s) for s in external_sizes]
        self.external_names = [str(n) for n in external_names]
        self.epoch_length = int(epoch_length)
        self.p_manip_start = float(p_manip_start)
        self.p_manip_end = float(p_manip_end)
        self.warmup_start = max(0, int(warmup_start))
        self.warmup_end = max(self.warmup_start, int(warmup_end))
        self.seed = int(seed)

        # ConcatDataset cumulative offsets:
        #   _offsets[0]   = 0                                (start of Manip)
        #   _offsets[i+1] = manip_size + sum(external_sizes[:i])  (start of external i)
        self._offsets: List[int] = [0]
        cum = self.manip_size
        for size in self.external_sizes:
            self._offsets.append(cum)
            cum += size
        self._total = cum

        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Step / weight inspection (used by the training loop for logging
    # and by tests / dry-runs).
    # ------------------------------------------------------------------
    def set_global_step(self, step: int) -> None:
        """Called by the training loop after each optimizer step.

        The new step is picked up at the next ``__iter__`` call (= next epoch),
        which lines up naturally with how DataLoader requests indices.
        """
        self._global_step = int(step)

    @property
    def global_step(self) -> int:
        return self._global_step

    def get_p_manip(self, step: Optional[int] = None) -> float:
        s = self._global_step if step is None else int(step)
        if s <= self.warmup_start or self.warmup_end == self.warmup_start:
            return self.p_manip_start
        if s >= self.warmup_end:
            return self.p_manip_end
        ratio = (s - self.warmup_start) / (self.warmup_end - self.warmup_start)
        return self.p_manip_start + ratio * (self.p_manip_end - self.p_manip_start)

    def get_dataset_weights(self, step: Optional[int] = None) -> Dict[str, float]:
        """Per-source effective draw probability at the given step.

        Returned dict always contains ``"manip"``; one key per external too.
        """
        p_m = self.get_p_manip(step)
        out: Dict[str, float] = {"manip": p_m}
        n_ext = len(self.external_sizes)
        if n_ext == 0:
            out["manip"] = 1.0
            return out
        p_ext_each = (1.0 - p_m) / n_ext
        for name in self.external_names:
            out[name] = p_ext_each
        return out

    # ------------------------------------------------------------------
    # PyTorch Sampler protocol
    # ------------------------------------------------------------------
    def __iter__(self):
        # Per-epoch RNG seeded from (seed, step) so successive epochs at the
        # same step still differ. & 0xFFFFFFFF keeps it inside the uint32
        # range numpy expects.
        rng = np.random.default_rng((self.seed ^ self._global_step) & 0xFFFFFFFF)
        p_manip = self.get_p_manip()
        n_ext = len(self.external_sizes)
        manip_offset = self._offsets[0]
        manip_size = self.manip_size

        for _ in range(self.epoch_length):
            if n_ext == 0 or rng.random() < p_manip:
                local = int(rng.integers(0, manip_size))
                yield manip_offset + local
            else:
                ds = int(rng.integers(0, n_ext))
                local = int(rng.integers(0, self.external_sizes[ds]))
                yield self._offsets[1 + ds] + local

    def __len__(self) -> int:
        return self.epoch_length
