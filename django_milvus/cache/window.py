"""
Window size control for the cache.

"Window" means two different, independently controllable things:

**Capacity window** - what fraction of the cache is reserved for freshly
admitted keys before they must prove their worth. In W-TinyLFU this is
``admission_ratio``; inside the main region, ``probation_ratio`` splits
unproven from protected entries. A large window favours bursty workloads
where new keys are about to become hot; a small window favours a stable
hot set, spending nearly all capacity on proven entries.

**Temporal window** - ``sample_interval``, the period over which hit rate
is measured. Every decision below is made against the *recent* hit rate
from the sliding window in :mod:`.stats`, never a lifetime average, which
would take hours to react to a workload change.

Set ``admission_ratio`` yourself, or leave ``adaptive: True`` and let
:class:`WindowController` hill-climb it.
"""

import logging

logger = logging.getLogger("django_milvus.cache")

#: The window may never exceed this share of capacity; beyond it the
#: policy degenerates towards plain LRU and loses its admission filter.
MAX_ADMISSION_RATIO = 0.8
MIN_ADMISSION_RATIO = 0.0

#: Below this many samples in an interval the hit rate is noise, so the
#: controller holds its position rather than chasing it.
MIN_SAMPLES = 50


class WindowController:
    """Hill-climbs ``admission_ratio`` toward the best observed hit rate.

    Each tick compares the recent hit rate against the previous tick:

    * improved -> keep stepping in the same direction;
    * worsened -> reverse direction and halve the step, so the search
      converges instead of oscillating;
    * step decayed to nothing -> reset it, letting the controller escape a
      stale optimum after the workload shifts.

    This is the approach Caffeine uses, and it needs no model of the
    workload - only the hit rate it can already measure.
    """

    def __init__(self, config, stats=None):
        self.config = config
        self.stats = stats
        self.enabled = config.adaptive

        self.admission_ratio = config.admission_ratio
        self.initial_step = config.step or 0.05
        self.step = self.initial_step
        self.direction = 1

        self._previous_hit_rate = None
        self._previous_lookups = 0
        self.ticks = 0
        self.adjustments = 0
        self.best_hit_rate = 0.0
        self.best_ratio = config.admission_ratio

        self._listeners = []

    def subscribe(self, callback):
        """Register a callback invoked with each new admission ratio."""
        self._listeners.append(callback)
        return callback

    def _notify(self):
        for callback in self._listeners:
            try:
                callback(self.admission_ratio)
            except Exception:  # pragma: no cover - listener bugs
                logger.warning(
                    "django-milvus cache window listener failed", exc_info=True
                )

    def tick(self, hit_rate=None, lookups=None):
        """Advance the controller one sample interval.

        Returns the (possibly unchanged) admission ratio.
        """
        self.ticks += 1
        if not self.enabled:
            return self.admission_ratio

        if hit_rate is None:
            if self.stats is None:
                return self.admission_ratio
            hit_rate = self.stats.recent_hit_rate()
            lookups = self.stats.hits + self.stats.misses

        # Too little traffic this interval to draw a conclusion from.
        if lookups is not None:
            sampled = lookups - self._previous_lookups
            self._previous_lookups = lookups
            if sampled < MIN_SAMPLES:
                return self.admission_ratio

        if hit_rate > self.best_hit_rate:
            self.best_hit_rate = hit_rate
            self.best_ratio = self.admission_ratio

        previous = self._previous_hit_rate
        self._previous_hit_rate = hit_rate
        if previous is None:
            # First real sample: take a step to get a gradient to compare.
            self._apply(self.admission_ratio + self.direction * self.step)
            return self.admission_ratio

        if hit_rate > previous:
            # Moving the right way; keep going.
            pass
        elif hit_rate < previous:
            self.direction *= -1
            self.step = max(self.step / 2, self.initial_step / 16)
        else:
            # Flat: nothing to learn, so stop nudging and wait.
            return self.admission_ratio

        self._apply(self.admission_ratio + self.direction * self.step)

        # Once the step has decayed away the search has converged. Restore
        # it so a later workload change can still be tracked.
        if self.step <= self.initial_step / 16:
            self.step = self.initial_step

        return self.admission_ratio

    def _apply(self, ratio):
        ratio = max(MIN_ADMISSION_RATIO, min(MAX_ADMISSION_RATIO, ratio))
        if abs(ratio - self.admission_ratio) < 1e-9:
            return
        self.admission_ratio = ratio
        self.adjustments += 1
        self._notify()

    def reset(self):
        self.admission_ratio = self.config.admission_ratio
        self.step = self.initial_step
        self.direction = 1
        self._previous_hit_rate = None
        self._previous_lookups = 0
        self.ticks = 0
        self.adjustments = 0
        self.best_hit_rate = 0.0
        self.best_ratio = self.config.admission_ratio
        self._notify()

    def stats_dict(self):
        return {
            "adaptive": self.enabled,
            "admission_ratio": round(self.admission_ratio, 4),
            "probation_ratio": round(self.config.probation_ratio, 4),
            "sample_interval": self.config.sample_interval,
            "step": round(self.step, 4),
            "direction": self.direction,
            "ticks": self.ticks,
            "adjustments": self.adjustments,
            "best_hit_rate": round(self.best_hit_rate, 4),
            "best_ratio": round(self.best_ratio, 4),
        }

    def __repr__(self):
        return (
            f"<WindowController ratio={self.admission_ratio:.3f} "
            f"step={self.step:.3f} adaptive={self.enabled}>"
        )
