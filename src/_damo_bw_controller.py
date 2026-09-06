# SPDX-License-Identifier: GPL-2.0
"""
Closed-loop bandwidth interleave controller.

Hill-climbing bandwidth-feedback controller (total-bandwidth objective).

ratio: 0..100 (100 = all near/DRAM).
BW values: MB/s (int or float).
"""

import collections
import math

try:
    import numpy as np
    _have_numpy = True
except ImportError:
    _have_numpy = False

MAX_STEP = 8
MIN_STEP = 2
MIN_THROTTLE_STEP = MAX_STEP // 2   # = 4
THROTTLE_THRESHOLD = 90
BW_PERCENTILE = 100
MAX_ADJUST_INTERVAL_MS = 30000
# Longest span of bandwidth history the interval selector looks at.  Stage 2
# transforms the whole history, so this also bounds that cost.
FFT_WINDOW_LEN_MS = 90000

MAGNITUDE_FACTOR = 5.0 / 4.0
MIN_MAGNITUDE = 5
HAMMING_WINDOW_THRESH = 64
# Adjust intervals are counted in bandwidth samples, since the transforms
# resolve periods in units of the sampling period.  Two samples is the shortest
# interval whose spectrum has anything but a DC bin.
MIN_ADJUST_SAMPLES = 2
# Resolution of the between-bin period sweep, in samples.  One is as fine as the
# sampled signal can distinguish.
GOERTZEL_STEP = 1


def process_signal(bw_history, window_size, hamming_window):
    """Condition a bandwidth history into a real signal of exactly window_size.

    When the history is shorter than the window the OLDEST end is zero-filled
    and the DC mean is still taken over the full window, so a short history
    reads as a signal that was quiet before it started rather than as a shorter
    signal.  Both matter: the transforms below index by period in samples, and a
    window whose length silently varied would make a period mean two different
    spans of time on consecutive calls.

    Bandwidth is converted MB/s -> GB/s to keep magnitudes in the range the
    MIN_MAGNITUDE threshold was chosen for.
    """
    if window_size <= 0:
        return []
    signal = [0.0] * window_size
    i = window_size - 1
    for x in reversed(list(bw_history)):
        if i < 0:
            break
        signal[i] = x / 1000.0
        i -= 1
    mean = sum(signal) / window_size
    for i in range(window_size):
        signal[i] -= mean
        # The window term divides by window_size - 1, so a one-sample window has
        # no Hamming shape to apply.  do_fft() asks for windowing unconditionally,
        # so this is reached rather than merely defensive.
        if hamming_window and window_size > 1:
            signal[i] *= (0.54
                          - 0.46 * math.cos(2 * math.pi * i / (window_size - 1)))
    return signal


def goertzel_magnitude(signal, target_freq):
    """Goertzel algorithm for a single normalized target frequency [0,1)."""
    n = len(signal)
    if n == 0:
        return 0.0
    omega = 2.0 * math.pi * target_freq
    coeff = 2.0 * math.cos(omega)
    s_prev2, s_prev1 = 0.0, 0.0
    for x in signal:
        s = x + coeff * s_prev1 - s_prev2
        s_prev2, s_prev1 = s_prev1, s
    # Power = s_prev1^2 + s_prev2^2 - coeff*s_prev1*s_prev2
    power = (s_prev1 * s_prev1 + s_prev2 * s_prev2
             - coeff * s_prev1 * s_prev2)
    return math.sqrt(max(0.0, power))


def do_goertzel(bw_history, periods, window_size):
    """Magnitude at each requested period, in samples, over the last window_size.

    Periods need not be integers and need not land on a DFT bin -- that freedom
    is the whole reason this exists alongside do_fft(), which can only report the
    window_size/k bins.

    No Hamming window below HAMMING_WINDOW_THRESH samples: windowing widens the
    main lobe in the frequency domain, and on a short window that smears
    together exactly the between-bin periods being probed here.
    """
    signal = process_signal(bw_history, window_size,
                            window_size >= HAMMING_WINDOW_THRESH)
    if not signal:
        return [0.0] * len(periods)
    return [goertzel_magnitude(signal, 1.0 / p) if p > 0 else 0.0
            for p in periods]


def do_fft(bw_history, window_size):
    """Magnitudes of bins 0..window_size//2 over the last window_size samples.

    Bin k is the period window_size/k, so index 1 is the longest period the
    transform can see and the resolution between low bins is coarse -- which is
    why the caller uses do_goertzel() when it needs to look between them.
    """
    signal = process_signal(bw_history, window_size, True)
    if not signal:
        return []
    if _have_numpy:
        return list(np.abs(np.fft.rfft(np.array(signal, dtype=float))))
    # Pure-Python DFT fallback: same output, no numpy dependency.
    out = []
    for k in range(window_size // 2 + 1):
        re = sum(signal[j] * math.cos(2 * math.pi * k * j / window_size)
                 for j in range(window_size))
        im = sum(signal[j] * math.sin(2 * math.pi * k * j / window_size)
                 for j in range(window_size))
        out.append(math.sqrt(re * re + im * im))
    return out


class BwController:
    def __init__(self, init_ratio, bw_cutoff, min_ratio=0, max_ratio=100,
                 history_len=256):
        self.bw_cutoff = bw_cutoff
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        # Hill-climb state carried across step() calls
        self.last_bw = 1
        self.last_ratio = init_ratio
        self.last_step = -(MAX_STEP * 2)
        # Every bandwidth sample, one entry each, most recent last.
        # choose_adjust_interval() resolves periods in units of the entry
        # spacing, so the entries have to be the samples themselves for a period
        # shorter than an adjust cycle to be expressible at all.
        self.bw_history = collections.deque(
            maxlen=max(MIN_ADJUST_SAMPLES, history_len))

    def step(self, readings):
        """Compute next ratio from this cycle's readings (most-recent last).

        Returns new ratio (int, clamped to [min_ratio, max_ratio]).
        Updates internal state (last_bw, last_ratio, last_step).

        The hill-climb acts on one representative value per cycle.  Appending the
        readings to bw_history is the caller's, since only the caller knows their
        order relative to the settle-wait samples.
        """
        if not readings:
            return self.last_ratio

        # BW_PERCENTILE=100 -> take the max (last element of sorted)
        sorted_bw = sorted(readings)
        nth_idx = max(0, int(len(sorted_bw) * BW_PERCENTILE / 100) - 1)
        cur_bw = sorted_bw[nth_idx]
        if cur_bw == 0:
            cur_bw = 1

        last_bw = self.last_bw if self.last_bw != 0 else cur_bw
        last_ratio = self.last_ratio if self.last_ratio != 0 else 1
        last_step = self.last_step

        # Relative changes (x10000 for resolution).  Truncate toward zero
        # (int(a/b)), not floor (a//b): the ratios can be negative and we want
        # magnitude comparisons symmetric about zero.  The same applies to the
        # step arithmetic below: halving a negative step with // rounds away
        # from zero, so a reversal would be one larger than the step it undoes
        # and the smallest step would never reach zero.
        bw_change = int(10000 * (last_bw - cur_bw) / last_bw)
        interleave_change = last_step * -100
        big_bw_change = (interleave_change != 0 and
                         abs(bw_change) > 5 * abs(interleave_change))

        # TOTALBW: good_step = cur_bw > last_bw
        good_step = cur_bw > last_bw

        ratio = self.last_ratio
        touchup_update = False

        if cur_bw < self.bw_cutoff:
            # Unsaturated: increase local ratio
            if last_ratio == 100:
                cur_step = 0
            elif last_step == 0 and bw_change > 0:
                cur_step = int(ratio * int(bw_change / 100) / 100)
                if abs(cur_step) < MIN_STEP:
                    cur_step = MIN_STEP
                elif abs(cur_step) > MAX_STEP // 2:
                    cur_step = MAX_STEP // 2 if cur_step > 0 else -(MAX_STEP // 2)
            elif last_step <= 0:
                cur_step = max(abs(last_step) // 2, MIN_STEP)
            else:
                cur_step = last_step
        elif last_step == 0:
            # Stopped: check if BW changed enough to search again
            cur_step = int(ratio * int(bw_change / 100) / 100)
            if abs(cur_step) < 4:
                cur_step = 0
            elif abs(cur_step) > MAX_STEP:
                cur_step = MAX_STEP if cur_step > 0 else -MAX_STEP
        elif last_ratio == 100:
            # Probe downward
            cur_step = -(abs(last_step) // 2)
        elif good_step:
            bw_int_ratio = abs(int(bw_change * 100 / interleave_change)) if interleave_change != 0 else 0
            do_throttle = (bw_int_ratio < THROTTLE_THRESHOLD and
                           abs(last_step) > MIN_THROTTLE_STEP)
            if do_throttle:
                throttle_step = int(bw_int_ratio * last_step / 100)
                if abs(throttle_step) < MIN_THROTTLE_STEP:
                    throttle_step = MIN_THROTTLE_STEP if last_step > 0 else -MIN_THROTTLE_STEP
                cur_step = throttle_step
            else:
                cur_step = last_step
        else:
            # Bad step: reverse
            cur_step = -int(last_step / 2)

        # Clamp step
        if big_bw_change and abs(cur_step) < MAX_STEP // 2:
            cur_step = -MAX_STEP if cur_step < 0 else MAX_STEP
        elif abs(cur_step) < MIN_STEP:
            cur_step = 0
        elif abs(cur_step) > MAX_STEP:
            cur_step = -MAX_STEP if cur_step < 0 else MAX_STEP

        # TOTALBW: if cur_step==0 and last_bw > cur_bw, undo last step
        if cur_step == 0 and last_bw > cur_bw:
            ratio -= last_step
            touchup_update = last_step != 0
        else:
            ratio += cur_step

        # Update last_bw only when moving
        if last_step != 0 or cur_step != 0:
            self.last_bw = cur_bw

        # Clamp ratio
        ratio = max(self.min_ratio, min(self.max_ratio, ratio))

        self.last_step = cur_step
        self.last_ratio = ratio
        return ratio

    def floor_reading(self, bw):
        """Raise a below-cutoff reading to just under the cutoff.

        Below the cutoff the memory system is not what is limiting the workload,
        so how far below it the reading sat says nothing about the ratio -- but it
        is a large relative change, and both the step size and the spectrum are
        computed from relative changes.  Collapsing the whole below-cutoff range
        to one value keeps the unsaturated branch reachable while leaving an idle
        stretch flat instead of loud.
        """
        if bw < self.bw_cutoff:
            return float(self.bw_cutoff - 1)
        return bw

    def _stage1_periods(self, cur_int):
        """Candidate periods for stage 1, longest first.

        The DFT bins of a cur_int-long window are cur_int/k, so between the
        first two bins -- cur_int and cur_int/2 -- there is nothing at all, and
        that gap covers every period from "a little shorter than now" to "half as
        long as now".  Those are the interesting ones, so the two widest gaps are
        filled in at single-sample resolution; Goertzel can evaluate them because
        it does not need its frequencies to land on a bin.

        Descending order is load-bearing: the caller selects the LONGEST period
        within a factor of the strongest one, which is then simply the first
        qualifying index.
        """
        periods = []
        for i in range(1, max(1, cur_int // 2) + 1):
            periods.append(cur_int / float(i))
            if i > 2:
                continue
            j = cur_int // i - GOERTZEL_STEP
            while j > cur_int // (i + 1):
                periods.append(float(j))
                j -= GOERTZEL_STEP
        return periods

    def choose_adjust_interval(self, bw_history, cur_int,
                               min_int=MIN_ADJUST_SAMPLES, max_int=None):
        """Pick the next adjust interval from the bandwidth spectrum.

        Everything here -- cur_int, min_int, max_int, the return value -- counts
        bandwidth samples, and bw_history holds one entry per sample.  A
        transform resolves periods in units of its sampling period, so the
        shortest period this can report is two entries of that history.

        Two stages, because the two questions need different windows:

          1. Can the interval be SHORTER?  Answered over the samples since the
             last ratio adjustment only, so our own actuation is not part of the
             signal.  That window is short, hence Goertzel over a fine sweep
             rather than an FFT limited to its handful of bins.
          2. Should it be LONGER?  Stage 1 cannot answer this, since the longest
             period any transform can report is the length of the window it was
             given -- so a workload period longer than the current interval shows
             up as "the period is exactly the current interval".  That specific
             answer is the trigger for stage 2, which transforms the whole
             history: many samples, so bin resolution is adequate and an FFT is
             the cheaper choice.
        """
        if max_int is None:
            max_int = MAX_ADJUST_INTERVAL_MS
        history = list(bw_history) if bw_history is not None else []
        n = len(history)
        # Below two samples there is no spectrum to read but a DC bin, so the
        # stage-1 sweep would be empty and every path would fall to min_int.
        cur_int = max(MIN_ADJUST_SAMPLES, int(cur_int))
        if n < MIN_ADJUST_SAMPLES:
            return cur_int

        # ------------------------------------------------- stage 1: since last
        periods = self._stage1_periods(cur_int)
        magnitudes = do_goertzel(history, periods, cur_int)

        # Seeded at the SHORTEST period so that a signal with nothing but a DC
        # component falls through to the minimum interval rather than to
        # whichever period happened to be first.
        dominant = len(periods) - 1
        max_magnitude = 0.0
        for i, mag in enumerate(magnitudes):
            if mag > max_magnitude:
                max_magnitude, dominant = mag, i
        if max_magnitude < MIN_MAGNITUDE:
            return min_int

        # Prefer the longest period whose magnitude is within MAGNITUDE_FACTOR of
        # the strongest: when two periods are comparably strong, the longer one
        # contains the shorter, and adjusting on it errs toward acting less often.
        new_int = None
        for i in range(dominant):
            if magnitudes[i] * MAGNITUDE_FACTOR > max_magnitude:
                if i == 0:
                    # The current interval is already as good as the best on
                    # offer, so there is nothing to gain from stage 2 either.
                    return self._clamp_interval(cur_int, min_int, max_int)
                dominant = i
                break
        new_int = int(periods[dominant])
        if new_int < cur_int:
            return self._clamp_interval(new_int, min_int, max_int)

        # Falling through means the strongest period was the current interval,
        # i.e. stage 1 hit its own ceiling.
        if n == cur_int:
            # The history is exactly the window stage 1 just looked at, so there
            # is nothing longer for stage 2 to see.  A history shorter than the
            # interval does go on to stage 2, which resolves to the history
            # length and brings the interval down to what can be observed.
            return self._clamp_interval(cur_int, min_int, max_int)

        # ------------------------------------------------- stage 2: full history
        magnitudes = do_fft(history, n)
        # Bin i is the period n/i, so a LOWER index is a LONGER period.
        # There are n // 2 + 1 bins, and the loops below read cur_idx inclusive,
        # so an odd n at cur_int 2 rounds one bin past the last one that exists.
        cur_idx = min(len(magnitudes) - 1, int(math.ceil(float(n) / cur_int)))
        # Longest period worth checking; never below 1, since bin 0 is DC and the
        # rising-edge test below reads i-1.
        max_idx = max(2, n // max_int)

        # A ratio change, or a workload phase change, leaves a large monotonically
        # decaying skirt in the lowest bins.  Skip it by starting at the first bin
        # that rises above its neighbour, which is the first one carrying a real
        # period rather than the tail of that skirt.
        #
        # When max_idx > cur_idx both loops below are empty and dominant stays at
        # cur_idx, so the interval resolves to the whole history: the longest
        # period there is any evidence for.
        dominant = cur_idx
        max_magnitude = 0.0
        for i in range(max_idx, cur_idx + 1):
            if magnitudes[i] > magnitudes[i - 1] * MAGNITUDE_FACTOR:
                max_magnitude, dominant = magnitudes[i], i
                break
        for i in range(dominant, cur_idx + 1):
            if magnitudes[i] > max_magnitude:
                max_magnitude, dominant = magnitudes[i], i
        return self._clamp_interval(n // dominant, min_int, max_int)

    @staticmethod
    def _clamp_interval(new_int, min_int, max_int):
        """Add the headroom that lets the interval observe its own period, clamp.

        A window of exactly one period cannot resolve that period -- it needs
        strictly more than one -- so the chosen interval would keep re-deriving
        itself and never grow.  Ten percent is enough to break that.
        """
        new_int = int(new_int + max(math.ceil(new_int * 0.1), 1))
        return max(min_int, min(max_int, new_int))
