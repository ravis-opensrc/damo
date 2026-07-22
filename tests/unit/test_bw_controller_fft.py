# SPDX-License-Identifier: GPL-2.0
"""Unit tests for the spectral adjust-interval selector.

Everything here counts bandwidth samples: cur_int, min_int, max_int and the
return value are all sample counts, and the history holds one entry per sample.
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import _damo_bw_controller as ctrl_mod
from _damo_bw_controller import BwController, MIN_ADJUST_SAMPLES

CUTOFF = 200


def sine(period, n, amp=10000, dc=20000):
    """A bandwidth series in MB/s with one dominant period, in samples."""
    return [dc + amp * math.sin(2 * math.pi * i / period) for i in range(n)]


def ctrl():
    return BwController(init_ratio=50, bw_cutoff=CUTOFF)


# ------------------------------------------------------------- signal handling
def test_process_signal_removes_dc():
    out = ctrl_mod.process_signal([100.0] * 32, 32, False)
    assert all(abs(x) < 1e-9 for x in out), 'DC not removed'


def test_process_signal_pads_the_old_end():
    """A short history fills the OLDEST end, so the newest sample stays last."""
    out = ctrl_mod.process_signal([1000.0, 2000.0], 8, False)
    assert len(out) == 8
    # mean over the whole window = 3.0 GB/s / 8 = 0.375
    assert abs(out[-1] - (2.0 - 0.375)) < 1e-9
    assert abs(out[-2] - (1.0 - 0.375)) < 1e-9
    assert all(abs(x - (-0.375)) < 1e-9 for x in out[:6])


def test_process_signal_length_is_the_window():
    """Length is window_size regardless of history length, so a period means the
    same span of time on every call."""
    for hist_len in (1, 5, 32, 100):
        out = ctrl_mod.process_signal([1000.0] * hist_len, 32, True)
        assert len(out) == 32, 'history %d gave %d' % (hist_len, len(out))


def test_hamming_is_skipped_on_short_windows():
    """Windowing widens the main lobe, which would smear together exactly the
    between-bin periods stage 1 probes."""
    n = ctrl_mod.HAMMING_WINDOW_THRESH - 1
    hist = sine(8, n)
    plain = ctrl_mod.process_signal(hist, n, False)
    windowed = ctrl_mod.process_signal(hist, n, True)
    assert plain != windowed, 'the flag must actually do something'
    # do_goertzel picks by window length, and n is below the threshold.
    assert (ctrl_mod.do_goertzel(hist, [8.0], n)[0]
            == ctrl_mod.goertzel_magnitude(plain, 1.0 / 8))


def test_hamming_is_applied_on_long_windows():
    n = ctrl_mod.HAMMING_WINDOW_THRESH
    hist = sine(8, n)
    windowed = ctrl_mod.process_signal(hist, n, True)
    assert (ctrl_mod.do_goertzel(hist, [8.0], n)[0]
            == ctrl_mod.goertzel_magnitude(windowed, 1.0 / 8))


def test_goertzel_beats_the_fft_between_bins():
    """The reason stage 1 uses Goertzel: a period that is not window_size/k."""
    n = 40
    period = 13.0            # 40/13 is not an integer, so there is no bin here
    hist = sine(period, n)
    at_period = ctrl_mod.do_goertzel(hist, [period], n)[0]
    fft = ctrl_mod.do_fft(hist, n)      # nearest bins: 40/3=13.33, 40/4=10
    assert at_period > fft[3], 'Goertzel at the true period must beat bin 3'
    assert at_period > fft[4]


def test_fft_dc_bin_is_removed():
    assert ctrl_mod.do_fft([20000.0] * 64, 64)[0] < 1e-6


def test_fft_fallback_matches_numpy():
    """The pure-Python DFT is the same transform, not an approximation."""
    if not ctrl_mod._have_numpy:
        return
    hist = sine(9, 32)
    ref = ctrl_mod.do_fft(hist, 32)
    ctrl_mod._have_numpy = False
    try:
        alt = ctrl_mod.do_fft(hist, 32)
    finally:
        ctrl_mod._have_numpy = True
    assert len(ref) == len(alt)
    for a, b in zip(ref, alt):
        assert abs(a - b) < 1e-6 * max(1.0, abs(a)), '%r vs %r' % (ref, alt)


# ----------------------------------------------------- stage 1: can it shorten?
def test_stage1_periods_descend():
    """Longest first, so "longest within MAGNITUDE_FACTOR" is the first hit."""
    periods = ctrl()._stage1_periods(20)
    assert periods == sorted(periods, reverse=True)
    assert periods[0] == 20.0
    assert min(periods) >= 2.0


def test_stage1_periods_fill_the_first_two_gaps():
    """Between bins 1 and 2 (periods 20 and 10) an FFT has nothing at all."""
    periods = ctrl()._stage1_periods(20)
    for p in range(11, 21):
        assert float(p) in periods, 'missing period %d' % p


def test_shortens_to_a_period_below_the_current_interval():
    result = ctrl().choose_adjust_interval(sine(5, 200), 40,
                                          min_int=2, max_int=128)
    assert 5 <= result <= 8, 'expected ~5 samples plus headroom, got %d' % result


def test_pure_dc_falls_to_the_minimum():
    """No periodicity at all: adjust as often as allowed rather than holding."""
    assert ctrl().choose_adjust_interval([20000.0] * 200, 40, min_int=3,
                                         max_int=128) == 3


def test_prefers_the_longer_of_two_comparable_periods():
    """Two similar-magnitude periods: take the longer, which contains both."""
    n = 400
    hist = [20000 + 10000 * math.sin(2 * math.pi * i / 24)
            + 10000 * math.sin(2 * math.pi * i / 12) for i in range(n)]
    result = ctrl().choose_adjust_interval(hist, 48, min_int=2, max_int=256)
    assert result >= 24, 'expected the 24-sample period, got %d' % result


def test_below_min_magnitude_is_treated_as_dc():
    """Periodic but too weak to act on: fall to min_int, not to its period."""
    tiny = [20000 + 0.001 * math.sin(2 * math.pi * i / 5) for i in range(200)]
    assert ctrl().choose_adjust_interval(tiny, 40, min_int=4, max_int=128) == 4


# ---------------------------------------------------- stage 2: can it lengthen?
def test_lengthens_past_the_stage1_ceiling():
    """The period is LONGER than the current interval.

    Stage 1 cannot say so -- the longest period it can report is the window it
    was given -- so this is the case that must reach stage 2.

    It takes more than one call: from a short interval the current one still
    leaks enough magnitude to be within MAGNITUDE_FACTOR of the peak, which holds
    the interval and grows it only by the headroom.  Once it is long enough for
    the true period to dominate, stage 2 lands on it and it stays.
    """
    period, cur, c = 40, 20, ctrl()
    hist = sine(period, 400)
    seen = []
    for _ in range(8):
        cur = c.choose_adjust_interval(hist, cur, min_int=2, max_int=256)
        seen.append(cur)
    assert seen[-1] > 20, 'never lengthened the interval: %r' % (seen,)
    assert abs(seen[-1] - period) <= period * 0.35, \
        'expected ~%d samples, got %r' % (period, seen)


def test_no_history_beyond_the_interval_keeps_it():
    """Nothing longer to look at: hold, do not invent a longer period."""
    result = ctrl().choose_adjust_interval(sine(20, 20), 20,
                                           min_int=2, max_int=256)
    assert result == 22, 'expected cur_int plus headroom, got %d' % result


def test_interval_longer_than_the_history_is_pulled_back():
    """An interval longer than the history cannot be observed, so it must come
    down to the history length rather than keep growing off the end of it.

    Only "history == interval" holds; "history < interval" goes on to stage 2,
    where the bin range is empty and the answer is the whole history -- the
    longest period there is any evidence for.
    """
    c, hist = ctrl(), sine(20, 20)
    seen = []
    cur = 20
    for _ in range(10):
        cur = c.choose_adjust_interval(hist, cur, min_int=2, max_int=256)
        seen.append(cur)
    assert max(seen) <= 2 * len(hist), \
        'ran away past the history length: %r' % (seen,)
    assert min(seen) <= len(hist) + 2, \
        'never came back to the observable range: %r' % (seen,)


def test_low_frequency_skirt_is_skipped():
    """A ratio change leaves a large decaying skirt in the lowest bins, which
    must not read as a very long workload period.  Here a one-off ramp (our own
    actuation) sits under a real 30-sample period."""
    n = 300
    hist = [20000 + 10000 * math.sin(2 * math.pi * i / 30) + 40000 * (1 - i / n)
            for i in range(n)]
    result = ctrl().choose_adjust_interval(hist, 15, min_int=2, max_int=256)
    assert result < 120, \
        'read the actuation skirt as the period (got %d)' % result


def test_headroom_lets_the_interval_see_its_own_period():
    """A window of exactly one period cannot resolve that period, so the chosen
    interval is always strictly longer than the period it came from."""
    for period in (5, 17, 40):
        got = BwController._clamp_interval(period, 2, 10000)
        assert got > period, 'period %d gave %d' % (period, got)


# ------------------------------------------------------------ bounds/degenerate
def test_clamped_to_bounds():
    for hist in ([20000.0] * 200, sine(3, 200), sine(64, 400)):
        got = ctrl().choose_adjust_interval(hist, 16, min_int=5, max_int=20)
        assert 5 <= got <= 20, 'out of bounds: %d' % got


def test_empty_and_tiny_history_keep_the_interval():
    assert ctrl().choose_adjust_interval([], 7) == 7
    assert ctrl().choose_adjust_interval([100.0], 7) == 7
    assert ctrl().choose_adjust_interval(None, 7) == 7


def test_min_int_is_honoured_as_the_floor():
    """min_int is the floor, including on the DC path, which returns it directly.

    An interval of one sample has no spectrum but DC, so the caller passes
    MIN_ADJUST_SAMPLES rather than 1 and the floor is a property of that choice,
    not something this function second-guesses.
    """
    c = ctrl()
    assert c.choose_adjust_interval([20000.0] * 100, 8, min_int=1,
                                    max_int=64) == 1
    assert c.choose_adjust_interval([20000.0] * 100, 8,
                                    min_int=MIN_ADJUST_SAMPLES,
                                    max_int=64) == MIN_ADJUST_SAMPLES


def test_interval_is_stable_under_repetition():
    """A steady periodic signal must converge, not oscillate forever."""
    c, hist, cur, seen = ctrl(), sine(24, 400), 8, []
    for _ in range(12):
        cur = c.choose_adjust_interval(hist, cur, min_int=2, max_int=256)
        seen.append(cur)
    assert seen[-1] == seen[-2] == seen[-3], 'did not settle: %r' % seen


def test_repeated_calls_reach_a_bounded_limit_cycle():
    """Not every signal settles on a single value.

    Some periods end on a short limit cycle instead: the chosen interval is a
    little longer than the period, the transform then reports that inflated
    interval as its own dominant period, and the next call trims it back.  The
    cycle is bounded, not divergent, so what is asserted is that the sequence
    becomes exactly periodic and stays inside a narrow band, not that it
    reaches a fixed point.
    """
    for period in (5, 13, 24, 40, 60, 97):
        c, hist, cur, seen = ctrl(), sine(period, 400), 8, []
        for _ in range(120):
            cur = c.choose_adjust_interval(hist, cur, min_int=2, max_int=256)
            seen.append(cur)
        tail = seen[-40:]
        cycle = next((n for n in range(1, 21)
                      if all(tail[i] == tail[i - n]
                             for i in range(n, len(tail)))), None)
        assert cycle is not None, \
            'period %d never became periodic: %r' % (period, tail)
        band = max(tail) - min(tail)
        assert band <= max(4, 0.3 * min(tail)), \
            'period %d cycles too widely: %r' % (period, sorted(set(tail)))


# ----------------------------------------------------------- small-input bounds
# The three boundaries below are all cases where a window or an interval is too
# small for the transform it feeds, and each is guarded in exactly one place.
def test_odd_history_at_the_shortest_interval_stays_in_the_bins():
    """cur_int 2 over an odd history rounds one past the last bin that exists.

    The stage-2 loops read the current-interval index inclusive, so without the
    clamp this indexes off the end of the transform output.
    """
    for n in range(3, 40, 2):
        hist = sine(5, n)
        got = ctrl().choose_adjust_interval(hist, 2, min_int=2, max_int=256)
        assert 2 <= got <= 256, 'n=%d gave %d' % (n, got)


def test_interval_below_two_samples_is_raised_to_two():
    """One sample per interval has no spectrum but DC, so stage 1 would have no
    periods to sweep and every path would fall to min_int."""
    hist = [90000.0, 10000.0] * 3
    for cur in (0, 1, 2):
        got = ctrl().choose_adjust_interval(hist, cur, min_int=1, max_int=256)
        assert got == 3, 'cur_int %d gave %d' % (cur, got)


def test_single_sample_window_has_no_hamming_shape():
    """The window term divides by window_size - 1."""
    assert ctrl_mod.process_signal([50000.0], 1, True) == [0.0]
    assert ctrl_mod.do_fft([50000.0], 1) == [0.0]


# ------------------------------------------------------------ history plumbing
def test_step_leaves_the_history_to_its_caller():
    """The hill-climb reads only the readings it was handed; appending them to
    the history is the caller's."""
    c = ctrl()
    c.step([100, 200, 300])
    assert list(c.bw_history) == []


def test_history_is_bounded():
    c = BwController(init_ratio=50, bw_cutoff=CUTOFF, history_len=10)
    for i in range(50):
        c.bw_history.append(i)
    assert len(c.bw_history) == 10
    assert list(c.bw_history)[-1] == 49


if __name__ == '__main__':
    # test.sh runs each file with python3, so the tests need an explicit
    # runner here.
    import traceback
    failed = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith('test_') or not callable(fn):
            continue
        try:
            fn()
        except Exception:
            failed += 1
            print('FAIL %s' % name)
            traceback.print_exc()
    if failed:
        sys.exit(1)
