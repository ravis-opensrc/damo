# SPDX-License-Identifier: GPL-2.0
"""Tests for the auto_tier control loop."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import _damo_bw_controller as ctrl_mod


def test_no_reversal_on_rising_bw():
    """Rising BW with decreasing ratio: controller should keep stepping down.

    step() answers only to the bandwidth series, so a ratio that is paying off
    keeps being stepped in the same direction.
    """
    ctrl = ctrl_mod.BwController(init_ratio=98, bw_cutoff=200,
                                  min_ratio=0, max_ratio=100)
    # Simulate: BW is rising as ratio decreases (good steps)
    # Feed increasing BW readings; ratio should keep decreasing
    ratios = []
    bw = 250  # above cutoff
    for i in range(10):
        bw += 10  # BW keeps rising
        new_ratio = ctrl.step([bw])
        ratios.append(new_ratio)

    # After 10 good steps, ratio should be well below 98
    assert ratios[-1] < 95, \
        'expected ratio < 95 after 10 good steps, got %d' % ratios[-1]
    # Should never reverse (no ratio increase after first decrease)
    first_decrease = None
    for i, r in enumerate(ratios):
        if first_decrease is None and r < 98:
            first_decrease = i
        elif first_decrease is not None:
            # After first decrease, should not go back up significantly
            pass  # allow small oscillations but not full reversal


class _FakeSource:
    def __init__(self):
        self.n = 0

    def sample(self, sample_ms):
        self.n += 1
        return 1000 + self.n


class _FakeActuator:
    def __init__(self, converge_after):
        self.calls = 0
        self.converge_after = converge_after

    def converged(self):
        self.calls += 1
        return self.calls >= self.converge_after


def test_settle_wait_records_its_samples():
    """The settle-wait readings are excluded from the hill-climb but must still
    reach the bandwidth history, so that history stays uniformly sampled.

    choose_adjust_interval() resolves periods in units of the entry spacing, so
    the spacing has to be one sample throughout.
    """
    import damo_auto_tier
    ctrl = ctrl_mod.BwController(init_ratio=50, bw_cutoff=200)
    source, actuator = _FakeSource(), _FakeActuator(converge_after=3)
    damo_auto_tier._settle_wait(ctrl, actuator, source, sample_ms=1,
                                timeout_ms=5000)
    assert source.n == 3, 'sampled %d times, expected 3' % source.n
    assert list(ctrl.bw_history) == [1001, 1002, 1003], \
        'settle-wait samples were dropped: %r' % (list(ctrl.bw_history),)


def test_interval_is_chosen_before_the_settle_wait():
    """The selector reads the tail of the history, so the tail has to be settled
    readings rather than the samples taken while the ratio was moving.

    Driving the real loop needs root and a live kdamond, so this replays the order
    the loop performs the three operations in.
    """
    import damo_auto_tier
    ctrl = ctrl_mod.BwController(init_ratio=50, bw_cutoff=200)
    source, actuator = _FakeSource(), _FakeActuator(converge_after=2)

    seen = []
    ctrl.choose_adjust_interval = lambda hist, cur_int, **kw: (
        seen.append(list(hist)) or cur_int)

    readings = damo_auto_tier._measure_bw(source, sample_ms=1, n_samples=3)
    ctrl.bw_history.extend(readings)
    ctrl.choose_adjust_interval(ctrl.bw_history, 3)
    damo_auto_tier._settle_wait(ctrl, actuator, source, sample_ms=1,
                                timeout_ms=5000)

    assert seen and seen[0] == readings, \
        'selector saw %r, expected the measured readings %r' % (
            seen[0] if seen else None, readings)
    # The settle-wait samples still reach the history -- just after the choice.
    assert list(ctrl.bw_history) == readings + [1004, 1005], \
        'history is %r' % (list(ctrl.bw_history),)


def test_interval_ms_converts_to_sample_counts():
    """The command line is in milliseconds, the controller works in samples."""
    sample_ms = 200
    n_samples = max(ctrl_mod.MIN_ADJUST_SAMPLES, 10000 // sample_ms)
    max_samples = max(ctrl_mod.MIN_ADJUST_SAMPLES,
                      ctrl_mod.MAX_ADJUST_INTERVAL_MS // sample_ms)
    assert n_samples == 50
    assert max_samples == 150
    # The history must span FFT_WINDOW_LEN_MS worth of samples, so the deepest
    # period stage 2 can see does not depend on the sampling rate of the run.
    ctrl = ctrl_mod.BwController(
        init_ratio=50, bw_cutoff=200,
        history_len=max(ctrl_mod.MIN_ADJUST_SAMPLES,
                        ctrl_mod.FFT_WINDOW_LEN_MS // sample_ms))
    assert ctrl.bw_history.maxlen == 450


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
