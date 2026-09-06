# SPDX-License-Identifier: GPL-2.0
"""Hermetic tests for BwController algorithm."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from _damo_bw_controller import BwController, MAX_STEP, MAX_ADJUST_INTERVAL_MS

CUTOFF = 200  # MB/s

def test_rising_bw_ratio_climbs():
    """Rising BW -> ratio should decrease (explore more far memory)."""
    ctrl = BwController(init_ratio=100, bw_cutoff=CUTOFF)
    # Feed rising BW series above cutoff
    bw_series = [210, 220, 230, 240, 250, 260, 270, 280]
    ratios = []
    for bw in bw_series:
        r = ctrl.step([bw])
        ratios.append(r)
    # Ratio should have moved from 100 (probing downward)
    assert ratios[-1] < 100, 'ratio should decrease when BW is above cutoff'

def test_below_cutoff_ratio_increases():
    """BW below cutoff -> ratio should increase toward 100."""
    ctrl = BwController(init_ratio=50, bw_cutoff=CUTOFF)
    bw_series = [100, 110, 120, 130, 140]
    ratios = []
    for bw in bw_series:
        r = ctrl.step([bw])
        ratios.append(r)
    assert ratios[-1] > 50, 'ratio should increase when BW is below cutoff'

def test_plateau_ratio_stabilizes():
    """Plateau BW -> step goes to 0, ratio stops changing."""
    ctrl = BwController(init_ratio=70, bw_cutoff=CUTOFF)
    # First move the controller
    for bw in [210, 220, 230]:
        ctrl.step([bw])
    # Now plateau
    plateau_bw = 235
    ratios = []
    for _ in range(10):
        r = ctrl.step([plateau_bw])
        ratios.append(r)
    # After plateau, step should converge to 0 and ratio stabilize
    assert ratios[-1] == ratios[-2], 'ratio should stabilize on plateau'

def test_ratio_clamped():
    """Ratio must stay in [0, 100]."""
    ctrl = BwController(init_ratio=100, bw_cutoff=CUTOFF)
    for _ in range(50):
        r = ctrl.step([300])  # always above cutoff, always probing down
        assert 0 <= r <= 100

def test_ratio_clamped_custom_bounds():
    ctrl = BwController(init_ratio=80, bw_cutoff=CUTOFF, min_ratio=20, max_ratio=90)
    for bw in [100] * 20:  # below cutoff, ratio should climb but not exceed 90
        r = ctrl.step([bw])
        assert 20 <= r <= 90

def test_choose_interval_ignores_hill_climb_state():
    # The interval comes from the bandwidth spectrum alone.  Whether the ratio
    # happens to be moving is not part of that signal, so last_step must not
    # change the answer.
    flat = [500.0] * 8
    results = []
    for last_step in (0, 4, -4):
        ctrl = BwController(init_ratio=70, bw_cutoff=CUTOFF)
        ctrl.last_step = last_step
        results.append(ctrl.choose_adjust_interval(flat, 16, min_int=2,
                                                   max_int=256))
    assert len(set(results)) == 1, 'last_step changed the interval: %r' % results

def test_choose_interval_counts_samples():
    # cur_int, min_int, max_int and the return value are sample counts, so a flat
    # signal falls to min_int rather than to some millisecond value.
    ctrl = BwController(init_ratio=70, bw_cutoff=CUTOFF)
    flat = [500.0] * 8
    assert ctrl.choose_adjust_interval(flat, 16, min_int=2, max_int=256) == 2

def test_step_returns_int():
    ctrl = BwController(init_ratio=60, bw_cutoff=CUTOFF)
    r = ctrl.step([250])
    assert isinstance(r, int)

def test_floor_reading_flattens_below_cutoff():
    ctrl = BwController(init_ratio=70, bw_cutoff=CUTOFF)
    assert ctrl.floor_reading(10) == ctrl.floor_reading(150) == CUTOFF - 1
    assert ctrl.floor_reading(CUTOFF) == CUTOFF
    assert ctrl.floor_reading(500) == 500


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
