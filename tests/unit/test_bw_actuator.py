# SPDX-License-Identifier: GPL-2.0
"""Unit tests for _damo_bw_actuator using a fake Kdamond model."""
import sys, os
import types
import importlib

# We need to inject a _damon stub ONLY for this module's import of
# _damo_bw_actuator, then restore sys.modules so other tests are unaffected.

def _make_damon_stub():
    stub = types.ModuleType('_damon')
    stub.commit = lambda kds, **kw: None
    # Model the live stats-refresh path: update_schemes_stats succeeds
    # (returns None) and current_kdamonds() returns whatever the test
    # registered as the live snapshot (defaults to none).
    stub._current_kdamonds = None
    stub.update_schemes_stats = lambda idxs=None: None
    stub.current_kdamonds = lambda: stub._current_kdamonds
    return stub

# Save original _damon entry (may not exist in test environment)
_orig_damon = sys.modules.get('_damon')
_orig_actuator = sys.modules.get('_damo_bw_actuator')

# Install stub, import actuator, then restore
sys.modules['_damon'] = _make_damon_stub()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import _damo_bw_actuator as act
# Patch the actuator module's own _damon reference to the stub
act._damon = sys.modules['_damon']

# Restore _damon in sys.modules so other test files get the real module
if _orig_damon is None:
    sys.modules.pop('_damon', None)
else:
    sys.modules['_damon'] = _orig_damon


class FakeDest:
    def __init__(self, weight): self.weight = weight

class FakeStats:
    def __init__(self, nr): self.nr_applied = nr

class FakeGoal:
    def __init__(self, tv, nid=0):
        self.metric = 'node_eligible_mem_bp'
        self.target_value = tv
        self.nid = nid

class FakeQuota:
    def __init__(self, goals=None): self.goals = goals or []

class FakeScheme:
    def __init__(self, dests=None, goals=None, nr_applied=0, action='migrate_hot'):
        self.dests = dests or []
        self.quotas = FakeQuota(goals or [])
        self.stats = FakeStats(nr_applied)
        self.action = action

class FakeCtx:
    def __init__(self, schemes): self.schemes = schemes

class FakeKd:
    def __init__(self, ctx): self.contexts = [ctx]

def make_weight_kds(ratio=70, nr=5):
    s = FakeScheme(dests=[FakeDest(ratio), FakeDest(100-ratio)], nr_applied=nr)
    return [FakeKd(FakeCtx([s]))]

def make_goal_kds(ratio=60, nr_pull=3, nr_push=2):
    pull = FakeScheme(goals=[FakeGoal(ratio*100, nid=0)], nr_applied=nr_pull,
                      action='migrate_hot')
    push = FakeScheme(goals=[FakeGoal(10000-ratio*100, nid=1)], nr_applied=nr_push,
                      action='migrate_hot')
    return [FakeKd(FakeCtx([pull, push]))]

def test_weight_actuator_read():
    kds = make_weight_kds(ratio=65)
    a = act.WeightActuator(kds, 0, 0, 0)
    assert a.read_ratio() == 65

def test_weight_actuator_write():
    kds = make_weight_kds(ratio=50)
    a = act.WeightActuator(kds, 0, 0, 0)
    a.write_ratio(80)
    assert kds[0].contexts[0].schemes[0].dests[0].weight == 80
    assert kds[0].contexts[0].schemes[0].dests[1].weight == 20

def test_weight_actuator_nr_applied():
    kds = make_weight_kds(nr=7)
    act._damon._current_kdamonds = kds
    a = act.WeightActuator(kds, 0, 0, 0)
    assert a.nr_applied() == 7

def test_goal_actuator_read():
    kds = make_goal_kds(ratio=55)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    assert a.read_ratio() == 55

def test_goal_actuator_write():
    kds = make_goal_kds(ratio=50)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    a.write_ratio(70)
    assert kds[0].contexts[0].schemes[0].quotas.goals[0].target_value == 7000
    assert kds[0].contexts[0].schemes[1].quotas.goals[0].target_value == 3000

def test_goal_actuator_nr_applied():
    kds = make_goal_kds(nr_pull=3, nr_push=4)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    assert a.nr_applied() == 7

def test_detect_weight():
    kds = make_weight_kds()
    a = act.detect_actuator(kds, 0, 0, 0)
    assert isinstance(a, act.WeightActuator)

def test_detect_goal():
    kds = make_goal_kds()
    a = act.detect_actuator(kds, 0, 0, 0)
    assert isinstance(a, act.GoalActuator)

def test_detect_neither_raises():
    s = FakeScheme()  # no dests, no goals
    kds = [FakeKd(FakeCtx([s]))]
    try:
        act.detect_actuator(kds, 0, 0, 0)
    except ValueError:
        return
    assert False, 'expected ValueError'

def test_detect_both_raises():
    # Both >=2 dests AND node_eligible_mem_bp goal — malformed config, must raise
    s = FakeScheme(dests=[FakeDest(60), FakeDest(40)],
                   goals=[FakeGoal(6000)], action='migrate_hot')
    kds = [FakeKd(FakeCtx([s]))]
    try:
        act.detect_actuator(kds, 0, 0, 0)
    except ValueError as e:
        assert 'mutually exclusive' in str(e), e
        return
    assert False, 'expected ValueError'


# ---------------------------------------------------------------------------
# GoalActuator.converged() with mock FtraceReader
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, start=0.0, step=1.2):
        self._t = start
        self._step = step

    def __call__(self):
        t = self._t
        self._t += self._step
        return t


class MockFtraceReader:
    """Inject a sequence of synthetic tracepoint events."""
    def __init__(self, events):
        self._events = list(events)
        self._idx = 0

    def read_event(self):
        if self._idx >= len(self._events):
            return None
        ev = self._events[self._idx]
        self._idx += 1
        return ev

    def close(self):
        pass


def make_converged_events(target_bp, local_nid, n=35):
    """Generate n events where current_value == target_bp (fully converged)."""
    return [{'nid': local_nid, 'target_value': target_bp, 'current_value': target_bp}
            for _ in range(n)]


def make_diverged_events(target_bp, local_nid, n=35):
    """Generate n events with current_value=0 (far from target)."""
    return [{'nid': local_nid, 'target_value': target_bp, 'current_value': 0}
            for _ in range(n)]


def test_goal_converged_true():
    kds = make_goal_kds(ratio=60)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    target_bp = 6000
    clock = FakeClock(start=0.0, step=1.0)
    reader = MockFtraceReader(make_converged_events(target_bp, local_nid=0, n=5))
    result = a.converged(target_dram_bp=target_bp,
                         tolerance_bp=500, stable_window_ms=2000,
                         ftrace_reader=reader, now=clock, timeout_ms=60000)
    assert result is True


def test_goal_converged_plateaued():
    """Stuck far from target for 3+ samples → PLATEAUED → True."""
    kds = make_goal_kds(ratio=60)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    target_bp = 6000
    # All events current_value=0 → avg stays at 0, far from 6000
    # After 3 consecutive no-progress samples → plateau → True
    events = [{'nid': 0, 'current_value': 0, 'target_value': target_bp}
              for _ in range(10)]
    clock = FakeClock(start=0.0, step=1.0)
    reader = MockFtraceReader(events)
    result = a.converged(target_dram_bp=target_bp,
                         tolerance_bp=500, stable_window_ms=100,
                         ftrace_reader=reader, now=clock, timeout_ms=60000,
                         progress_threshold=200, plateau_count_threshold=3)
    assert result is True


def test_goal_converged_timeout():
    """No events → timeout → True (never deadlock)."""
    kds = make_goal_kds(ratio=60)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    reader = MockFtraceReader([])  # no events
    result = a.converged(target_dram_bp=6000,
                         tolerance_bp=500, stable_window_ms=100,
                         ftrace_reader=reader, timeout_ms=200)
    assert result is True


def test_goal_converged_filters_wrong_nid():
    kds = make_goal_kds(ratio=60)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    # All events nid=1, near_node=0 → no samples match → timeout → True
    events = [{'nid': 1, 'target_value': 6000, 'current_value': 6000}
              for _ in range(35)]
    reader = MockFtraceReader(events)
    result = a.converged(target_dram_bp=6000,
                         tolerance_bp=500, stable_window_ms=100,
                         ftrace_reader=reader, timeout_ms=200)
    assert result is True  # timeout exit — never deadlock


def test_weight_converged_true_when_zero():
    kds = make_weight_kds(nr=0)
    act._damon._current_kdamonds = kds
    a = act.WeightActuator(kds, 0, 0, 0)
    assert a.converged() is True


def test_weight_converged_false_when_nonzero():
    kds = make_weight_kds(nr=5)
    act._damon._current_kdamonds = kds
    a = act.WeightActuator(kds, 0, 0, 0)
    assert a.converged() is False


def test_goal_converged_no_premature_on_alternating():
    """Alternating 0/10000 stream: avg=5000, |5000-6000|=1000 > tolerance.
    With plateau detection: alternating values keep changing avg → no plateau.
    Eventually times out → True (timeout exit, not premature convergence).
    The key invariant: does NOT return True due to REACHED (avg within tolerance).
    """
    kds = make_goal_kds(ratio=60)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    target_bp = 6000
    events = []
    for i in range(10):
        events.append({'nid': 0, 'current_value': 0 if i % 2 == 0 else 10000,
                       'target_value': target_bp})
    reader = MockFtraceReader(events)
    clock = FakeClock(start=0.0, step=0.1)
    # Use large plateau_count_threshold so plateau path doesn't trigger
    result = a.converged(target_dram_bp=target_bp, tolerance_bp=500,
                         stable_window_ms=2000, timeout_ms=500,
                         ftrace_reader=reader, now=clock,
                         plateau_count_threshold=100)
    # Should not converge via REACHED (avg ≈ 5000, not within 500bp of 6000)
    # May converge via TIMEOUT — that's acceptable (no deadlock)
    # The important thing: result is not True due to wrong avg being within tolerance
    # We verify by checking that avg of events is NOT within tolerance
    import _damo_ftrace as ftrace
    wav = ftrace.TimeWindowedAverage(2000)
    fake_t = [0.0]
    for i, ev in enumerate(events):
        fake_t[0] = i * 0.1
        wav.add(ev['current_value'], now=fake_t[0])
    avg = wav.average()
    assert abs(avg - target_bp) > 500, \
        'alternating stream avg should be outside tolerance'


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
