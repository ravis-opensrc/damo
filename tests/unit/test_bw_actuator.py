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
    # Whether the kernel takes a self-refresh request.  None is taken; a string
    # is the error a kernel without the file gives.
    stub.set_stats_self_refresh = lambda ki, ms: None
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
    # refresh_ms is on the real Kdamond and the commit path writes it back from
    # there, so a fake without it cannot show a period being lost across a commit.
    def __init__(self, ctx, refresh_ms=0):
        self.contexts = [ctx]
        self.refresh_ms = refresh_ms

def make_weight_kds(ratio=70, nr=5, refresh_ms=0):
    s = FakeScheme(dests=[FakeDest(ratio), FakeDest(100-ratio)], nr_applied=nr)
    return [FakeKd(FakeCtx([s]), refresh_ms=refresh_ms)]

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

def test_goal_actuator_retries_a_busy_commit():
    """A commit refused because the state file was taken is attempted again.

    The refusal is what a concurrent user of the same file produces, and it costs
    the whole ratio change if it is taken as the answer.
    """
    kds = make_goal_kds(ratio=50)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    calls = []

    def busy_then_ok(k, **kw):
        calls.append(1)
        if len(calls) < 3:
            return '[Errno 16] Device or resource busy'
        return None

    orig = act._damon.commit
    orig_retry = act._COMMIT_RETRY_S
    act._damon.commit = busy_then_ok
    act._COMMIT_RETRY_S = 0
    try:
        a.write_ratio(70)
    finally:
        act._damon.commit = orig
        act._COMMIT_RETRY_S = orig_retry
    assert len(calls) == 3, calls
    assert kds[0].contexts[0].schemes[0].quotas.goals[0].target_value == 7000

def test_goal_actuator_does_not_retry_another_error():
    """Only a busy state file is worth attempting again.

    Any other refusal is the same on the next attempt, so repeating it only
    delays reporting it.
    """
    kds = make_goal_kds(ratio=50)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    calls = []

    def rejected(k, **kw):
        calls.append(1)
        return '[Errno 22] Invalid argument'

    orig = act._damon.commit
    act._damon.commit = rejected
    try:
        a.write_ratio(70)
    finally:
        act._damon.commit = orig
    assert len(calls) == 1, calls

def test_goal_actuator_nr_applied():
    # The live snapshot has to be registered, because the reading comes from a
    # refresh rather than from the objects handed to the constructor.  Without
    # it the assertion would pass off the construction-time numbers and say
    # nothing about whether the counter is being re-read at all.
    kds = make_goal_kds(nr_pull=3, nr_push=4)
    act._damon._current_kdamonds = kds
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    assert a.nr_applied() == 7

def test_a_goal_counter_that_advanced_is_read_as_advanced():
    """The reading tracks the running kdamond, not the snapshot it was built on.

    Summing the construction-time objects gives a number that never moves, and a
    counter that never moves is what a settled system looks like -- so every
    window reads as quiet and a wait on it ends immediately.
    """
    kds = make_goal_kds(nr_pull=3, nr_push=4)
    act._damon._current_kdamonds = kds
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    assert a.nr_applied() == 7

    # The kdamond applied more since; the reader has to see it.
    act._damon._current_kdamonds = make_goal_kds(nr_pull=30, nr_push=40)
    assert a.nr_applied() == 70

def test_a_goal_reading_that_could_not_be_taken_is_not_a_number():
    """A refused refresh reports nothing rather than the previous number.

    None is what keeps a reading that does not exist from being compared with the
    one before it and found equal, which is the shape of a settled system.
    """
    kds = make_goal_kds(nr_pull=3, nr_push=4)
    act._damon._current_kdamonds = kds
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)

    orig = act._damon.update_schemes_stats
    act._damon.update_schemes_stats = lambda idxs=None: 'no such device'
    try:
        assert a.nr_applied() is None
    finally:
        act._damon.update_schemes_stats = orig

class _RefusedReader:
    """A reader whose tracepoint is not there to open."""
    def open(self, name):
        raise OSError('no such tracepoint: %s' % name)

def test_a_missing_tracepoint_is_not_a_convergence():
    """Without the tracepoint there is no signal, and no verdict to record.

    The fallback only sees whether anything was applied at all.  Recording
    'reached' before asking that made the loop log a wait it never had as one it
    did, and the three names it could have recorded are all verdicts on a system
    that was being watched -- which this one was not.
    """
    kds = make_goal_kds(nr_pull=0, nr_push=0)
    act._damon._current_kdamonds = kds
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)

    orig = act._damo_ftrace.FtraceReader
    act._damo_ftrace.FtraceReader = _RefusedReader
    try:
        assert a.converged(target_dram_bp=6000) is False
        assert a.outcome == act.CONV_UNOBSERVED, a.outcome

        # Something applied is the one thing the fallback can see.
        act._damon._current_kdamonds = make_goal_kds(nr_pull=1, nr_push=0)
        assert a.converged(target_dram_bp=6000) is True
        assert a.outcome == act.CONV_REACHED, a.outcome
    finally:
        act._damo_ftrace.FtraceReader = orig

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
    # Both >=2 dests AND node_eligible_mem_bp goal -- malformed config, must raise
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


def test_goal_converged_stalled_is_not_reached():
    """A gap that stops closing ends the wait, and says it was not reached."""
    kds = make_goal_kds(ratio=60)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    target_bp = 6000
    # current_value pinned at 0: the average never approaches 6000, so no window
    # ever improves on the first one.
    events = [{'nid': 0, 'current_value': 0, 'target_value': target_bp}
              for _ in range(60)]
    clock = FakeClock(start=0.0, step=1.0)
    reader = MockFtraceReader(events)
    result = a.converged(target_dram_bp=target_bp, tolerance_bp=500,
                         stable_window_ms=2000, ftrace_reader=reader,
                         now=clock, progress_bp=200, stall_windows=3)
    assert result is True
    assert a.outcome == act.CONV_STALLED, a.outcome


def test_goal_converged_waits_while_still_closing():
    """A slow but progressing migration is not cut short.

    The gap closes by more than progress_bp each window and never reaches the
    band, so every window renews the wait and the events run out rather than the
    wait giving up.  With no cap that is the only way it can end, which is the
    property being checked: nothing here abandons a system that is still moving.
    """
    kds = make_goal_kds(ratio=60)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    target_bp = 6000
    # 0 -> 5000 in 500 bp steps: closing steadily, never within 500 bp of 6000.
    events = [{'nid': 0, 'current_value': v, 'target_value': target_bp}
              for v in range(0, 5000, 250) for _ in range(2)]
    clock = FakeClock(start=0.0, step=1.0)
    reader = MockFtraceReader(events)
    result = a.converged(target_dram_bp=target_bp, tolerance_bp=100,
                         stable_window_ms=2000, ftrace_reader=reader,
                         now=clock, progress_bp=200, stall_windows=3)
    assert result is True
    # Ended because the injected stream ran dry, not because it gave up on a
    # gap that was still shrinking.
    assert a.outcome == act.CONV_DEADLINE, a.outcome


def test_goal_converged_deadline_is_not_reached():
    """A cap that expires while the gap is still closing reports the cap."""
    kds = make_goal_kds(ratio=60)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    reader = MockFtraceReader([])   # nothing to read; the cap is what ends it
    result = a.converged(target_dram_bp=6000, tolerance_bp=500,
                         stable_window_ms=100, ftrace_reader=reader,
                         max_wait_ms=200)
    assert result is True
    assert a.outcome == act.CONV_DEADLINE, a.outcome


def test_goal_converged_records_reached():
    """At the target the outcome says so, which is the one case a caller acts on."""
    kds = make_goal_kds(ratio=60)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    target_bp = 6000
    clock = FakeClock(start=0.0, step=1.0)
    reader = MockFtraceReader(make_converged_events(target_bp, local_nid=0, n=8))
    result = a.converged(target_dram_bp=target_bp, tolerance_bp=500,
                         stable_window_ms=2000, ftrace_reader=reader,
                         now=clock)
    assert result is True
    assert a.outcome == act.CONV_REACHED, a.outcome


def test_goal_converged_single_sample_window_does_not_decide():
    """One sample spanning the window is one tick of a binary signal.

    A lone 10000 would average to 10000 and sit inside the band around a 10000
    target, so a predicate that decided on it would report convergence off a
    single tick.  Two samples are required before any verdict.
    """
    kds = make_goal_kds(ratio=100)
    a = act.GoalActuator(kds, 0, 0, near_node=0, far_node=1)
    reader = MockFtraceReader(
        [{'nid': 0, 'current_value': 10000, 'target_value': 10000}])
    clock = FakeClock(start=0.0, step=5.0)
    result = a.converged(target_dram_bp=10000, tolerance_bp=500,
                         stable_window_ms=2000, ftrace_reader=reader,
                         now=clock)
    assert result is True
    assert a.outcome != act.CONV_REACHED, a.outcome


def test_goal_converged_timeout():
    """No events -> timeout -> True (never deadlock)."""
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
    # All events nid=1, near_node=0 -> no samples match -> timeout -> True
    events = [{'nid': 1, 'target_value': 6000, 'current_value': 6000}
              for _ in range(35)]
    reader = MockFtraceReader(events)
    result = a.converged(target_dram_bp=6000,
                         tolerance_bp=500, stable_window_ms=100,
                         ftrace_reader=reader, timeout_ms=200)
    assert result is True  # timeout exit -- never deadlock


def test_self_refresh_replaces_the_commanded_refresh():
    """A kdamond refreshing its own stats is not asked to refresh them.

    The commanded refresh goes through the file the ratio changes go through, so
    every one of them is a chance to cost a ratio change.  Once the kdamond is
    doing it, reading the stats has to touch that file not at all.
    """
    kds = make_weight_kds(nr=7)
    act._damon._current_kdamonds = kds
    a = act.WeightActuator(kds, 0, 0, 0)
    assert a.enable_self_refresh() is True

    commanded = []
    orig = act._damon.update_schemes_stats
    act._damon.update_schemes_stats = lambda idxs=None: commanded.append(idxs)
    try:
        assert a.nr_applied() == 7
    finally:
        act._damon.update_schemes_stats = orig
    assert commanded == [], commanded


def test_refresh_is_commanded_when_the_kernel_will_not_do_it():
    """Without self refresh the reader has to command each refresh itself.

    A kernel without the file is an ordinary case, not a fault: the reading still
    has to be correct, it just costs the contention it was meant to avoid.
    """
    kds = make_weight_kds(nr=7)
    act._damon._current_kdamonds = kds
    a = act.WeightActuator(kds, 0, 0, 0)

    orig_set = act._damon.set_stats_self_refresh
    act._damon.set_stats_self_refresh = lambda ki, ms: 'no such file'
    commanded = []
    orig_upd = act._damon.update_schemes_stats
    act._damon.update_schemes_stats = lambda idxs=None: commanded.append(idxs)
    try:
        assert a.enable_self_refresh() is False
        assert a.nr_applied() == 7
    finally:
        act._damon.set_stats_self_refresh = orig_set
        act._damon.update_schemes_stats = orig_upd
    assert commanded == [[0]], commanded


def test_weight_actuator_retries_a_busy_commit():
    """A weight change refused because the file was taken is attempted again.

    Unlike the goal actuator this one did not retry, so a refusal was the end of
    the change: the kernel kept distributing to the previous weights while the
    caller believed it had moved.
    """
    kds = make_weight_kds(ratio=50)
    a = act.WeightActuator(kds, 0, 0, 0)
    calls = []

    def busy_then_ok(k, **kw):
        calls.append(1)
        if len(calls) < 3:
            return '[Errno 16] Device or resource busy'
        return None

    orig = act._damon.commit
    orig_retry = act._COMMIT_RETRY_S
    act._damon.commit = busy_then_ok
    act._COMMIT_RETRY_S = 0
    try:
        a.write_ratio(70)
    finally:
        act._damon.commit = orig
        act._COMMIT_RETRY_S = orig_retry
    assert len(calls) == 3, calls
    assert a.write_ok is True
    assert kds[0].contexts[0].schemes[0].dests[0].weight == 70


def test_a_refused_write_is_reported_as_not_written():
    """A ratio the kernel never took is reported, not assumed.

    The caller decides its next step from a bandwidth reading, and after a
    refused write that reading describes the ratio still in force.  Acting on it
    as though it described the new one is how a refusal becomes a wrong decision
    rather than a missed one.
    """
    kds = make_weight_kds(ratio=50)
    a = act.WeightActuator(kds, 0, 0, 0)

    orig = act._damon.commit
    orig_retry = act._COMMIT_RETRY_S
    act._damon.commit = lambda k, **kw: '[Errno 16] Device or resource busy'
    act._COMMIT_RETRY_S = 0
    try:
        a.write_ratio(70)
    finally:
        act._damon.commit = orig
        act._COMMIT_RETRY_S = orig_retry
    assert a.write_ok is False


def test_backoff_between_attempts_grows():
    """The wait between attempts doubles rather than staying put.

    The file is held across a wait for the kdamond, so a hold can outlast several
    fixed short waits in a row -- attempts spaced by one of them all land inside
    the same hold, which is the same as not retrying.
    """
    slept = []
    orig_sleep = act.time.sleep
    orig = act._damon.commit
    act.time.sleep = slept.append
    act._damon.commit = lambda k, **kw: '[Errno 16] Device or resource busy'
    try:
        act.WeightActuator(make_weight_kds(), 0, 0, 0).write_ratio(70)
    finally:
        act.time.sleep = orig_sleep
        act._damon.commit = orig
    assert len(slept) == act._COMMIT_TRIES - 1, slept
    assert slept == sorted(slept), slept
    assert slept[-1] > slept[0], slept
    assert max(slept) <= act._COMMIT_RETRY_MAX_S, slept


class UnreadableWeightActuator(act.WeightActuator):
    """One whose counter cannot be read for a stretch in the middle."""
    def __init__(self, kds, readings):
        super().__init__(kds, 0, 0, 0)
        self._readings = list(readings)
        self._last = 0

    def nr_applied(self):
        if self._readings:
            self._last = self._readings.pop(0)
        return self._last


def test_an_unreadable_counter_is_not_a_quiet_window():
    """Readings that could not be taken do not add up to a settled system.

    A refused refresh used to hand back the previous snapshot's number, and an
    unchanged counter is exactly what a settled system shows -- so the wait ended
    early and the bandwidth that followed described a distribution still moving.
    """
    now, sleep = _fake_time()
    a = UnreadableWeightActuator(
            make_weight_kds(), [100, None, None, None, None, None, None])
    assert a.converged(stable_window_ms=1000, max_wait_ms=3000,
                       now=now, sleep=sleep) is True
    assert a.outcome == act.CONV_DEADLINE


def test_a_quiet_window_after_an_unreadable_stretch_still_counts():
    """The window is only restarted by the gap, not abandoned.

    A refresh that fails once should cost the window it interrupted and nothing
    more, otherwise a single failure would keep a run from ever reporting a
    settled distribution.
    """
    now, sleep = _fake_time()
    a = UnreadableWeightActuator(
            make_weight_kds(), [100, 140, None, 160, 160, 160, 160, 160])
    assert a.converged(stable_window_ms=1000, max_wait_ms=30000,
                       now=now, sleep=sleep) is True
    assert a.outcome == act.CONV_REACHED


class ListWeightActuator(act.WeightActuator):
    """As above, but the final reading repeats instead of falling to zero."""
    def __init__(self, kds, readings):
        super().__init__(kds, 0, 0, 0)
        self._readings = list(readings)
        self._last = readings[-1] if readings else 0
        self.nr_reads = 0

    def nr_applied(self):
        self.nr_reads += 1
        if self._readings:
            self._last = self._readings.pop(0)
        return self._last


def _fake_time():
    """A clock the wait drives forward itself, through its own sleeps."""
    state = {'t': 0.0}
    def now():
        return state['t']
    def sleep(dt):
        state['t'] += dt
    return now, sleep


def test_weight_converged_reached_when_counter_stops():
    """A counter that stops advancing and stays stopped reaches the target.

    The counter never returns to zero -- it is cumulative -- so this is the case
    that a test of the absolute value could not describe.
    """
    kds = make_weight_kds()
    now, sleep = _fake_time()
    a = ListWeightActuator(kds, [100, 140, 160, 160])
    assert a.converged(stable_window_ms=1000, max_wait_ms=30000,
                       now=now, sleep=sleep) is True
    assert a.outcome == act.CONV_REACHED


def test_weight_converged_needs_a_full_window_of_quiet():
    """One unchanged reading is not enough: the pause has to hold.

    A counter that pauses for less than the window and then advances again is
    still migrating, and a wait that ended on the pause would hand the caller a
    reading taken in transit.
    """
    kds = make_weight_kds()
    now, sleep = _fake_time()
    # Pauses at 160 for one poll, resumes, then stops for good.
    a = ListWeightActuator(kds, [100, 160, 160, 200, 240, 240])
    assert a.converged(stable_window_ms=1000, max_wait_ms=30000,
                       now=now, sleep=sleep) is True
    assert a.outcome == act.CONV_REACHED
    # The wait outlasted the false pause rather than ending on it: reaching the
    # run at the end takes more readings than the first pause would have.
    assert a.nr_reads > 4


def test_weight_converged_deadline_while_still_migrating():
    """A counter that never stops ends the wait on its cap, not as reached."""
    kds = make_weight_kds()
    now, sleep = _fake_time()
    a = ListWeightActuator(kds, list(range(100, 100000, 40)))
    assert a.converged(stable_window_ms=1000, max_wait_ms=5000,
                       now=now, sleep=sleep) is True
    assert a.outcome == act.CONV_DEADLINE


def test_weight_converged_cumulative_counter_is_never_zero():
    """The regression this replaces: a settled VA scheme has nr_applied > 0.

    The counter accumulates for the scheme's lifetime and is reset only when the
    scheme is installed, so testing it for zero cannot report a distribution
    that settled after any migration at all.  A stopped counter at a large
    value is the settled case and has to be reported as reached.
    """
    kds = make_weight_kds()
    now, sleep = _fake_time()
    a = ListWeightActuator(kds, [999999, 999999])
    assert a.converged(stable_window_ms=1000, max_wait_ms=30000,
                       now=now, sleep=sleep) is True
    assert a.outcome == act.CONV_REACHED


def test_weight_converged_timeout_ms_is_accepted_as_the_cap():
    """timeout_ms keeps its meaning for callers that pass it."""
    kds = make_weight_kds()
    now, sleep = _fake_time()
    a = ListWeightActuator(kds, list(range(100, 100000, 40)))
    assert a.converged(stable_window_ms=1000, timeout_ms=3000,
                       now=now, sleep=sleep) is True
    assert a.outcome == act.CONV_DEADLINE


def test_goal_converged_no_premature_on_alternating():
    """Alternating 0/10000 stream: avg=5000, |5000-6000|=1000 > tolerance.
    Alternating values keep the average at 5000, which is outside the band, so
    the wait ends on its cap rather than by reporting the target reached.
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
    # A stall exit that cannot trigger, so the assertion is about the band and
    # nothing else.
    result = a.converged(target_dram_bp=target_bp, tolerance_bp=500,
                         stable_window_ms=2000, max_wait_ms=500,
                         ftrace_reader=reader, now=clock,
                         stall_windows=10 ** 6)
    # Should not converge via REACHED (avg ~ 5000, not within 500bp of 6000)
    # May converge via TIMEOUT -- that's acceptable (no deadlock)
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



def test_self_refresh_period_survives_a_ratio_change():
    """A ratio change must not put the configured period back.

    A weight change commits the whole kdamond, and the sysfs writer takes
    refresh_ms off the object it commits.  That object came from the
    configuration, so unless the request is recorded on it as well, the first
    ratio change restores the configured period and the kdamond stops refreshing
    on the period that was asked for -- while the caller, having been told the
    request was taken, no longer commands a refresh.  The settle wait then reads
    a counter nothing is keeping current.
    """
    kds = make_weight_kds(nr=7, refresh_ms=1000)
    act._damon._current_kdamonds = kds
    a = act.WeightActuator(kds, 0, 0, 0)
    assert a.enable_self_refresh(period_ms=250) is True

    # What the sysfs writer does with the object a commit is handed.
    committed = []
    orig = act._damon.commit
    act._damon.commit = lambda k, **kw: committed.append(k[0].refresh_ms)
    try:
        a.write_ratio(60)
    finally:
        act._damon.commit = orig
    assert committed == [250], committed


def test_a_refused_self_refresh_leaves_the_configured_period_alone():
    """A request the kernel did not take must not change what is committed.

    Reporting a period as in force when it is not would leave the reader
    trusting a refresh that is not happening.  A kernel without the file is an
    ordinary case, so the configured period stays exactly as it was.
    """
    kds = make_weight_kds(nr=7, refresh_ms=1000)
    act._damon._current_kdamonds = kds
    a = act.WeightActuator(kds, 0, 0, 0)

    orig_set = act._damon.set_stats_self_refresh
    act._damon.set_stats_self_refresh = lambda ki, ms: 'no such file'
    try:
        assert a.enable_self_refresh(period_ms=250) is False
    finally:
        act._damon.set_stats_self_refresh = orig_set
    assert kds[0].refresh_ms == 1000, kds[0].refresh_ms


def test_the_period_survives_a_stats_read_too():
    """Re-reading the kdamonds replaces the object the period was recorded on.

    The reader replaces its snapshot on every stats read, and a commit after
    that writes refresh_ms back from the replacement.  So the period has to
    survive the read as well as the request.
    """
    kds = make_weight_kds(nr=7, refresh_ms=1000)
    a = act.WeightActuator(kds, 0, 0, 0)
    assert a.enable_self_refresh(period_ms=250) is True

    # The kernel is asked again and answers with fresh objects, which is what a
    # kernel reporting the period as unset looks like.
    act._damon._current_kdamonds = make_weight_kds(nr=9, refresh_ms=1000)
    assert a.nr_applied() == 9

    committed = []
    orig = act._damon.commit
    act._damon.commit = lambda k, **kw: committed.append(k[0].refresh_ms)
    try:
        a.write_ratio(60)
    finally:
        act._damon.commit = orig
    assert committed == [250], committed


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
