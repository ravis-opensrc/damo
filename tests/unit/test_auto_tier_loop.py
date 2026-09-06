# SPDX-License-Identifier: GPL-2.0
"""Tests for the auto_tier control loop."""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import _damo_bw_controller as ctrl_mod
import _damo_bw_actuator as act_mod


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


class _SleepySource(_FakeSource):
    """A source whose sample costs the interval, as a real one does.

    The sampler paces itself on how long a sample takes, so a source that
    returns instantly would spin and the counts below would say nothing.
    """

    def sample(self, sample_ms):
        time.sleep(sample_ms / 1000.0)
        return _FakeSource.sample(self, sample_ms)


class _FakeActuator:
    """Reaches its target on the converge_after'th call, or never.

    outcome carries which of the CONV_* states ended the wait, as the real
    actuators do; never=True stands for a system that answers but is not at the
    target, which is what a stalled or capped wait looks like to the caller.
    """

    def __init__(self, converge_after, never=False):
        self.calls = 0
        self.converge_after = converge_after
        self.never = never
        self.outcome = None
        self.budgets = []

    def read_ratio(self):
        return 50

    def converged(self, **kwargs):
        self.calls += 1
        if 'max_wait_ms' in kwargs:
            self.budgets.append(kwargs['max_wait_ms'])
        if self.never:
            self.outcome = act_mod.CONV_STALLED
            return True
        reached = self.calls >= self.converge_after
        self.outcome = act_mod.CONV_REACHED if reached else None
        return reached


def test_sampler_keeps_sampling_while_nobody_is_asking():
    """The readings taken between requests are the ones worth having.

    They cover the part of the cycle spent writing the ratio and waiting for it
    to take effect, which is when the ratio being measured is in force.  A
    sampler the loop drives cannot see that span at all.
    """
    import damo_auto_tier
    source = _SleepySource()
    sampler = damo_auto_tier._BwSampler(source, sample_ms=10)
    sampler.start()
    try:
        time.sleep(0.25)
        first = sampler.take()
        # Nothing is asked of it here, standing in for the ratio being written
        # and waited on.
        time.sleep(0.25)
        second = sampler.take()
    finally:
        sampler.stop()
        sampler.join(1)
    assert len(first) > 1, 'took %r in the first span' % (first,)
    assert len(second) > 1, 'nothing accumulated while unattended: %r' % (
            second,)
    # Handed over once each, oldest first, with no repeats across the two.
    assert first == sorted(first) and second == sorted(second)
    assert max(first) < min(second), '%r then %r' % (first, second)


class _FakeCtx:
    def __init__(self, ops, targets=None):
        self.ops = ops
        self.targets = targets or []


class _FakeKdamond:
    def __init__(self, ops, n_ctxs=1):
        self.contexts = [_FakeCtx(ops) for _ in range(n_ctxs)]


def _mon_group_args(**kw):
    class _Args:
        pass
    args = _Args()
    args.kdamond_idx = 0
    args.ctx_idx = 0
    args.resctrl_mon_group = None
    args.resctrl_system_wide = False
    args.verbose = False
    for key, val in kw.items():
        setattr(args, key, val)
    return args


def test_bandwidth_is_read_at_the_scope_the_schemes_act_on():
    """The read has to cover the tasks the schemes can move memory for.

    A physical-address context acts on a node's memory whatever holds it, so the
    root group is the matching read.  A virtual-address context acts on the
    processes its targets name, so reading the root group would attribute every
    other task's traffic to its own placement decisions -- which is why it gets a
    group of its own without having to be told to.
    """
    import damo_auto_tier

    name, ctx, err = damo_auto_tier._resolve_mon_group(
            _mon_group_args(), [_FakeKdamond('paddr')])
    assert (name, ctx, err) == (None, None, None), (name, ctx, err)

    # An operator who wants the machine's traffic for a vaddr context says so,
    # and is then not talked out of it.
    name, ctx, err = damo_auto_tier._resolve_mon_group(
            _mon_group_args(resctrl_system_wide=True),
            [_FakeKdamond('vaddr')])
    assert (name, ctx, err) == (None, None, None), (name, ctx, err)


def test_an_unreadable_context_is_an_error_not_a_scope():
    """Falling back to the root group would be the mismatch this exists to stop.

    A scope that cannot be determined is not a scope of the machine, so it is
    reported rather than guessed at.
    """
    import damo_auto_tier
    name, ctx, err = damo_auto_tier._resolve_mon_group(
            _mon_group_args(kdamond_idx=7), [_FakeKdamond('vaddr')])
    assert (name, ctx) == (None, None), (name, ctx)
    assert err is not None and 'context' in err, err


def test_the_default_group_name_names_the_context_it_holds():
    """Two controllers on two contexts must not populate one group.

    Membership is what the reading is, so a group shared between contexts gives
    each of them a total covering the other's processes -- the scope mismatch a
    group exists to remove, reintroduced.  The name therefore carries the indices
    the operator addresses the context by, which also makes the group
    identifiable from outside the run.
    """
    import damo_auto_tier

    assert damo_auto_tier.default_mon_group(0, 0) == 'damo_auto_tier_k0c0'
    assert damo_auto_tier.default_mon_group(1, 2) == 'damo_auto_tier_k1c2'

    # Distinct for every context there could be, not merely for the first two.
    names = set()
    for kd_idx in range(4):
        for ctx_idx in range(4):
            names.add(damo_auto_tier.default_mon_group(kd_idx, ctx_idx))
    assert len(names) == 16, sorted(names)

    # The kernel rejects only '\n' in a group name, and kernfs additionally
    # rejects '/' and NUL, so the format has to stay clear of all three.
    for name in names:
        assert '\n' not in name and '/' not in name and '\0' not in name, name


def test_the_group_holds_only_its_own_context_s_targets():
    """A suffixed name differentiates nothing unless membership follows it.

    The sync that populates the group is the other half of the change: walking
    every context would fill both groups with both contexts' processes, leaving
    two differently-named groups holding the same tasks.
    """
    import _damo_resctrl

    class _T:
        def __init__(self, pid):
            self.pid = pid
            self.obsolete = False

    kd = _FakeKdamond('vaddr', n_ctxs=2)
    kd.contexts[0].targets = [_T(11), _T(12)]
    kd.contexts[1].targets = [_T(21)]

    assert _damo_resctrl.ctx_target_pids(kd.contexts[0]) == ['11', '12']
    assert _damo_resctrl.ctx_target_pids(kd.contexts[1]) == ['21']
    # The whole-tree walk is still there for the caller that wants it, and is
    # what the per-context read has to differ from.
    assert _damo_resctrl.target_pids([kd]) == ['11', '12', '21']

    # A target that has exited is not a target: its threads are gone, and
    # counting it would report a group that was populated when it was not.
    kd.contexts[0].targets[1].obsolete = True
    assert _damo_resctrl.ctx_target_pids(kd.contexts[0]) == ['11']


class _ExplodingSource:
    """Fails its first two reads, as a counter file that went away would."""

    def __init__(self):
        self.n = 0

    def sample(self, sample_ms):
        self.n += 1
        if self.n <= 2:
            raise OSError('counter read failed')
        time.sleep(sample_ms / 1000.0)
        return 1000.0 + self.n


def test_a_failed_reading_does_not_end_the_sampling():
    """A sampler that dies takes the loop with it, silently.

    The loop then waits for readings that will never arrive while the ratio last
    written stays in force and DAMON keeps migrating to it -- a run that looks
    like it is holding a converged point and is in fact no longer measuring
    anything.  A freshly created monitoring group makes this reachable: the
    kernel answers its counter reads with a word rather than a number until the
    hardware has counted for it.
    """
    import damo_auto_tier
    source = _ExplodingSource()
    sampler = damo_auto_tier._BwSampler(source, sample_ms=10)
    sampler.start()
    try:
        # Asked for with a deadline rather than open-endedly, because a sampler
        # that has died never satisfies the count and the wait is the symptom
        # under test: without the deadline this test would hang where it should
        # fail.
        deadline = time.monotonic() + 2
        readings = sampler.take_at_least(
                2, running=lambda: time.monotonic() < deadline)
    finally:
        sampler.stop()
        sampler.join(1)
    assert len(readings) >= 2, 'the thread died on the failures: %r' % (
            readings,)
    # And the failures were the ones the source raised, not a source that never
    # got called: a test passing because nothing was read would prove nothing.
    assert source.n > 2, 'source was read %d times' % source.n


class _WarmingUpSource:
    """A source with no reading for its first three windows.

    Which is a monitoring group that has just been created: the kernel answers
    its counter reads with a word rather than a number until the hardware has
    counted for it, and the reader turns that into no reading at all.
    """

    def __init__(self):
        self.n = 0

    def sample(self, sample_ms):
        self.n += 1
        time.sleep(sample_ms / 1000.0)
        if self.n <= 3:
            return None
        return 1000.0 + self.n


def test_a_window_with_no_reading_is_not_a_window_with_no_bandwidth():
    """A warming-up counter must not reach the controller as a zero.

    Zero bandwidth is what an unsaturated system looks like, and that is a state
    the hill-climb steers by -- so a fabricated zero in the opening windows of a
    run is a decision taken on a reading that does not exist.  The windows are
    dropped instead, and the cycle waits for readings that are real.
    """
    import damo_auto_tier
    source = _WarmingUpSource()
    sampler = damo_auto_tier._BwSampler(source, sample_ms=10)
    sampler.start()
    try:
        deadline = time.monotonic() + 2
        readings = sampler.take_at_least(
                2, running=lambda: time.monotonic() < deadline)
    finally:
        sampler.stop()
        sampler.join(1)
    assert len(readings) >= 2, 'the empty windows were never replaced: %r' % (
            readings,)
    # Every reading handed over is one the source actually took.
    assert all(r is not None and r > 1000.0 for r in readings), readings
    # And the empty windows really did happen, so the test is about them.
    assert source.n > 3, 'source was read %d times' % source.n


def test_settle_wait_takes_no_samples_of_its_own():
    """The sampler does not stop for the wait, so the wait does not sample.

    Sampling in both places would double-count the span into the history the
    interval selector reads, and that history has to stay one sample per
    interval for the periods it reports to mean anything.
    """
    import damo_auto_tier
    ctrl = ctrl_mod.BwController(init_ratio=50, bw_cutoff=200)
    source = _FakeSource()
    damo_auto_tier._settle_wait(_FakeActuator(converge_after=3),
                                timeout_ms=5000)
    assert source.n == 0, 'the wait sampled %d times' % source.n
    assert not list(ctrl.bw_history), \
        'the wait wrote to the history: %r' % (list(ctrl.bw_history),)


def test_a_cycle_takes_the_interval_worth_of_readings():
    """A cycle is a span of time, and the interval is what names that span.

    With the sampler on its own thread the way to spend the interval is to wait
    for its samples, so asking for the interval's worth is what keeps the
    interval meaning the same thing in time as the sampling rate changes.
    """
    import damo_auto_tier
    source = _SleepySource()
    sampler = damo_auto_tier._BwSampler(source, sample_ms=10)
    sampler.start()
    try:
        readings = sampler.take_at_least(5)
    finally:
        sampler.stop()
        sampler.join(1)
    assert len(readings) >= 5, 'took %r' % (readings,)


def test_readings_taken_in_transit_are_recorded_but_not_compared():
    """A cycle decides on readings of the ratio it is deciding about.

    What accumulated while the last ratio was being written and waited on is of
    a point in transit, so it is kept for the history -- which has to stay
    uniformly sampled for the periods the interval selector reports to mean
    anything -- and left out of the comparison.
    """
    import damo_auto_tier
    source = _SleepySource()
    sampler = damo_auto_tier._BwSampler(source, sample_ms=10)
    sampler.start()
    try:
        # Nothing is asked of it, standing in for the ratio being written and
        # waited on.
        time.sleep(0.25)
        settled, in_transit = sampler.take_settled(3)
    finally:
        sampler.stop()
        sampler.join(1)
    assert len(in_transit) > 1, 'nothing accumulated in transit: %r' % (
            in_transit,)
    assert len(settled) >= 3, 'took %r for the interval' % (settled,)
    # The two do not overlap, and the interval's are the later ones.
    assert max(in_transit) < min(settled), '%r then %r' % (in_transit, settled)


def test_a_cycle_ends_early_when_the_run_is_ending():
    """A shutdown does not wait out an interval it will not use.

    The loop asks with the run's own flag, so a run stopped part way through a
    cycle returns rather than sleeping until the count is met.
    """
    import damo_auto_tier
    source = _SleepySource()
    sampler = damo_auto_tier._BwSampler(source, sample_ms=10)
    sampler.start()
    try:
        readings = sampler.take_at_least(1000, lambda: False)
    finally:
        sampler.stop()
        sampler.join(1)
    assert len(readings) < 1000, 'waited out the whole interval: %d' % len(
            readings)


def test_interval_is_chosen_before_the_settle_wait():
    """The selector reads the tail of the history, so the tail has to be the
    readings of a settled point rather than of a ratio still moving.

    Driving the real loop needs root and a live kdamond, so this replays the order
    the loop performs the three operations in.
    """
    import damo_auto_tier
    ctrl = ctrl_mod.BwController(init_ratio=50, bw_cutoff=200)
    actuator = _FakeActuator(converge_after=2)

    seen = []
    ctrl.choose_adjust_interval = lambda hist, cur_int, **kw: (
        seen.append(list(hist)) or cur_int)

    source = _SleepySource()
    sampler = damo_auto_tier._BwSampler(source, sample_ms=10)
    sampler.start()
    try:
        readings = sampler.take_at_least(3)
    finally:
        sampler.stop()
        sampler.join(1)
    ctrl.bw_history.extend(readings)
    ctrl.choose_adjust_interval(ctrl.bw_history, 3)
    damo_auto_tier._settle_wait(actuator, timeout_ms=5000)

    assert seen and seen[0] == readings, \
        'selector saw %r, expected the readings of the cycle %r' % (
            seen[0] if seen else None, readings)
    # The wait adds nothing of its own, so the tail is still the cycle's.
    assert list(ctrl.bw_history) == readings, \
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


def test_settle_wait_returns_whether_the_target_was_reached():
    """The verdict distinguishes a settled point from one still in transit.

    The caller reads bandwidth either way, so the verdict is what tells a run
    apart from one whose every reading was taken mid-move.
    """
    import damo_auto_tier
    reached = damo_auto_tier._settle_wait(
            _FakeActuator(converge_after=2), timeout_ms=5000)
    assert reached is True

    unsettled = damo_auto_tier._settle_wait(
            _FakeActuator(converge_after=1, never=True), timeout_ms=5000)
    assert unsettled is False


def test_only_a_cycle_that_moved_the_ratio_waits():
    """A cycle that left the ratio alone has no transition to wait out.

    The wait is what separates one cycle's reading from the next, so spending it
    on a cycle that wrote nothing costs the interval and returns nothing -- and
    the climber spends most of its cycles at one operating point.

    Driving the real loop needs root and a live kdamond, so this replays the two
    steps the loop makes the decision in, over a controller fed a constant
    bandwidth until it holds.
    """
    import damo_auto_tier
    ctrl = ctrl_mod.BwController(init_ratio=80, bw_cutoff=200)
    actuator = _FakeActuator(converge_after=1)
    cur_ratio = 80
    moved = 0
    held = 0

    for _ in range(8):
        new_ratio = ctrl.step([1000.0])
        actuated = new_ratio != cur_ratio
        if actuated:
            cur_ratio = new_ratio
            moved += 1
            damo_auto_tier._settle_wait(actuator, timeout_ms=100)
        else:
            held += 1

    assert moved and held, 'moved=%d held=%d' % (moved, held)
    assert actuator.calls == moved, \
        'waited %d times for %d ratio changes' % (actuator.calls, moved)


def test_settle_wait_passes_down_its_remaining_budget():
    """One deadline for the wait, so the inner call cannot outlast the outer.

    Each budget handed down is what was left at that moment, which is both
    within the wait's own span and shrinking.
    """
    import damo_auto_tier
    actuator = _FakeActuator(converge_after=3)
    damo_auto_tier._settle_wait(actuator, timeout_ms=5000)
    assert len(actuator.budgets) == 3, actuator.budgets
    assert all(0 < b <= 5000 for b in actuator.budgets), actuator.budgets
    assert actuator.budgets == sorted(actuator.budgets, reverse=True), \
            actuator.budgets


def test_every_generated_config_key_can_be_read_back():
    """A key the generator writes has to be a key the command reads.

    The merge tells "the config file may speak" from "the command line has
    already spoken" by testing for None, so a tunable carrying its default in
    argparse arrives non-None whether or not it was typed, and the file's value
    for it is dropped.  Checked against what the generator actually emits, so a
    key added there and nowhere else fails here.
    """
    import argparse
    import damo_auto_tier
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tools'))
    import damon_tier_gen

    # The address ranges are passed rather than read from /proc/iomem, which is
    # masked to an unprivileged reader; nothing here depends on their values.
    written = damon_tier_gen.build_config(
        'ibs', near_node=0, far_node=1, bw_cutoff_mbps=38400,
        local_pa_start=0, local_pa_end=1 << 30,
        far_pa_start=1 << 30, far_pa_end=2 << 30)['auto_tier']
    parser = argparse.ArgumentParser()
    damo_auto_tier.set_argparser(parser)
    defaults = {a.dest: a.default for a in parser._actions}

    unreadable = sorted(k for k in written
                        if k in defaults and defaults[k] is not None)
    assert not unreadable, \
        'the generator writes these and the command cannot read them: %s' \
        % unreadable
    unknown = sorted(k for k in written if k not in defaults)
    assert not unknown, 'the generator writes keys with no flag: %s' % unknown


def test_a_tunable_with_no_default_anywhere_would_be_none_at_use():
    """Every flag the merge leaves as None has to be filled in afterwards.

    Stripping a default out of argparse is what lets the config file be heard,
    so something else has to supply it -- otherwise a run that names neither
    reaches the loop holding None.
    """
    import argparse
    import damo_auto_tier

    parser = argparse.ArgumentParser()
    damo_auto_tier.set_argparser(parser)
    args = parser.parse_args([])
    for key, val in damo_auto_tier.DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    missing = sorted(k for k in damo_auto_tier.DEFAULTS
                     if getattr(args, k, None) is None)
    assert not missing, missing
    # And the other direction: a key in the table whose flag still carries its
    # default in argparse is a key the config file cannot be heard on, which is
    # the whole reason the table exists.
    defaults = {a.dest: a.default for a in parser._actions}
    held_by_argparse = sorted(k for k in damo_auto_tier.DEFAULTS
                              if defaults.get(k) is not None)
    assert not held_by_argparse, \
        'these still carry an argparse default: %s' % held_by_argparse


def test_paddr_ranges_name_one_node_each_not_an_interleaved_span():
    """A near range has to name the near node and no other node.

    A machine that splits one socket's DRAM across several nodes presents them
    as adjacent System RAM in the address map, so a range derived from the map
    covers all of them.  Every scheme here is scoped by an address filter, and a
    filter carrying that span admits the scheme on memory the near node does not
    hold -- pages get moved toward a node that was never the target, and the
    controller's ratio stops describing the split it is steering.

    Built here rather than read from the machine: the topology that shows the
    difference is four DRAM nodes and a fifth far node, which the box running
    this test does not have.
    """
    import shutil
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tools'))
    import damon_tier_gen

    block = 1 << 28
    # Four DRAM nodes of 4 blocks each, then a far node -- one socket's memory
    # divided the way an NPS4 partitioning divides it, so nodes 0..3 are
    # adjacent and indistinguishable in the address map.
    layout = {0: range(0, 4), 1: range(4, 8), 2: range(8, 12),
              3: range(12, 16), 4: range(16, 24)}
    root = tempfile.mkdtemp()
    try:
        with open(os.path.join(root, 'block_size_bytes'), 'w') as f:
            f.write('%x\n' % block)
        for nid, blocks in layout.items():
            node_dir = os.path.join(root, 'node%d' % nid)
            os.mkdir(node_dir)
            for b in blocks:
                os.mkdir(os.path.join(node_dir, 'memory%d' % b))

        saved = (damon_tier_gen._SYSFS_NODE_ROOT,
                 damon_tier_gen._SYSFS_MEMORY_ROOT)
        damon_tier_gen._SYSFS_NODE_ROOT = root
        damon_tier_gen._SYSFS_MEMORY_ROOT = root
        try:
            cfg = damon_tier_gen.build_config(
                    'ibs', near_node=0, far_node=4, cold_demote=True,
                    cold_demote_mode='proactive', bw_cutoff_mbps=38400)
        finally:
            (damon_tier_gen._SYSFS_NODE_ROOT,
             damon_tier_gen._SYSFS_MEMORY_ROOT) = saved
    finally:
        shutil.rmtree(root)

    near = (4096, 4 * block)
    far = (16 * block, 24 * block)
    ctx = cfg['kdamonds'][0]['contexts'][0]

    regions = [(int(r['start']), int(r['end']))
               for r in ctx['targets'][0]['regions']]
    assert regions == [near, far], regions

    # Every filter, not the first one: a scheme scoped to the wrong node is
    # wrong whichever scheme it is, and the near and far schemes are scoped from
    # different ends.
    filters = [(s['action'], f['address_range'])
               for s in ctx['schemes'] for f in s['filters']
               if f['filter_type'] == 'addr']
    assert filters, 'no addr filter emitted -- every scheme has to be scoped'
    for action, r in filters:
        got = (int(r['start']), int(r['end']))
        assert got in (near, far), '%s scoped to %s' % (action, got)

    # The demotion scheme is the one that has to be scoped to the near node: it
    # moves pages off it, so admitting it on the far node's range would demote
    # pages already there.
    cold = [s for s in ctx['schemes'] if s['action'] == 'migrate_cold']
    assert len(cold) == 1, [s['action'] for s in ctx['schemes']]
    cold_addr = [f['address_range'] for f in cold[0]['filters']
                 if f['filter_type'] == 'addr']
    assert len(cold_addr) == 1, cold[0]['filters']
    assert (int(cold_addr[0]['start']),
            int(cold_addr[0]['end'])) == near, cold_addr[0]


def test_an_explicit_range_overrides_the_node_topology():
    """The flags stay authoritative, because a machine can misreport.

    A node whose blocks do not describe the memory the operator means -- a
    partial onlining, a firmware that lays the map out differently -- is the
    reason the flags exist, so what they name is used unexamined.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tools'))
    import damon_tier_gen

    near, far = damon_tier_gen.read_pa_ranges(
            near_node=0, far_node=1,
            local_pa_start=1 << 20, local_pa_end=1 << 30,
            far_pa_start=1 << 30, far_pa_end=1 << 31)
    assert near == [(1 << 20, 1 << 30)], near
    assert far == [(1 << 30, 1 << 31)], far


def test_overlapping_near_and_far_ranges_are_refused():
    """Two ranges that overlap cannot both scope a scheme.

    A filter is an allow-list over addresses, so overlapping ranges admit each
    scheme on part of the other node's memory: the promotion and demotion
    schemes would contend over the same pages and the ratio would describe
    neither node.  Better to refuse the config than to emit one whose schemes
    fight.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tools'))
    import damon_tier_gen

    try:
        damon_tier_gen.read_pa_ranges(
                near_node=0, far_node=1,
                local_pa_start=0, local_pa_end=2 << 30,
                far_pa_start=1 << 30, far_pa_end=3 << 30)
    except ValueError as e:
        assert 'overlap' in str(e), str(e)
    else:
        assert False, 'overlapping ranges were accepted'

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
