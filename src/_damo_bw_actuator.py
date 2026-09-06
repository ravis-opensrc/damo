# SPDX-License-Identifier: GPL-2.0
"""
Actuator abstractions for DAMON-based memory-bandwidth interleave control.

Two actuator implementations:
  WeightActuator  -- mutates DamosDest weights on a scheme with >=2 dests.
  GoalActuator    -- sets node_eligible_mem_bp quota goal target_value on
                    PULL/PUSH scheme pairs identified by goal.nid.

detect_actuator() auto-selects based on the live kdamond topology.
VA and PA modes are mutually exclusive: a config with both >=2 dests and
node_eligible_mem_bp goals is malformed and raises ValueError.
"""
import logging
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import _damon
import _damo_ftrace


# What a convergence wait observed.  Three states rather than one bit, because
# the caller's next reading means something different in each: at the target it
# describes the new operating point, and in the other two it describes a system
# that had not stopped moving.
CONV_REACHED = 'reached'
CONV_STALLED = 'stalled'
CONV_DEADLINE = 'deadline'

# How many times a commit refused for a busy state file is attempted, how long
# to leave before the first retry, and the ceiling the wait doubles up to.  The
# holder of the file is not doing a single write: it is waiting for the kdamond
# to service its command, and holds the file across that wait, so a hold can
# last an aggregation interval.  A fixed short wait spends every attempt inside
# one hold, which is the same as not retrying at all -- hence the doubling.
_COMMIT_TRIES = 7
_COMMIT_RETRY_S = 0.05
_COMMIT_RETRY_MAX_S = 1.0

# How long to leave between two readings of a migration counter, and the cap on
# a wait whose caller named none.  The gap has to outlast an aggregation
# interval for the difference between readings to describe one.
_POLL_MS = 500
_DEFAULT_WAIT_MS = 30000

# The period a kdamond is asked to refresh its own stats at.  Shorter than the
# gap between two readings of them, so a reading never describes a snapshot
# taken before the reading that preceded it.
_SELF_REFRESH_MS = 250


def _busy_err(err):
    """Whether a commit error is the state file reporting itself busy.

    The error arrives as text from the sysfs write, so it is matched on the
    errno name and number rather than on an exception type.
    """
    s = str(err)
    return 'EBUSY' in s or 'Errno 16' in s or 'resource busy' in s.lower()


def _commit_with_retry(commit):
    """Commit, attempting again while the state file reports itself busy.

    Returns None once the commit is taken, or the error that ended the
    attempts.  The wait between attempts doubles up to a ceiling, because a
    refusal means some other caller holds the file across a wait of its own and
    attempts spaced by a fixed short interval would all land inside that one
    hold.

    A refusal for any other reason is reported on the first attempt, since
    repeating it would only repeat the answer.
    """
    delay = _COMMIT_RETRY_S
    err = None
    for attempt in range(_COMMIT_TRIES):
        err = commit()
        if err is None:
            return None
        if not _busy_err(err) or attempt == _COMMIT_TRIES - 1:
            return err
        time.sleep(delay)
        delay = min(delay * 2, _COMMIT_RETRY_MAX_S)
    return err


class Actuator:
    """Abstract base: ratio is 0..100 (100 = all near/DRAM)."""

    def read_ratio(self):
        raise NotImplementedError

    def write_ratio(self, ratio):
        raise NotImplementedError

    def nr_applied(self):
        raise NotImplementedError

    # What the most recent convergence wait observed, one of the CONV_* names.
    outcome = None

    # Whether the most recent write_ratio() reached the kernel.  A write that
    # did not leaves the kernel at the ratio it was already at, which the caller
    # has to know: waiting for a distribution nothing asked for, and then
    # reading the bandwidth of the ratio still in force, is what turns a refused
    # write into a wrong decision rather than a missed one.
    write_ok = True

    # Whether the kdamond is refreshing its own stats.  Until it says it is, a
    # reader of those stats has to command each refresh itself.
    _self_refresh = False

    # The period that was asked for, kept so that a commit rewriting the kdamond
    # writes back the period in force rather than the configured one.
    _self_refresh_ms = 0

    def enable_self_refresh(self, period_ms=_SELF_REFRESH_MS):
        """Ask the kdamond to keep its own stats current; report whether it will.

        Reading a stat the kdamond does not refresh means commanding a refresh,
        and that command goes through the file the ratio changes go through --
        which refuses a caller while another holds it.  The refreshes are
        frequent and the ratio changes are rare, so it is the ratio changes that
        lose.  A kdamond refreshing its own stats does it from inside its own
        loop, where it yields the file instead of waiting for it, and so cannot
        cost a ratio change.
        """
        err = _damon.set_stats_self_refresh(self._ki, period_ms)
        self._self_refresh = err is None
        if not self._self_refresh:
            logging.info(
                    'kdamond %d will not refresh its own stats (%s); each '
                    'refresh will be commanded instead', self._ki, err)
            return self._self_refresh
        # A ratio change commits the whole kdamond, and that writes refresh_ms
        # back from the object being committed.  Unless the period is recorded
        # there too, the first ratio change after this request restores the
        # period the configuration named and the kdamond stops refreshing on
        # the one that was asked for.
        if self._ki < len(self._kds):
            self._kds[self._ki].refresh_ms = period_ms
        self._self_refresh_ms = period_ms
        return self._self_refresh

    def _refresh_stats(self):
        """Bring the live stats up to date, and report whether that worked.

        The snapshot taken at construction does not track the running kdamond,
        so it has to be re-read either way.  What self refresh changes is only
        whether a refresh has to be commanded first.

        Returns False when the stats could not be brought up to date.  Reporting
        that, rather than handing back the previous snapshot's numbers, is what
        keeps a counter that could not be read from being mistaken for a counter
        that stopped advancing -- which is what a settled system looks like.
        """
        if not self._self_refresh:
            err = _damon.update_schemes_stats([self._ki])
            if err is not None:
                logging.warning('could not refresh scheme stats: %s', err)
                return False
        self._kds = _damon.current_kdamonds()
        # The objects just read replace the ones the period was recorded on, and
        # a later commit writes refresh_ms back from these.  What the kernel
        # reports is the period in force, so this only matters on a kernel that
        # reports it as unset; recording it again costs nothing either way.
        if self._self_refresh and self._ki < len(self._kds):
            self._kds[self._ki].refresh_ms = self._self_refresh_ms
        return bool(self._kds)

    def converged(self, **kwargs):
        """Return True if the actuator has reached its target state.

        Default implementation: always True (no convergence wait needed).
        Subclasses override for mode-specific convergence detection.

        Keyword arguments describing a wait are accepted and ignored, so that a
        caller can hand the same arguments to any actuator without knowing
        which one it holds.
        """
        self.outcome = CONV_REACHED
        return True


class WeightActuator(Actuator):
    """Drives dest0.weight=ratio, dest1.weight=100-ratio on one scheme."""

    def __init__(self, kdamonds, kdamond_idx, ctx_idx, scheme_idx):
        self._kds = kdamonds
        self._ki = kdamond_idx
        self._ci = ctx_idx
        self._si = scheme_idx

    def _scheme(self):
        return self._kds[self._ki].contexts[self._ci].schemes[self._si]

    def read_ratio(self):
        s = self._scheme()
        if not s.dests or len(s.dests) < 2:
            return 100
        return s.dests[0].weight

    def write_ratio(self, ratio):
        s = self._scheme()
        s.dests[0].weight = ratio
        s.dests[1].weight = 100 - ratio
        # Retried for the reason the goal actuator retries: the state file is
        # refused rather than queued while another caller holds it, and a weight
        # change that is dropped leaves the kernel distributing to the previous
        # weights while the caller believes it moved.
        err = _commit_with_retry(lambda: _damon.commit(self._kds))
        self.write_ok = err is None
        if err is not None:
            logging.warning('WeightActuator commit failed: %s', err)

    def nr_applied(self):
        # None rather than a number when the stats could not be re-read, so that
        # a reading which does not exist cannot be compared with the one before
        # it and found equal.
        if not self._refresh_stats():
            return None
        s = self._scheme()
        if hasattr(s, 'stats') and s.stats is not None:
            return s.stats.nr_applied
        return 0

    def converged(self, stable_window_ms=3000, max_wait_ms=None,
                  timeout_ms=None, poll_ms=_POLL_MS, now=time.monotonic,
                  sleep=time.sleep, **kwargs):
        """VA mode convergence, from the migration counter's rate of increase.

        Returns True once the wait is over and records which outcome ended it
        in self.outcome:

          CONV_REACHED   the counter stopped advancing and stayed stopped for a
                         full stable_window_ms
          CONV_DEADLINE  the cap elapsed while it was still advancing

        Only CONV_REACHED means the requested distribution is in force.

        The counter accumulates over the scheme's lifetime, so a single reading
        of it describes every migration since the scheme was installed rather
        than the interval just past.  What describes the interval is the
        increase between two readings, which is what is compared here.

        A weight change is reached when the increase falls to zero, because a
        folio already on the destination its weight selects is left where it is
        and so is not counted.  One zero increase is not enough to say that has
        happened: two readings can fall either side of an aggregation interval
        and see nothing in between while migration is still under way.  The
        increase therefore has to hold at zero for stable_window_ms, spanning
        several aggregation intervals, before the distribution is called
        settled.

        stable_window_ms: how long the increase has to stay at zero.
        max_wait_ms: cap on the whole wait; timeout_ms is accepted as the cap
            when it is not given, so existing callers keep their meaning.
        poll_ms: how long to leave between readings.
        now, sleep: injectable clock and delay, for the tests.
        """
        self.outcome = None
        cap_ms = max_wait_ms if max_wait_ms is not None else timeout_ms
        if cap_ms is None:
            cap_ms = _DEFAULT_WAIT_MS
        deadline = now() + cap_ms / 1000.0
        window_s = stable_window_ms / 1000.0
        poll_s = poll_ms / 1000.0

        prev = self.nr_applied()
        # When the run of unchanged readings began, or None while the counter is
        # still advancing.  The window is measured from the first reading that
        # matched its predecessor, so a full window of quiet is required after
        # migration appears to have stopped rather than at the moment it does.
        quiet_since = None
        while True:
            if now() >= deadline:
                self.outcome = CONV_DEADLINE
                return True
            sleep(min(poll_s, max(0.0, deadline - now())))
            cur = self.nr_applied()
            if cur is None:
                # Nothing is known about whether the counter advanced.  Treated
                # as advancing rather than as quiet: a window of quiet made up
                # of readings that were never taken would report a distribution
                # as settled while it was still moving.
                quiet_since = None
                continue
            if cur == prev:
                if quiet_since is None:
                    quiet_since = now()
                elif now() - quiet_since >= window_s:
                    self.outcome = CONV_REACHED
                    return True
            else:
                quiet_since = None
            prev = cur


class GoalActuator(Actuator):
    """Drives node_eligible_mem_bp quota goals on PULL+PUSH schemes.

    PULL scheme: goal.nid == near_node (DRAM); target_value = ratio * 100.
    PUSH scheme: goal.nid == far_node  (CXL);  target_value = 10000 - ratio * 100.

    Both schemes use migrate_hot action; identification is by goal.nid,
    not by action name.
    """

    def __init__(self, kdamonds, kdamond_idx, ctx_idx,
                 near_node=0, far_node=1):
        self._kds = kdamonds
        self._ki = kdamond_idx
        self._ci = ctx_idx
        self._near_node = near_node
        self._far_node = far_node
        # What the most recent wait observed, one of the CONV_* names.  Read it
        # alongside the boolean when the distinction matters.
        self.outcome = None

    def _ctx(self):
        return self._kds[self._ki].contexts[self._ci]

    def read_ratio(self):
        for s in self._ctx().schemes:
            for g in s.quotas.goals:
                if hasattr(g, 'metric') and 'node_eligible_mem_bp' in str(g.metric):
                    if hasattr(g, 'nid') and g.nid == self._near_node:
                        return g.target_value // 100
        return 100

    def write_ratio(self, ratio):
        self.outcome = None
        for s in self._ctx().schemes:
            for g in s.quotas.goals:
                if hasattr(g, 'metric') and 'node_eligible_mem_bp' in str(g.metric):
                    if hasattr(g, 'nid'):
                        if g.nid == self._near_node:   # PULL
                            g.target_value = ratio * 100
                        elif g.nid == self._far_node:  # PUSH
                            g.target_value = 10000 - ratio * 100
        err = _commit_with_retry(
                lambda: _damon.commit(self._kds, commit_quota_goals_only=True))
        self.write_ok = err is None
        if err is not None:
            logging.warning('GoalActuator commit failed: %s', err)

    def nr_applied(self):
        total = 0
        for s in self._ctx().schemes:
            for g in s.quotas.goals:
                if hasattr(g, 'metric') and 'node_eligible_mem_bp' in str(g.metric):
                    if hasattr(s, 'stats') and s.stats is not None:
                        total += s.stats.nr_applied
                    break
        return total

    def converged(self, target_dram_bp=None, tolerance_bp=500,
                  stable_window_ms=3000, ftrace_reader=None,
                  now=time.monotonic, timeout_ms=None,
                  progress_bp=200, stall_windows=4, max_wait_ms=None):
        """PA-mode convergence via the damos_node_eligible_mem_bp tracepoint.

        Returns True once the wait is over and records which of the three
        outcomes ended it in self.outcome:

          CONV_REACHED   the trailing average held within tolerance_bp of the
                         target across a full window
          CONV_STALLED   the distance to the target stopped shrinking for
                         stall_windows consecutive windows
          CONV_DEADLINE  max_wait_ms elapsed while still closing the gap

        Only CONV_REACHED means the system is at the requested distribution.
        False is returned only while it is still on its way there.

        A decision needs a window that spans the full stable_window_ms and holds
        at least two samples.  Per tick the tracepoint reports 0 or 10000,
        because one tick samples one region; only the average over a span of
        ticks is the share of eligible hot memory on the near node.  Deciding on
        a window that has just started filling therefore decides on a single
        tick of a binary signal.

        The metric is a share of the scheme's own eligible memory, not of the
        node: 10000 bp means all of the hot set is on the near node, whatever
        fraction of the node that occupies.  So the bytes a given bp target has
        to move are the hot footprint's, and the same target is quick on a small
        working set and slow on a large one.  A span that is generous on one
        workload is therefore short on the next, and waiting here is bounded by
        progress instead: each window that comes at least progress_bp closer to
        the target renews the wait, so a migration is waited out however long it
        needs.  max_wait_ms is the cap for the case where the gap keeps closing
        but never arrives; it defaults to no cap, since the stall exit already
        covers a system that has stopped moving.

        target_dram_bp: defaults to the PULL goal's own target.
        tolerance_bp: half-width of the band counted as at-target.
        stable_window_ms: how long the average has to hold before it is read.
        progress_bp: reduction in distance that counts as progress.
        stall_windows: consecutive windows without progress before giving up.
        now: injectable clock, for the tests.
        timeout_ms: accepted as the cap when max_wait_ms is not given, so
            existing callers keep their meaning.
        """
        if target_dram_bp is None:
            target_dram_bp = self.read_ratio() * 100
        if max_wait_ms is None:
            max_wait_ms = timeout_ms

        own_reader = ftrace_reader is None
        reader = ftrace_reader
        if own_reader:
            reader = _damo_ftrace.FtraceReader()
            try:
                reader.open('damos_node_eligible_mem_bp')
            except OSError as e:
                logging.warning(
                    'GoalActuator.converged: tracepoint unavailable (%s); '
                    'falling back to nr_applied > 0', e)
                self.outcome = CONV_REACHED
                return self.nr_applied() > 0

        wav = _damo_ftrace.TimeWindowedAverage(stable_window_ms)
        started = time.monotonic()
        # The distance at the last window that was read, and how many windows
        # since it last improved.
        best_gap = None
        idle_windows = 0
        window_end = None

        try:
            while True:
                if max_wait_ms is not None and \
                        (time.monotonic() - started) * 1000.0 >= max_wait_ms:
                    self.outcome = CONV_DEADLINE
                    logging.info(
                        'convergence on %d bp still in progress after %d ms; '
                        'proceeding', target_dram_bp, max_wait_ms)
                    return True

                event = reader.read_event()
                if event is None:
                    if own_reader:
                        time.sleep(0.1)
                        continue
                    # A reader with nothing left to give cannot be waited on.
                    self.outcome = CONV_DEADLINE
                    return True
                # The PULL goal reports the near node's share; the PUSH goal
                # reports the far node and is a different quantity.
                if event.get('nid') != self._near_node:
                    continue
                cv = event.get('current_value')
                if cv is None:
                    continue

                t = now()
                wav.add(cv, now=t)
                # Two ways of asking whether there is enough to average over: a
                # window can span the full period and still hold one sample.
                if not wav.is_stable(stable_window_ms, now=t):
                    continue
                if wav.count() < 2:
                    continue

                gap = abs(wav.average() - target_dram_bp)
                if gap <= tolerance_bp:
                    self.outcome = CONV_REACHED
                    return True

                # One verdict per window, so that stall_windows counts spans of
                # time rather than however many samples happened to arrive.
                if window_end is not None and t < window_end:
                    continue
                window_end = t + stable_window_ms / 1000.0

                if best_gap is None or gap <= best_gap - progress_bp:
                    best_gap = gap
                    idle_windows = 0
                    continue

                idle_windows += 1
                if idle_windows >= stall_windows:
                    self.outcome = CONV_STALLED
                    logging.info(
                        'convergence on %d bp stopped progressing at %d bp '
                        'away; proceeding', target_dram_bp, int(gap))
                    return True
        finally:
            if own_reader:
                reader.close()


def detect_actuator(kdamonds, kdamond_idx, ctx_idx, scheme_idx,
                    near_node=0, far_node=1):
    """Auto-detect actuator type from live kdamond topology.

    >=2 dests on scheme_idx -> WeightActuator
    node_eligible_mem_bp goal on any scheme -> GoalActuator
    both -> raise ValueError (VA and PA modes are mutually exclusive)
    neither -> raise ValueError
    """
    ctx = kdamonds[kdamond_idx].contexts[ctx_idx]
    scheme = ctx.schemes[scheme_idx]

    has_dests = hasattr(scheme, 'dests') and len(scheme.dests) >= 2

    has_goals = False
    for s in ctx.schemes:
        for g in s.quotas.goals:
            if hasattr(g, 'metric') and 'node_eligible_mem_bp' in str(g.metric):
                has_goals = True
                break
        if has_goals:
            break

    if has_dests and has_goals:
        raise ValueError(
            'malformed config: dest-weights and node_eligible_mem_bp goals both detected; '
            'VA and PA modes are mutually exclusive')
    elif has_goals:
        return GoalActuator(kdamonds, kdamond_idx, ctx_idx,
                            near_node=near_node, far_node=far_node)
    elif has_dests:
        return WeightActuator(kdamonds, kdamond_idx, ctx_idx, scheme_idx)
    else:
        raise ValueError(
            'detect_actuator: scheme %d has neither >=2 dests nor '
            'node_eligible_mem_bp goals; cannot determine actuator type'
            % scheme_idx)
