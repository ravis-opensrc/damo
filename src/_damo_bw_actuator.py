# SPDX-License-Identifier: GPL-2.0
"""
Actuator abstractions for DAMON-based memory-bandwidth interleave control.

Two actuator implementations:
  WeightActuator  — mutates DamosDest weights on a scheme with >=2 dests.
  GoalActuator    — sets node_eligible_mem_bp quota goal target_value on
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


class Actuator:
    """Abstract base: ratio is 0..100 (100 = all near/DRAM)."""

    def read_ratio(self):
        raise NotImplementedError

    def write_ratio(self, ratio):
        raise NotImplementedError

    def nr_applied(self):
        raise NotImplementedError

    def converged(self):
        """Return True if the actuator has reached its target state.

        Default implementation: always True (no convergence wait needed).
        Subclasses override for mode-specific convergence detection.
        """
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
        err = _damon.commit(self._kds)
        if err:
            logging.warning('WeightActuator commit failed: %s', err)

    def nr_applied(self):
        # Refresh live scheme stats before reading: the kdamonds snapshot
        # captured at construction does not track the running kdamond.
        err = _damon.update_schemes_stats([self._ki])
        if err is None:
            self._kds = _damon.current_kdamonds()
        s = self._scheme()
        if hasattr(s, 'stats') and s.stats is not None:
            return s.stats.nr_applied
        return 0

    def converged(self):
        """VA mode: converged when nr_applied == 0.

        Weight writes take effect on the next DAMON aggregation interval.
        nr_applied == 0 means no folios were migrated in the last interval,
        indicating the system has settled at the new ratio.
        """
        return self.nr_applied() == 0


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
        self._prev_avg = None
        self._plateau_count = 0

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
        self._prev_avg = None
        self._plateau_count = 0
        for s in self._ctx().schemes:
            for g in s.quotas.goals:
                if hasattr(g, 'metric') and 'node_eligible_mem_bp' in str(g.metric):
                    if hasattr(g, 'nid'):
                        if g.nid == self._near_node:   # PULL
                            g.target_value = ratio * 100
                        elif g.nid == self._far_node:  # PUSH
                            g.target_value = 10000 - ratio * 100
        err = _damon.commit(self._kds, commit_quota_goals_only=True)
        if err:
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
                  progress_threshold=200, plateau_count_threshold=3):
        """PA-mode convergence via damos_node_eligible_mem_bp tracepoint.

        Three exit conditions (in priority order):
        1. REACHED:   windowed avg within tolerance_bp of target → True
        2. PLATEAUED: no progress for plateau_count_threshold consecutive
                      samples (|delta| < progress_threshold bp) → True
        3. TIMEOUT:   max wait exceeded → True (never deadlock)

        Returns False only while actively progressing toward target.

        At ratio=100 the PULL goal target=10000 but current_value≈0
        (workload already on DRAM, nothing to migrate).  The plateau
        path exits after ~3 aggr_intervals instead of deadlocking.

        stable_window_ms: size of the trailing windowed-average (also sets the default
            timeout); a decision is made once the window has data, it is not required to be full.
        now: injectable clock (default time.monotonic) for unit tests.
        timeout_ms: deadline in ms; defaults to max(stable_window_ms*5, 20000).
        progress_threshold: bp change below which a sample is "no progress".
        plateau_count_threshold: consecutive no-progress samples before exit.
        """
        if target_dram_bp is None:
            target_dram_bp = self.read_ratio() * 100

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
                return self.nr_applied() > 0

        if timeout_ms is None:
            timeout_ms = max(stable_window_ms * 5, 20000)

        wav = _damo_ftrace.TimeWindowedAverage(stable_window_ms)
        deadline = time.monotonic() + timeout_ms / 1000.0

        try:
            while time.monotonic() < deadline:
                event = reader.read_event()
                if event is None:
                    if own_reader:
                        time.sleep(0.1)
                    continue
                if event.get('nid') != self._near_node:
                    continue
                cv = event.get('current_value')
                if cv is None:
                    continue
                t = now()
                wav.add(cv, now=t)
                if not wav.has_data():
                    continue
                avg = wav.average()

                # REACHED: within tolerance
                if abs(avg - target_dram_bp) <= tolerance_bp:
                    self._prev_avg = None
                    self._plateau_count = 0
                    return True

                # PLATEAUED: no progress for K consecutive samples
                if self._prev_avg is not None:
                    if abs(avg - self._prev_avg) < progress_threshold:
                        self._plateau_count += 1
                        if self._plateau_count >= plateau_count_threshold:
                            self._prev_avg = None
                            self._plateau_count = 0
                            return True
                    else:
                        self._plateau_count = 0
                self._prev_avg = avg

            # TIMEOUT: never block forever
            self._prev_avg = None
            self._plateau_count = 0
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
