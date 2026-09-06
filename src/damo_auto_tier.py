# SPDX-License-Identifier: GPL-2.0
"""
damo auto_tier -- closed-loop memory-bandwidth interleave controller.

Adjusts the DRAM:far-memory interleave ratio of a running DAMON kdamond
to maximise aggregate bandwidth, using live MBM bandwidth feedback and
actuator convergence detection (nr_applied for VA/weight mode;
damos_node_eligible_mem_bp tracepoint windowed average for PA/goal mode).

Hill-climb algorithm: ISMM'26, doi 10.1145/3814942.3816137.
"""
import atexit
import collections
import logging
import signal
import sys
import os
import threading
import time
sys.path.insert(0, os.path.dirname(__file__))

try:
    import yaml
    _have_yaml = True
except ImportError:
    _have_yaml = False

import _damon
import _damon_args
import _damo_bw_actuator as actuator_mod
import _damo_bw_controller as ctrl_mod
import _damo_bw_source as source_mod
import _damo_resctrl
from _damo_bw_controller import MAX_ADJUST_INTERVAL_MS

_running = True

# How close to the target counts as at-target, and how long the share has to
# hold there.  The eligible-memory share is reported about once a second, so a
# window shorter than a few seconds cannot hold the two samples a verdict needs.
DEFAULT_TOLERANCE_BP = 200
DEFAULT_STABLE_WINDOW_MS = 5000

# What each tunable is when nobody named it, applied after the config file has
# been read rather than by argparse.
#
# The precedence wanted is command line, then config file, then these -- and the
# merge below can only tell "the config file may speak" from "the command line
# has already spoken" by testing the value for None.  So a tunable that carries
# its default in argparse arrives non-None whether or not the user typed it, the
# merge reads that as a command-line value, and the config file's value for it is
# dropped without a word.  Every key the generator writes has to appear here, or
# it is a key that can be written and cannot be read.
DEFAULTS = {
    'near_node': 0,
    'far_node': 1,
    'sample_interval_ms': 1000,
    'adjust_interval_ms': 10000,
    'min_ratio': 0,
    'max_ratio': 100,
    'tolerance_bp': DEFAULT_TOLERANCE_BP,
    'stable_window_ms': DEFAULT_STABLE_WINDOW_MS,
}

def _sighandler(sig, frame):
    global _running
    _running = False

def load_yaml_config(path):
    """Load kdamonds + auto_tier params from a yaml config file.

    Returns (kdamonds_kv, auto_tier_kv) or raises ValueError on parse error.
    """
    if not _have_yaml:
        raise ValueError('pyyaml not installed; run: pip install pyyaml')
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get('kdamonds'), data.get('auto_tier', {})


class _BwSampler:
    """Sample bandwidth without pausing for the rest of the loop.

    The samples the loop wants are of the ratio that is in force, and the ratio
    is in force from the moment it is written -- including while the actuator is
    being waited on.  A sampler the loop drives can only read between those
    waits, so it reads a fraction of each cycle and calls it the cycle.  This
    one runs alongside instead and hands over whatever has arrived since it was
    last asked.

    Readings accumulate while nobody is asking, which is the point: they are the
    part of the cycle a caller-driven sampler could not see.
    """

    def __init__(self, source, sample_ms):
        self._source = source
        self._sample_ms = sample_ms
        self._pending = collections.deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            # A sampler that dies takes the loop with it: the loop waits for
            # readings that will never arrive, while the ratio last written stays
            # in force and DAMON keeps migrating to it.  So a failed read is
            # reported and the next one attempted, and only a stop ends this.
            try:
                bw = self._source.sample(self._sample_ms)
            except Exception as e:
                logging.warning('bandwidth sample failed: %s', e)
                # The sleep a sample owns may not have happened, so it is spent
                # here instead: a failure that recurs must not become a spin.
                self._stop.wait(self._sample_ms / 1000.0)
                continue
            if bw is None:
                # A window with no reading is not a window with no bandwidth, so
                # it is dropped rather than queued.  The sleep already happened
                # inside the sample, so this is not a spin.  The caller waits for
                # the count it asked for, which is what has to be true here: a
                # cycle decides on readings, not on however many attempts it took
                # to get them.
                continue
            with self._lock:
                self._pending.append(bw)

    def take(self):
        """Return the readings that arrived since the last call, oldest first."""
        with self._lock:
            out = list(self._pending)
            self._pending.clear()
        return out

    def take_at_least(self, n, running=lambda: True):
        """Return at least n readings, waiting for them to arrive.

        A cycle is a span of time, and with the sampler on its own thread the
        way to spend that span is to wait for its samples rather than to count
        calls that each carry their own sleep.  Returns early if the run is
        ending, so a shutdown does not have to wait out an interval.
        """
        out = []
        waited_ms = 0.0
        warn_at_ms = 10 * self._sample_ms
        while len(out) < n and running():
            out.extend(self.take())
            if len(out) < n:
                step = self._sample_ms / 1000.0 / 2
                time.sleep(step)
                waited_ms += step * 1000.0
                # Readings that never arrive would otherwise hold the cycle here
                # with nothing said.  The wait is the right behaviour -- a cycle
                # decides on readings or not at all -- but it has to be visible,
                # because from the outside it looks the same as a run that is
                # merely slow.
                if waited_ms >= warn_at_ms:
                    logging.warning(
                            'waited %d ms for %d bandwidth readings, have %d; '
                            'the counters may have nothing to report yet',
                            int(waited_ms), n, len(out))
                    warn_at_ms *= 2
        return out

    def take_settled(self, n, running=lambda: True):
        """Return the interval's readings, and separately what preceded them.

        What preceded them is whatever accumulated while the caller was not
        asking -- the ratio being written and waited on.  Those readings are of
        a point in transit, so they are handed back to be recorded rather than
        compared, and the interval is then spent on readings of the ratio the
        caller is deciding about.
        """
        skipped = self.take()
        return self.take_at_least(n, running), skipped

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        self._thread.join(timeout)


class _MonGroupSync:
    """Keep a resctrl monitoring group holding the monitored tasks' threads.

    resctrl assigns an RMID per thread rather than per process, so a thread that
    appears after the group was populated is counted nowhere the controller
    reads: its traffic leaves the group's total and the reading stops describing
    the tasks the schemes act on.  Membership is therefore refreshed for as long
    as the run lasts.

    On its own thread, and at a cadence of its own, because the loop's cycle is
    an interval plus however long the actuator takes to settle -- a span chosen
    for the control problem, not for how fast a workload can spawn threads.

    Writing the group's `tasks` file does not commit the kdamond, so this
    competes with nothing the loop does.
    """

    def __init__(self, mon_group, ctx, poll_s=3):
        self._mon_group = mon_group
        self._ctx = ctx
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self._poll_s):
            err = _damo_resctrl.sync_ctx_targets_to_group(
                    self._mon_group, self._ctx)
            if err is not None:
                # Warned rather than fatal: a sync that found nothing to assign
                # is what a run looks like once its targets have exited, and the
                # readings taken until then are still the readings that were
                # taken.
                logging.warning('resctrl group sync: %s', err)

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        self._thread.join(timeout)


MON_GROUP_PREFIX = 'damo_auto_tier'


def default_mon_group(kdamond_idx, ctx_idx):
    """The group name a run uses when the operator did not name one.

    Carries the context's own indices, because the group holds one context's
    processes and two controllers steering different contexts must not share
    one: each would then read a total covering the other's processes, which is
    the scope mismatch a group exists to remove.  The indices are the ones the
    operator addresses the context by on the command line, so the group is
    identifiable from the outside without consulting the run.
    """
    return '%s_k%dc%d' % (MON_GROUP_PREFIX, kdamond_idx, ctx_idx)


def _resolve_mon_group(args, kds):
    """Which resctrl monitoring group the bandwidth is to be read from.

    The scope of the read has to match the scope the schemes act on, or the
    controller is steering on traffic it cannot move.  A physical-address
    context acts on a node's memory wherever it is in use, so its scope is the
    machine and the root group is the right read.  A virtual-address context
    acts on the processes its targets name and on nothing else, so its scope is
    those processes and the read has to be a group holding exactly their
    threads.

    Returns (mon_group_name_or_None, context_or_None, error).  A None name names
    the root group; the context is the one the group is scoped to, and is what
    keeps the group's membership that context's own.
    """
    if getattr(args, 'resctrl_system_wide', False):
        return None, None, None
    try:
        ctx = kds[args.kdamond_idx].contexts[args.ctx_idx]
    except (IndexError, AttributeError) as e:
        return None, None, 'cannot read the context to scope bandwidth to (%s)' % e
    name = getattr(args, 'resctrl_mon_group', None)
    if name is None:
        if ctx.ops != 'vaddr':
            return None, None, None
        name = default_mon_group(args.kdamond_idx, args.ctx_idx)
    err = _damo_resctrl.ensure_mon_group(name)
    if err is not None:
        return None, None, err
    err = _damo_resctrl.sync_ctx_targets_to_group(name, ctx)
    if err is not None:
        return None, None, err
    return name, ctx, None


def _settle_wait(actuator, timeout_ms=30000, **conv_kwargs):
    """Wait until the actuator has converged OR the timeout expires.

    Bandwidth is not sampled here.  The sampler runs on its own thread and does
    not stop for this wait, so the readings taken during it reach the next cycle
    on their own -- which is what they are for: the ratio being waited on is the
    ratio in force, so those readings belong to the cycle that follows.

    Returns whether the actuator reported reaching its target.  The caller does
    not gate the next step on it, but a run whose waits keep ending short of the
    target is measuring transitions, and that is only visible if it is recorded.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        remaining_ms = (deadline - time.monotonic()) * 1000.0
        if remaining_ms <= 0:
            return False
        # One deadline for the whole wait, not one here and another inside:
        # the inner call would otherwise derive its own and could spend this
        # budget in a single pass.
        if actuator.converged(max_wait_ms=int(remaining_ms), **conv_kwargs):
            return actuator.outcome == actuator_mod.CONV_REACHED


def main(args):
    global _running
    _running = True

    # Root + initialized check.
    # ensure_root_and_initialized must be called before stage_kdamonds
    # so that sysinfo is seeded.
    _damon.ensure_root_and_initialized(args)

    # Detect config path from deducible_target (positional arg from _damon_args)
    config_path = None
    if hasattr(args, 'deducible_target') and args.deducible_target:
        if (args.deducible_target.endswith('.yaml') or
                args.deducible_target.endswith('.json')):
            config_path = args.deducible_target

    # yaml config: load kdamonds spec and auto_tier overrides
    yaml_kdamonds = None
    if config_path:
        try:
            kdamonds_kv, auto_tier_kv = load_yaml_config(config_path)
        except (OSError, ValueError) as e:
            print('failed to load yaml config: %s' % e, file=sys.stderr)
            sys.exit(1)

        # one-shot yaml detection -- error if kdamonds present but no
        # auto_tier section (generated with --target_bp for damo start, not
        # damo auto_tier).
        if kdamonds_kv is not None and not auto_tier_kv:
            print('error: this yaml was generated for one-shot use '
                  '(--target_bp). '
                  'Use "damo start cfg.yaml" instead of '
                  '"damo auto_tier cfg.yaml".',
                  file=sys.stderr)
            sys.exit(1)

        # Override args fields from auto_tier_kv only if not already set by CLI
        for key, val in auto_tier_kv.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, val)
            elif key not in DEFAULTS and hasattr(args, key):
                # A key the config file may set that this command cannot honour,
                # because it reaches here already carrying a value.  Say so
                # rather than run with a number the file did not ask for.
                logging.warning(
                    'auto_tier: ignoring %s=%s from the config file; it is '
                    'already set', key, val)
        # Parse kdamonds from kv if provided
        if kdamonds_kv is not None:
            try:
                yaml_kdamonds = [_damon.Kdamond.from_kvpairs(kv)
                                 for kv in kdamonds_kv]
            except Exception as e:
                print('failed to parse kdamonds from yaml: %s' % e,
                      file=sys.stderr)
                sys.exit(1)

    # Whatever neither the command line nor the config file named.  Last, so that
    # both of those had their chance at it first.
    for key, val in DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    # Auto-start: if no live kdamond and yaml provided kdamonds, start DAMON
    _auto_started = False
    if not _damon.any_kdamond_running():
        if yaml_kdamonds is not None:
            err = _damon.stage_kdamonds(yaml_kdamonds)
            if not err:
                err = _damon.turn_damon_on(
                    ['%d' % i for i in range(len(yaml_kdamonds))])
            if err:
                print('failed to start DAMON: %s' % err, file=sys.stderr)
                sys.exit(1)
            _auto_started = True
            def _stop_on_exit():
                _damon.turn_damon_off(
                    ['%d' % i for i in range(len(yaml_kdamonds))])
            atexit.register(_stop_on_exit)
        else:
            print('no kdamond is running', file=sys.stderr)
            sys.exit(1)

    # HMAT auto-derive bw_cutoff_mbps when not set by CLI or yaml
    if not args.bw_cutoff_mbps:
        near = getattr(args, 'near_node', 0)
        hmat_path = ('/sys/devices/system/node/node%d/access0/initiators/'
                     'write_bandwidth' % near)
        try:
            with open(hmat_path) as f:
                write_bw = int(f.read().strip())
            frac = getattr(args, 'bw_cutoff_frac', 3) or 3
            args.bw_cutoff_mbps = write_bw // frac
            logging.info('bw_cutoff auto-derived from HMAT node%d '
                         'write_bandwidth: %d MB/s', near, args.bw_cutoff_mbps)
        except FileNotFoundError:
            print('error: --bw_cutoff_mbps not set and HMAT write_bandwidth '
                  'not found at %s. Pass --bw_cutoff_mbps explicitly.'
                  % hmat_path, file=sys.stderr)
            sys.exit(1)

    kds = _damon.current_kdamonds()
    if not kds:
        print('failed to read kdamonds', file=sys.stderr)
        sys.exit(1)

    try:
        actuator = actuator_mod.detect_actuator(
            kds, args.kdamond_idx, args.ctx_idx, args.scheme_idx,
            near_node=args.near_node, far_node=args.far_node)
    except ValueError as e:
        print('actuator detection failed: %s' % e, file=sys.stderr)
        sys.exit(1)

    # The settle wait reads a migration counter to decide when a ratio change has
    # taken effect.  Commanding a refresh of that counter goes through the same
    # file the ratio changes go through, and that file refuses a caller while
    # another holds it -- so the refreshes and the ratio changes compete, twice a
    # second against once a cycle.  A kdamond can refresh its own stats instead,
    # from inside its own loop, where it cannot block anything.  Asked for once
    # here; the wait then only reads.
    actuator.enable_self_refresh()

    # Resolved before the source is built, because the source opens the counter
    # files at construction and so decides then which scope it reads for the rest
    # of the run.
    mon_group, mon_ctx, err = _resolve_mon_group(args, kds)
    if err is not None:
        print('resctrl mon group setup failed: %s' % err, file=sys.stderr)
        sys.exit(1)
    args.resctrl_mon_group = mon_group
    group_sync = None
    if mon_group is not None:
        if args.verbose:
            print('auto_tier: bandwidth scoped to mon_groups/%s' % mon_group)
        group_sync = _MonGroupSync(mon_group, mon_ctx)
        group_sync.start()
        atexit.register(group_sync.stop)

    # Opened before the source, since the source writes to it from the window it
    # measures.  One row per window rather than one per cycle: a cycle reports a
    # single value chosen from several windows, so the window that carried an
    # impossible reading is not the row the cycle wrote.
    window_log_file = None
    if getattr(args, 'bw_window_log', None):
        try:
            window_log_file = open(args.bw_window_log, 'a')
            window_log_file.write('ts,delta_total_bytes,delta_local_bytes,'
                                  'elapsed_s,state\n')
            window_log_file.flush()
        except OSError as e:
            print('bw window log open failed: %s' % e, file=sys.stderr)
            sys.exit(1)

    try:
        source = source_mod.detect_bw_source(args, window_log=window_log_file)
    except (OSError, NotImplementedError, ValueError) as e:
        print('bw source init failed: %s' % e, file=sys.stderr)
        sys.exit(1)

    init_ratio = actuator.read_ratio()
    ctrl = ctrl_mod.BwController(
        init_ratio=init_ratio,
        bw_cutoff=args.bw_cutoff_mbps,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        # Bound the history by the span of time it covers rather than by a count,
        # so the deepest period stage 2 can see does not depend on how fast this
        # particular run happens to sample.
        history_len=max(ctrl_mod.MIN_ADJUST_SAMPLES,
                        ctrl_mod.FFT_WINDOW_LEN_MS // args.sample_interval_ms),
    )

    signal.signal(signal.SIGINT, _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    sample_ms = args.sample_interval_ms
    # The controller works in samples, the command line in milliseconds.
    n_samples = max(ctrl_mod.MIN_ADJUST_SAMPLES,
                    int(args.adjust_interval_ms) // sample_ms)
    # Half the interval that was configured, computed once.  A floor derived from
    # the current interval would fall with it, so each shortening would permit the
    # next one and the interval would walk to the absolute minimum with no
    # symmetric way back up.
    min_samples = max(ctrl_mod.MIN_ADJUST_SAMPLES, n_samples // 2)
    max_samples = max(min_samples, MAX_ADJUST_INTERVAL_MS // sample_ms)

    # Long enough to hold several windows, so that a wait ending early is the
    # actuator's verdict rather than this cap.
    settle_timeout_ms = max(30000, args.stable_window_ms * 6)

    cur_ratio = init_ratio

    # Started before the first cycle so that cycle has a full interval behind
    # it, rather than reading a window that began when it did.
    sampler = _BwSampler(source, sample_ms)
    sampler.start()
    atexit.register(sampler.stop)

    if args.verbose:
        print('auto_tier: actuator=%s init_ratio=%d cutoff=%d MB/s' % (
            type(actuator).__name__, init_ratio, args.bw_cutoff_mbps))

    # --bw_log CSV output
    bw_log_file = None
    if getattr(args, 'bw_log', None):
        bw_log_file = open(args.bw_log, 'a')
        bw_log_file.write('ts,bw_mbps,ratio,step\n')

    while _running:
        # 1. Take the interval's readings.  Whatever accumulated while the last
        # ratio was being written and waited on comes back separately: it is
        # recorded, because the history has to stay uniformly sampled for the
        # periods the interval selector reports to mean anything, but it is not
        # compared, because it describes a point in transit.  The hill-climb acts
        # on one value per cycle, taken from the interval's own readings.
        settled, in_transit = sampler.take_settled(n_samples, lambda: _running)
        if not settled:
            break
        ctrl.bw_history.extend(ctrl.floor_reading(r) for r in in_transit)
        readings = [ctrl.floor_reading(r) for r in settled]
        ctrl.bw_history.extend(readings)
        bw = max(readings)

        if args.verbose:
            print('  bw=%.1f MB/s ratio=%d' % (bw, cur_ratio))

        # 2. Step
        new_ratio = ctrl.step(readings)

        # 3. Adaptive interval, in samples.  Chosen before the settle-wait so the
        # newest entries in the history are the settled readings from step 1 and
        # not the samples taken while the ratio was moving.
        n_samples = ctrl.choose_adjust_interval(
            ctrl.bw_history, n_samples,
            min_int=min_samples, max_int=max_samples)
        if args.verbose:
            print('  adjust interval -> %d samples (%d ms)'
                  % (n_samples, n_samples * sample_ms))

        # 4. Actuate
        actuated = new_ratio != cur_ratio
        if actuated:
            actuator.write_ratio(new_ratio)
            if actuator.write_ok:
                cur_ratio = new_ratio
                if args.verbose:
                    print('  -> ratio=%d (step=%d)'
                          % (cur_ratio, ctrl.last_step))
            else:
                # The write was refused, so the kernel is still at the ratio it
                # was at.  There is no transition to wait out and no new
                # operating point to measure: the readings that follow describe
                # the ratio still in force.  The controller is told where the
                # kernel actually is, because one stepping from a ratio that was
                # never installed compares its next reading against a point that
                # does not exist.
                actuated = False
                ctrl.last_ratio = cur_ratio
                logging.warning('ratio %d was not installed; still at %d',
                                new_ratio, cur_ratio)

        # 5. Log
        if bw_log_file:
            bw_log_file.write('%s,%.2f,%d,%d\n' % (
                time.strftime('%Y-%m-%dT%H:%M:%S'),
                bw, cur_ratio, ctrl.last_step))
            bw_log_file.flush()

        # 6. Settle-wait until the actuator converges or the timeout expires, so
        # the next cycle measures the new operating point and not the
        # transition.  Only a cycle that moved the ratio has a transition to
        # wait out; waiting on a cycle that left it alone costs the wait and
        # returns nothing, and it costs it every cycle the climber spends at its
        # operating point, which is most of them.  A run that holds its ratio
        # therefore advances at the interval it chose.
        if actuated:
            settled = _settle_wait(actuator,
                                   timeout_ms=settle_timeout_ms,
                                   tolerance_bp=args.tolerance_bp,
                                   stable_window_ms=args.stable_window_ms)
            if not settled:
                # Recorded rather than acted on: the reading that follows is of
                # a point that had not stopped moving, which is worth knowing
                # when reading the run back.  Reported at warning level so it
                # reaches a log that records only warnings -- a wait whose
                # verdict is invisible cannot be told from one that succeeded.
                logging.warning('ratio %d did not settle within the wait (%s)',
                                cur_ratio, actuator.outcome)

        if args.once:
            break

    if bw_log_file:
        bw_log_file.close()

    # Stopped before the source is closed, since the sampler reads the fds the
    # close releases.  It sleeps for up to one sample interval, so it is joined
    # rather than assumed to have noticed.
    sampler.stop()
    sampler.join(sample_ms / 1000.0 * 2)

    if group_sync is not None:
        group_sync.stop()
        group_sync.join(6)

    if hasattr(source, 'close'):
        source.close()

    # Closed after the source, which writes to it, and after the sampler that
    # drives the source has been joined.
    if window_log_file:
        window_log_file.close()

def set_argparser(parser):
    _damon_args.set_argparser(parser, add_record_options=False, min_help=True)
    parser.add_argument('--kdamond_idx', type=int, default=0,
                        metavar='<n>', help='kdamond index (default: 0)')
    parser.add_argument('--ctx_idx', type=int, default=0,
                        metavar='<n>', help='context index (default: 0)')
    parser.add_argument('--scheme_idx', type=int, default=0,
                        metavar='<n>', help='scheme index for weight actuator (default: 0)')
    parser.add_argument('--bw_source', default='auto',
                        choices=['auto', 'resctrl', 'perf'],
                        help='bandwidth source (default: auto)')
    parser.add_argument('--resctrl_mon_group', default=None,
                        metavar='<name>',
                        help='resctrl mon_group to read MBM from; a '
                             'virtual-address context creates and uses one by '
                             'default so the reading covers the monitored '
                             'processes, a physical-address context reads the '
                             'root group')
    parser.add_argument('--resctrl_system_wide', action='store_true',
                        help='read MBM from the root group whatever the '
                             'monitoring mode, so the reading covers every task '
                             'on the machine')
    parser.add_argument('--sample_interval_ms', type=int, default=None,
                        metavar='<ms>', help='BW sampling interval ms (default: %d)'
                        % DEFAULTS['sample_interval_ms'])
    parser.add_argument('--adjust_interval_ms', default=None,
                        metavar='<ms>',
                        help='control loop interval ms (default: %d)'
                        % DEFAULTS['adjust_interval_ms'])
    parser.add_argument('--bw_cutoff_mbps', type=int, default=None,
                        metavar='<MB/s>',
                        help='BW saturation threshold MB/s '
                             '(default: auto-derived from HMAT write_bandwidth // bw_cutoff_frac)')
    parser.add_argument('--bw_cutoff_frac', type=int, default=3,
                        metavar='<N>',
                        help='denominator for HMAT auto-derive (default 3)')
    parser.add_argument('--near_node', type=int, default=None,
                        metavar='<node>', help='near (DRAM) NUMA node (default: %d)'
                        % DEFAULTS['near_node'])
    parser.add_argument('--far_node', type=int, default=None,
                        metavar='<node>', help='far (CXL) NUMA node (default: %d)'
                        % DEFAULTS['far_node'])
    parser.add_argument('--tolerance_bp', type=int, default=None,
                        metavar='<bp>',
                        help='half-width of the band counted as at-target '
                        '(default: %d)' % DEFAULTS['tolerance_bp'])
    parser.add_argument('--stable_window_ms', type=int, default=None,
                        metavar='<ms>',
                        help='how long the eligible-memory share has to hold '
                        'before it is read (default: %d)'
                        % DEFAULTS['stable_window_ms'])
    parser.add_argument('--min_ratio', type=int, default=None,
                        metavar='<0-100>', help='minimum interleave ratio (default: %d)'
                        % DEFAULTS['min_ratio'])
    parser.add_argument('--max_ratio', type=int, default=None,
                        metavar='<0-100>', help='maximum interleave ratio (default: %d)'
                        % DEFAULTS['max_ratio'])
    parser.add_argument('--once', action='store_true',
                        help='run one control iteration then exit')
    parser.add_argument('--verbose', action='store_true',
                        help='print per-sample BW and ratio changes')
    parser.add_argument('--bw_log', default=None, metavar='<file>',
                        help='append CSV rows (ts,bw_mbps,ratio,step) to file')
    parser.add_argument('--bw_window_log', default=None, metavar='<file>',
                        help='append one row per bandwidth window '
                        '(ts,delta_total_bytes,delta_local_bytes,elapsed_s,'
                        'state) to file; records the counter reads a reading is '
                        'derived from, and whether the counter confirmed them')
    parser.description = ('Tune memory bandwidth interleave ratio using '
                          'DAMON feedback. Pass a .yaml config file as '
                          'positional argument to load kdamonds + auto_tier '
                          'parameters.')
    return parser
