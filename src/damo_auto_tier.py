# SPDX-License-Identifier: GPL-2.0
"""
damo auto_tier — closed-loop memory-bandwidth interleave controller.

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
from _damo_bw_controller import MAX_ADJUST_INTERVAL_MS

_running = True

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


def _measure_bw(source, sample_ms, n_samples):
    """Sample BW n_samples times, return list of readings."""
    readings = []
    for _ in range(n_samples):
        readings.append(source.sample(sample_ms))
    return readings


def _settle_wait(ctrl, actuator, source, sample_ms, timeout_ms=30000):
    """Sample BW until the actuator has converged OR the timeout expires.

    The readings are not fed to the hill-climb -- they were taken while the
    ratio was still moving, so they describe neither the old nor the new
    operating point.  They are appended to the bandwidth history, since
    choose_adjust_interval() reads that history as a uniformly sampled signal
    and a hole in it would misplace every period the transform reports.

    Stage 1 of that selection reads the tail of the history, and these samples
    are the actuation it exists to exclude, so the caller chooses the interval
    before settling rather than after.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        ctrl.bw_history.append(source.sample(sample_ms))
        if actuator.converged():
            break


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

        # one-shot yaml detection — error if kdamonds present but no
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
        # Parse kdamonds from kv if provided
        if kdamonds_kv is not None:
            try:
                yaml_kdamonds = [_damon.Kdamond.from_kvpairs(kv)
                                 for kv in kdamonds_kv]
            except Exception as e:
                print('failed to parse kdamonds from yaml: %s' % e,
                      file=sys.stderr)
                sys.exit(1)

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

    try:
        source = source_mod.detect_bw_source(args)
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
    min_samples = ctrl_mod.MIN_ADJUST_SAMPLES
    max_samples = max(min_samples, MAX_ADJUST_INTERVAL_MS // sample_ms)

    cur_ratio = init_ratio

    if args.verbose:
        print('auto_tier: actuator=%s init_ratio=%d cutoff=%d MB/s' % (
            type(actuator).__name__, init_ratio, args.bw_cutoff_mbps))

    # --bw_log CSV output
    bw_log_file = None
    if getattr(args, 'bw_log', None):
        bw_log_file = open(args.bw_log, 'a')
        bw_log_file.write('ts,bw_mbps,ratio,step\n')

    while _running:
        # 1. Measure settled BW.  Every reading joins the history the interval
        # selector reads; the hill-climb acts on one value per cycle.
        readings = _measure_bw(source, sample_ms, n_samples)
        ctrl.bw_history.extend(readings)
        bw = max(readings) if readings else 0.0

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
        if new_ratio != cur_ratio:
            actuator.write_ratio(new_ratio)
            cur_ratio = new_ratio
            if args.verbose:
                print('  -> ratio=%d (step=%d)' % (cur_ratio, ctrl.last_step))

        # 5. Log
        if bw_log_file:
            bw_log_file.write('%s,%.2f,%d,%d\n' % (
                time.strftime('%Y-%m-%dT%H:%M:%S'),
                bw, cur_ratio, ctrl.last_step))
            bw_log_file.flush()

        # 6. Settle-wait until the actuator converges or the timeout expires, so
        # the next cycle measures the new operating point and not the transition.
        _settle_wait(ctrl, actuator, source, sample_ms, timeout_ms=30000)

        if args.once:
            break

    if bw_log_file:
        bw_log_file.close()

    if hasattr(source, 'close'):
        source.close()

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
                        help='resctrl mon_group name for scoped MBM (default: root)')
    parser.add_argument('--sample_interval_ms', type=int, default=1000,
                        metavar='<ms>', help='BW sampling interval ms (default: 1000)')
    parser.add_argument('--adjust_interval_ms', default=10000,
                        metavar='<ms>',
                        help='control loop interval ms (default: 10000)')
    parser.add_argument('--bw_cutoff_mbps', type=int, default=None,
                        metavar='<MB/s>',
                        help='BW saturation threshold MB/s '
                             '(default: auto-derived from HMAT write_bandwidth // bw_cutoff_frac)')
    parser.add_argument('--bw_cutoff_frac', type=int, default=3,
                        metavar='<N>',
                        help='denominator for HMAT auto-derive (default 3)')
    parser.add_argument('--near_node', type=int, default=0,
                        metavar='<node>', help='near (DRAM) NUMA node (default: 0)')
    parser.add_argument('--far_node', type=int, default=1,
                        metavar='<node>', help='far (CXL) NUMA node (default: 1)')
    parser.add_argument('--min_ratio', type=int, default=0,
                        metavar='<0-100>', help='minimum interleave ratio (default: 0)')
    parser.add_argument('--max_ratio', type=int, default=100,
                        metavar='<0-100>', help='maximum interleave ratio (default: 100)')
    parser.add_argument('--once', action='store_true',
                        help='run one control iteration then exit')
    parser.add_argument('--verbose', action='store_true',
                        help='print per-sample BW and ratio changes')
    parser.add_argument('--bw_log', default=None, metavar='<file>',
                        help='append CSV rows (ts,bw_mbps,ratio,step) to file')
    parser.description = ('Tune memory bandwidth interleave ratio using '
                          'DAMON feedback. Pass a .yaml config file as '
                          'positional argument to load kdamonds + auto_tier '
                          'parameters.')
    return parser
