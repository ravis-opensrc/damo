# SPDX-License-Identifier: GPL-2.0
"""
Bandwidth source abstractions for damo auto_tier.

BwSource.sample(sample_ms) -> float MB/s (owns the sleep).

ResctrlMbmSource: reads MBM total bytes from resctrl mon_data,
  snapshots before/after sleep, returns delta/elapsed in MB/s.
"""
import logging
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import _damo_resctrl as resctrl

class BwSource:
    def sample(self, sample_ms):
        """Sleep sample_ms, return MB/s as float, or None for no reading.

        None is not zero.  A source that could not read its counters has nothing
        to say about the bandwidth in that window, and a caller told zero would
        act on it: zero bandwidth is what an unsaturated system looks like, and
        that is a state the controller steers by.
        """
        raise NotImplementedError

class ResctrlMbmSource(BwSource):
    def __init__(self, mon_group=None, resctrl_root=None, window_log=None):
        self._mon_group = mon_group
        self._resctrl_root = resctrl_root
        self._tfds = resctrl.open_mbm_counters(
            mon_group=mon_group, resctrl_root=resctrl_root)
        # A second reading of the same window, and the span it was divided by.
        # Total bytes is the only reading the controller acts on, so these are
        # opened for the record and not for the decision: a bandwidth figure the
        # hardware could not have produced is either the byte delta or the span
        # it was divided by, and the two cannot be told apart from the quotient
        # alone.  Local bytes counts the same traffic from a second counter, so a
        # window where total moves and local does not is a byte-delta fault,
        # while one where both move together is not.  Measured on two hosts, a
        # corrupt read moves both together, so local is kept for the record and
        # the ordering check is what a window is rejected on.
        self._lfds = resctrl.open_mbm_local_counters(
            mon_group=mon_group, resctrl_root=resctrl_root)
        self._window_log = window_log

    def _log_window(self, dtotal, dlocal, elapsed, ok=True):
        if self._window_log is None:
            return
        try:
            self._window_log.write('%s,%s,%s,%.6f,%s\n' % (
                time.strftime('%Y-%m-%dT%H:%M:%S'),
                '' if dtotal is None else dtotal,
                '' if dlocal is None else dlocal,
                elapsed,
                'ok' if ok else 'unordered'))
            self._window_log.flush()
        except Exception as e:
            # A record that cannot be written must not take the reading with it.
            logging.warning('bandwidth window log write failed: %s', e)
            self._window_log = None

    def sample(self, sample_ms):
        t0, ok0 = resctrl.read_mbm_confirmed(self._tfds)
        l0 = resctrl.read_mbm_sum(self._lfds)
        ts0 = time.monotonic()
        time.sleep(sample_ms / 1000.0)
        t1, ok1 = resctrl.read_mbm_confirmed(self._tfds)
        l1 = resctrl.read_mbm_sum(self._lfds)
        ts1 = time.monotonic()
        elapsed = ts1 - ts0
        dlocal = None if l0 is None or l1 is None else l1 - l0
        dtotal = None if t0 is None or t1 is None else t1 - t0
        ordered = ok0 and ok1
        self._log_window(dtotal, dlocal, elapsed, ok=ordered)
        if not ordered:
            # A boundary read that the counter itself contradicts.  The window is
            # dropped rather than repaired: the confirming read establishes that
            # one of the two disagreed, not which one, and a window built from
            # the survivor would span an interval no clock here measured.  There
            # is another reading along in a moment, and None is already how this
            # source says a window carried none.
            return None
        if elapsed <= 0:
            return 0.0
        if t0 is None or t1 is None:
            # A counter that had nothing to report is not a counter that reported
            # zero.  A monitoring group that has just been created is in exactly
            # this state until the hardware has counted for it, so the opening
            # windows of a run land here -- and a zero handed to the controller
            # there reads as an unsaturated system, which is a state it steers
            # by.  None says the window has no reading, and is dropped rather
            # than acted on.
            return None
        delta_bytes = t1 - t0
        if delta_bytes < 0:
            # Counter wrap (unlikely on 64-bit but be safe)
            delta_bytes = 0
        return delta_bytes / elapsed / 1_000_000

    def close(self):
        resctrl.close_mbm_counters(self._tfds)
        resctrl.close_mbm_counters(self._lfds)

def detect_bw_source(args, window_log=None):
    """Return a BwSource based on args.bw_source."""
    src = getattr(args, 'bw_source', 'auto')
    mon_group = getattr(args, 'resctrl_mon_group', None)
    if src in ('auto', 'resctrl'):
        return ResctrlMbmSource(mon_group=mon_group, window_log=window_log)
    elif src == 'perf':
        raise NotImplementedError('perf BW source is a follow-up')
    else:
        raise ValueError('unknown bw_source: %s' % src)
