# SPDX-License-Identifier: GPL-2.0
"""
Bandwidth source abstractions for damo auto_tier.

BwSource.sample(sample_ms) -> float MB/s (owns the sleep).

ResctrlMbmSource: reads MBM total bytes from resctrl mon_data,
  snapshots before/after sleep, returns delta/elapsed in MB/s.
"""
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import _damo_resctrl as resctrl

class BwSource:
    def sample(self, sample_ms):
        """Sleep sample_ms, return MB/s as float."""
        raise NotImplementedError

class ResctrlMbmSource(BwSource):
    def __init__(self, mon_group=None, resctrl_root=None):
        self._mon_group = mon_group
        self._resctrl_root = resctrl_root
        self._tfds = resctrl.open_mbm_counters(
            mon_group=mon_group, resctrl_root=resctrl_root)

    def sample(self, sample_ms):
        t0 = resctrl.read_mbm_total(self._tfds)
        ts0 = time.monotonic()
        time.sleep(sample_ms / 1000.0)
        t1 = resctrl.read_mbm_total(self._tfds)
        ts1 = time.monotonic()
        elapsed = ts1 - ts0
        if elapsed <= 0:
            return 0.0
        delta_bytes = t1 - t0
        if delta_bytes < 0:
            # Counter wrap (unlikely on 64-bit but be safe)
            delta_bytes = 0
        return delta_bytes / elapsed / 1_000_000

    def close(self):
        resctrl.close_mbm_counters(self._tfds)

def detect_bw_source(args):
    """Return a BwSource based on args.bw_source."""
    src = getattr(args, 'bw_source', 'auto')
    mon_group = getattr(args, 'resctrl_mon_group', None)
    if src in ('auto', 'resctrl'):
        return ResctrlMbmSource(mon_group=mon_group)
    elif src == 'perf':
        raise NotImplementedError('perf BW source is a follow-up')
    else:
        raise ValueError('unknown bw_source: %s' % src)
