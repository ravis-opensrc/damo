# SPDX-License-Identifier: GPL-2.0
"""Unit tests for extended DamonPrep (perf_event PMU attrs)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import _damon


def test_damonprep_perf_event_roundtrip():
    p = _damon.DamonPrep(
            prep_action='perf_event',
            type=11, config=0, config1=0, config2=0,
            sample_period=10000, freq=0, sample_phys_addr=1,
            precise_ip=0, wakeup_events=1,
            exclude_kernel=0, exclude_hv=0)
    kv = p.to_kvpairs()
    p2 = _damon.DamonPrep.from_kvpairs(kv)
    assert p == p2
    assert p2.type == 11
    assert p2.sample_period == 10000
    assert p2.config2 == 0


def test_damonprep_set_pgidle_no_pmu_attrs():
    p = _damon.DamonPrep(prep_action='set_pgidle')
    kv = p.to_kvpairs()
    assert 'type' not in kv
    assert 'config' not in kv
    p2 = _damon.DamonPrep.from_kvpairs(kv)
    assert p2.type is None
    assert p2.config is None


def test_damonprep_perf_event_eq():
    p1 = _damon.DamonPrep(prep_action='perf_event', type=11, config=0)
    p2 = _damon.DamonPrep(prep_action='perf_event', type=11, config=0)
    p3 = _damon.DamonPrep(prep_action='perf_event', type=6, config=0)
    assert p1 == p2
    assert p1 != p3


def test_damonprep_perf_event_neq_set_pgidle():
    p1 = _damon.DamonPrep(prep_action='perf_event', type=11)
    p2 = _damon.DamonPrep(prep_action='set_pgidle')
    assert p1 != p2


def test_sample_control_no_perf_events_attr():
    sc = _damon.DamonSampleControl()
    assert not hasattr(sc, 'perf_events') or getattr(sc, 'perf_events', None) is None
    kv = sc.to_kvpairs()
    assert 'perf_events' not in kv


def test_damonprep_partial_pmu_attrs():
    """Only set type and sample_period — others stay None."""
    p = _damon.DamonPrep(prep_action='perf_event', type=11, sample_period=4096)
    kv = p.to_kvpairs()
    assert kv['type'] == 11
    assert kv['sample_period'] == 4096
    assert 'config2' not in kv or kv.get('config2') is None
    p2 = _damon.DamonPrep.from_kvpairs(kv)
    assert p2.type == 11
    assert p2.sample_period == 4096


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
