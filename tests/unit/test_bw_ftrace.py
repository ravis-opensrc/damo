# SPDX-License-Identifier: GPL-2.0
"""Unit tests for _damo_ftrace FtraceReader and WindowedAverage."""
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import _damo_ftrace as ftrace


def temp_dir():
    """A fresh empty directory as a pathlib.Path, and its cleanup callable."""
    d = tempfile.mkdtemp()
    return pathlib.Path(d), lambda: shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# WindowedAverage tests
# ---------------------------------------------------------------------------

def test_windowed_average_basic():
    w = ftrace.WindowedAverage(3)
    w.add(10000)
    w.add(0)
    w.add(10000)
    assert w.full()
    assert abs(w.average() - 10000 / 1.5) < 1  # ~6666.7

def test_windowed_average_not_full():
    w = ftrace.WindowedAverage(5)
    w.add(100)
    assert not w.full()
    assert w.average() == 100.0

def test_windowed_average_evicts_old():
    w = ftrace.WindowedAverage(3)
    for v in [0, 0, 0, 10000]:
        w.add(v)
    # Window is [0, 0, 10000] -> avg = 10000/3
    assert abs(w.average() - 10000 / 3) < 1

def test_windowed_average_clear():
    w = ftrace.WindowedAverage(3)
    w.add(5000)
    w.clear()
    assert w.average() is None
    assert not w.full()


# ---------------------------------------------------------------------------
# TimeWindowedAverage tests
# ---------------------------------------------------------------------------

def test_time_windowed_average_basic():
    w = ftrace.TimeWindowedAverage(2000)  # 2s window
    w.add(6000, now=0.0)
    w.add(6000, now=1.0)
    assert w.has_data()
    assert w.average() == 6000.0

def test_time_windowed_average_prunes_old():
    w = ftrace.TimeWindowedAverage(2000)  # 2s window
    w.add(0, now=0.0)
    w.add(10000, now=3.0)  # 0.0 is now 3s old -> pruned (cutoff = 3.0 - 2.0 = 1.0)
    assert w.has_data()
    assert w.average() == 10000.0

def test_time_windowed_average_empty():
    w = ftrace.TimeWindowedAverage(1000)
    assert not w.has_data()
    assert w.average() is None

def test_time_windowed_average_clear():
    w = ftrace.TimeWindowedAverage(1000)
    w.add(5000, now=0.0)
    w.clear()
    assert not w.has_data()
    assert w.average() is None

def test_time_windowed_average_all_in_window():
    w = ftrace.TimeWindowedAverage(5000)  # 5s window
    for i in range(5):
        w.add(i * 1000, now=float(i))
    # All 5 samples within window (oldest at t=0, newest at t=4, cutoff=4-5=-1)
    assert w.has_data()
    assert w.average() == (0 + 1000 + 2000 + 3000 + 4000) / 5


# ---------------------------------------------------------------------------
# FtraceReader._parse_line tests (no filesystem needed)
# ---------------------------------------------------------------------------

def test_parse_line_extracts_fields():
    r = ftrace.FtraceReader()
    line = ('          damo-1234  [003]  1234.567890: damos_node_eligible_mem_bp: '
            'nid=0 target_value=6000 current_value=10000')
    ev = r._parse_line(line)
    assert ev['nid'] == 0
    assert ev['target_value'] == 6000
    assert ev['current_value'] == 10000

def test_parse_line_comment_returns_raw():
    r = ftrace.FtraceReader()
    ev = r._parse_line('# tracer: nop')
    assert ev['raw'].startswith('#')

def test_parse_line_missing_fields():
    r = ftrace.FtraceReader()
    ev = r._parse_line('some random line without fields')
    assert 'nid' not in ev
    assert 'raw' in ev


# ---------------------------------------------------------------------------
# FtraceReader with fake filesystem
# ---------------------------------------------------------------------------

def make_fake_ftrace(tmp_path, tracepoint='damos_node_eligible_mem_bp'):
    """Create a minimal fake ftrace tree under tmp_path."""
    inst_dir = tmp_path / 'instances'
    inst_dir.mkdir()
    event_dir = tmp_path / 'events' / 'damon' / tracepoint
    event_dir.mkdir(parents=True)
    (event_dir / 'enable').write_text('0\n')
    # trace_pipe is a regular file in the fake tree
    pipe = tmp_path / 'trace_pipe'
    pipe.write_text('')
    return str(tmp_path)


def test_ftrace_reader_open_close():
    tmp_path, cleanup = temp_dir()
    try:
        root = make_fake_ftrace(tmp_path)
        inst = tmp_path / 'instances'
        r = ftrace.FtraceReader(ftrace_root=root)
        # Point the reader at the fake tree instead of creating a real
        # instance under /sys/kernel/tracing.
        r._instance_dir = str(inst / 'damo-test')
        os.makedirs(r._instance_dir, exist_ok=True)
        ev_dir = os.path.join(r._instance_dir, 'events', 'damon',
                              'damos_node_eligible_mem_bp')
        os.makedirs(ev_dir, exist_ok=True)
        with open(os.path.join(ev_dir, 'enable'), 'w') as f:
            f.write('0\n')
        pipe_path = os.path.join(r._instance_dir, 'trace_pipe')
        with open(pipe_path, 'w') as f:
            f.write('')
        r._tracepoint = 'damos_node_eligible_mem_bp'
        r._pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
        r.close()
        assert r._pipe_fd is None
        assert r._instance_dir is None
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# TimeWindowedAverage.is_stable tests
# ---------------------------------------------------------------------------

def test_time_windowed_average_is_stable():
    fake_time = [0.0]
    def now():
        return fake_time[0]
    wav = ftrace.TimeWindowedAverage(window_ms=2000)
    # No samples -- not stable
    assert not wav.is_stable(2000, now=now())
    # Add sample at t=0
    wav.add(5000, now=now())
    fake_time[0] = 1.0
    # Only 1s elapsed -- not stable (need 2s)
    assert not wav.is_stable(2000, now=now())
    fake_time[0] = 2.1
    # 2.1s elapsed -- stable
    assert wav.is_stable(2000, now=now())


def test_time_windowed_average_is_stable_keeps_adding():
    # The test above adds ONE sample, so nothing is ever pruned.  The callers add
    # a sample per report for as long as they wait, and pass the same value as
    # the retention window and as the threshold -- so the answer has to come from
    # when sampling started and not from how much of it is still held.
    #
    # The period here does not divide the window on purpose: that is the case a
    # span-of-retained-samples answer gets wrong, because pruning caps that span
    # just under the window and it never reaches the threshold.
    period, window_ms = 0.37, 5000
    wav = ftrace.TimeWindowedAverage(window_ms=window_ms)
    t = 0.0
    for _ in range(400):
        wav.add(5000, now=t)
        t += period
        assert wav.is_stable(window_ms, now=t) == (t * 1000 >= window_ms), \
            'is_stable disagreed with elapsed sampling time at t=%.3f' % t
    # And the retained span is indeed capped below the window, which is what
    # makes it the wrong thing to measure.
    assert wav.count() * period * 1000 < window_ms + period * 1000

    # clear() puts it back to having never sampled.
    wav.clear()
    assert not wav.is_stable(window_ms, now=t)
    wav.add(5000, now=t)
    assert not wav.is_stable(window_ms, now=t + 1.0)
    assert wav.is_stable(window_ms, now=t + 5.5)


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
