# SPDX-License-Identifier: GPL-2.0
"""Unit tests for _damo_resctrl MBM reader helpers."""
import os, sys, tempfile
import pathlib
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import _damo_resctrl as resctrl

def temp_dir():
    """A fresh empty directory as a pathlib.Path, and its cleanup callable."""
    d = tempfile.mkdtemp()
    return pathlib.Path(d), lambda: shutil.rmtree(d, ignore_errors=True)

def make_fake_resctrl(tmp_path, n_l3=3, total_vals=None, local_vals=None):
    """Create a fake /sys/fs/resctrl tree under tmp_path."""
    total_vals = total_vals or [1000 * (i+1) for i in range(n_l3)]
    mon_data = tmp_path / 'mon_data'
    mon_data.mkdir(parents=True)
    for i in range(n_l3):
        d = mon_data / ('mon_L3_%02d' % i)
        d.mkdir()
        (d / 'mbm_total_bytes').write_text('%d\n' % total_vals[i])
        if local_vals is not None:
            (d / 'mbm_local_bytes').write_text('%d\n' % local_vals[i])
    return str(tmp_path)

def test_open_and_read():
    tmp_path, cleanup = temp_dir()
    try:
        root = make_fake_resctrl(tmp_path, n_l3=3,
                                 total_vals=[1000, 2000, 3000])
        tfds = resctrl.open_mbm_counters(resctrl_root=root)
        assert len(tfds) == 3
        assert resctrl.read_mbm_total(tfds) == 6000
        resctrl.close_mbm_counters(tfds)
    finally:
        cleanup()

def test_open_no_counters_raises():
    tmp_path, cleanup = temp_dir()
    try:
        # Empty mon_data -- no L3 dirs
        (tmp_path / 'mon_data').mkdir()
        try:
            resctrl.open_mbm_counters(resctrl_root=str(tmp_path))
        except OSError:
            return
        assert False, 'expected OSError'
    finally:
        cleanup()

def test_sparse_domain_numbering():
    # The directory name carries a cache id, not an index, so a gap in the
    # numbering is legal.  Every domain past the gap has to be found.
    tmp_path, cleanup = temp_dir()
    try:
        mon_data = tmp_path / 'mon_data'
        mon_data.mkdir()
        for i, val in ((0, 1000), (5, 2000), (17, 3000)):
            d = mon_data / ('mon_L3_%02d' % i)
            d.mkdir()
            (d / 'mbm_total_bytes').write_text('%d\n' % val)
        tfds = resctrl.open_mbm_counters(resctrl_root=str(tmp_path))
        assert len(tfds) == 3, len(tfds)
        assert resctrl.read_mbm_total(tfds) == 6000
        resctrl.close_mbm_counters(tfds)
    finally:
        cleanup()

def test_non_integer_counter_reads_as_no_reading():
    # The kernel writes these three words into the counter file in place of a
    # number, and a group that has just been created reports one of them until
    # the hardware has counted for it.
    for sentinel in ['Error', 'Unavailable', 'Unassigned']:
        tmp_path, cleanup = temp_dir()
        try:
            root = make_fake_resctrl(tmp_path, n_l3=2, total_vals=[1000, 2000])
            (tmp_path / 'mon_data' / 'mon_L3_01'
             / 'mbm_total_bytes').write_text('%s\n' % sentinel)
            tfds = resctrl.open_mbm_counters(resctrl_root=root)
            # None rather than 1000: a domain that did not answer is not a
            # domain that carried no traffic, and the partial sum would read as
            # a real bandwidth figure.
            assert resctrl.read_mbm_total(tfds) is None, sentinel
            resctrl.close_mbm_counters(tfds)
        finally:
            cleanup()

def test_mon_group_path():
    tmp_path, cleanup = temp_dir()
    try:
        grp = tmp_path / 'mon_groups' / 'shell' / 'mon_data'
        grp.mkdir(parents=True)
        d = grp / 'mon_L3_00'
        d.mkdir()
        (d / 'mbm_total_bytes').write_text('9999\n')
        tfds = resctrl.open_mbm_counters(mon_group='shell',
                                         resctrl_root=str(tmp_path))
        assert resctrl.read_mbm_total(tfds) == 9999
        resctrl.close_mbm_counters(tfds)
    finally:
        cleanup()


def test_local_counter_is_optional():
    # A host that does not offer local bytes is a host with one reading, not one
    # that cannot run: the reading the controller acts on is total bytes.
    tmp_path, cleanup = temp_dir()
    try:
        root = make_fake_resctrl(tmp_path, n_l3=2, total_vals=[1000, 2000])
        lfds = resctrl.open_mbm_local_counters(resctrl_root=root)
        assert lfds == [], lfds
        # None rather than 0: a counter that is not there reported nothing, and
        # zero bytes in a window is a reading.
        assert resctrl.read_mbm_sum(lfds) is None
        resctrl.close_mbm_counters(lfds)
    finally:
        cleanup()

def test_local_counter_is_summed_across_domains():
    tmp_path, cleanup = temp_dir()
    try:
        root = make_fake_resctrl(tmp_path, n_l3=3, total_vals=[10, 20, 30],
                                 local_vals=[1, 2, 3])
        lfds = resctrl.open_mbm_local_counters(resctrl_root=root)
        assert len(lfds) == 3, len(lfds)
        assert resctrl.read_mbm_sum(lfds) == 6
        resctrl.close_mbm_counters(lfds)
    finally:
        cleanup()


class _FlakyFd:
    """A counter fd whose Nth read returns a value the counter never held.

    Modelled on what was measured rather than on what is easy to inject: the
    corrupt read came back high, and the following read came back normal, so the
    counter had not stepped.
    """
    def __init__(self, values):
        self._values = list(values)
        self._n = 0

    def seek(self, _):
        pass

    def read(self):
        v = self._values[min(self._n, len(self._values) - 1)]
        self._n += 1
        return b'%d\n' % v

    def close(self):
        pass

def test_a_confirming_read_passes_a_counter_that_only_moves_forward():
    tmp_path, cleanup = temp_dir()
    try:
        root = make_fake_resctrl(tmp_path, n_l3=2, total_vals=[1000, 2000])
        tfds = resctrl.open_mbm_counters(resctrl_root=root)
        val, ok = resctrl.read_mbm_confirmed(tfds)
        # The value is the first read, unaltered: the second read confirms and is
        # not averaged in, so a window that passes reads as it would have without
        # the check.
        assert val == 3000, val
        assert ok is True
        resctrl.close_mbm_counters(tfds)
    finally:
        cleanup()

def test_a_read_the_counter_contradicts_is_not_confirmed():
    # First read high, second read back at the real value.  This is the shape
    # that was measured on hardware, where the window after the event read
    # normally and so the counter had never stepped.
    fds = [_FlakyFd([19298259099648, 358896144384, 358896144384])]
    val, ok = resctrl.read_mbm_confirmed(fds)
    assert val == 19298259099648, val
    assert ok is False, 'a read above the counter must not be confirmed'

if __name__ == '__main__':
    # test.sh runs each file with python3, so the tests need an explicit
    # runner here.
    import sys
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
