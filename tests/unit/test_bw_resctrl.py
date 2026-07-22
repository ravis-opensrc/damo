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

def make_fake_resctrl(tmp_path, n_l3=3, total_vals=None):
    """Create a fake /sys/fs/resctrl tree under tmp_path."""
    total_vals = total_vals or [1000 * (i+1) for i in range(n_l3)]
    mon_data = tmp_path / 'mon_data'
    mon_data.mkdir(parents=True)
    for i in range(n_l3):
        d = mon_data / ('mon_L3_%02d' % i)
        d.mkdir()
        (d / 'mbm_total_bytes').write_text('%d\n' % total_vals[i])
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
