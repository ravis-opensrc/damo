# SPDX-License-Identifier: GPL-2.0

"""
Sync DAMON VADDR monitoring-target PIDs (and their threads) into a resctrl
MBM monitoring group, so external bandwidth readers (e.g. an MBM-based
tiering controller) can measure bandwidth scoped to exactly the tasks DAMON
is monitoring, rather than system-wide.

Intended to be called on the same cadence as child-task target refresh in
`damo start --include_child_tasks`, keeping the resctrl group membership in
lockstep with the (dynamic) DAMON target set.
"""

import os

RESCTRL_ROOT = '/sys/fs/resctrl'


def resctrl_mounted():
    return os.path.isdir(os.path.join(RESCTRL_ROOT, 'mon_data'))


def ensure_mon_group(name):
    '''Create resctrl mon_groups/<name> if needed.  Returns error string or None.'''
    if not resctrl_mounted():
        return ('resctrl not mounted (try: sudo mount -t resctrl resctrl %s)'
                % RESCTRL_ROOT)
    grp = os.path.join(RESCTRL_ROOT, 'mon_groups', name)
    if not os.path.isdir(grp):
        try:
            os.mkdir(grp)
        except Exception as e:
            return 'failed to create mon group %s (%s)' % (grp, e)
    return None


def _thread_ids(pid):
    '''All thread IDs of a process (resctrl assigns per-thread RMIDs, so the
    whole thread group must be added, not just the tgid).'''
    try:
        return os.listdir('/proc/%s/task' % pid)
    except Exception:
        return []


def sync_pids_to_group(name, pids):
    '''Write every thread of every pid into mon_groups/<name>/tasks.

    resctrl 'tasks' is append-on-write: writing a tid assigns it to this
    group; already-present tids are harmless.  Threads that exited are
    dropped by the kernel automatically.  Returns error string or None.
    '''
    tasks_path = os.path.join(RESCTRL_ROOT, 'mon_groups', name, 'tasks')
    assigned = 0
    for pid in pids:
        for tid in _thread_ids(pid):
            try:
                with open(tasks_path, 'w') as f:
                    f.write('%s' % tid)
                assigned += 1
            except Exception:
                # thread may have exited between listdir and write; ignore
                pass
    if assigned == 0:
        return 'no threads assigned to mon group %s' % name
    return None


def ctx_target_pids(ctx):
    '''Live (non-obsolete) VADDR target PIDs of one context.'''
    pids = []
    for t in ctx.targets:
        if getattr(t, 'obsolete', False):
            continue
        if t.pid is not None:
            pids.append('%s' % t.pid)
    return pids


def target_pids(kdamonds):
    '''Live (non-obsolete) VADDR target PIDs across all kdamonds/contexts.'''
    pids = []
    for kd in kdamonds:
        for ctx in kd.contexts:
            pids.extend(ctx_target_pids(ctx))
    return pids


def sync_targets_to_group(name, kdamonds):
    '''Convenience: sync all live target PIDs (+threads) into the group.'''
    return sync_pids_to_group(name, target_pids(kdamonds))


def sync_ctx_targets_to_group(name, ctx):
    '''Sync one context's live target PIDs (+threads) into the group.

    A group scoped to a context rather than to every context there is: a reader
    measuring one context's placement decisions against a group holding another
    context's processes is reading traffic it cannot move, which is the mismatch
    a scoped group exists to remove.
    '''
    return sync_pids_to_group(name, ctx_target_pids(ctx))


# MBM bandwidth counter reader helpers

RESCTRL_MAX_L3 = 32

def _mon_data_dir(mon_group=None, resctrl_root=None):
    root = resctrl_root or RESCTRL_ROOT
    if mon_group is None:
        return os.path.join(root, 'mon_data')
    return os.path.join(root, 'mon_groups', mon_group, 'mon_data')


def _open_mbm_files(counter, mon_group=None, resctrl_root=None):
    """Open every mon_L3_NN/<counter> fd under mon_data/.

    The directory names carry a cache id rather than an index, so the numbering
    is not guaranteed to be dense or to start at zero.  Stopping at the first
    absent index would therefore drop every domain past a gap and return a sum
    that looks complete, so the whole range is walked.
    """
    base = _mon_data_dir(mon_group=mon_group, resctrl_root=resctrl_root)
    fds = []
    for i in range(RESCTRL_MAX_L3):
        p = os.path.join(base, 'mon_L3_%02d' % i, counter)
        try:
            fds.append(open(p, 'rb'))
        except OSError:
            continue
    return fds, base


def open_mbm_counters(mon_group=None, resctrl_root=None):
    """Open every mon_L3_NN/mbm_total_bytes fd under mon_data/.

    mon_group: if None, use root mon_data; else mon_groups/<mon_group>/mon_data.
    resctrl_root: override for testing (default: RESCTRL_ROOT).
    Returns a list of total_fds. Raises OSError if none found.
    """
    total_fds, base = _open_mbm_files('mbm_total_bytes', mon_group=mon_group,
                                      resctrl_root=resctrl_root)
    if not total_fds:
        raise OSError('no mbm counters found under %s' % base)
    return total_fds


def open_mbm_local_counters(mon_group=None, resctrl_root=None):
    """Open every mon_L3_NN/mbm_local_bytes fd under mon_data/.

    Returns a list of fds, empty if the counter is not there.  Empty rather
    than an error: local bytes is a second reading of the same window taken for
    comparison, not the reading a controller acts on, and a host that does not
    offer it is a host with one reading rather than a host that cannot run.
    """
    local_fds, _ = _open_mbm_files('mbm_local_bytes', mon_group=mon_group,
                                   resctrl_root=resctrl_root)
    return local_fds

def _read_mbm_fd(fd):
    """Read a single MBM counter fd.  Returns int bytes, or None.

    The counter files do not always hold a number: the kernel writes 'Error',
    'Unavailable' or 'Unassigned' into them, and a monitoring group that has
    just been created reports 'Unavailable' until the hardware has counted
    anything for it.  Parsing that as an integer raises, so a caller reading on
    a thread would lose the thread rather than the reading.  None says the
    counter had nothing to report.
    """
    fd.seek(0)
    raw = fd.read().strip()
    try:
        return int(raw)
    except ValueError:
        return None

def read_mbm_total(total_fds):
    """Sum mbm_total_bytes across all L3 instances.

    Returns bytes, or None if any instance had nothing to report.  None rather
    than a partial sum: a domain that did not answer is not a domain that
    carried no traffic, and adding up the rest yields an undercount that reads
    as a real bandwidth figure.
    """
    total = 0
    for f in total_fds:
        val = _read_mbm_fd(f)
        if val is None:
            return None
        total += val
    return total


def read_mbm_sum(fds):
    """Sum any one MBM counter across L3 instances.  None if none were opened.

    Same rule as read_mbm_total for a domain that did not answer, and None as
    well for an empty fd list, so a counter this host does not offer reads as
    absent rather than as zero bytes.
    """
    if not fds:
        return None
    return read_mbm_total(fds)


def read_mbm_confirmed(fds):
    """Read the counter twice and return (value, confirmed).

    These counters are cumulative, so a second read of the same boundary cannot
    come back lower than the first.  A read that disagrees with the counter
    breaks that ordering, and the ordering is a property of the counter rather
    than of the platform it sits in: checking it needs no bandwidth ceiling to
    compare against, and so no per-host figure to be told or to get wrong.

    The value returned is the first read.  The second read is a confirmation and
    not a measurement, so it is not averaged in and not substituted -- a window
    that passes therefore reads exactly as it would have without the second
    read, and the check only ever rejects.
    """
    first = read_mbm_total(fds)
    if first is None:
        return None, False
    second = read_mbm_total(fds)
    if second is None:
        return first, False
    return first, second >= first

def close_mbm_counters(total_fds):
    """Close all open MBM fds."""
    for f in total_fds:
        try:
            f.close()
        except Exception:
            pass
