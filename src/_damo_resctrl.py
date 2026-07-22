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
