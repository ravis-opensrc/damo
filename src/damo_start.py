# SPDX-License-Identifier: GPL-2.0

"""
Start DAMON with given parameters.
"""

import os
import signal
import time

import _damo_sysinfo
import _damon
import _damon_args
import _damon_modules
import _damo_resctrl

def handle_modules():
    sysinfo = _damo_sysinfo.get_sysinfo_or_panic()
    for module in os.listdir(os.path.join(sysinfo.sysfs_path, 'module')):
        if not module.startswith('damon_'):
            continue
        if not _damon_modules.module_running(module):
            continue
        print('Cannot turn on damon since %s is running.  '
              'You should disable it first.' % module)
        answer = input('May I disable it for you? [Y/n] ')
        if answer.lower() == 'n':
            print('Ok, see you later')
            exit(1)
        print('Ok, disabling it')
        _damon_modules.module_disable(module)
        print('Disabled it.  Continue starting DAMON')

def sighandler(signum, frame):
    print('\nsingal %s received' % signum)
    exit(0)

def main(args):
    _damon.ensure_root_and_initialized(args)
    handle_modules()

    err, kdamonds = _damon_args.turn_damon_on(args)
    if err:
        print('could not turn on damon (%s)' % err)
        exit(1)

    mon_group = args.resctrl_mon_group
    if mon_group is not None:
        err = _damo_resctrl.ensure_mon_group(mon_group)
        if err is not None:
            print('resctrl mon group setup failed: %s' % err)
            exit(1)
        err = _damo_resctrl.sync_targets_to_group(mon_group, kdamonds)
        if err is not None:
            print('resctrl initial sync warning: %s' % err)
        print('resctrl: syncing DAMON target PIDs into mon_groups/%s' %
              mon_group)

    if args.include_child_tasks is True or mon_group is not None:
        signal.signal(signal.SIGINT, sighandler)
        signal.signal(signal.SIGTERM, sighandler)
        if args.include_child_tasks is True:
            print('Continue monitoring child tasks and updating DAMON targets '
                  '(refresh every %ds)' % args.child_refresh_interval)
        print('Press Ctrl+C to stop')
        refresh_int = args.child_refresh_interval
        poll = 3                       # resctrl-sync cadence (no kdamond commit)
        since_refresh = refresh_int    # force a child refresh on the first pass
        while True:
            if args.include_child_tasks is True and since_refresh >= refresh_int:
                # This refresh commits the kdamond, so it is kept infrequent:
                # another writer adjusting the same kdamond needs the
                # kdamonds/N/state file to itself.  Exited PIDs are dropped
                # here; resctrl drops their RMIDs regardless.
                _damon.add_commit_vaddr_child_targets(kdamonds)
                since_refresh = 0
            if mon_group is not None:
                # resctrl group membership sync does NOT commit the kdamond,
                # so it is safe to run frequently for fresh BW attribution.
                _damo_resctrl.sync_targets_to_group(mon_group, kdamonds)
            time.sleep(poll)
            since_refresh += poll

def set_argparser(parser):
    _damon_args.set_argparser(parser, add_record_options=False, min_help=True)
    parser.add_argument('--include_child_tasks', action='store_true',
                        help='add child tasks as monitoring target')
    parser.add_argument('--child_refresh_interval', metavar='<seconds>',
                        type=int, default=3,
                        help='how often to (re)scan child tasks and commit them '
                        'as DAMON targets; raise it (e.g. 60) when an external '
                        'controller commits the same kdamond to avoid state '
                        'write collisions')
    parser.add_argument('--resctrl_mon_group', metavar='<name>', default=None,
                        help='sync DAMON target PIDs (and their threads, '
                        'incl. child tasks) into resctrl mon_groups/<name> so '
                        'MBM bandwidth can be measured scoped to exactly the '
                        'monitored tasks')
    parser.description = 'Start DAMON with specified parameters'
    return parser
