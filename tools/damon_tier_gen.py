#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""
damon_tier_gen.py -- unified DAMON tiering config generator.

Supports multiple hotness sources (pebs, ibs, pte), including multi-probe
combinations.

Usage:
  damon_tier_gen.py --hotness SOURCE[:WEIGHT][,SOURCE2[:WEIGHT2]] \\
                    [--pid P[,P2,...]] \\
                    [--near_node N] [--far_node N] \\
                    [--target_bp BP] \\
                    [--cold_demote] \\
                    [--local_pa_start ADDR] [--local_pa_end ADDR] \\
                    [--far_pa_start ADDR] [--far_pa_end ADDR] \\
                    -o OUTPUT.yaml
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import _damon

try:
    import yaml
    _have_yaml = True
except ImportError:
    _have_yaml = False

# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

# ops mode for each source
_SOURCE_OPS = {
    'pebs': 'vaddr',
    'pte':  'vaddr',
    'ibs':  'paddr',
}

# PMU device name for auto-detect (None = no prep needed)
_SOURCE_PMU = {
    'pebs': 'cpu',
    'ibs':  'ibs_op',
    'pte':  None,
}

# How often the kdamond refreshes its own scheme statistics.  One second is a
# period a person reading the numbers as a run proceeds can use, and it is far
# longer than the interval at which anything in the loop changes, so it costs
# nothing to leave on.
_STATS_REFRESH_MS = 1000

# Where the node and memory-block topology is read from.  Named rather than
# spelled inline so a test can point them at a tree it built, which is the only
# way to exercise a topology the machine running the test does not have.
_SYSFS_NODE_ROOT = '/sys/devices/system/node'
_SYSFS_MEMORY_ROOT = '/sys/devices/system/memory'


def resolve_pmu_type(name):
    """Read PMU type from /sys/bus/event_source/devices/<name>/type.

    Returns int or None if not found.
    """
    devices_dir = '/sys/bus/event_source/devices'
    if not os.path.isdir(devices_dir):
        return None
    for entry in sorted(os.listdir(devices_dir)):
        if entry == name or entry.startswith(name + '_'):
            type_file = os.path.join(devices_dir, entry, 'type')
            if os.path.isfile(type_file):
                with open(type_file) as f:
                    return int(f.read().strip())
    return None


def build_prep(source):
    """Build DamonPrep for a hotness source.  Returns None for 'pte'."""
    if source == 'pte':
        return None
    if source == 'pebs':
        pmu_type = resolve_pmu_type('cpu')
        if pmu_type is None:
            print('warning: cpu PMU not found; using type=4', file=sys.stderr)
            pmu_type = 4
        return _damon.DamonPrep(
            prep_action='perf_event',
            type=pmu_type,
            config=0x20d1,   # MEM_LOAD_RETIRED.L3_MISS
            config1=0, config2=0,
            freq=1,
            sample_freq=5003,
            sample_phys_addr=0,
            sample_weight_struct=0,
            precise_ip=2,
            wakeup_events=1,
            exclude_kernel=0,
            exclude_hv=0,
        )
    if source == 'ibs':
        pmu_type = resolve_pmu_type('ibs_op')
        if pmu_type is None:
            print('warning: ibs_op PMU not found; using type=11',
                  file=sys.stderr)
            pmu_type = 11
        return _damon.DamonPrep(
            prep_action='perf_event',
            type=pmu_type,
            config=0, config1=0, config2=0,
            freq=0,
            sample_period=262144,
            sample_phys_addr=1,
            sample_weight_struct=0,
            precise_ip=0,
            wakeup_events=1,
            exclude_kernel=0,
            exclude_hv=0,
        )
    raise ValueError('unknown hotness source: %r' % source)


def parse_hotness(hotness_str):
    """Parse 'ibs:1,pte:2' -> [('ibs', 1), ('pte', 2)].

    A source written without a weight yields None, so the caller can tell an
    unspecified weight from an explicit one and pick a default that suits the
    schemes it is about to build.
    """
    result = []
    for part in hotness_str.split(','):
        part = part.strip()
        if ':' in part:
            name, weight_s = part.split(':', 1)
            result.append((name.strip(), int(weight_s.strip())))
        else:
            result.append((part, None))
    return result


def resolve_probe_weights(sources, cold_demote):
    """Fill in unspecified probe weights.

    A nonzero weight hands region merging and aging to the probe's own hit
    counts and turns off the software access check, which leaves a cold scheme
    with nothing to age.  Cold demotion therefore needs weight 0, which keeps
    nr_accesses the readable signal for both schemes: the probe still credits
    the regions it reports, and the regions it does not report decay.
    """
    default = 0 if cold_demote else 1
    resolved = []
    for name, weight in sources:
        if weight is None:
            weight = default
        elif cold_demote and weight != 0:
            raise ValueError(
                'probe weight %d for source %r cannot be combined with '
                '--cold_demote: a nonzero weight disables the software access '
                'check, so no region ages and the cold scheme never matches. '
                'Use %s:0 or omit the weight.' % (weight, name, name))
        resolved.append((name, weight))
    return resolved


# ---------------------------------------------------------------------------
# /proc/iomem parsing
# ---------------------------------------------------------------------------

def read_node_pa_range(nid):
    """The physical range a node's own memory blocks cover, from sysfs.

    Returns (start, end) or None if the node reports no blocks.

    A node names its blocks by index; multiplied by the block size they give the
    range that node holds and no other.  A node's blocks need not be contiguous,
    so the span from the first to the last can enclose a hole, but it cannot
    enclose another node -- which is the property that matters here, and the one
    a bandwidth-derived or iomem-derived span does not have.
    """
    node_dir = os.path.join(_SYSFS_NODE_ROOT, 'node%d' % nid)
    try:
        with open(os.path.join(_SYSFS_MEMORY_ROOT, 'block_size_bytes')) as f:
            block_size = int(f.read().strip(), 16)
        blocks = sorted(int(m.group(1)) for m in
                        (re.match(r'^memory([0-9]+)$', n)
                         for n in os.listdir(node_dir)) if m)
    except (OSError, ValueError):
        return None
    if not blocks:
        return None
    return blocks[0] * block_size, (blocks[-1] + 1) * block_size


def read_pa_ranges(near_node=0, far_node=1,
                   local_pa_start=None, local_pa_end=None,
                   far_pa_start=None, far_pa_end=None):
    """Resolve the near and far physical ranges, one range per node.

    Three sources, in decreasing order of authority: the explicit overrides, the
    two nodes' own memory blocks, and /proc/iomem.

    The nodes' own blocks are preferred over /proc/iomem because iomem describes
    the address map, not the NUMA split: a machine interleaving one socket's DRAM
    across several nodes presents them as adjacent System RAM lines that coalesce
    into one span covering all of them.  A near range derived that way names
    nodes the workload never allocates from, and every scheme scoped by it is
    admitted on memory neither tier holds.  A node's blocks name that node.
    """
    def _to_int(v):
        if v is None:
            return None
        return int(v, 0) if isinstance(v, str) else int(v)

    near = None
    if local_pa_start is not None and local_pa_end is not None:
        near = (_to_int(local_pa_start), _to_int(local_pa_end))
    far = None
    if far_pa_start is not None and far_pa_end is not None:
        far = (_to_int(far_pa_start), _to_int(far_pa_end))

    if near is None:
        near = read_node_pa_range(near_node)
    if far is None:
        far = read_node_pa_range(far_node)

    if near is None or far is None:
        # No sysfs block list for one of the nodes.  Fall back to the address
        # map, which is less precise but is all that is left.
        dram_ranges, cxl_ranges = read_pa_ranges_from_iomem(
                local_pa_start=local_pa_start, local_pa_end=local_pa_end,
                far_pa_start=far_pa_start, far_pa_end=far_pa_end)
        if near is None:
            near = dram_ranges[0]
        if far is None:
            far = cxl_ranges[0]

    # Page 0 is not a migration candidate, and a range starting at 0 makes the
    # first region straddle it.
    near = (max(near[0], 4096), near[1])
    far = (max(far[0], 4096), far[1])

    if near[0] >= near[1] or far[0] >= far[1]:
        raise ValueError(
            'empty physical range (near=%s far=%s) -- pass '
            '--local_pa_start/--local_pa_end and --far_pa_start/--far_pa_end '
            'explicitly' % (near, far))
    if near[0] < far[1] and far[0] < near[1]:
        raise ValueError(
            'the near and far physical ranges overlap (near=%s far=%s), so a '
            'scheme scoped to one would be admitted on the other' % (near, far))
    return [near], [far]


def read_pa_ranges_from_iomem(local_pa_start=None, local_pa_end=None,
                               far_pa_start=None, far_pa_end=None):
    """Parse /proc/iomem for System RAM and CXL Window entries.

    Returns (dram_ranges, cxl_ranges) as lists of (start, end) int tuples.
    Manual overrides take precedence.  Raises ValueError if the CXL window
    (or, without an override, a usable System RAM range) cannot be found --
    /proc/iomem masks physical addresses to 0 for non-root readers, so this
    fails loudly rather than emitting a bogus range that matches nothing.
    """
    def _to_int(v):
        if v is None:
            return None
        return int(v, 0) if isinstance(v, str) else int(v)

    if local_pa_start is not None and local_pa_end is not None:
        dram_ranges = [(_to_int(local_pa_start), _to_int(local_pa_end))]
    else:
        dram_ranges = None  # will parse from iomem

    if far_pa_start is not None and far_pa_end is not None:
        cxl_ranges = [(_to_int(far_pa_start), _to_int(far_pa_end))]
        if dram_ranges is None:
            dram_ranges = _parse_dram_from_iomem(cxl_ranges)
        return dram_ranges, cxl_ranges

    # Parse /proc/iomem
    cxl_ranges = []
    dax_ranges = []
    parsed_dram = []
    try:
        with open('/proc/iomem') as f:
            for line in f:
                stripped = line.strip()
                if ':' not in stripped:
                    continue
                addr_part, name = stripped.split(':', 1)
                name = name.strip()
                try:
                    lo_s, hi_s = addr_part.strip().split('-')
                    lo, hi = int(lo_s, 16), int(hi_s, 16)
                except ValueError:
                    continue
                if 'cxl window' in name.lower():
                    cxl_ranges.append((lo, hi))
                elif 'dax' in name.lower() or 'kmem' in name.lower():
                    # CXL/hmem as devdax onlined system-ram (no 'CXL Window'
                    # line); dax0.0 and its 'System RAM (kmem)' child share the
                    # span (deduped).
                    dax_ranges.append((lo, hi))
                elif 'System RAM' in name and lo >= 0x1000:
                    parsed_dram.append((lo, hi))
    except FileNotFoundError:
        pass

    if not cxl_ranges and dax_ranges:
        cxl_ranges = sorted(set(dax_ranges))
    if not cxl_ranges:
        raise ValueError(
            'CXL Window not found in /proc/iomem -- '
            'pass --far_pa_start/--far_pa_end explicitly')

    # /proc/iomem masks all physical addresses to 0 for non-root readers.
    # A discovered range of (0, 0) is that mask, not a real window: emitting
    # it would produce a bogus addr filter (start=end=0) that matches nothing,
    # so the migrate scheme silently applies to no regions.  Fail loudly.
    if any(lo == 0 and hi == 0 for lo, hi in cxl_ranges):
        raise ValueError(
            '/proc/iomem physical addresses are masked to 0 (run as root, '
            'e.g. `sudo python3 damon_tier_gen.py ...`, or pass '
            '--far_pa_start/--far_pa_end and --local_pa_start/--local_pa_end '
            'explicitly)')

    if dram_ranges is None:
        dram_ranges = _coalesce_dram(parsed_dram, cxl_ranges)
    return dram_ranges, cxl_ranges


def _coalesce_dram(parsed_dram, cxl_ranges):
    """Reduce parsed System RAM fragments to a single near-memory DRAM span.

    A CXL window's nested 'System RAM' child is onlined system-ram but belongs
    to the far tier; drop any fragment overlapping a CXL window so it is not
    duplicated into both tiers.  Coalesce the remaining (often many) fragments
    into one contiguous [min_lo, max_hi] span: DAMON tiers on the whole near
    range, and emitting per-fragment regions would both explode the region
    count and, taking only the first fragment downstream, cover almost nothing.
    """
    near = [(lo, hi) for (lo, hi) in parsed_dram
            if not any(lo < chi and clo < hi for clo, chi in cxl_ranges)]
    if not near:
        raise ValueError(
            'no usable near-memory System RAM range parsed from /proc/iomem '
            '(unprivileged reads mask addresses to 0x0); run as root or pass '
            '--local_pa_start/--local_pa_end explicitly')
    return [(min(lo for lo, _ in near), max(hi for _, hi in near))]


def _parse_dram_from_iomem(cxl_ranges=None):
    ranges = []
    try:
        with open('/proc/iomem') as f:
            for line in f:
                stripped = line.strip()
                if ':' not in stripped:
                    continue
                addr_part, name = stripped.split(':', 1)
                if 'System RAM' not in name:
                    continue
                try:
                    lo_s, hi_s = addr_part.strip().split('-')
                    lo, hi = int(lo_s, 16), int(hi_s, 16)
                    if lo >= 0x1000:
                        ranges.append((lo, hi))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return _coalesce_dram(ranges, cxl_ranges or [])


# ---------------------------------------------------------------------------
# HMAT bw_cutoff auto-derive
# ---------------------------------------------------------------------------

def derive_bw_cutoff(near_node=0, frac=3):
    """Auto-derive bw_cutoff_mbps from HMAT write_bandwidth // frac."""
    path = ('/sys/devices/system/node/node%d/access0/initiators/'
            'write_bandwidth' % near_node)
    try:
        with open(path) as f:
            write_bw = int(f.read().strip())
        cutoff = write_bw // frac
        print('bw_cutoff auto-derived from HMAT node%d write_bandwidth: '
              '%d MB/s -> cutoff=%d MB/s' % (near_node, write_bw, cutoff),
              file=sys.stderr)
        return cutoff
    except FileNotFoundError:
        raise FileNotFoundError(
            'HMAT write_bandwidth not found for node %d '
            '(path: %s). '
            'Pass --bw_cutoff_mbps explicitly or enable HMAT in BIOS.'
            % (near_node, path))


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

# A region this size resolves a sub-region of a few tens of MiB with room to
# spare, which is the scale at which a workload's hot set and the memory it has
# stopped touching sit inside one mapping.  Smaller costs sampling work linearly
# for resolution nothing here needs.
_TARGET_REGION_BYTES = 256 << 20
# Below this the span is small enough that the floor is not what limits
# resolution, and above it the per-region marker cost stops being worth the
# finer start.
_MIN_NR_REGIONS_FLOOR = 1000
_MIN_NR_REGIONS_CAP = 20000


def _vaddr_pid_span(pid):
    """The span the kdamond will monitor for one process, from its mappings.

    A vaddr target carries no regions in the config: the kdamond derives three
    of them itself, by cutting the process's whole mapped range at its two
    largest unmapped gaps.  That is reproducible from /proc/<pid>/maps, so the
    span is computed the same way here rather than guessed -- the outer range
    less those two gaps.  Summing the mappings instead would ignore every hole
    small enough to survive the cut, and on a process whose heap and stack sit
    at opposite ends of the address space those holes are most of the span.

    None if the process is gone or its maps cannot be read, so a caller falls
    back rather than sizing a bound from a partial read.
    """
    try:
        with open('/proc/%d/maps' % pid) as f:
            bounds = []
            for line in f:
                lo, _, hi = line.partition(' ')[0].partition('-')
                bounds.append((int(lo, 16), int(hi, 16)))
    except (OSError, ValueError):
        return None
    if not bounds:
        return None
    gaps = sorted((bounds[i + 1][0] - bounds[i][1]
                   for i in range(len(bounds) - 1)), reverse=True)
    return (bounds[-1][1] - bounds[0][0]) - sum(gaps[:2])


def _monitored_span(ops, targets):
    """Total address span the context will monitor, or None if not yet known.

    A paddr context names its ranges up front, so the span is exact.  A vaddr
    context names processes instead, and the span is what the kdamond will
    derive from their mappings -- read here from the same source it reads, so
    both kinds of context size their region bound against the memory they will
    actually monitor.  A vaddr target naming no process is the case that cannot
    be measured, and that one falls back.
    """
    if ops == 'paddr':
        span = 0
        for t in targets:
            for r in (t.regions or []):
                span += int(r.end) - int(r.start)
        return span or None
    span = 0
    for t in targets:
        if t.pid is None:
            return None
        one = _vaddr_pid_span(int(t.pid))
        if one is None:
            return None
        span += one
    return span or None


def build_hot_context(sources, ops, near_node=0, far_node=1,
                      pids=None, target_bp=None,
                      local_pa_start=None, local_pa_end=None,
                      far_pa_start=None, far_pa_end=None,
                      cold_demote=False, cold_demote_mode='reactive',
                      min_nr_regions=None, max_nr_regions=None):
    """Build the hot-tracking context, optionally with cold demotion.

    With cold_demote the context enables the page-fault primitive alongside
    the probe and carries the cold scheme in the same scheme list, so one
    kdamond drives both directions.  The two signals are additive: the probe
    credits the regions its PMU reports, the page-fault primitive's marker
    reports the rest, and a region neither reports decays to zero accesses,
    which is what the cold scheme matches on.
    """
    # Build probes
    probes = []
    for source, weight in sources:
        prep = build_prep(source)
        preps = [prep] if prep is not None else []
        probe = _damon.DamonProbe(filters=[], weight=weight, preps=preps)
        probes.append(probe)

    intervals = _damon.DamonIntervals(sample='5ms', aggr='100ms',
                                      ops_update='1s')

    near_ranges = None
    if ops == 'paddr':
        dram_ranges, cxl_ranges = read_pa_ranges(
            near_node=near_node, far_node=far_node,
            local_pa_start=local_pa_start, local_pa_end=local_pa_end,
            far_pa_start=far_pa_start, far_pa_end=far_pa_end)
        near_ranges = dram_ranges
        # Targets: explicit PA regions
        regions = []
        for lo, hi in dram_ranges + cxl_ranges:
            regions.append(_damon.DamonRegion(lo, hi))
        targets = [_damon.DamonTarget(pid=None, regions=regions)]
        schemes = _build_paddr_schemes(dram_ranges, cxl_ranges,
                                       near_node, far_node, target_bp)
    else:
        # vaddr: per-PID targets
        if pids:
            targets = [_damon.DamonTarget(pid=p, regions=[]) for p in pids]
        else:
            targets = [_damon.DamonTarget(pid=None, regions=[])]
        schemes = _build_vaddr_schemes(near_node, far_node, target_bp=target_bp)

    sample_control = None
    if cold_demote:
        # The cold scheme goes last so the hot schemes keep the indices the
        # bandwidth controller addresses them by.
        schemes = schemes + [build_cold_scheme(near_node, far_node,
                                               mode=cold_demote_mode,
                                               near_ranges=near_ranges)]
        # Exactly one primitive may be enabled.  The page-fault primitive is
        # the one that produces the absence of an access under a probe: it is
        # the only path whose zero-access decay still runs when a probe is
        # attached.
        sample_control = _damon.DamonSampleControl(
            primitives_enabled=_damon.DamonPrimitivesEnabled(
                page_table=False, page_fault=True))

    # Region bounds do double duty when the page-fault primitive is enabled:
    # they set the resolution a hot or cold sub-region has to survive at, and
    # they set the sampling rate, because the primitive installs one marker per
    # region per sampling interval.
    #
    # The resolution is what decides which scheme a region is offered to, so a
    # bound too coarse for the span silently swaps the two.  A region wide
    # enough to hold both the memory a process is using and the memory it has
    # stopped using carries one access count for all of it, and whichever
    # signature wins drags the rest along: a nonzero count admits the promotion
    # scheme over the idle part and, because the cold scheme matches only a zero
    # count, excludes demotion from those pages entirely.  The idle memory then
    # moves under the promotion scheme's weights instead of being demoted, which
    # looks like demotion working and is not.  A count that decays to zero does
    # the reverse and demotes memory that is still being read.
    #
    # So the floor is derived from the span rather than fixed, to start the
    # regions small enough that the two signatures separate.  The kdamond splits
    # and merges from there by access rate; this only sets where it begins, and
    # beginning too coarse is not recoverable, because a region is only ever
    # split on a difference the region itself was able to show.
    if min_nr_regions is None and cold_demote:
        span = _monitored_span(ops, targets)
        if span:
            min_nr_regions = max(_MIN_NR_REGIONS_FLOOR,
                                 min(_MIN_NR_REGIONS_CAP,
                                     span // _TARGET_REGION_BYTES))
    if min_nr_regions is None:
        min_nr_regions = _MIN_NR_REGIONS_FLOOR if cold_demote else 10
    if max_nr_regions is None:
        # Headroom to split into, not a second bound to tune: the kdamond needs
        # room above the floor or it merges regions it has just split.
        max_nr_regions = max(20000, min_nr_regions * 4) if cold_demote else 1000
    nr_regions = _damon.DamonNrRegionsRange(min_=min_nr_regions,
                                            max_=max_nr_regions)

    return _damon.DamonCtx(
        ops=ops,
        intervals=intervals,
        nr_regions=nr_regions,
        targets=targets,
        schemes=schemes,
        probes=probes,
        sample_control=sample_control)


def _build_paddr_schemes(dram_ranges, cxl_ranges, near_node, far_node,
                         target_bp=None):
    """PULL+PUSH schemes with addr filters and node_eligible_mem_bp goals."""
    # Closed-loop (no explicit target) starts all-DRAM (ratio=100): the
    # controller hill-climbs the ratio down from 100 toward the bandwidth
    # optimum, so the workload begins single-tier and the gradient is visible.
    closed_loop = target_bp is None
    if closed_loop:
        target_bp = 10000

    def addr_filter(ranges):
        lo, hi = ranges[0]
        return _damon.DamosFilter(
            filter_type='addr',
            matching=True,
            allow=True,
            address_range=_damon.DamonRegion(lo, hi))

    def make_scheme(nid, goal_bp, filter_ranges):
        goal = _damon.DamosQuotaGoal(
            metric=_damon.qgoal_node_eligible_mem_bp,
            target_value=str(goal_bp),
            nid=str(nid))
        quotas = _damon.DamosQuotas(
            time_ms=0,
            sz_bytes=5368709120,
            reset_interval_ms=1000,
            goals=[goal],
            goal_tuner='temporal')
        filt = addr_filter(filter_ranges)
        ap = _damon.DamosAccessPattern(
            sz_bytes=['4096', 'max'],
            # nr_accesses min=1 as an ABSOLUTE sample count (unit_samples),
            # not '1 %': a percent-of-max value rounds down to 0 samples on
            # sysfs write (e.g. 1% of max=20 -> 0), which would match every
            # region including nr_accesses==0 ('not sampled by perf').
            nr_accesses=['1', 'max'],
            nr_accesses_unit=_damon.unit_samples,
            age=['0', 'max'])
        return _damon.Damos(
            action='migrate_hot',
            target_nid=nid,
            access_pattern=ap,
            quotas=quotas,
            filters=[filt])

    # Closed-loop always emits both PULL+PUSH so the controller can move the
    # ratio in either direction; the initial goals encode ratio=100 (all-DRAM):
    # PULL target=10000 (pull all hot CXL pages back to DRAM), PUSH target=0.
    if closed_loop:
        return [
            make_scheme(near_node, 10000, cxl_ranges),
            make_scheme(far_node, 0, dram_ranges),
        ]
    # One-shot endpoints: a single scheme is sufficient.
    # target_bp==10000 => 100% near/DRAM: only the PULL scheme (CXL->DRAM) is
    # meaningful (PUSH goal would be 0 bp).  target_bp==0 => only PUSH.
    # Otherwise emit both PULL+PUSH for the requested interior split.
    if target_bp >= 10000:
        return [make_scheme(near_node, 10000, cxl_ranges)]
    if target_bp <= 0:
        return [make_scheme(far_node, 10000, dram_ranges)]
    return [
        make_scheme(near_node, target_bp, cxl_ranges),
        make_scheme(far_node, 10000 - target_bp, dram_ranges),
    ]


def _build_vaddr_schemes(near_node, far_node, target_bp=None):
    """Single migrate_hot scheme with 2 DamosDest weights.

    V6-9: dests reflect target_bp when provided (near=bp//100, far=100-near).
    Without target_bp, initial closed-loop state is near=100, far=0.

    Weight-mode (vaddr): NO quota -- unlimited migration.
    Reference: intel_pa_setup.sh lines 227-229: ms=0, bytes=0, reset_interval_ms=0
    """
    if target_bp is not None:
        near_weight = target_bp // 100
        far_weight = 100 - near_weight
    else:
        near_weight, far_weight = 100, 0
    dests = [
        _damon.DamosDest(id=near_node, weight=near_weight),
        _damon.DamosDest(id=far_node, weight=far_weight),
    ]
    # Weight-mode: unlimited quota (sz=0, reset_interval_ms=0)
    quotas = _damon.DamosQuotas(
        time_ms=0,
        sz_bytes=0,
        reset_interval_ms=0,
        goals=[])
    ap = _damon.DamosAccessPattern(
        sz_bytes=['4096', 'max'],
        nr_accesses=['1', 'max'],
        nr_accesses_unit=_damon.unit_samples,
        age=['0', 'max'])
    return [_damon.Damos(
        action='migrate_hot',
        access_pattern=ap,
        quotas=quotas,
        dests=dests)]


def build_cold_scheme(near_node=0, far_node=1, mode='reactive',
                      near_ranges=None):
    """Build the cold-demote scheme.

    Demotes unused (cold, not-recently-accessed) near-node pages to the far
    node.  The scheme has two gates:

      1. access_rate 0%..0% + age >= 5s -- region-level cold: no accesses
         credited for at least 5s, by either the probe or the page-fault
         primitive.
      2. quota goal node_mem_free_bp on the near node -- the memory-pressure
         trigger.  Goal feedback grows the effective demotion quota as the
         near node's free memory falls toward the 1% target, so demotion
         only ramps up under real pressure.

    The scheme lives in the hot context.  What makes nr_accesses==0 mean 'cold'
    rather than 'not sampled by perf' is the page-fault primitive the caller
    enables alongside the probe: it reports accesses the probe's PMU does not,
    so a region only reaches zero when neither source reported it.

    Coldness therefore rests on those two sources and on nothing else.  A
    reject-young filter would add a third: it rechecks each folio's PTE Accessed
    bit at apply time and clears it, which is the bit the page-table primitive
    owns -- and that primitive is the one this configuration disables in favour
    of the page-fault one.  Consulting it here would decide coldness by a PTE
    scan running underneath the sources above, and clearing it would take
    accesses away from them, so no such filter is installed.

    near_ranges scopes the scheme to the near node's physical addresses and is
    required in paddr mode: there the monitoring target spans both nodes, so an
    unscoped scheme is admitted on far-node regions too and would demote pages
    already at their destination.  In vaddr mode the target is a process's
    address space, which carries no node, so there is nothing to scope by and
    target_nid alone decides where a page goes.
    """
    # Cold = not accessed for >= 5s.  access_rate 0%..0% keeps only regions
    # with zero credited accesses.
    ap = _damon.DamosAccessPattern(
        sz_bytes=['4096', 'max'],
        nr_accesses=['0 %', '0 %'],
        age=['5s', 'max'])
    # Demotion pace depends on mode:
    #   reactive  - node_mem_free_bp goal on the NEAR node gates demotion by
    #               memory pressure.  As near (DRAM) free memory falls toward
    #               1% (=100 bp), goal feedback raises the effective demotion
    #               size, so cold pages spill to the far node only under real
    #               pressure.  reset_interval must be non-zero to actuate.
    #   proactive - no goal; the fixed quota alone applies, so cold pages are
    #               demoted continuously (up to the ceiling) regardless of
    #               pressure, keeping DRAM headroom free for bursts.
    if mode == 'proactive':
        goals = []
    else:
        goals = [_damon.DamosQuotaGoal(
            metric=_damon.qgoal_node_mem_free_bp,
            target_value='1 %',
            nid=str(near_node))]
    quotas = _damon.DamosQuotas(
        time_ms=1000,
        sz_bytes=10737418240,
        reset_interval_ms=1000,
        goals=goals,
        goal_tuner='temporal')
    filters = []
    if near_ranges:
        lo, hi = near_ranges[0]
        filters = [_damon.DamosFilter(
            filter_type='addr', matching=True, allow=True,
            address_range=_damon.DamonRegion(lo, hi))]
    dests = [_damon.DamosDest(id=far_node, weight=100)]
    return _damon.Damos(
        action='migrate_cold',
        target_nid=far_node,
        access_pattern=ap,
        apply_interval_us='1s',
        quotas=quotas,
        filters=filters,
        dests=dests)


def build_config(hotness_str, near_node=0, far_node=1,
                 pids=None, target_bp=None, cold_demote=False,
                 cold_demote_mode='reactive',
                 bw_cutoff_mbps=None, bw_cutoff_frac=3,
                 local_pa_start=None, local_pa_end=None,
                 far_pa_start=None, far_pa_end=None,
                 min_nr_regions=None, max_nr_regions=None):
    """Build complete config dict with kdamonds + optional auto_tier."""
    sources = resolve_probe_weights(parse_hotness(hotness_str), cold_demote)

    # Validate: all sources must share same ops mode
    ops_modes = set(_SOURCE_OPS.get(s, 'vaddr') for s, _ in sources)
    if len(ops_modes) > 1:
        raise ValueError(
            'mixed ops modes in --hotness %r: %s. '
            'All sources must be vaddr (pebs, pte) or paddr (ibs).'
            % (hotness_str, {s: _SOURCE_OPS.get(s) for s, _ in sources}))
    ops = ops_modes.pop()

    # Validate: --pid only for vaddr
    if pids and ops == 'paddr':
        raise ValueError(
            '--pid is only valid for vaddr sources (pebs, pte); '
            'paddr sources (ibs) use PA regions, not PIDs')

    # One context, one kdamond: the cold scheme rides in the hot context so
    # both directions see the same region list and the same access rates.
    hot_ctx = build_hot_context(
        sources, ops, near_node=near_node, far_node=far_node,
        pids=pids, target_bp=target_bp,
        local_pa_start=local_pa_start, local_pa_end=local_pa_end,
        far_pa_start=far_pa_start, far_pa_end=far_pa_end,
        cold_demote=cold_demote, cold_demote_mode=cold_demote_mode,
        min_nr_regions=min_nr_regions, max_nr_regions=max_nr_regions)

    # The kdamond keeps its own scheme statistics current on this period.  A
    # reader that does not ask for this has to command each refresh through the
    # same file a configuration change goes through, and that file refuses a
    # caller while another holds it -- so the refreshes, which are frequent,
    # take it from the configuration changes, which are not.  A kdamond
    # refreshing from inside its own loop cannot cost a change that way, and the
    # numbers a run reports at the end are the ones it really finished with
    # rather than whichever commanded refresh last succeeded.
    kdamond = _damon.Kdamond(state='off', pid=None, contexts=[hot_ctx],
                             refresh_ms=_STATS_REFRESH_MS)
    kdamonds_kv = [kdamond.to_kvpairs(raw=True)]

    result = {'kdamonds': kdamonds_kv}

    if target_bp is None:
        # Closed-loop mode: emit auto_tier section
        if bw_cutoff_mbps is None:
            bw_cutoff_mbps = derive_bw_cutoff(near_node=near_node,
                                              frac=bw_cutoff_frac)
        result['auto_tier'] = {
            'bw_cutoff_mbps': bw_cutoff_mbps,
            'stable_window_ms': 5000,
            'tolerance_bp': 200,
            'near_node': near_node,
            'far_node': far_node,
            'sample_interval_ms': 1000,
            'adjust_interval_ms': 10000,
            'min_ratio': 0,
            'max_ratio': 100,
        }

    return result


# ---------------------------------------------------------------------------
# yaml output
# ---------------------------------------------------------------------------

def _yaml_dump(config, stream=None):
    import collections
    try:
        from yaml.representer import Representer
        import yaml

        class _OrderedDumper(yaml.Dumper):
            pass

        def _represent_ordered_dict(dumper, data):
            return dumper.represent_mapping(
                'tag:yaml.org,2002:map', data.items())

        _OrderedDumper.add_representer(
            collections.OrderedDict, _represent_ordered_dict)

        return yaml.dump(config, stream, Dumper=_OrderedDumper,
                         default_flow_style=False, sort_keys=False)
    except ImportError:
        raise ImportError('pyyaml not installed; run: pip install pyyaml')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if not _have_yaml:
        print('error: pyyaml not installed; run: pip install pyyaml',
              file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hotness', required=True,
                        help='hotness source(s): pebs, ibs, pte '
                             '(with optional :weight, comma-separated)')
    parser.add_argument('--pid', type=int, action='append',
                        dest='pids', metavar='PID',
                        help='target PID (repeatable; vaddr sources only)')
    parser.add_argument('--near_node', type=int, default=0,
                        help='DRAM NUMA node (default 0)')
    parser.add_argument('--far_node', type=int, default=1,
                        help='CXL NUMA node (default 1)')
    parser.add_argument('--target_bp', type=int, default=None,
                        help='one-shot target_dram_bp (0-10000); '
                             'omit for closed-loop auto_tier mode')
    parser.add_argument('--cold_demote', action='store_true',
                        help='add a cold-demote scheme and the page fault '
                             'primitive to the hot context')
    parser.add_argument('--cold_demote_mode', default='reactive',
                        choices=['reactive', 'proactive'],
                        help='reactive: gate demotion by node_mem_free_bp '
                             'memory-pressure goal (default); proactive: '
                             'demote cold pages continuously (no goal)')
    parser.add_argument('--local_pa_start', default=None,
                        help='DRAM PA range start (hex or decimal)')
    parser.add_argument('--local_pa_end', default=None,
                        help='DRAM PA range end (hex or decimal)')
    parser.add_argument('--far_pa_start', default=None,
                        help='CXL PA range start (hex or decimal)')
    parser.add_argument('--far_pa_end', default=None,
                        help='CXL PA range end (hex or decimal)')
    parser.add_argument('--min_nr_regions', type=int, default=None,
                        help='min_nr_regions (default 10; with --cold_demote, '
                             'derived from the monitored span for ~%d MiB '
                             'regions, clamped to %d..%d)'
                             % (_TARGET_REGION_BYTES >> 20,
                                _MIN_NR_REGIONS_FLOOR, _MIN_NR_REGIONS_CAP))
    parser.add_argument('--max_nr_regions', type=int, default=None,
                        help='max_nr_regions (default 1000, or 4x the min for '
                             '--cold_demote)')
    parser.add_argument('--bw_cutoff_mbps', type=int, default=None,
                        help='BW cutoff MB/s (default: auto from HMAT)')
    parser.add_argument('--bw_cutoff_frac', type=int, default=3,
                        help='HMAT write_bw denominator (default 3)')
    parser.add_argument('-o', '--output', default='-',
                        help='output file (default: stdout)')
    args = parser.parse_args()

    def _pa_int(s):
        return int(s, 0) if s is not None else None

    try:
        config = build_config(
            args.hotness,
            near_node=args.near_node,
            far_node=args.far_node,
            pids=args.pids,
            target_bp=args.target_bp,
            cold_demote=args.cold_demote,
            cold_demote_mode=args.cold_demote_mode,
            bw_cutoff_mbps=args.bw_cutoff_mbps,
            bw_cutoff_frac=args.bw_cutoff_frac,
            local_pa_start=_pa_int(args.local_pa_start),
            local_pa_end=_pa_int(args.local_pa_end),
            far_pa_start=_pa_int(args.far_pa_start),
            far_pa_end=_pa_int(args.far_pa_end),
            min_nr_regions=args.min_nr_regions,
            max_nr_regions=args.max_nr_regions)
    except (ValueError, FileNotFoundError) as e:
        print('error: %s' % e, file=sys.stderr)
        sys.exit(1)

    if args.output == '-':
        _yaml_dump(config, sys.stdout)
    else:
        with open(args.output, 'w') as f:
            _yaml_dump(config, f)
        print('written to %s' % args.output)


if __name__ == '__main__':
    main()
