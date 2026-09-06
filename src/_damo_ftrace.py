# SPDX-License-Identifier: GPL-2.0
"""
Ftrace reader for DAMON tracepoints.

Provides FtraceReader: creates a private ftrace instance, enables a
tracepoint, and reads events from trace_pipe in a non-blocking fashion.
Used by GoalActuator.converged() to read damos_node_eligible_mem_bp
events for PA-mode convergence detection.

The ftrace_root parameter allows unit tests to redirect to a synthetic
tree without touching /sys/kernel/tracing.
"""

import os
import errno
import re
import time

FTRACE_ROOT = '/sys/kernel/tracing'
_seq = 0


def _next_seq():
    global _seq
    _seq += 1
    return _seq


class FtraceReader:
    """Private ftrace instance reader for a single tracepoint.

    Usage:
        r = FtraceReader()
        r.open('damos_node_eligible_mem_bp', ftrace_root='/sys/kernel/tracing')
        event = r.read_event()   # returns dict or None (non-blocking)
        r.close()

    The instance directory is named
    ``instances/damo-<pid>-<seq>/`` to avoid collisions when
    multiple GoalActuators run concurrently.
    """

    def __init__(self, ftrace_root=None):
        self._root = ftrace_root or FTRACE_ROOT
        self._instance_dir = None
        self._pipe_fd = None
        self._buf = ''
        self._tracepoint = None

    def open(self, tracepoint):
        """Create a private ftrace instance and enable *tracepoint*.

        tracepoint is the bare event name, e.g.
        ``'damos_node_eligible_mem_bp'``.  The subsystem ``damon/`` is
        prepended automatically.
        """
        self._tracepoint = tracepoint
        name = 'damo-%d-%d' % (os.getpid(), _next_seq())
        self._instance_dir = os.path.join(self._root, 'instances', name)
        os.makedirs(self._instance_dir, exist_ok=True)

        # Enable the tracepoint
        event_enable = os.path.join(
            self._instance_dir, 'events', 'damon', tracepoint, 'enable')
        try:
            with open(event_enable, 'w') as f:
                f.write('1\n')
        except OSError as e:
            self.close()
            raise OSError(
                'failed to enable tracepoint %s: %s' % (tracepoint, e))

        # Open trace_pipe non-blocking
        pipe_path = os.path.join(self._instance_dir, 'trace_pipe')
        self._pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)

    def read_event(self):
        """Read one event line from trace_pipe.

        Returns a dict of parsed fields, or None if no event is
        available right now (non-blocking).  The dict always contains
        at least ``{'raw': <line>}``; known fields are also extracted.

        For ``damos_node_eligible_mem_bp`` the dict contains:
          ``nid``, ``target_value``, ``current_value`` (all int).
        """
        if self._pipe_fd is None:
            return None

        # Try to read more data into the buffer
        try:
            chunk = os.read(self._pipe_fd, 4096)
            self._buf += chunk.decode('utf-8', errors='replace')
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
            # No data available
            if '\n' not in self._buf:
                return None

        if '\n' not in self._buf:
            return None

        line, self._buf = self._buf.split('\n', 1)
        line = line.strip()
        if not line or line.startswith('#'):
            return None

        return self._parse_line(line)

    def _parse_line(self, line):
        """Parse a ftrace event line into a dict.

        damos_node_eligible_mem_bp format (kernel tracepoint):
          ... damos_node_eligible_mem_bp: nid=0 target_value=6000 current_value=10000
        """
        event = {'raw': line}
        # Extract nid, target_value, current_value
        for field in ('nid', 'target_value', 'current_value'):
            m = re.search(r'\b' + field + r'=(\d+)', line)
            if m:
                event[field] = int(m.group(1))
        return event

    def close(self):
        """Disable the tracepoint, close trace_pipe, remove the instance."""
        if self._pipe_fd is not None:
            try:
                os.close(self._pipe_fd)
            except OSError:
                pass
            self._pipe_fd = None

        if self._instance_dir is not None and os.path.isdir(self._instance_dir):
            # Disable the tracepoint before removing the instance
            if self._tracepoint:
                event_enable = os.path.join(
                    self._instance_dir, 'events', 'damon',
                    self._tracepoint, 'enable')
                try:
                    with open(event_enable, 'w') as f:
                        f.write('0\n')
                except OSError:
                    pass
            try:
                os.rmdir(self._instance_dir)
            except OSError:
                pass
            self._instance_dir = None


class TimeWindowedAverage:
    """Time-based sliding window average.

    Keeps (timestamp, value) pairs; prunes entries older than window_ms on each add.
    """

    def __init__(self, window_ms):
        self._window_ms = window_ms
        self._samples = []  # list of (timestamp, value)
        # When the first sample arrived, kept separately from the samples
        # because pruning removes it.  See is_stable.
        self._started = None

    def add(self, value, now=None):
        if now is None:
            now = time.monotonic()
        if self._started is None:
            self._started = now
        self._samples.append((now, value))
        cutoff = now - self._window_ms / 1000.0
        self._samples = [(t, v) for t, v in self._samples if t >= cutoff]

    def average(self):
        if not self._samples:
            return None
        return sum(v for _, v in self._samples) / len(self._samples)

    def has_data(self):
        return len(self._samples) > 0

    def count(self):
        return len(self._samples)

    def is_stable(self, stable_window_ms, now=None):
        """True once samples have been arriving for at least stable_window_ms.

        Measured from the FIRST sample ever added, not from the oldest one still
        held.  The two differ, and using the oldest held sample makes the answer
        depend on arithmetic it should not: add() prunes anything older than
        window_ms, so the retained span is bounded ABOVE by window_ms and this
        would be asking whether a span capped at window_ms has reached
        stable_window_ms.  The callers pass the same value for both, so the
        question became "is the span exactly at its cap", which is true only when
        a sample happens to land on the boundary.  With a sample period that
        divides the window that is most ticks; with one that does not it is
        never, and the caller waits out its whole budget on every call however
        long the signal has actually been steady.

        Asking when sampling STARTED is the intended question and does not depend
        on the retention window at all.
        """
        if self._started is None:
            return False
        if now is None:
            now = time.monotonic()
        return (now - self._started) * 1000 >= stable_window_ms

    def clear(self):
        self._samples = []
        self._started = None


class WindowedAverage:
    """Sliding window average over the last *maxlen* samples."""

    def __init__(self, maxlen):
        self._maxlen = maxlen
        self._samples = []

    def add(self, value):
        self._samples.append(value)
        if len(self._samples) > self._maxlen:
            self._samples.pop(0)

    def average(self):
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    def full(self):
        return len(self._samples) >= self._maxlen

    def clear(self):
        self._samples = []
