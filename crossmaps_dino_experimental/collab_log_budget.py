"""Bounded recent diagnostic logs in one explicitly owned run directory.

Only direct *.log and ros/ or ros_logs/ descendants are eligible. Evidence,
worlds, assets, other runs, links and non-log files are never touched. Compaction
keeps the recent tail in-place so append-mode child writers keep working. It is
diagnostic retention, not lossless recording: concurrent lines may be dropped
or reordered during copy/truncate. Caps are checked periodically, not per write.
"""
from contextlib import ExitStack
import os
from pathlib import Path
import stat
import time


MIB = 1024 ** 2
NOTICE = b'\n[CrossMaps: older diagnostic log lines compacted; recent tail follows]\n'


class ScopedLogBudget:
    def __init__(self, root, *, max_file_bytes=8*MIB, max_total_bytes=32*MIB,
                 interval_s=1., clock=time.monotonic):
        self.root = Path(root).absolute()
        st = self.root.lstat()
        if not stat.S_ISDIR(st.st_mode):
            raise ValueError('Log scope must be an existing real directory')
        self.identity = st.st_dev, st.st_ino
        if (type(max_file_bytes) is not int or type(max_total_bytes) is not int
                or min(max_file_bytes, max_total_bytes) < 256):
            raise ValueError('Log budgets must be integer byte limits of at least256')
        self.max_file_bytes, self.max_total_bytes = max_file_bytes, max_total_bytes
        self.interval_s, self.clock, self.last = max(0., float(interval_s)), clock, -float('inf')
        self.report = dict(files=0, bytes=0, compactions=0, dropped_bytes=0, scope=str(self.root))

    def _files(self, stack):
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, root_fd)
        st = os.fstat(root_fd)
        if (st.st_dev, st.st_ino) != self.identity:
            raise RuntimeError('Owned log directory changed; refusing log compaction')
        records, count = [], 0

        def walk(folder_fd, prefix='', depth=0):
            nonlocal count
            if depth > 8:
                raise RuntimeError('Owned log directory nesting exceeds safe limit')
            with os.scandir(folder_fd) as entries:
                names = sorted(entry.name for entry in entries)
            for name in names:
                count += 1
                if count > 4096:
                    raise RuntimeError('Owned log inventory exceeds safe limit')
                try:
                    before = os.stat(name, dir_fd=folder_fd, follow_symlinks=False)
                    if stat.S_ISDIR(before.st_mode) and (depth or name in ('ros', 'ros_logs')):
                        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=folder_fd)
                        try:
                            walk(child, prefix+name+'/', depth+1)
                        finally:
                            os.close(child)
                    elif name.endswith('.log') and stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                        if len(records) >= 128:
                            raise RuntimeError('Too many owned log files to compact safely')
                        fd = os.open(name, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW, dir_fd=folder_fd)
                        stack.callback(os.close, fd)
                        after = os.fstat(fd)
                        if (not stat.S_ISREG(after.st_mode) or after.st_nlink != 1
                                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)):
                            continue
                        records.append((fd, prefix+name))
                except FileNotFoundError:
                    continue  # A child can finish/remove a transient log.
        walk(root_fd)
        return records

    @staticmethod
    def _compact(fd, target):
        before = os.fstat(fd).st_size
        if before <= target:
            return 0
        target = max(0, target)
        notice = NOTICE if target >= len(NOTICE) else b''
        keep = max(0, target-len(notice))
        tail = os.pread(fd, keep, max(0, before-keep)) if keep else b''
        os.ftruncate(fd, 0)
        data = memoryview(notice+tail)
        while data:
            data = data[os.write(fd, data):]
        return max(0, before-os.fstat(fd).st_size)

    def tick(self, *, force=False):
        now = self.clock()
        if not force and now-self.last < self.interval_s:
            return dict(self.report)
        self.last = now
        with ExitStack() as stack:
            records = self._files(stack)
            compactions, dropped = 0, 0
            for fd, _ in records:
                if os.fstat(fd).st_size > self.max_file_bytes:
                    dropped += self._compact(fd, self.max_file_bytes//2)
                    compactions += 1
            total = sum(os.fstat(fd).st_size for fd, _ in records)
            if total > self.max_total_bytes:
                for fd, _ in sorted(records, key=lambda row: os.fstat(row[0]).st_size, reverse=True):
                    size = os.fstat(fd).st_size
                    target = max(0, size-(total-self.max_total_bytes))
                    if target < size:
                        removed = self._compact(fd, target)
                        dropped += removed
                        total -= removed
                        compactions += 1
                    if total <= self.max_total_bytes:
                        break
            self.report.update(files=len(records), bytes=sum(os.fstat(fd).st_size for fd, _ in records),
                               compactions=self.report['compactions']+compactions,
                               dropped_bytes=self.report['dropped_bytes']+dropped)
        return dict(self.report)
