"""Prevent two local CrossMaps pipelines from publishing conflicting maps."""
import fcntl
import os
from pathlib import Path


class PipelineInstance:
    def __init__(self, path):
        self.path=Path(path)
        self.stream=None

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        stream=self.path.open('a+')
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0);owner=stream.read().strip()
            stream.close()
            raise RuntimeError('Another CrossMaps perception process already owns this pipeline '
                               f'(PID {owner or "unknown"}). Stop it before starting a replacement.') from None
        stream.seek(0);stream.truncate();stream.write(str(os.getpid()));stream.flush()
        self.stream=stream
        return self

    def __exit__(self,*_):
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
            self.stream.close();self.stream=None
        # Keep the inode: unlinking a lockfile permits different owners to lock
        # different inodes. A stale PID is harmless; the OS lock is authoritative.


def check_existing_publishers(node, spin_once, seconds=2.):
    """Also detect older builds/remote processes that do not hold our file lock."""
    import time
    end=time.monotonic()+seconds
    topics=('/crossmaps/diagnostics/pipeline_health','/crossmaps/object_graph/descriptors')
    while time.monotonic()<end:
        spin_once(node,timeout_sec=.1)
        if any(node.count_publishers(topic)>0 for topic in topics):
            raise RuntimeError('An existing perception publisher already uses /crossmaps topics. '
                               'Refusing to mix different world maps and segment IDs. '
                               'Stop the previous perception node before starting this one.')
