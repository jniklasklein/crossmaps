"""Bounded recording inventory and explicitly confirmed, scoped cleanup.

No automatic eviction or filesystem-wide search. All targets are direct children
of one caller-supplied recording root, and cleanup tokens name a frozen inventory
of inactive recordings. File locks protect live runs and recorded replays.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import threading
import time


GIB = 1024 ** 3
MARKER = '.crossmaps-run.json'
SCHEMA = 'crossmaps_managed_recording_v1'
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
FRAME_FILE = re.compile(r'frame-[0-9a-f]{20}\.npz(?:\.tmp)?\Z')
DISCARD_DIR = re.compile(r'\.discarded-evidence-[0-9a-f]{16}\Z')
MAX_SCAN_FILES = 20000
STATE_RESERVE_BYTES = 4 * 1024 ** 2


class StorageError(ValueError):
    pass


def _json_file(path, maximum=16 * 1024 ** 2):
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_size > maximum:
        raise StorageError('Invalid recording metadata')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'r') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise StorageError('Recording metadata must be an object')
    return value


def _scan_tree(path):
    """Never follow links; return a footprint and mutation fingerprint."""
    root_stat = path.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise StorageError('Recording must be a real directory, not a symlink')
    total, newest, count = 0, root_stat.st_mtime, 0
    digest = hashlib.sha256()
    files = []
    digest.update(f'{root_stat.st_dev}:{root_stat.st_ino}'.encode())
    first_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened_stat = os.fstat(first_fd)
    if (opened_stat.st_dev, opened_stat.st_ino) != (root_stat.st_dev, root_stat.st_ino):
        os.close(first_fd)
        raise StorageError('Recording changed during inventory')
    pending = [(first_fd, '')]
    try:
        while pending:
            folder_fd, prefix = pending.pop()
            try:
                with os.scandir(folder_fd) as listing:
                    entries = sorted(listing, key=lambda e: e.name)
                for entry in entries:
                    count += 1
                    if count > MAX_SCAN_FILES:
                        raise StorageError('Recording inventory exceeds the safe scan limit')
                    st = entry.stat(follow_symlinks=False)
                    relative = prefix + entry.name
                    digest.update(f'{relative}:{st.st_dev}:{st.st_ino}:{st.st_mode}:{st.st_size}:{st.st_mtime_ns}'.encode())
                    newest = max(newest, st.st_mtime)
                    if stat.S_ISDIR(st.st_mode):
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=folder_fd)
                        pending.append((child, relative+'/'))
                    elif stat.S_ISREG(st.st_mode):
                        total += st.st_size
                        files.append(relative)
                    else:
                        raise StorageError('Symlinks or special files protect this recording from cleanup')
            finally:
                os.close(folder_fd)
    finally:
        for fd, _ in pending:
            os.close(fd)
    return total, newest, digest.hexdigest(), files


def _remove_checked_tree(parent, name, expected_inode):
    """Use fd-relative, symlink-resistant rmtree; never broaden the target."""
    if not shutil.rmtree.avoids_symlink_attacks:
        raise StorageError('This Python cannot safely remove recording trees')
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(st.st_mode) or (st.st_dev, st.st_ino) != expected_inode:
            raise StorageError('Recording changed; refresh the cleanup preview')
        shutil.rmtree(name, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


class StorageManager:
    def __init__(self, root, run_dir, *, readonly=False, max_total_bytes=4 * GIB,
                 min_free_bytes=GIB, max_runs=10, owned_domain=None,
                 domain_lock_held=False, cache_seconds=2., clock=time.monotonic):
        root_path = Path(root).expanduser().absolute()
        if root_path.is_symlink():
            raise StorageError('Recording root cannot be a symlink')
        root_path.mkdir(parents=True, exist_ok=True)
        self.root = root_path.resolve()
        root_stat = self.root.lstat()
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        candidate = Path(run_dir).expanduser().absolute()
        if candidate.is_symlink() or candidate.parent.resolve() != self.root or not RUN_ID.fullmatch(candidate.name):
            raise StorageError('Run must be a direct, non-symlink child of the recording root')
        self.run_dir = self.root / candidate.name
        self.run_id = candidate.name
        self.readonly = bool(readonly)
        for name, value, minimum in [('max_total_bytes', max_total_bytes, 1),
                                     ('min_free_bytes', min_free_bytes, 0), ('max_runs', max_runs, 1)]:
            if type(value) is not int or value < minimum:
                raise StorageError(f'{name} must be an integer >= {minimum}')
        if owned_domain is not None and (type(owned_domain) is not int or not 0 <= owned_domain <= 101):
            raise StorageError('Invalid owned ROS domain')
        if domain_lock_held and owned_domain is None:
            raise StorageError('A held domain lock requires its domain')
        self.max_total_bytes, self.min_free_bytes, self.max_runs = max_total_bytes, min_free_bytes, max_runs
        self.owned_domain = owned_domain if domain_lock_held else None
        self.clock, self.cache_seconds = clock, max(0., float(cache_seconds))
        self._mutex = threading.RLock()
        self._cache = None
        self._cache_at = 0.
        self._plans = {}
        self._closed = False
        self._active_fd = None
        self._root_fd = None
        self._run_identity = None
        self._lock_dir = self.root / '.runlocks'
        if self._lock_dir.is_symlink():
            raise StorageError('Recording lock directory cannot be a symlink')
        self._lock_dir.mkdir(exist_ok=True)
        self._root_fd = os.open(self.root / '.storage.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            with self._locked():
                if self.readonly and not self.run_dir.is_dir():
                    raise StorageError('Replay recording does not exist')
                self.run_dir.mkdir(exist_ok=True)
                _scan_tree(self.run_dir)
                run_stat = self.run_dir.lstat()
                self._run_identity = (run_stat.st_dev, run_stat.st_ino)
                self._active_fd = self._open_run_lock(self.run_id)
                try:
                    fcntl.flock(self._active_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise StorageError('This recording is already active or open for replay') from None
                if not self.readonly:
                    # Deliberate opening of a run claims ownership; this does not
                    # merge it with any other recording or adopt other folders.
                    self._write_marker(reserved_bytes=0)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _locked(self):
        with self._mutex:
            if self._closed:
                raise StorageError('Storage manager is closed')
            root_stat = self.root.lstat()
            if (not stat.S_ISDIR(root_stat.st_mode)
                    or (root_stat.st_dev, root_stat.st_ino) != self._root_identity):
                raise StorageError('Recording root changed; restart the application')
            if self._run_identity is not None:
                run_stat = self.run_dir.lstat()
                if (not stat.S_ISDIR(run_stat.st_mode)
                        or (run_stat.st_dev, run_stat.st_ino) != self._run_identity):
                    raise StorageError('Current recording directory changed; restart the application')
            fcntl.flock(self._root_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._root_fd, fcntl.LOCK_UN)

    def _open_run_lock(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise StorageError('Invalid recording ID')
        if self._lock_dir.is_symlink() or not self._lock_dir.is_dir():
            raise StorageError('Recording lock directory changed')
        return os.open(self._lock_dir / f'{run_id}.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)

    @contextmanager
    def _inactive_lock(self, run_id):
        if run_id == self.run_id:
            raise StorageError('The current recording cannot be deleted')
        fd = self._open_run_lock(run_id)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise StorageError('Recording is active or open for replay') from None
            yield
        finally:
            os.close(fd)

    def _write_marker(self, *, reserved_bytes):
        path = self.run_dir / MARKER
        if path.exists() or path.is_symlink():
            previous = _json_file(path, 65536)
            if previous.get('schema') != SCHEMA or previous.get('run_id') != self.run_id:
                raise StorageError('Unrecognized recording ownership marker')
        else:
            previous = {'schema': SCHEMA, 'run_id': self.run_id, 'created_unix': time.time()}
        previous.update(reserved_bytes=int(reserved_bytes), pid=os.getpid())
        temporary = self.run_dir / (MARKER + '.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(previous, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self._cache = None

    def _legacy_domain(self, path, files):
        # Recognize only the old application's exact recording layout. An
        # unknown user file makes the folder ineligible for recursive cleanup.
        for name in files:
            if name == 'config.json' or name in ('evidence/state.json', 'evidence/state.json.tmp'):
                continue
            if name.startswith('evidence/') and '/' not in name[len('evidence/'):] and FRAME_FILE.fullmatch(name[len('evidence/'):]):
                continue
            if name.startswith('ros_logs/') and (name.endswith('.log') or name.endswith('/latest')):
                continue
            raise StorageError('Unrecognized files: legacy recording is protected')
        config = _json_file(path / 'config.json', 1024 * 1024)
        state = _json_file(path / 'evidence' / 'state.json')
        domain = config.get('central_domain')
        if (type(domain) is not int or not 0 <= domain <= 101
                or not isinstance(config.get('robots'), list)
                or state.get('schema') != 'crossmaps_collaborative_store_v1'
                or not isinstance(state.get('sessions'), dict)):
            raise StorageError('Unrecognized legacy recording metadata')
        return domain

    @contextmanager
    def _legacy_locks(self, domain, acquired_paths=None):
        fds = []
        acquired_paths = set() if acquired_paths is None else acquired_paths
        owned_paths = []
        try:
            paths = list(self.root.parent.glob('collaboration_review_*.lock'))
            if domain != self.owned_domain:
                paths.append(self.root.parent / f'collaboration_domain_{domain}.lock')
            for path in sorted(paths):
                if path in acquired_paths:
                    continue
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                fds.append(fd)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise StorageError('An older coordinator or replay may still use this recording') from None
                acquired_paths.add(path)
                owned_paths.append(path)
            yield
        finally:
            for fd in reversed(fds):
                os.close(fd)
            for path in owned_paths:
                acquired_paths.remove(path)

    def _entry(self, path, *, check_locks=True):
        entry = dict(id=path.name, bytes=0, modified=0., current=path.name == self.run_id,
                     active=path.name == self.run_id, managed=False, legacy=False, deletable=False,
                     protected_reason=None, reserved_bytes=0)
        try:
            size, modified, fingerprint, files = _scan_tree(path)
            entry.update(bytes=size, modified=modified, _fingerprint=fingerprint)
            marker = path / MARKER
            if marker.exists() or marker.is_symlink():
                record = _json_file(marker, 65536)
                if record.get('schema') != SCHEMA or record.get('run_id') != path.name:
                    raise StorageError('Unrecognized recording ownership marker')
                entry['managed'] = True
                entry['reserved_bytes'] = max(0, int(record.get('reserved_bytes', 0)))
                config_path = path / 'config.json'
                if config_path.exists() or config_path.is_symlink():
                    config = _json_file(config_path, 1024 * 1024)
                    domain = config.get('central_domain')
                    if type(domain) is int and 0 <= domain <= 101 and isinstance(config.get('robots'), list):
                        # A recording can later be opened by a pre-manager
                        # application build. Its old domain/replay locks still
                        # protect it, even though that build ignores our marker.
                        entry['_domain'] = domain
                    else:
                        raise StorageError('Unrecognized recording configuration; cleanup is protected')
            else:
                entry['_domain'] = self._legacy_domain(path, files)
                entry['legacy'] = True
            if entry['current']:
                entry['protected_reason'] = 'Current recording; use Rebuild maps instead'
            elif check_locks:
                with self._inactive_lock(path.name):
                    if '_domain' in entry:
                        with self._legacy_locks(entry['_domain']):
                            pass
                entry['deletable'] = True
                entry['reserved_bytes'] = 0
        except (OSError, ValueError, TypeError) as error:
            entry['protected_reason'] = str(error)
            if 'active' in str(error) or 'older coordinator' in str(error):
                entry['active'] = True
            if not entry.get('_fingerprint'):
                entry['unmeasured'] = True
        return entry

    def _inventory(self):
        entries = []
        for path in sorted(self.root.iterdir()):
            if path.name.startswith('.') or not RUN_ID.fullmatch(path.name):
                continue
            if path.is_dir() or path.is_symlink():
                entries.append(self._entry(path))
        total = sum(r['bytes'] for r in entries)
        reserved = sum(r['reserved_bytes'] for r in entries)
        free = shutil.disk_usage(self.root).free
        return dict(current_run_id=self.run_id, current_bytes=next((r['bytes'] for r in entries if r['current']), 0),
                    total_bytes=total, reserved_bytes=reserved, max_total_bytes=self.max_total_bytes,
                    available_bytes=max(0, min(self.max_total_bytes-total-reserved, free-self.min_free_bytes)),
                    disk_free_bytes=free, min_free_bytes=self.min_free_bytes, max_runs=self.max_runs,
                    managed_runs=sum(r['managed'] for r in entries), runs=entries,
                    auto_delete=False, readonly=self.readonly,
                    policy='Bounded evidence; no automatic deletion. Cleanup permanently removes only confirmed inactive recordings.')

    def inventory(self, *, force=False):
        with self._locked():
            if force or self._cache is None or self.clock()-self._cache_at >= self.cache_seconds:
                self._cache = self._inventory()
                self._cache_at = self.clock()
            result = deepcopy(self._cache)
            for run in result['runs']:
                run.pop('_fingerprint', None)
                run.pop('_domain', None)
            return result

    def check_write(self, additional_bytes):
        """Reserve prospective bytes until finish_write() (call in a finally).

        The small metadata reserve covers the following state checkpoint. Active
        reservations are visible to other managers, preventing concurrent runs
        from independently consuming the same final bytes of the quota.
        """
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise StorageError('Write size must be a nonnegative integer')
        with self._locked():
            if self.readonly:
                return False, 'Recorded replay is read-only'
            self._write_marker(reserved_bytes=0)
            report = self._inventory()
            if any(run.get('unmeasured') for run in report['runs']):
                return False, 'Recording storage cannot be measured safely; inspect protected folders'
            if report['managed_runs'] > self.max_runs:
                return False, f'Recording count limit reached ({self.max_runs}); free inactive recordings'
            required = additional_bytes + STATE_RESERVE_BYTES
            if report['total_bytes'] + report['reserved_bytes'] + required > self.max_total_bytes:
                return False, 'Total recording storage limit reached; free inactive recordings or rebuild maps'
            if report['disk_free_bytes'] - report['reserved_bytes'] - required < self.min_free_bytes:
                return False, 'Free disk reserve reached; free storage before retaining more evidence'
            self._write_marker(reserved_bytes=required)
            return True, None

    def finish_write(self):
        with self._locked():
            if not self.readonly:
                self._write_marker(reserved_bytes=0)

    def prepare_cleanup(self, run_ids):
        if self.readonly:
            raise StorageError('Cleanup is disabled in recorded replay')
        if not isinstance(run_ids, list) or not 1 <= len(run_ids) <= 100 or len(set(run_ids)) != len(run_ids):
            raise StorageError('Select between one and 100 distinct recordings')
        with self._locked():
            runs = []
            for run_id in run_ids:
                if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
                    raise StorageError('Invalid recording ID')
                entry = self._entry(self.root / run_id)
                if not entry['deletable']:
                    raise StorageError(entry['protected_reason'] or 'Recording is protected')
                runs.append(entry)
            return self._plan('cleanup', runs)

    def prepare_reset(self):
        if self.readonly:
            raise StorageError('Rebuild is disabled in recorded replay')
        with self._locked():
            path = self.run_dir / 'evidence'
            size, modified, _, _ = _scan_tree(path) if path.exists() or path.is_symlink() else (0, 0, None, [])
            inode = None if not path.exists() else (path.lstat().st_dev, path.lstat().st_ino)
            for old in self.run_dir.iterdir():
                if DISCARD_DIR.fullmatch(old.name):
                    size += _scan_tree(old)[0]
            return self._plan('reset', [dict(id=self.run_id, bytes=size, modified=modified, _inode=inode)])

    def _plan(self, kind, runs):
        now = self.clock()
        self._plans = {k: v for k, v in self._plans.items() if v['expires'] > now}
        # A bounded number of pending dialogs, never an unbounded request cache.
        while len(self._plans) >= 16:
            self._plans.pop(next(iter(self._plans)))
        token = secrets.token_urlsafe(32)
        self._plans[token] = dict(kind=kind, runs=runs, expires=now+300.)
        return dict(token=token, kind=kind, runs=[{k: run[k] for k in ('id', 'bytes')} for run in runs],
                    bytes=sum(run['bytes'] for run in runs), expires_in_s=300, permanent=True)

    def _consume(self, token, kind):
        if not isinstance(token, str):
            raise StorageError('A cleanup preview token is required')
        plan = self._plans.pop(token, None)
        if not plan or plan['kind'] != kind or plan['expires'] <= self.clock():
            raise StorageError('Cleanup preview expired; refresh and confirm again')
        return plan

    def cleanup(self, token):
        if self.readonly:
            raise StorageError('Cleanup is disabled in recorded replay')
        with self._locked(), ExitStack() as stack:
            plan = self._consume(token, 'cleanup')
            checked = []
            acquired_legacy_paths = set()
            # Lock and validate every target before deleting the first one.
            for expected in plan['runs']:
                run_id = expected['id']
                stack.enter_context(self._inactive_lock(run_id))
                path = self.root / run_id
                current = self._entry(path, check_locks=False)
                if '_domain' in current:
                    stack.enter_context(self._legacy_locks(current['_domain'], acquired_legacy_paths))
                if (current['protected_reason'] or current.get('_fingerprint') != expected['_fingerprint']
                        or not (current['managed'] or current['legacy'])):
                    raise StorageError('Recording changed or is protected; refresh the cleanup preview')
                st = path.lstat()
                checked.append((run_id, current['bytes'], (st.st_dev, st.st_ino)))
            removed, freed, error = [], 0, None
            for run_id, size, inode in checked:
                try:
                    _remove_checked_tree(self.root, run_id, inode)
                except (OSError, ValueError) as failure:
                    error = f'Cleanup stopped at {run_id}: {failure}. This target may be partially removed; refresh storage.'
                    break
                removed.append(run_id)
                freed += size
            self._cache = None
            return dict(ok=error is None, removed=removed, freed_bytes=freed, recoverable=False, error=error)

    def reset_current(self, token):
        """Caller must quiesce inference, then replace in-memory maps on success.

        Only the current recording's evidence is removed. Robot SLAM maps,
        config, logs, model weights, and all other runs remain untouched.
        """
        if self.readonly:
            raise StorageError('Rebuild is disabled in recorded replay')
        with self._locked():
            plan = self._consume(token, 'reset')
            path = self.run_dir / 'evidence'
            expected = plan['runs'][0]
            self._write_marker(reserved_bytes=0)
            size = 0
            if path.exists() or path.is_symlink():
                size, _, _, _ = _scan_tree(path)
                st = path.lstat()
                if (st.st_dev, st.st_ino) != expected['_inode']:
                    raise StorageError('Evidence generation changed; refresh the rebuild preview')
                discarded = self.run_dir / ('.discarded-evidence-' + secrets.token_hex(8))
                os.rename(path, discarded)
                try:
                    path.mkdir()
                except BaseException:
                    os.rename(discarded, path)
                    raise
            elif expected['_inode'] is not None:
                raise StorageError('Evidence changed; refresh the rebuild preview')
            else:
                path.mkdir()
            # Logical reset is now committed. Cleanup failure must not tempt
            # callers to revive the previous in-memory graph. Pending trees
            # remain accounted for and the next explicit rebuild retries them.
            removed, freed, warnings = [], 0, []
            try:
                discarded_trees = list(self.run_dir.iterdir())
            except OSError as error:
                warnings.append(f'Old evidence cleanup pending: {error}')
                discarded_trees = []
            for old in discarded_trees:
                if not DISCARD_DIR.fullmatch(old.name):
                    continue
                try:
                    old_size, _, _, _ = _scan_tree(old)
                    st = old.lstat()
                    _remove_checked_tree(self.run_dir, old.name, (st.st_dev, st.st_ino))
                    removed.append(f'{self.run_id}/{old.name}')
                    freed += old_size
                except (OSError, ValueError) as error:
                    warnings.append(f'Old evidence cleanup pending: {error}')
            self._cache = None
            return dict(ok=True, reset_applied=True, removed=removed, freed_bytes=freed,
                        recoverable=False, warning='; '.join(warnings) or None)

    def close(self):
        with self._mutex:
            if self._closed:
                return
            if self._active_fd is not None:
                os.close(self._active_fd)
                self._active_fd = None
            if self._root_fd is not None:
                os.close(self._root_fd)
                self._root_fd = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
