"""Preflight for the laptop's compressed camera relay. No robot configuration writes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes as c
import getpass
import os
import re
import shlex
import shutil
import subprocess
import sys


PASSWORD_ENV = 'CROSSMAPS_RELAY_PASSWORD'  # optional explicit, legacy session-only input
SERVICE = b'org.crossmaps.collaborative.camera-relay.v1'


def safe_environment(environ=None):
    return {k: v for k, v in (os.environ if environ is None else environ).items()
            if k not in (PASSWORD_ENV, 'SSHPASS')}


def ssh_args(host, remote, password=False):
    if not re.fullmatch(r'(?:[A-Za-z_][\w.-]{0,63}@)?[A-Za-z0-9][A-Za-z0-9.-]{0,252}', host, re.ASCII):
        raise ValueError('invalid camera relay SSH host')
    options = ('StrictHostKeyChecking=yes', 'UpdateHostKeys=no', 'ClearAllForwardings=yes',
               'ForwardAgent=no', 'ForwardX11=no', 'ControlMaster=no', 'ControlPath=none',
               'ConnectTimeout=5', 'ConnectionAttempts=1', 'ServerAliveInterval=3',
               'ServerAliveCountMax=2', 'NumberOfPasswordPrompts=1',
               'BatchMode='+('no' if password else 'yes'),
               'PreferredAuthentications='+('password,keyboard-interactive' if password else 'publickey'))
    return ['ssh', '-T', *(v for option in options for v in ('-o',option)), host, remote]


def _password_bytes(password):
    data = password.encode('utf8')
    if not data or len(data) > 1024 or any(v in data for v in (b'\r', b'\n', b'\0')):
        raise ValueError('Camera relay needs a nonempty single-line password (up to 1024 bytes)')
    return data


def launch_ssh(argv, env, password, **kwargs):
    """sshpass -d reads a private inherited fd. Never use argv, temp files or SSHPASS."""
    if not password:
        return subprocess.Popen(argv, env=safe_environment(env), **kwargs)
    secret = _password_bytes(password)
    readfd, writefd = os.pipe()
    try:
        os.write(writefd, secret+b'\n')  # bounded below the pipe's capacity
        return subprocess.Popen(['sshpass', '-d', str(readfd), *argv], env=safe_environment(env),
                                pass_fds=(readfd,), **kwargs)
    finally:
        os.close(writefd)
        os.close(readfd)


class KeychainError(RuntimeError):
    pass


class MacKeychain:
    """Generic passwords in the user's default keychain. Never enable UI/unlock it."""
    def __init__(self):
        self.sec = c.CDLL('/System/Library/Frameworks/Security.framework/Security')
        self.cf = c.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        p, u, b = c.c_void_p, c.c_uint32, c.c_ubyte
        self.cf.CFRelease.argtypes, self.cf.CFRelease.restype = [p], None
        for name, args in {
                'SecKeychainGetUserInteractionAllowed':[c.POINTER(b)],
                'SecKeychainSetUserInteractionAllowed':[b], 'SecKeychainCopyDefault':[c.POINTER(p)],
                'SecKeychainFindGenericPassword':[p,u,p,u,p,c.POINTER(u),c.POINTER(p),c.POINTER(p)],
                'SecKeychainAddGenericPassword':[p,u,p,u,p,u,p,c.POINTER(p)],
                'SecKeychainItemModifyAttributesAndData':[p,p,u,p], 'SecKeychainItemFreeContent':[p,p],
        }.items():
            fn = getattr(self.sec, name)
            fn.argtypes, fn.restype = args, c.c_int32

    @staticmethod
    def _check(status):
        if status:
            raise KeychainError(f'macOS Keychain returned {status}; unlock the login keychain or use SSH keys')

    @contextmanager
    def _open(self):
        # Called serially during startup/enrollment, before ROS/model threads exist.
        was, kc = c.c_ubyte(), c.c_void_p()
        self._check(self.sec.SecKeychainGetUserInteractionAllowed(c.byref(was)))
        self._check(self.sec.SecKeychainSetUserInteractionAllowed(False))
        try:
            self._check(self.sec.SecKeychainCopyDefault(c.byref(kc)))
            yield kc
        finally:
            if kc: self.cf.CFRelease(kc)
            self.sec.SecKeychainSetUserInteractionAllowed(was)

    def get(self, host):
        account, n, ptr = host.encode('ascii'), c.c_uint32(), c.c_void_p()
        with self._open() as kc:
            result = self.sec.SecKeychainFindGenericPassword(kc,len(SERVICE),SERVICE,len(account),account,
                                                             c.byref(n),c.byref(ptr),None)
            if result == -25300:  # errSecItemNotFound
                return None
            self._check(result)
            try:
                if not 0 < n.value <= 1024:
                    raise KeychainError('Camera relay Keychain item has invalid size')
                return c.string_at(ptr,n.value).decode('utf8')
            finally:
                if ptr: self.sec.SecKeychainItemFreeContent(None,ptr)

    def put(self, host, password):
        account, data = host.encode('ascii'), _password_bytes(password)
        with self._open() as kc:
            result = self.sec.SecKeychainAddGenericPassword(kc,len(SERVICE),SERVICE,len(account),account,
                                                            len(data),data,None)
            if result != -25299:  # errSecDuplicateItem
                self._check(result)
                return
            item = c.c_void_p()
            self._check(self.sec.SecKeychainFindGenericPassword(kc,len(SERVICE),SERVICE,len(account),account,
                                                                None,None,c.byref(item)))
            try:
                self._check(self.sec.SecKeychainItemModifyAttributesAndData(item,None,len(data),data))
            finally:
                self.cf.CFRelease(item)


def probe(robot, password):
    """One read-only, bounded SSH/Docker permission check, never start a camera tap."""
    relay = robot['camera_relay']
    argv = ssh_args(relay['ssh_host'],shlex.join(['docker','exec',relay['container'],'true']),bool(password))
    if not shutil.which('ssh'):
        return 'no_ssh'
    if password and not shutil.which('sshpass'):
        return 'no_sshpass'
    try:
        with launch_ssh(argv,os.environ,password,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,start_new_session=True) as proc:
            try:
                _, errors = proc.communicate(timeout=9)
            except subprocess.TimeoutExpired:
                import signal
                try: os.killpg(proc.pid,signal.SIGKILL)  # only our private SSH group
                except ProcessLookupError: pass
                proc.communicate()
                return 'offline'
        if proc.returncode == 0:
            return 'ok'
        msg = errors[-8192:].lower()
        if b'host key verification failed' in msg or b'remote host identification has changed' in msg:
            return 'host_key'
        if b'docker' in msg or b'container' in msg:
            return 'docker'
        if proc.returncode == 5 or b'permission denied' in msg:
            return 'denied'
        return 'offline'
    except OSError:
        return 'offline'


def _batch(robots, passwords, verify):
    with ThreadPoolExecutor(max_workers=min(len(robots),4)) as pool:
        jobs = {r['id']:pool.submit(verify, r, passwords.get(r['id'])) for r in robots}
        return {k:v.result() for k,v in jobs.items()}


def prepare(robots, *, setup=False, force_prompt=False, store=None, interactive=False, session_password=None,
            verify=probe, prompt=getpass.getpass, report=lambda msg: print(msg,file=sys.stderr,flush=True)):
    """Return ephemeral passwords keyed by robot ID; never mutate the recording config."""
    robots = [r for r in robots if r['camera_relay']['mode'] != 'off']
    if not robots:
        if setup or force_prompt: raise ValueError('No relay configured in this profile')
        return {}
    if setup and store is None:
        raise ValueError('Keychain enrollment requires macOS; use an SSH key or a session-only password')
    if (setup or force_prompt) and not interactive:
        raise ValueError('Camera relay enrollment/password prompt needs an interactive terminal')
    passwords, methods, need, unverified = {}, {}, [], set()
    key_status = _batch(robots,{},verify)
    for r in robots:
        rid, host = r['id'], r['camera_relay']['ssh_host']
        status = key_status[rid]
        # Never send a password after a host-key failure, or prompt for Docker / network errors.
        if status == 'offline':
            unverified.add(rid)
        if status == 'ok' and not force_prompt:
            methods[rid] = 'SSH key'
            continue
        if status in ('host_key','docker','no_ssh'):
            raise ValueError(f'{rid}: relay preflight failed ({status}); verify host key, SSH and container access')
        if setup and status == 'offline':
            raise ValueError(f'{rid}: rover unreachable; cannot verify and enroll a password')
        saved = None
        if not (setup or force_prompt or session_password) and store is not None:
            try:
                saved = store.get(host)
            except (KeychainError, UnicodeError) as error:
                report(f'{rid}: saved camera relay credential unavailable ({error})')
        password = session_password or saved
        if password and not setup and not force_prompt:
            passwords[rid] = password
            methods[rid] = 'session password' if session_password else 'macOS Keychain'
        elif status == 'offline' and not force_prompt:
            methods[rid] = 'SSH key'
        else:
            need.append(r)
    # Validate stored credentials on reachable endpoints; offer correction before mapping.
    ready = [r for r in robots if r['id'] in passwords and key_status[r['id']] != 'offline']
    for rid, result in (_batch(ready,passwords,verify) if ready else {}).items():
        if result == 'denied':
            del passwords[rid]
            need.append(next(r for r in ready if r['id']==rid))
        elif result == 'offline':
            unverified.add(rid)
        elif result != 'ok':
            raise ValueError(f'{rid}: relay preflight failed ({result}); check SSH, sshpass and container access')
    if need:
        if not interactive:
            raise ValueError('Camera recovery requires SSH authentication for '+', '.join(r['id'] for r in need)+
                             '; configure SSH keys, run --setup-relay-auth on macOS, or start in an interactive terminal')
        if force_prompt:  # explicitly requested legacy shared-password mode
            hosts = ', '.join(r['camera_relay']['ssh_host'] for r in need)
            password = prompt(f'Shared SSH password for {hosts} (this run only): ')
            _password_bytes(password)
            passwords.update({r['id']:password for r in need})
        else:
            scope = 'save to macOS Keychain' if setup else 'this run only'
            for r in need:
                password = prompt(f'SSH password for {r["camera_relay"]["ssh_host"]} ({scope}): ')
                _password_bytes(password)
                passwords[r['id']] = password
        results = _batch(need,passwords,verify)
        for rid, result in results.items():
            if result == 'offline' and not setup:
                unverified.add(rid)
            elif result != 'ok':
                raise ValueError(f'{rid}: relay preflight failed ({result}); no password saved')
        # Write only after each intended host authenticated successfully.
        for r in need:
            rid, host = r['id'], r['camera_relay']['ssh_host']
            if setup:
                store.put(host,passwords[rid])
            methods[rid] = 'macOS Keychain' if setup else 'session password'
        if not setup and store:
            report('For unattended launches, run --setup-relay-auth once (no robot changes).')
    for r in robots:
        rid = r['id']
        state = 'unverified (rover unreachable)' if rid in unverified else 'ready'
        report(f'{rid} compressed camera relay: {state}, {methods[rid]}.')
    return passwords
