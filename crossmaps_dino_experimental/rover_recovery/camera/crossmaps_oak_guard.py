"""Linux ROS 2 Humble sensor supervisor. Only manages a PRIVATE OAK launch group."""
import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


class FrameProbe:
    """Require two different, advancing acquisition stamps on *both* outputs."""
    def __init__(self):
        self.first = {}
        self.good = set()

    def observe(self, name, stamp):
        if stamp <= 0:
            return
        if name in self.first and stamp > self.first[name]:
            self.good.add(name)
        self.first[name] = stamp

    def complete(self):
        return self.good == {'rgb', 'depth'}


def proc_identity(pid):
    try:
        # stat.comm may contain ')' and spaces. state is field 3, starttime field 22.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] in ('Z', 'X') else fields[19]
    except (OSError, IndexError):
        return None


def camera_containers(name, domain):
    """Fail closed if an original driver or an externally owned twin is live."""
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\x00')
            if Path(os.fsdecode(argv[0])).name not in ('component_container', 'component_container_mt'):
                continue
            if os.fsencode(f'__node:={name}_container') not in argv:
                continue
            env = (entry / 'environ').read_bytes().split(b'\x00')
            domain_var = next((v for v in env if v.startswith(b'ROS_DOMAIN_ID=')), b'ROS_DOMAIN_ID=0')
            if domain_var == os.fsencode(f'ROS_DOMAIN_ID={domain}'):
                found.append(entry.name)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_private_group(child):
    # This PID was returned by our own Popen(start_new_session=True); never use pgrep or signal a parent's group.
    for signum, delay in ((signal.SIGINT, 6), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        child.poll()
        if not group_exists(child.pid):
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signum)
        end = time.monotonic() + delay
        while time.monotonic() < end:
            child.poll()
            if not group_exists(child.pid):
                return
            time.sleep(.1)


class ProbeSubscriptions:
    """Brief local compressed-only probes avoid keeping image encoders active while idle."""
    def __init__(self, node, name, profile, msg_type):
        self.node = node
        self.probe = FrameProbe()
        self.subs = []
        for kind, suffix in [('rgb', 'rgb/image_rect/compressed'), ('depth', 'stereo/image_raw/compressedDepth')]:
            def receive(msg, key=kind):
                stamp = msg.header.stamp
                self.probe.observe(key, stamp.sec * 1_000_000_000 + stamp.nanosec)
            self.subs.append(node.create_subscription(msg_type, f'/{name}/{suffix}', receive, profile))

    def close(self):
        for sub in self.subs:
            self.node.destroy_subscription(sub)
        self.subs.clear()


def supervise(args, domain):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import CompressedImage

    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    owner = proc_identity(args.owner_pid)
    if not owner:
        raise SystemExit('OAK guard: owner already gone, refusing to launch')
    native = Path(__file__).resolve().with_name('crossmaps_oak_native.launch.py')
    cmd = ['ros2', 'launch', str(native), 'name:=' + args.name, 'rectify_rgb:=' + args.rectify_rgb]
    if args.params_file:
        cmd.append('params_file:=' + args.params_file)

    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(args.name + '_recovery_guard', enable_rosout=False, start_parameter_services=False)
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    child = probe = None
    attempts = 0
    retry_at = 0.
    try:
        while not stopping and proc_identity(args.owner_pid) == owner:
            now = time.monotonic()
            if child is None:
                if now < retry_at:
                    time.sleep(.25)
                    continue
                twins = camera_containers(args.name, domain)
                if twins:
                    print(f'OAK guard: camera already owned by PID(s) {twins}; waiting, not stopping them.', flush=True)
                    retry_at = now + 15
                    continue
                child = subprocess.Popen(cmd, start_new_session=True)
                print(f'OAK guard: started private camera launch PID {child.pid}', flush=True)
                started = next_probe = now
                first_healthy = None

            reason = None
            if child.poll() is not None:
                reason = f'camera launch exited ({child.returncode})'
            elif now >= next_probe:
                if probe is None:
                    probe = ProbeSubscriptions(node, args.name, qos, CompressedImage)
                    # Allow camera boot and the driver's own diagnostics reconnect to finish.
                    deadline = now + (75 if first_healthy is None else 35)
                rclpy.spin_once(node, timeout_sec=.2)
                if probe.probe.complete():
                    if first_healthy is None:
                        print('OAK guard: RGB and depth acquisition stamps advancing', flush=True)
                        first_healthy = now
                    if now - first_healthy >= 60:
                        attempts = 0
                    probe.close()
                    probe = None
                    next_probe = now + 12
                elif now >= deadline:
                    missing = sorted({'rgb', 'depth'} - probe.probe.good)
                    reason = f'no advancing {missing} stamps in probe window ({now-started:.0f}s since launch)'
            else:
                time.sleep(.25)
            if reason is not None:
                print(f'OAK guard: {reason}; recycling only the camera', flush=True)
                if probe is not None:
                    probe.close()
                    probe = None
                stop_private_group(child)
                child = None
                pause = min(60, 3 * 2**min(attempts, 5))
                attempts += 1
                retry_at = time.monotonic() + pause
    finally:
        if child is not None:
            stop_private_group(child)
        if probe is not None:
            probe.close()
        node.destroy_node()
        rclpy.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner-pid', type=int, required=True, help='SLAM launch PID; guard exits if this process ends')
    p.add_argument('--name', default='oak')
    p.add_argument('--rectify-rgb', default='False')
    p.add_argument('--params-file')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', args.name):
        p.error('name must be a bare ROS node name')
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    path = f'/tmp/crossmaps-oak-{domain}-{args.name}.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, 'r+'):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit('OAK guard: an owner is already supervising this camera')
        supervise(args, domain)


if __name__ == '__main__':
    main()
