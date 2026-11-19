"""One-time handoff for an *existing* ROS launch. Normal future launches need no bridge.

Runs inside the same PID namespace as RTAB-Map. Only the explicitly identified
sensor is stopped. Its replacement is tied to this particular mapping session.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        tail = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if tail[0] == 'Z' else (int(tail[1]), tail[19])  # parent PID, start ticks
    except (FileNotFoundError, ProcessLookupError):
        return None


def argv(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner', required=True, type=int)
    p.add_argument('--old-sensor', required=True, type=int)
    p.add_argument('--domain', required=True, choices=['0', '1'])
    p.add_argument('--sensor', required=True, choices=['lidar', 'camera'])
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    old, parent = identity(a.old_sensor), identity(a.owner)
    old_args, owner_args = argv(a.old_sensor), argv(a.owner)
    owner_env = dict(os.fsdecode(e).split('=', 1) for e in
                     Path(f'/proc/{a.owner}/environ').read_bytes().split(b'\0') if e)
    if (not old or not parent or old[0] != a.owner or
        not any(owner_args[i:i+3] == ['launch', 'ugv_slam', 'rtabmap_rgbd.launch.py']
                for i in range(len(owner_args))) or owner_env.get('ROS_DOMAIN_ID','0') != a.domain):
        p.error('Owner, domain or process ancestry changed; inspect again.')
    if a.sensor == 'lidar':
        executable = '/home/ws/ugv_ws/install/ldlidar/lib/ldlidar/ldlidar_node'
        ros_name = '__node:=LD19'
        package, launch = 'ldlidar', 'ld19.launch.py'
    else:
        executable = '/opt/ros/humble/lib/rclcpp_components/component_container'
        ros_name = '__node:=oak_container'
        package, launch = 'ugv_vision', 'oak_d_lite.launch.py'
    if old_args[0] != executable or ros_name not in old_args:
        p.error('Expected sensor command line does not match; refusing to signal.')
    target = ['/opt/ros/humble/bin/ros2','launch',package,launch]
    print(f'Validated domain {a.domain}, SLAM launch {a.owner}, {a.sensor} {a.old_sensor}',flush=True)
    if not a.apply:
        return
    stop = False
    def shutdown(sig, frame):
        nonlocal stop
        stop = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, shutdown)
    with open(f'/run/crossmaps-handoff-{a.domain}-{a.sensor}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # pidfd pins the identity across exit and PID reuse, including during escalation.
        pidfd = os.pidfd_open(a.old_sensor)
        try:
            for sig, grace in [(signal.SIGINT, 9), (signal.SIGTERM, 5), (signal.SIGKILL, 3)]:
                if identity(a.old_sensor) != old or identity(a.owner) != parent or stop:
                    break
                signal.pidfd_send_signal(pidfd, sig)
                end = time.monotonic()+grace
                while time.monotonic()<end and identity(a.old_sensor)==old and not stop:
                    time.sleep(.2)
        finally:
            os.close(pidfd)
        if stop or identity(a.owner) != parent:
            return
        if identity(a.old_sensor)==old:
            raise RuntimeError('Old sensor still present; refused to create a duplicate.')
        print('Starting replacement:', target, flush=True)
        proc = subprocess.Popen(target, env=owner_env, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while proc.poll() is None and not stop and identity(a.owner) == parent:
                time.sleep(.25)
        finally:
            if proc.poll() is None:
                for sig, grace in [(signal.SIGINT,15),(signal.SIGTERM,8),(signal.SIGKILL,3)]:
                    if proc.poll() is not None:
                        break
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        proc.wait(timeout=grace)
                    except subprocess.TimeoutExpired:
                        pass
        if identity(a.owner) == parent and not stop:
            raise SystemExit(proc.returncode or 1)


if __name__ == '__main__':
    main()
