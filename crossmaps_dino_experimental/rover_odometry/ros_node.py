"""ROS-facing adapter for the Jetson's headerless encoders and *physical* gyro."""
import argparse
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time


def proc_start(pid):
    try:
        f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if f[0] in ('Z', 'X') else f[19]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def planar_pose(msg):
    if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_footprint':
        raise ValueError('unexpected seed frame IDs')
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    if not all(map(math.isfinite, (p.x, p.y, p.z, q.x, q.y, q.z, q.w))):
        raise ValueError('nonfinite seed')
    if abs(p.z) > .01 or abs(q.x) > .005 or abs(q.y) > .005 or abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1) > .002:
        raise ValueError('invalid or nonplanar seed')
    return (p.x, p.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))


def parked(history, now):
    recent = [row for row in history if 0 <= now-row[0] <= 1.5]
    return (len(recent) >= 3 and now-recent[-1][0] < .4 and recent[-1][0]-recent[0][0] >= .8
            and all(max(r[i] for r in recent)-min(r[i] for r in recent) <= .001 for i in (1, 2)))


def verify_legacy(pid, owner_pid, domain):
    root = Path(f'/proc/{pid}')
    args = (root/'cmdline').read_bytes().split(b'\x00')
    exe = os.fsdecode(args[0])
    if not exe.endswith('/install/ugv_base_node/lib/ugv_base_node/base_node'):
        raise ValueError('not the stock ugv odometry executable: '+exe)
    if int((root/'stat').read_text().rsplit(')',1)[1].split()[1]) != owner_pid:
        raise ValueError('legacy node is not a direct child of the nominated SLAM launch')
    env = (root/'environ').read_bytes().split(b'\x00')
    d = next((x.split(b'=',1)[1] for x in env if x.startswith(b'ROS_DOMAIN_ID=')), b'0')
    if int(d) != domain:
        raise ValueError('legacy odometry domain differs')
    return proc_start(pid)


def run(args):
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float32MultiArray, String
    from tf2_msgs.msg import TFMessage
    from tf2_ros import TransformBroadcaster
    try:
        from .odometry_core import Config, Estimator
    except ImportError:
        from odometry_core import Config, Estimator

    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    # No duplicate supervisor even in the short DDS discovery window.
    suffix = 'shadow' if args.mode == 'shadow' else 'live'
    lock_file = os.fdopen(os.open(f'/tmp/crossmaps-odom-{domain}-{suffix}.lock',
                                  os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600), 'r+')
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner = proc_start(args.owner_pid)
    if owner is None:
        raise ValueError('owner does not exist')
    legacy = verify_legacy(args.legacy_pid, args.owner_pid, domain) if args.mode == 'handoff' else None
    if args.mode == 'handoff' and not legacy:
        raise ValueError('legacy process already gone: cannot seed a handoff')
    cfg = Config(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    prefix = '/crossmaps/odometry_shadow' if args.mode == 'shadow' else '/crossmaps/odometry'
    rclpy.init(args=[])
    node = rclpy.create_node('crossmaps_odometry_shadow' if args.mode == 'shadow' else 'crossmaps_odometry')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    status_pub = node.create_publisher(String, prefix+'/status', 2)
    created = time.monotonic()
    state = 'waiting'
    error = ''
    estimator = Estimator(config=cfg, initial_pose=(0.,0.,0.))
    wheel_history = deque(maxlen=256)
    gyro_history = deque(maxlen=300)
    seed = None
    old_gid = None
    last_stamp_ns = 0
    last_gyro_stamp_ns = 0
    wrong_frame = late_gyro = bad_wheel = published = rejected_stamp = frozen_frames = 0
    imu_payload = None
    imu_unchanged_since = None
    wheel_held = None
    wheel_unchanged_since = None
    last_output = None
    last_source = None
    ready_logged = False
    pub = tf = None
    legacy_sub = None
    fail_reason = ''
    sent_tf_stamps = deque(maxlen=256)
    activation_ns = 0
    saw_existing_tf = False

    def fail(msg):
        nonlocal state, fail_reason
        state, fail_reason = 'failed', msg
        node.get_logger().error(msg)

    def now_ns():
        return node.get_clock().now().nanoseconds

    def activate(pose, seed_stamp=0):
        nonlocal estimator, pub, tf, state, last_stamp_ns, legacy_sub, activation_ns
        if legacy_sub is not None:
            node.destroy_subscription(legacy_sub)
            legacy_sub = None
        estimator = Estimator(config=cfg, initial_pose=pose)
        for when, wz, stamp in gyro_history:
            estimator.gyro(when, wz, acquisition_s=stamp)
        if wheel_history:
            estimator.wheel(*wheel_history[-1])
        last_stamp_ns = seed_stamp
        pub = node.create_publisher(Odometry, prefix+'/odom' if args.mode=='shadow' else '/odom', 5)
        if args.mode != 'shadow' and args.publish_tf:
            tf = TransformBroadcaster(node)
        activation_ns = now_ns()
        state = 'active'
        print(f'ODOMETRY_ACTIVE mode={args.mode} seed={pose} TF={tf is not None}', flush=True)

    def gyro(msg):
        nonlocal last_gyro_stamp_ns, wrong_frame, late_gyro, imu_payload, imu_unchanged_since
        if msg.header.frame_id != 'base_imu_link':
            wrong_frame += 1
            return
        ns = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        age = (now_ns()-ns)*1e-9
        wz = float(msg.angular_velocity.z)
        if ns <= last_gyro_stamp_ns or not -.15 <= age <= .5 or not math.isfinite(wz) or msg.angular_velocity_covariance[0] < 0:
            late_gyro += 1
            return
        last_gyro_stamp_ns = ns
        now = time.monotonic()
        signature = tuple(float(getattr(vec, axis)) for vec in
                          (msg.linear_acceleration, msg.angular_velocity) for axis in ('x','y','z'))
        if not all(map(math.isfinite, signature)):
            late_gyro += 1
            return
        if signature != imu_payload:
            imu_payload, imu_unchanged_since = signature, now
        # A vendor JSON parse failure can stamp and resend the entire previous
        # six-axis sample. A genuinely constant yaw reading *alone* means nothing.
        if now-imu_unchanged_since >= 3.:
            return
        data = (now, wz, ns*1.e-9)
        gyro_history.append(data)
        estimator.gyro(*data[:2], acquisition_s=data[2])

    def wheels(msg):
        nonlocal published, last_output, last_source, last_stamp_ns, rejected_stamp, bad_wheel
        nonlocal wheel_held, wheel_unchanged_since, frozen_frames
        if len(msg.data) < 2 or not all(math.isfinite(v) for v in msg.data[:2]):
            bad_wheel += 1
            return
        now = time.monotonic()
        data = (now, float(msg.data[0]), float(msg.data[1]))
        if data[1:] != wheel_held:
            wheel_held, wheel_unchanged_since = data[1:], now
        wheel_history.append(data)
        if state != 'active':
            return
        # No upstream hardware stamp exists for this message: receipt on the rover
        # is the only honest timestamp. Never restamp a cached pose from a timer.
        ns = now_ns()
        s = estimator.wheel(*data)
        if (imu_unchanged_since is not None and now-imu_unchanged_since >= 3.
                and now-wheel_unchanged_since >= 3.):
            frozen_frames += 1
            return
        if ns <= last_stamp_ns:
            rejected_stamp += 1
            return
        if s is None:
            return
        m = Odometry()
        m.header.stamp.sec, m.header.stamp.nanosec = divmod(ns, 1_000_000_000)
        m.header.frame_id, m.child_frame_id = 'odom', 'base_footprint'
        p = m.pose.pose
        p.position.x, p.position.y = s.x, s.y
        p.orientation.z, p.orientation.w = math.sin(s.yaw/2), math.cos(s.yaw/2)
        m.twist.twist.linear.x, m.twist.twist.angular.z = s.v, s.w
        m.pose.covariance, m.twist.covariance = list(s.pose_cov), list(s.twist_cov)
        pub.publish(m)
        if tf:
            t = TransformStamped()
            t.header, t.child_frame_id = m.header, m.child_frame_id
            t.transform.translation.x, t.transform.translation.y = s.x, s.y
            t.transform.rotation = p.orientation
            sent_tf_stamps.append(ns)
            tf.sendTransform(t)
        last_stamp_ns, last_output, last_source = ns, (s.x,s.y,s.yaw), s.source
        published += 1
        if args.mode=='handoff' and published==1:
            print(f'HANDOFF_ACTIVE stamp_ns={ns} pose={last_output} tf={tf is not None}', flush=True)

    def receive_legacy(msg):
        nonlocal seed, error
        now = time.monotonic()
        if old_gid is None or not wheel_history or proc_start(args.legacy_pid) != legacy:
            return
        try:
            p = planar_pose(msg)
            ns = msg.header.stamp.sec*1_000_000_000 + msg.header.stamp.nanosec
            if ns <= 0 or not -.15 <= (now_ns()-ns)*1e-9 <= .6 or (seed and ns <= seed[1]):
                return
            if now-wheel_history[-1][0] > .4:
                return
            seed = (now, ns, p, wheel_history[-1][1:])
        except ValueError as ex:
            error = str(ex)

    def publishers():
        return [(p.node_namespace,p.node_name,bytes(p.endpoint_gid)) for p in node.get_publishers_info_by_topic('/odom')]

    def inspect_tf(msg):
        nonlocal saw_existing_tf
        for t in msg.transforms:
            if (t.header.frame_id,t.child_frame_id) != ('odom','base_footprint'):
                continue
            if state=='waiting' and args.mode=='live':
                saw_existing_tf=True
            ns=t.header.stamp.sec*1_000_000_000+t.header.stamp.nanosec
            if state=='active' and ns>activation_ns and ns not in sent_tf_stamps:
                fail('Foreign odom->base_footprint TF detected; withdrawing replacement')
                return

    def guard():
        nonlocal old_gid, ready_logged
        now = time.monotonic()
        if proc_start(args.owner_pid) != owner:
            fail('Owner ended; stopping odometry')
            return
        if state == 'failed' or args.mode == 'shadow':
            return
        peers = publishers()
        if state == 'active':
            # A gracefully exiting certified legacy process may remain briefly in the
            # DDS graph. Its PID/start time was already verified dead before handoff.
            other = [p for p in peers if not (old_gid is not None and p[2]==old_gid)]
            if len(other)>1 or any(p[:2] != ('/',node.get_name()) for p in other):
                fail('Another /odom publisher detected; withdrawing replacement')
            return
        if args.mode == 'live':
            if peers or saw_existing_tf:
                fail('Existing /odom publisher or odom->base TF: refusing fresh-origin startup')
            elif now-created >= 2:
                activate((0.,0.,0.))
            return
        if now-created > args.handoff_timeout:
            fail('Handoff timed out without a safe, recent seed')
            return
        old_alive = proc_start(args.legacy_pid) == legacy
        if old_alive:
            if len(peers)>1 or any(p[:2]!=('/','base_node') for p in peers):
                fail('Cannot identify a unique vendor /odom publisher')
                return
            if peers:
                if old_gid is not None and peers[0][2] != old_gid:
                    fail('Vendor publisher identity changed')
                    return
                old_gid = peers[0][2]
            ready = bool(seed and now-seed[0]<.5 and parked(wheel_history,now)
                         and gyro_history and now-gyro_history[-1][0]<.5 and peers)
            if ready and not ready_logged:
                print('HANDOFF_READY legacy_pid=%d seed=%s' % (args.legacy_pid, seed[2]), flush=True)
            ready_logged = ready
            return
        if any(p[2]!=old_gid for p in peers):
            fail('Unknown /odom publisher after vendor exit')
            return
        if not seed or not ready_logged or now-seed[0]>4 or not 0 <= (now_ns()-seed[1])*1e-9 <= 4:
            fail('Missing/expired handoff seed; refusing to jump to origin')
            return
        if not parked(wheel_history,now) or max(abs(a-b) for a,b in zip(wheel_history[-1][1:], seed[3]))>.001:
            fail('Rover moved during seed handoff; refusing discontinuity')
            return
        activate(seed[2],seed[1])

    def status():
        now = time.monotonic()
        payload = dict(estimator.health(now))
        payload.update(mode=args.mode, state=state, adapter_source=last_source, published=published,
                       last_pose=last_output, last_stamp_ns=last_stamp_ns, frame_id='odom',
                       child_frame_id='base_footprint', wrong_imu_frame=wrong_frame,
                       discarded_gyro=late_gyro, malformed_wheel=bad_wheel, clock_drops=rejected_stamp,
                       frozen_feedback_drops=frozen_frames,
                       identical_feedback_s=min(now-imu_unchanged_since,now-wheel_unchanged_since)
                       if imu_unchanged_since is not None and wheel_unchanged_since is not None else None,
                       adapter_wheel_age_s=(now-wheel_history[-1][0]) if wheel_history else None,
                       adapter_gyro_age_s=(now-gyro_history[-1][0]) if gyro_history else None,
                       seed_ready=ready_logged, last_error=fail_reason or error)
        if payload['identical_feedback_s'] is not None and payload['identical_feedback_s'] >= 3.:
            payload['status'] = 'frozen_encoder_and_imu_feedback'
        elif payload['adapter_wheel_age_s'] is not None and payload['adapter_wheel_age_s'] > cfg.wheel_gap:
            payload['status'] = 'stale_encoders'
        m = String()
        m.data = json.dumps(payload, allow_nan=False)
        status_pub.publish(m)
    sub_w = node.create_subscription(Float32MultiArray, '/odom/odom_raw', wheels, qos)
    sub_g = node.create_subscription(Imu, '/imu/data_raw', gyro, qos)
    sub_tf = node.create_subscription(TFMessage, '/tf', inspect_tf, qos) if args.mode!='shadow' and args.publish_tf else None
    if args.mode == 'handoff':
        legacy_sub = node.create_subscription(Odometry, '/odom', receive_legacy, qos)
    elif args.mode == 'shadow':
        activate((0.,0.,0.))
    timer_g = node.create_timer(.2,guard)
    timer_s = node.create_timer(1.,status)
    try:
        while rclpy.ok() and state != 'failed':
            rclpy.spin_once(node, timeout_sec=.25)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        lock_file.close()
    return 3 if state=='failed' else 0


def boolean(value):
    if value.lower() not in ('true','false','1','0'):
        raise argparse.ArgumentTypeError('expected true or false')
    return value.lower() in ('true','1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['shadow','live','handoff'], required=True)
    p.add_argument('--owner-pid', type=int, required=True)
    p.add_argument('--legacy-pid', '--incumbent-pid', type=int)
    p.add_argument('--publish-tf', type=boolean, default=True)
    p.add_argument('--handoff-timeout', type=float, default=120)
    p.add_argument('--config', help='optional JSON Config field overrides')
    a=p.parse_args()
    if a.mode=='handoff' and not a.legacy_pid:
        p.error('handoff requires --legacy-pid (still running)')
    return run(a)


if __name__ == '__main__':
    sys.exit(main())
