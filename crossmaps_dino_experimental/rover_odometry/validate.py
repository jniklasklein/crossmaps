#!/usr/bin/env python3
"""Read-only rover-side comparison or deterministic synthetic motion replay.

Examples (source ROS on the rover for live only):
  python3 validate.py live --seconds 30 --parked
  python3 validate.py live --seconds 45 --turn-deg 90   # independent measured CCW turn
  python3 validate.py replay

No TF, motor, parameter or other production publications. Reference values must
come from physical marks/measurements, not RF2O or the candidate estimator.
"""
import argparse
import json
import math
import random
import statistics
import sys
import time


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def integrate(x, y, h, d, a):
    # Exact circular arc; synthetic ground truth never calls estimator code.
    step = d if abs(a) < 1e-12 else d * math.sin(a / 2) / (a / 2)
    return x + step * math.cos(h + a / 2), y + step * math.sin(h + a / 2), h + a


def synthetic_replay():
    try:
        from .odometry_core import Config, Estimator
    except ImportError:  # script run directly on the Jetson
        from odometry_core import Config, Estimator
    rng = random.Random(0xC055)
    est = Estimator(Config())
    pose = (0., 0., 0.)
    left, right = 2., 3.   # existing cumulative MCU counters, not an initial displacement
    previous = None
    elapsed = 0.
    phases = []
    track, step = .175, .05
    def exercise(name, seconds, v, w, *, with_gyro=True, reset=False):
        nonlocal left, right, previous, pose, elapsed
        actual_start = None if previous is None else (previous.x, previous.y, previous.yaw)
        truth_start = pose
        max_park_deg = 0.
        if reset:
            left, right = 0., 0.
        for _ in range(round(seconds / step)):
            elapsed += step
            pose = integrate(*pose, v*step, w*step)
            left += (v-w*track/2)*step
            right += (v+w*track/2)*step
            # Deterministic gravity-aligned raw gyro in rad/s with measured
            # rover B-like DC bias and noise. Encoders rounded to centimeters.
            if with_gyro:
                est.gyro(elapsed, w-.0095+rng.gauss(0, .003), acquisition_s=100.+elapsed)
            sample = est.wheel(elapsed, round(left, 2), round(right, 2))
            if sample is None:
                continue
            if not all(map(math.isfinite, (sample.x,sample.y,sample.yaw,sample.v,sample.w))):
                raise ValueError("Nonfinite estimate during " + name)
            previous = sample
            if actual_start is None:
                actual_start = (sample.x, sample.y, sample.yaw)
            if v == 0. and w == 0.:
                max_park_deg = max(max_park_deg, abs(math.degrees(wrap(sample.yaw-actual_start[2]))))
        if previous is None or actual_start is None:
            raise RuntimeError("Estimator never reported in " + name)
        observed_turn = math.degrees(wrap(previous.yaw-actual_start[2]))
        true_turn = math.degrees(pose[2]-truth_start[2])
        displacement = math.hypot(previous.x-actual_start[0], previous.y-actual_start[1])
        wanted = math.hypot(pose[0]-truth_start[0], pose[1]-truth_start[1])
        turn_error = abs(math.degrees(wrap(math.radians(observed_turn-true_turn))))
        parked = v == w == 0
        passed = turn_error < (1.2 if parked else 6.) and abs(displacement-wanted) < (.022 if parked else .05)
        if parked:
            passed &= max_park_deg < 1.5
        phases.append(dict(phase=name, true_turn_deg=round(true_turn,2), observed_turn_deg=round(observed_turn,2),
                           truth_displacement_m=round(wanted,3), displacement_m=round(displacement,3),
                           max_park_excursion_deg=round(max_park_deg,2) if parked else None,
                           pass_test=bool(passed)))
    exercise("parked_with_gyro_bias", 8, 0, 0)
    exercise("forward", 3, .2, 0)
    exercise("left_quarter_turn", 4, 0, math.pi/8)
    exercise("park_after_turn", 3, 0, 0)
    exercise("reverse", 2, -.15, 0)
    exercise("right_arc", 3, .18, -math.pi/6)
    exercise("turn_with_missing_gyro", 3, 0, math.pi/6, with_gyro=False)
    exercise("parked_after_mcu_reset", 4, 0, 0, reset=True)
    # Wheel samples disappear completely across an unobserved interval; don't
    # integrate unknown travel or reuse an old angular speed when they return.
    elapsed += 4.
    left, right = left+1., right+1.
    exercise("wheel_outage_and_rebase", 3, 0, 0)
    err = math.hypot(previous.x-pose[0], previous.y-pose[1])
    yaw_error = abs(math.degrees(wrap(previous.yaw-pose[2])))
    hazards = stress_replay(Estimator, Config)
    return dict(test="synthetic_motion", seed="0xC055", phases=phases, hazards=hazards, endpoint_error_m=round(err, 4),
                endpoint_heading_error_deg=round(yaw_error, 3),
                pass_test=err < .08 and yaw_error < 6 and all(p["pass_test"] for p in phases+hazards))


def covariance_ok(values):
    """Cholesky tests complete ROS 6x6 covariance, including orientation axes."""
    if len(values)!=36 or not all(math.isfinite(v) for v in values):
        return False
    L = [[0.]*6 for _ in range(6)]
    for i in range(6):
        for j in range(i+1):
            if not math.isclose(values[6*i+j],values[6*j+i],rel_tol=1e-9,abs_tol=1e-12):
                return False
            remainder = values[6*i+j]-sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                if remainder <= 0:
                    return False
                L[i][j] = math.sqrt(remainder)
            else:
                L[i][j] = remainder/L[j][j]
    return True


def stress_replay(estimator_type, config_type):
    """Use cm-quantized wheel truth; do not substitute RF2O/old odom for truth."""
    results=[]
    for name in ('sparse_turn','gyro_sign_flip','alternating_turns_wrong_gyro',
                 'intermittent_gyro','repeated_sensor_gaps','stuck_gyro_at_rest'):
        e=estimator_type(config_type())
        t=l=r=true_yaw=0.
        max_error=0.
        withheld=late_outputs=produced=0
        jump=0.
        last_report_t=12.
        max_report_gap=0.
        last_pose=None
        gap_pose=None
        cov_valid=True
        checked_gap_growth=True
        for i in range(650):
            t+=.05
            after=t-12
            # Park 5s and turn one direction for 7s before inducing faults:
            # checks failure *after* axis confidence, not only cold start.
            if t<=5: w=0.
            elif t<=12: w=.49
            elif name=='alternating_turns_wrong_gyro': w=(1 if int(after/.3)%2==0 else -1)*.49
            elif name in ('stuck_gyro_at_rest','repeated_sensor_gaps'): w=0.
            elif name=='sparse_turn': w=.08  # >1s between 1cm encoder ticks
            else: w=(1 if int(after/2)%2==0 else -1)*.49
            gyro=(-w if name in ('gyro_sign_flip','alternating_turns_wrong_gyro') and after>0 else w)
            if name=='stuck_gyro_at_rest' and after>2:
                gyro=.2   # If this is a lifted rover the truth is unobservable.
            in_gap=name=='repeated_sensor_gaps' and after>0 and 1<after%6<2.4
            if in_gap:
                if gap_pose is None:
                    gap_pose=last_pose
                    # Unknown motion during dropout. Neither trusted wheel delta
                    # nor heading is recoverable from this fresh baseline.
                    l+=.2; r-=.1
                continue
            if not (name=='intermittent_gyro' and after>0 and int(after/.29)%3):
                e.gyro(t,gyro-.0095,acquisition_s=100+t)
            l-=w*.175*.05/2; r+=w*.175*.05/2
            true_yaw+=w*.05
            s=e.wheel(t,round(l,2),round(r,2))
            if s is None:
                if after>0: withheld+=1
                continue
            if after>0:
                produced+=1
                max_report_gap=max(max_report_gap,t-last_report_t)
                last_report_t=t
            cov_valid &= covariance_ok(s.pose_cov) and covariance_ok(s.twist_cov)
            cov_valid &= all(v>1e-8 for v in (s.pose_cov[0],s.pose_cov[7],s.pose_cov[35]))
            if gap_pose is not None:
                jump=max(jump,abs(math.degrees(wrap(s.yaw-gap_pose.yaw))))
                checked_gap_growth &= s.pose_cov[35] > gap_pose.pose_cov[35]
                gap_pose=None
            if after>0:
                max_error=max(max_error,abs(math.degrees(wrap(s.yaw-true_yaw))))
                if name=='stuck_gyro_at_rest' and after>4:
                    late_outputs+=1
            last_pose=s
        # Half-tick uncertainty on each side permits ~3.3° vs smooth truth.
        # Long unexplained gyro-only motion must neither accumulate yaw nor
        # produce an uninterrupted "healthy stationary" transform.
        passed=bool(cov_valid and max_error<6.1 and jump<1. and checked_gap_growth)
        if name=='stuck_gyro_at_rest':
            passed &= withheld>20 and late_outputs==0
        elif name=='sparse_turn':
            # Weak wheel evidence while gyro says turning isn't a reason to
            # deprive mapping of every observation in an otherwise sensed turn.
            passed &= produced>300 and max(max_report_gap,t-last_report_t)<.55
        if name=='repeated_sensor_gaps':
            passed &= withheld>=3
        results.append(dict(phase=name, max_heading_error_deg=round(max_error,3), outputs=produced,
                            withheld=withheld, outputs_during_sustained_unobserved_turn=late_outputs,
                            max_output_gap_s=round(max(max_report_gap,t-last_report_t),3),
                            max_reacquisition_jump_deg=round(jump,3),covariance_valid=bool(cov_valid),
                            gap_covariance_grew=bool(checked_gap_growth),pass_test=bool(passed)))
    return results


def pose_summary(rows):
    if not rows:
        return {}
    xy0 = rows[0][2:4]
    h0 = prev = rows[0][4]
    h = 0.
    turns = []
    radii = []
    for _, _, x, y, heading in rows:
        h += wrap(heading-prev)
        prev = heading
        turns.append(math.degrees(h))
        radii.append(math.hypot(x-xy0[0], y-xy0[1]))
    dx, dy = rows[-1][2]-xy0[0], rows[-1][3]-xy0[1]
    return dict(yaw_change_deg=turns[-1], max_heading_excursion_deg=max(map(abs,turns)),
                forward_displacement_m=math.cos(h0)*dx+math.sin(h0)*dy,
                lateral_displacement_m=-math.sin(h0)*dx+math.cos(h0)*dy,
                max_displacement_m=max(radii))


def continuity(rows, seconds, stamped=True):
    if not rows:
        return dict(n=0, fresh=False)
    times = [row[0] for row in rows]
    max_gap = max([times[0], seconds-times[-1]] + [v-u for u,v in zip(times,times[1:])])
    forward = sum(b[1]>a[1] for a,b in zip(rows,rows[1:])) if stamped else None
    ok = len(rows)>=5 and max_gap<1.2 and (forward is None or forward>=.90*(len(rows)-1))
    return dict(n=len(rows), max_gap_s=round(max_gap,3), advancing_intervals=forward, fresh=ok)


def live(args):
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float32MultiArray
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init()
    node = rclpy.create_node("crossmaps_odometry_validation", enable_rosout=False, start_parameter_services=False)
    capture = {k:[] for k in ("old","candidate","gyro","wheel")}
    topics = {"old": args.baseline, "candidate": args.candidate, "gyro": "/imu/data_raw", "wheel": "/odom/odom_raw"}
    if topics["old"] == topics["candidate"]:
        raise ValueError("Baseline and candidate must be different topics")
    begin = [None]
    bad = {k:0 for k in capture}
    subs = []
    def receive(k, m):
        if begin[0] is None: return
        t = time.monotonic()-begin[0]
        if k == 'wheel':
            if len(m.data)<2:
                bad[k]+=1
                return
            row = (t,None,*m.data[:2])
        else:
            st = m.header.stamp.sec*10**9 + m.header.stamp.nanosec
            if k == 'gyro':
                row = (t,st,m.angular_velocity.z)
            else:
                p,q = m.pose.pose.position,m.pose.pose.orientation
                if not .99 <= q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w <= 1.01:
                    bad[k]+=1
                    return
                row = (t,st,p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
        if all(math.isfinite(x) for x in row if x is not None):
            capture[k].append(row)
        else:
            bad[k]+=1
    try:
        types = dict(old=Odometry,candidate=Odometry,gyro=Imu,wheel=Float32MultiArray)
        for k,t in types.items():
            subs.append(node.create_subscription(t,topics[k],lambda m,k=k:receive(k,m),qos_profile_sensor_data))
        warm = time.monotonic()+2
        while time.monotonic()<warm:
            rclpy.spin_once(node,timeout_sec=.1)
        print("Measurement starts now; use independent physical marks for ground truth.",flush=True, file=sys.stderr)
        begin[0] = time.monotonic()
        while time.monotonic()-begin[0]<args.seconds:
            rclpy.spin_once(node,timeout_sec=.05)
        pubs = {k:[p.node_name for p in node.get_publishers_info_by_topic(v)] for k,v in topics.items()}
    finally:
        node.destroy_node()
        rclpy.shutdown()
    health = {k:continuity(rows,args.seconds,k!='wheel') for k,rows in capture.items()}
    poses = {k:pose_summary(capture[k]) for k in ('old','candidate')}
    issues = []
    for k in topics:
        if not health[k]['fresh'] or bad[k] or len(pubs[k])!=1:
            issues.append(f"{k}: missing, nonfinite, interrupted, repeated stamps or non-unique publisher")
    gyro = [r[2] for r in capture['gyro']]
    report = dict(test="live", seconds=args.seconds, topics=topics, publishers=pubs, stream_health=health,
                  invalid_messages=bad, poses=poses, gyro_median_rad_s=statistics.median(gyro) if gyro else None)
    if capture["wheel"]:
        first,last = capture["wheel"][0], capture["wheel"][-1]
        report["wheel_delta_m"] = (last[2]-first[2], last[3]-first[3])
        if args.turn_deg is not None and abs(args.turn_deg)>=30 and health['wheel']['fresh']:
            l_delta,r_delta=report["wheel_delta_m"]
            # Signed and intentionally not clamped: unexpected direction is
            # evidence to investigate, never an automatic calibration change.
            report["apparent_wheelbase_m"]=(r_delta-l_delta)/math.radians(args.turn_deg)
    if health['gyro']['fresh']:
        rows=capture['gyro']; bias=args.gyro_bias_rad_s or 0.
        angle=sum(((b[2]+a[2])*.5-bias)*(b[0]-a[0]) for a,b in zip(rows, rows[1:]))
        report['integrated_gyro_deg']=math.degrees(angle)
        report['gyro_bias_removed_rad_s']=args.gyro_bias_rad_s  # None = explicitly uncalibrated
    candidate = poses['candidate']
    if args.parked and candidate:
        if candidate['max_heading_excursion_deg']>1.0 or candidate['max_displacement_m']>.025:
            issues.append('Candidate moved during measured stationary interval')
        rows=capture['wheel']
        if rows and any(abs(r[j]-rows[0][j])>.022 for r in rows for j in (2,3)):
            issues.append('Encoder travel conflicts with parked reference')
    if args.turn_deg is not None and candidate and abs(candidate['yaw_change_deg']-args.turn_deg)>args.turn_tolerance_deg:
        issues.append('Candidate turn differs from independently measured signed turn')
    if args.straight_m is not None and candidate:
        if (abs(candidate['forward_displacement_m']-args.straight_m)>args.distance_tolerance_m or
                abs(candidate['lateral_displacement_m'])>args.distance_tolerance_m or
                candidate['max_heading_excursion_deg']>args.turn_tolerance_deg):
            issues.append('Candidate pose differs from independently measured straight line')
    return dict(**report, issues=issues, pass_test=not issues, reference=('parked' if args.parked else
                dict(measured_turn_deg=args.turn_deg) if args.turn_deg is not None else
                dict(measured_straight_m=args.straight_m) if args.straight_m is not None else
                'no physical reference: stream check only'))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('replay')
    l=sub.add_parser('live')
    l.add_argument('--seconds',type=float,default=30)
    l.add_argument('--baseline',default='/odom')
    l.add_argument('--candidate',default='/crossmaps/odometry_shadow/odom')
    g=l.add_mutually_exclusive_group()
    g.add_argument('--parked',action='store_true')
    g.add_argument('--turn-deg',type=float,help='Externally measured signed rotation, CCW positive')
    g.add_argument('--straight-m',type=float,help='Externally measured signed straight distance, forward positive')
    l.add_argument('--turn-tolerance-deg',type=float,default=6)
    l.add_argument('--distance-tolerance-m',type=float,default=.07)
    l.add_argument('--gyro-bias-rad-s',type=float,
                   help='Optional independent parked gyro bias, used only in the diagnostic angle integral')
    a=p.parse_args()
    if a.mode=='live' and (not 5<=a.seconds<=600 or not 0<a.turn_tolerance_deg<=45 or not 0<a.distance_tolerance_m<=1 or
                          any(v is not None and not math.isfinite(v) for v in (a.turn_deg,a.straight_m,a.gyro_bias_rad_s))):
        p.error('Expected 5–600 seconds and finite reference/tolerance values within range')
    report = live(a) if a.mode=='live' else synthetic_replay()
    print(json.dumps(report,indent=2,allow_nan=False))
    raise SystemExit(0 if report['pass_test'] else 1)


if __name__=='__main__':
    main()
