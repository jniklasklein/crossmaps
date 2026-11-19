"""Read-only audit spanning a parked, live odometry handoff. Run inside the rover.

Check exact-stamp agreement between the replacement odometry and its dynamic TF,
and the last-old to first-new pose difference. No camera or motion topics used.
"""
import argparse
from collections import Counter
import json
import math
import os
import time


def pose(v, q):
    return (v.x, v.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))


def difference(a, b):
    return (math.hypot(a[0]-b[0], a[1]-b[1]), abs(math.degrees(math.remainder(a[2]-b[2], 2*math.pi))))


def stamp(h):
    return h.stamp.sec*10**9+h.stamp.nanosec


def main():
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data as qos
    from tf2_msgs.msg import TFMessage
    from rtabmap_msgs.msg import Info
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--require-handoff', action='store_true')
    a = p.parse_args()
    if not 5 <= a.seconds <= 180:
        p.error('Use a 5–180 second window')
    rclpy.init(args=[])
    n = rclpy.create_node('crossmaps_odom_handoff_audit_'+str(os.getpid()), start_parameter_services=False, enable_rosout=False)
    tf = {}
    received, updates = [], []
    last_diag = {}
    def odom(m):
        if (m.header.frame_id, m.child_frame_id)!=('odom','base_footprint'):
            return
        # This Humble rclpy executor does not expose the publisher GID to
        # subscription callbacks. The two known vendor covariance signatures
        # identify legacy samples; independently inspect publisher owners below.
        diag = list(m.twist.covariance)[::7]
        label = ('vendor' if diag in ([1e-9,1e-3,1e6,1e6,1e6,1e-9],
                                      [1e-3,1e-3,1e6,1e6,1e6,1e3]) else 'replacement')
        row = (label, stamp(m.header), pose(m.pose.pose.position,m.pose.pose.orientation), time.monotonic())
        received.append(row)
        last_diag[label] = diag
    def transforms(m):
        for x in m.transforms:
            if (x.header.frame_id, x.child_frame_id) == ('odom','base_footprint'):
                key = stamp(x.header)
                tf[key] = pose(x.transform.translation,x.transform.rotation)
    n.create_subscription(Odometry,'/odom',odom,qos)
    n.create_subscription(TFMessage,'/tf',transforms,qos)
    n.create_subscription(Info,'/info',lambda m:updates.append((time.monotonic(),stamp(m.header))),qos)
    start = time.monotonic()
    switch_t = None
    try:
        while time.monotonic()-start<a.seconds:
            rclpy.spin_once(n,timeout_sec=.1)
            if received and received[0][0]!=received[-1][0] and switch_t is None:
                switch_t = received[-1][3]
            if switch_t and time.monotonic()-switch_t>12:
                break
        gids = list(dict.fromkeys(r[0] for r in received))
        jumps = [difference(r[2],s[2]) for r,s in zip(received,received[1:]) if r[0]!=s[0]]
        new = [r for r in received if r[0] == gids[-1]] if gids else []
        together = [(r,tf[r[1]]) for r in new if r[1] in tf]
        agrees = all(difference(r[2],v)[0]<1e-7 and difference(r[2],v)[1]<1e-5 for r,v in together)
        ref = new[0][3] if new else time.monotonic()
        info_count = len(set(key for t,key in updates if t>=ref))
        pubs = n.get_publishers_info_by_topic('/odom')
        ok = bool(len(pubs)==1 and len(together)>=5 and agrees and info_count>=2)
        if a.require_handoff:
            ok = ok and gids==['vendor','replacement'] and len(jumps)==1 and jumps[0][0]<.02 and jumps[0][1]<2
        report = dict(ok=ok, domain=os.getenv('ROS_DOMAIN_ID','0'),sample_s=round(time.monotonic()-start,1),
                      odom_samples=dict(Counter(r[0] for r in received)),
                      current_publishers=[x.node_name for x in pubs],jumps_m_deg=jumps,
                      final_twist_diag=last_diag.get(gids[-1]) if gids else None,
                      matched_tf_stamps=len(together),matched_tf_agrees=agrees,
                      rtab_updates_after_first_new=info_count)
        print(json.dumps(report),flush=True)
        return 0 if ok else 1
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
