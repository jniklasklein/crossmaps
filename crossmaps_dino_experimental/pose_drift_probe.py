"""Read-only, bounded IMU/odometry/TF recording; never commands the robot.

Run in the ROS environment from this experimental directory. Output, if asked
for, must remain here. Drift rates describe received data, not ground truth.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np


def yaw(q):
    norm = math.sqrt(sum(float(v)**2 for v in (q.x, q.y, q.z, q.w)))
    if not math.isfinite(norm) or norm < 1e-6:
        return None
    x,y,z,w = (float(v)/norm for v in (q.x,q.y,q.z,q.w))
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def summarize(rows):
    if not rows:
        return {'count': 0}
    result = {'count':len(rows), 'duration_s':round(rows[-1]['t']-rows[0]['t'],3),
              'frames':sorted({r.get('frame','') for r in rows}),
              'publishers':sorted({r.get('publisher','') for r in rows})}
    for key in ('yaw','wz','vx','vy','x','y','delay_s'):
        usable = [r for r in rows if r.get(key) is not None and math.isfinite(r[key])]
        if not usable:
            continue
        values = np.array([r[key] for r in usable])
        if key == 'yaw':
            values = np.degrees(np.unwrap(values))
        times = np.array([r['t'] for r in usable]); times -= times[0]
        slope = float(np.polyfit(times,values,1)[0]) if np.ptp(times)>1 else None
        result[key] = dict(first=float(values[0]),last=float(values[-1]),
                           delta=float(values[-1]-values[0]),mean=float(np.mean(values)),
                           std=float(np.std(values)),slope_per_s=slope)
    result['yaw_units']='degrees; wz remains rad/s'
    for key in ('yaw_cov','wz_cov','orientation_available'):
        values = [r[key] for r in rows if key in r]
        if values:result[key]=values[-1]
    return result


def main():
    import rclpy
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--output')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    output=Path(args.output).resolve() if args.output else None
    if output and (not output.is_relative_to(root) or output.exists()):
        parser.error('Output must be a new file inside the experimental directory')
    rclpy.init();node=rclpy.create_node('crossmaps_pose_drift_probe')
    started=time.monotonic();rows={};subs=[];publishers={}
    def add(key,msg,info,values):
        stamp=msg.header.stamp
        sample=dict(t=time.monotonic()-started,stamp_s=stamp.sec+stamp.nanosec*1e-9,
                    delay_s=time.time()-(stamp.sec+stamp.nanosec*1e-9),frame=msg.header.frame_id,
                    publisher=bytes(info.publisher_gid).hex() if info is not None else 'unavailable_in_callback',**values)
        bucket=rows.setdefault(key,[])
        if len(bucket)<30000:bucket.append(sample)
    def imu(topic):
        def callback(msg,info=None):
            available=bool(msg.orientation_covariance[0]>=0)
            add(topic,msg,info,dict(yaw=yaw(msg.orientation) if available else None,
                wz=msg.angular_velocity.z,orientation_available=available,
                yaw_cov=msg.orientation_covariance[8],wz_cov=msg.angular_velocity_covariance[8]))
        return callback
    def odom(topic):
        def callback(msg,info=None):
            p=msg.pose.pose;t=msg.twist.twist
            add(topic,msg,info,dict(yaw=yaw(p.orientation),x=p.position.x,y=p.position.y,
                wz=t.angular.z,vx=t.linear.x,vy=t.linear.y,
                yaw_cov=msg.pose.covariance[35],wz_cov=msg.twist.covariance[35],child=msg.child_frame_id))
        return callback
    def tf(msg,info=None):
        for transform in msg.transforms:
            p=transform.transform.translation
            add('TF '+transform.header.frame_id+' -> '+transform.child_frame_id,transform,info,
                dict(yaw=yaw(transform.transform.rotation),x=p.x,y=p.y))
    try:
        until=time.monotonic()+3
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.1)
        for topic,types in node.get_topic_names_and_types():
            cls,callback=(Imu,imu(topic)) if 'sensor_msgs/msg/Imu' in types else (
                (Odometry,odom(topic)) if 'nav_msgs/msg/Odometry' in types else (None,None))
            if cls:
                publishers[topic]=[dict(node=p.node_name,namespace=p.node_namespace,
                    gid=bytes(p.endpoint_gid).hex()) for p in node.get_publishers_info_by_topic(topic)]
                subs.append(node.create_subscription(cls,topic,callback,qos_profile_sensor_data))
        subs.append(node.create_subscription(TFMessage,'/tf',tf,qos_profile_sensor_data))
        static_qos=QoSProfile(depth=100,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        subs.append(node.create_subscription(TFMessage,'/tf_static',tf,static_qos))
        end=time.monotonic()+max(5,min(120,args.seconds));report=time.monotonic()+10
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.1)
            if time.monotonic()>=report:
                print(json.dumps({'elapsed_s':round(time.monotonic()-started,1),'counts':{k:len(v) for k,v in rows.items()}}),flush=True)
                report=time.monotonic()+10
        summary={key:summarize(value) for key,value in rows.items()}
        encoded=json.dumps(dict(summary=summary,publishers=publishers,samples=rows))
        if output:
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('x') as stream:
                stream.write(encoded)
        print(json.dumps({'summary':summary,'output':str(output) if output else None}),flush=True)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
