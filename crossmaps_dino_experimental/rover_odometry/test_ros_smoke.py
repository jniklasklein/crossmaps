"""Opt-in end-to-end ROS smoke test. Domain 74 + localhost ONLY; no motor publisher.

With this file beside ros_node.py and odometry_core.py, inside a sourced Linux
Humble container: ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 CROSSMAPS_ODOM_SMOKE=1
python3 -Bu test_ros_smoke.py
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(os.environ.get('CROSSMAPS_ODOM_SMOKE')=='1', 'explicit isolated Humble domain trial')
class SyntheticRos(unittest.TestCase):
    def test_odom_tf_and_foreign_owner(self):
        self.assertEqual(os.environ.get('ROS_DOMAIN_ID'),'74')
        self.assertEqual(os.environ.get('ROS_LOCALHOST_ONLY'),'1')
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data as qos
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Float32MultiArray, String
        from tf2_msgs.msg import TFMessage
        rclpy.init(args=[])
        n = rclpy.create_node('crossmaps_synthetic_odom_fixture',start_parameter_services=False)
        child=None
        try:
            t=time.monotonic()+1.1
            while time.monotonic()<t:
                rclpy.spin_once(n,timeout_sec=.05)
            for topic in ['/odom','/imu/data_raw','/odom/odom_raw']:
                self.assertFalse(n.get_publishers_info_by_topic(topic), f'{topic} is not isolated')
            wi=n.create_publisher(Float32MultiArray,'/odom/odom_raw',2)
            gy=n.create_publisher(Imu,'/imu/data_raw',2)
            poses, tf, status={}, {}, []
            def stamp(h):
                return h.stamp.sec*10**9+h.stamp.nanosec
            def co(p,q):
                return p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            def od(m):
                self.assertEqual((m.header.frame_id,m.child_frame_id),('odom','base_footprint'))
                self.assertTrue(all(v>0 and math.isfinite(v) for v in m.twist.covariance[::7]))
                poses[stamp(m.header)]=co(m.pose.pose.position,m.pose.pose.orientation)
            def transform(m):
                for u in m.transforms:
                    if (u.header.frame_id,u.child_frame_id)==('odom','base_footprint'):
                        tf[stamp(u.header)]=co(u.transform.translation,u.transform.rotation)
            n.create_subscription(Odometry,'/odom',od,qos)
            n.create_subscription(TFMessage,'/tf',transform,qos)
            n.create_subscription(String,'/crossmaps/odometry/status',lambda m:status.append(json.loads(m.data)),qos)
            counter=0
            def stream(seconds,left,right,rate=0.,freeze=False):
                nonlocal counter
                start=time.monotonic()
                end=start+seconds
                while time.monotonic()<end:
                    elapsed=time.monotonic()-start
                    m=Imu(); m.header.frame_id='base_imu_link'
                    m.header.stamp=n.get_clock().now().to_msg()
                    m.angular_velocity.z=rate
                    m.linear_acceleration.x=0. if freeze else (counter%5)*.0001
                    gy.publish(m)
                    w=Float32MultiArray()
                    w.data=[left(elapsed) if callable(left) else left,
                            right(elapsed) if callable(right) else right]
                    wi.publish(w);counter+=1
                    next_tick=time.monotonic()+.045
                    while time.monotonic()<next_tick:
                        rclpy.spin_once(n,timeout_sec=.006)
            with tempfile.TemporaryFile(mode='w+') as log:
                child=subprocess.Popen([sys.executable,'-Bu',str(Path(__file__).with_name('ros_node.py')),
                                        '--mode','live','--owner-pid',str(os.getpid())],
                                       stdout=log,stderr=subprocess.STDOUT)
                stream(5.,0.,0.)
                self.assertGreater(len(poses),8)
                self.assertLess(math.dist(next(reversed(poses.values())),(0.,0.,0.)),1e-6)
                stream(1.1,lambda t:.2*min(1.,t),lambda t:.2*min(1.,t))
                stream(.5,.2,.2)
                self.assertLess(math.dist(next(reversed(poses.values())),(.2,0.,0.)),.005)
                target=.6
                step=.175*.5*target
                stream(1.0,lambda t:.2-step*min(1.,t),lambda t:.2+step*min(1.,t),rate=target)
                stream(.8,.2-step,.2+step)
                x,y,yaw=next(reversed(poses.values()))
                self.assertLess(math.hypot(x-.2,y),.005); self.assertAlmostEqual(yaw,target,delta=.05)
                both=poses.keys() & tf.keys()
                self.assertGreater(len(both),20)
                self.assertTrue(all(math.dist(poses[k],tf[k])<1e-8 for k in both))
                print(f'PASS parked/straight/turn: pose {(x,y,yaw)}, {len(both)} matching odom/TF stamps',flush=True)
                stream(3.8,.2-step,.2+step,freeze=True)
                before=len(poses); stream(.65,.2-step,.2+step,freeze=True)
                self.assertEqual(len(poses),before)
                stream(1.25,.2-step,.2+step)
                self.assertGreater(len(poses),before+5)
                self.assertTrue(any(r.get('frozen_feedback_drops',0)>0 for r in status))
                print('PASS replayed full feedback withheld; fresh data resumed',flush=True)
                foreign=n.create_publisher(Odometry,'/odom',2)
                rc=child.wait(timeout=8)
                log.seek(0); text=log.read(); self.assertEqual(rc,3,text)
                self.assertIn('Another /odom publisher detected',text)
                n.destroy_publisher(foreign)
                print('PASS replacement withdrew on foreign publisher, no process restart',flush=True)
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill();child.wait(timeout=3)
            n.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    unittest.main(verbosity=2)
