#!/usr/bin/env python3
"""Read-only, sim-domain-only RGB-D capture for offline geometric diagnosis."""
import argparse
import math
import os
from pathlib import Path
import sys
import time
from run import apply_environment, environments, ros_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=('rover_a','rover_b'), default='rover_a')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=8.)
    args=parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 60:
        parser.error('--timeout must be finite, positive and at most 60 seconds')
    python=ros_python()
    if Path(sys.executable).resolve()!=Path(python).resolve():
        os.execv(python,[python,str(Path(__file__).resolve()),*sys.argv[1:]])
    _,env=environments(python,Path(__file__).parent/'.runtime/capture')
    apply_environment(env)
    os.environ['ROS_LOCALHOST_ONLY']='1'
    domain=80 if args.robot=='rover_a' else 81
    os.environ['ROS_DOMAIN_ID']=str(domain)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'crossmaps_dino_experimental'))
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CompressedImage,CameraInfo
    from tf2_ros import Buffer,TransformListener
    from collab_transport import decode_rgb,decode_depth,tf_matrix
    rclpy.init(args=[],domain_id=domain)
    node=Node('crossmaps_sim_capture_pair')
    buffer=Buffer()
    listener=TransformListener(buffer,node)
    pending={'rgb':{},'depth':{}}
    info=[]
    def image(kind,msg):
        tick=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        pending[kind][tick]=msg
        if len(pending[kind])>6:
            del pending[kind][min(pending[kind])]
    node.create_subscription(CompressedImage,'/oak/rgb/image_rect/compressed',lambda m:image('rgb',m),qos_profile_sensor_data)
    node.create_subscription(CompressedImage,'/oak/stereo/image_raw/compressedDepth',lambda m:image('depth',m),qos_profile_sensor_data)
    node.create_subscription(CameraInfo,'/oak/rgb/camera_info',lambda m:info.__setitem__(slice(None),[m]),qos_profile_sensor_data)
    try:
        deadline=time.monotonic()+args.timeout
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.01)
            common=pending['rgb'].keys()&pending['depth'].keys()
            if not common or not info:
                continue
            tick=max(common)
            rgb,depth=pending['rgb'][tick],pending['depth'][tick]
            try:
                pose=buffer.lookup_transform('odom',rgb.header.frame_id,Time.from_msg(rgb.header.stamp))
            except Exception:
                continue
            args.output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.output,rgb=decode_rgb(bytes(rgb.data)),depth=decode_depth(bytes(depth.data)),
                                K=np.array(info[0].k).reshape(3,3),T_local_camera=tf_matrix(pose),stamp_ns=tick)
            print(f'Saved {args.robot} capture {tick} to {args.output}')
            return 0
        raise RuntimeError('No matching RGB-D and capture-time TF before timeout')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    raise SystemExit(main())
