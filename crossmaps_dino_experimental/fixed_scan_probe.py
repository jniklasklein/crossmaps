"""Bounded diagnostic fixed-angle scan relay. Does not publish TF or commands.

Not wired into production. Target angular bounds/frame are frozen from the first
valid scan. Nearest-bin assignment preserves missing returns rather than filling
occlusions or inventing depth. Use a fresh RF2O process with this topic: changing
scan geometry under an existing RF2O instance is unsafe.
"""
import argparse
import collections
import copy
import json
import math
import time
import numpy as np


def fixed_ranges(ranges, source_min, source_increment, target_min, target_max,
                 bins, range_min, range_max):
    values=np.asarray(ranges,dtype=np.float64)
    if (values.ndim!=1 or len(values)<2 or bins<2 or
        not all(math.isfinite(v) for v in (source_min,source_increment,target_min,target_max,range_min,range_max)) or
        source_increment==0 or target_max<=target_min or range_max<=range_min):
        raise ValueError('Invalid scan geometry or range limits')
    step=(target_max-target_min)/(bins-1)
    angles=source_min+np.arange(len(values))*source_increment
    indices=np.floor((angles-target_min)/step+.5).astype(np.int64)
    valid=np.isfinite(values)&(values>=range_min)&(values<=range_max)&(indices>=0)&(indices<bins)
    output=np.full(bins,np.inf,dtype=np.float64)
    # For collisions retain the nearest obstacle, never average across edges.
    np.minimum.at(output,indices[valid],values[valid])
    output[~np.isfinite(output)]=np.nan
    return output.astype(np.float32)


def main():
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=90.)
    parser.add_argument('--bins',type=int,default=502)
    parser.add_argument('--topic',default='/crossmaps_diagnostics/scan_fixed')
    args=parser.parse_args()
    if not 32<=args.bins<=4096:parser.error('bins must be between 32 and 4096')
    if not args.topic.startswith('/crossmaps_diagnostics/'):
        parser.error('This probe may only publish inside /crossmaps_diagnostics/')
    rclpy.init();node=rclpy.create_node('crossmaps_fixed_scan_probe')
    publisher=node.create_publisher(LaserScan,args.topic,qos_profile_sensor_data)
    reference=None;counts=collections.Counter();accepted=0;rejected=0;last_stamp=-1
    def receive(msg):
        nonlocal reference,accepted,rejected,last_stamp
        counts[len(msg.ranges)]+=1
        stamp=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
        try:
            if stamp<=last_stamp or len(msg.ranges)<2 or msg.angle_increment<=0:
                raise ValueError('Nonadvancing or invalid scan')
            end_angle=msg.angle_min+(len(msg.ranges)-1)*msg.angle_increment
            if abs(end_angle-msg.angle_max)>abs(msg.angle_increment)*.1:
                raise ValueError('Angle bounds do not describe ranges')
            current=(msg.header.frame_id,float(msg.angle_min),float(msg.angle_max))
            if reference is None:reference=current
            if current[0]!=reference[0] or max(abs(current[i]-reference[i]) for i in (1,2))>1e-5:
                raise ValueError('Scan frame or field of view changed')
            output=copy.deepcopy(msg)
            output.ranges=fixed_ranges(msg.ranges,msg.angle_min,msg.angle_increment,
                reference[1],reference[2],args.bins,msg.range_min,msg.range_max).tolist()
            output.angle_increment=(reference[2]-reference[1])/(args.bins-1)
            output.angle_min=reference[1];output.angle_max=reference[2]
            output.intensities=[]  # No false correspondence after rebinning.
            output.time_increment=msg.time_increment*output.angle_increment/msg.angle_increment
            publisher.publish(output);accepted+=1;last_stamp=stamp
        except ValueError as error:
            rejected+=1
            if rejected<=3:print(str(error),flush=True)
    subscription=node.create_subscription(LaserScan,'/scan',receive,qos_profile_sensor_data)
    start=time.monotonic();report=start+10
    try:
        while time.monotonic()-start<max(5,min(300,args.seconds)):
            rclpy.spin_once(node,timeout_sec=.1)
            if time.monotonic()>report:
                print(json.dumps(dict(accepted=accepted,rejected=rejected,source_widths=dict(counts),fixed_bins=args.bins)),flush=True)
                report=time.monotonic()+10
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
