"""Parked rollout into a NEW RTAB-Map DB; never clear old data or restart SLAM."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from ros_node import proc_start, verify_legacy


def open_dbs(pid):
    paths = set()
    for p in Path(f'/proc/{pid}/fd').iterdir():
        try:
            dest = os.readlink(p)
        except FileNotFoundError:
            continue
        if dest.endswith('.db'):
            paths.add(dest)
    return paths


def main():
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from std_srvs.srv import Empty
    from rtabmap_msgs.srv import LoadDatabase
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner', required=True,type=int)
    p.add_argument('--mapper', required=True,type=int)
    p.add_argument('--incumbent', type=int, help='Only if the verified stock base_node is still running')
    p.add_argument('--db',required=True,type=Path,help='New unused absolute .db path under /home/ws')
    p.add_argument('--log',required=True,type=Path,help='New path for odometry stdout')
    p.add_argument('--domain',required=True,choices=['0','1'])
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    if not a.db.is_absolute() or a.db.suffix!='.db' or not a.db.resolve().is_relative_to('/home/ws') or a.db.exists() or a.log.exists():
        p.error('Refusing reused or unexpected database/log path')
    if not a.db.parent.is_dir() or not os.access(a.db.parent,os.W_OK):
        p.error('Database parent missing or not writable')
    owner=proc_start(a.owner); mapper=proc_start(a.mapper)
    old=verify_legacy(a.incumbent,a.owner,int(a.domain)) if a.incumbent else None
    args=Path(f'/proc/{a.owner}/cmdline').read_bytes().split(b'\0')
    if not owner or not mapper or not any(args[i:i+3]==[b'launch',b'ugv_slam',b'rtabmap_rgbd.launch.py'] for i in range(len(args))):
        p.error('Not the expected SLAM launch')
    mroot=Path(f'/proc/{a.mapper}')
    if not os.readlink(mroot/'exe').endswith('/rtabmap_slam/rtabmap') or int((mroot/'stat').read_text().rsplit(')',1)[1].split()[1])!=a.owner:
        p.error('Not the expected RTAB-Map child')
    env=dict(os.fsdecode(v).split('=',1) for v in Path(f'/proc/{a.owner}/environ').read_bytes().split(b'\0') if v)
    if env.get('ROS_DOMAIN_ID','0')!=a.domain or os.getenv('ROS_DOMAIN_ID','0')!=a.domain:
        p.error('Not in the owner domain')
    cmd=['/usr/bin/python3','-Bu','-m','ugv_bringup.crossmaps_odometry_node','--mode','live','--owner-pid',str(a.owner)]
    subprocess.run(['/usr/bin/python3','-B','-c','from ugv_bringup import odometry_core,crossmaps_odometry_node'],env=env,check=True)
    rclpy.init(args=[]); n=rclpy.create_node('fresh_map_preflight_'+str(os.getpid()),start_parameter_services=False,enable_rosout=False)
    clients={key:n.create_client(typ,'/rtabmap/'+key) for key,typ in [('pause',Empty),('load_database',LoadDatabase),('resume',Empty)]}
    msgs=[]
    n.create_subscription(Odometry,'/odom',lambda m:msgs.append(m),qos_profile_sensor_data)
    def wait_for(fn, seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until and not fn():
            rclpy.spin_once(n,timeout_sec=.1)
        return bool(fn())
    def owners():
        return [p.node_name for p in n.get_publishers_info_by_topic('/odom')]
    def rpc(key,request):
        call=clients[key].call_async(request)
        if not wait_for(call.done,150 if key=='load_database' else 15):
            raise RuntimeError(key+' timed out; do not resend without inspecting server state')
        if call.result() is None:
            raise RuntimeError(f'{key} failed: {call.exception()}')
        print(key+' returned',flush=True)
    def still_owners():
        return proc_start(a.owner)==owner and proc_start(a.mapper)==mapper
    try:
        for name,c in clients.items():
            if not c.wait_for_service(timeout_sec=12):
                raise RuntimeError(f'{name} service unavailable')
        end=time.monotonic()+2.0
        wait_for(lambda:time.monotonic()>=end,3.)
        before=open_dbs(a.mapper)
        if len(before)!=1 or owners()!=(['base_node'] if old else []) or not still_owners():
            raise RuntimeError(f'Unexpected DB or odometry owner: {before}, {owners()}')
        print('Validated active database:',before,'odom publishers:',owners(),flush=True)
        if not a.apply:
            return
        # Pause the consumer before changing the meaning of odom coordinates.
        rpc('pause',Empty.Request())
        if a.incumbent:
            fd=os.pidfd_open(a.incumbent)
            try:
                if not still_owners() or proc_start(a.incumbent)!=old:
                    raise RuntimeError('Process identities changed')
                signal.pidfd_send_signal(fd,signal.SIGINT)
            finally:
                os.close(fd)
            if not wait_for(lambda:proc_start(a.incumbent)!=old and not owners(),16):
                raise RuntimeError('Legacy /odom not retired; mapper is paused')
        if owners() or not still_owners():
            raise RuntimeError('Unexpected odometry owner before bind')
        # Switch exactly once. This version of the service has an empty reply:
        # the mapper's actual new database FD is the acknowledgement we need.
        if a.db.exists():
            raise RuntimeError('Database path appeared since preflight')
        req=LoadDatabase.Request();req.database_path=str(a.db);req.clear=False
        rpc('load_database',req)
        if not wait_for(lambda:open_dbs(a.mapper)=={str(a.db)},12):
            raise RuntimeError('Could not verify new DB on mapper; inspect open FDs, mapper stays paused')
        print('Verified new DB open:',a.db,flush=True)
        msgs.clear()
        with a.log.open('x') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        if not wait_for(lambda:proc.poll() is not None or (owners()==['crossmaps_odometry'] and len(msgs)>=10),12) or proc.poll() is not None:
            raise RuntimeError('New odometry did not start; inspect log, mapper stays paused')
        if not still_owners() or len({(x.header.stamp.sec,x.header.stamp.nanosec) for x in msgs[-10:]})!=10:
            raise RuntimeError('Odometry stamps not advancing')
        rpc('resume',Empty.Request())
        print('Resumed original mapper',a.mapper,'on the NEW graph; odometry PID',proc.pid,flush=True)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
