"""Opt-in Linux/Humble end-to-end test; never launches or subscribes to hardware.

Inside a sourced ROS environment, run CROSSMAPS_OAK_E2E=1 python3 -B
test_camera_guard_e2e.py. All ROS test processes are fixed to localhost/domain
74, publishing tiny synthetic messages under a unique camera name. The normal
production timeouts are used intentionally (approximately one minute).
"""
import contextlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


NATIVE = """
import os
from pathlib import Path
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    assert os.environ['ROS_LOCALHOST_ONLY'] == '1' and os.environ['ROS_DOMAIN_ID'] == '74'
    return LaunchDescription([
        DeclareLaunchArgument('name'),
        DeclareLaunchArgument('rectify_rgb', default_value='False'),
        ExecuteProcess(cmd=[sys.executable, '-u', str(Path(__file__).with_name('synthetic.py')),
                            LaunchConfiguration('name')], output='screen'),
    ])
"""
CAMERA = """
import os
from pathlib import Path
import sys
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
assert os.environ['ROS_LOCALHOST_ONLY'] == '1' and os.environ['ROS_DOMAIN_ID'] == '74'
name = sys.argv[1]
assert name.startswith('synthetic_oak_')
root = Path(__file__).parent
ledger = root / 'camera_runs'
first = not ledger.exists()
with ledger.open('a') as stream:
    print(os.getpid(), file=stream)
rclpy.init(args=[])
node = rclpy.create_node(name + '_source')
pubs = [node.create_publisher(CompressedImage, f'/{name}/' + suffix, qos_profile_sensor_data)
        for suffix in ('rgb/image_rect/compressed', 'stereo/image_raw/compressedDepth')]
msg = CompressedImage()
msg.data = [1]
frozen = False
while rclpy.ok():
    if not (first and (root / 'freeze').exists()):
        stamp = time.time_ns()
    elif not frozen:
        (root / 'frozen').touch()
        frozen = True
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
    for pub in pubs:
        pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=.1)
"""


def wait_for(label, func, timeout, guard, log):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if func():
            print('PASS:', label, flush=True)
            return
        if guard.poll() is not None:
            break
        time.sleep(.25)
    raise AssertionError(f'{label}: timed out or guard exited\n{log.read_text()[-5000:]}')


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


@unittest.skipUnless(os.environ.get('CROSSMAPS_OAK_E2E') == '1', 'explicit opt-in Linux/Humble test')
class CameraSupervisorEndToEnd(unittest.TestCase):
    def test_stalled_stamps_then_owner_shutdown(self):
        with tempfile.TemporaryDirectory(prefix='crossmaps-sim-oak-') as tmp:
            root = Path(tmp)
            self.assertTrue(sys.platform.startswith('linux'))
            name = f'synthetic_oak_{os.getpid()}'
            env = dict(os.environ, ROS_DOMAIN_ID='74', ROS_LOCALHOST_ONLY='1',
                       ROS_LOG_DIR=str(root / 'ros-logs'), PYTHONDONTWRITEBYTECODE='1')
            script = root / 'crossmaps_oak_guard.py'
            shutil.copyfile(Path(__file__).with_name(script.name), script)
            (root / 'crossmaps_oak_native.launch.py').write_text(NATIVE)
            (root / 'synthetic.py').write_text(CAMERA)
            log = root / 'guard.log'
            owner = subprocess.Popen(['sleep', '240'], start_new_session=True)
            unrelated = subprocess.Popen(['sleep', '240'], start_new_session=True)
            guard = None
            try:
                with log.open('w') as out:
                    guard = subprocess.Popen([sys.executable, '-u', str(script), '--name', name,
                                              '--owner-pid', str(owner.pid)], env=env, stdout=out,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                marker = 'RGB and depth acquisition stamps advancing'
                wait_for('both synthetic outputs initially advancing', lambda: marker in log.read_text(),
                         70, guard, log)
                ledger = root / 'camera_runs'
                self.assertEqual(len(ledger.read_text().splitlines()), 1)
                (root / 'freeze').touch()
                wait_for('synthetic publisher freezing acquisition time but still publishing',
                         (root / 'frozen').exists, 3, guard, log)
                wait_for('both outputs advancing after one automatic camera-only restart',
                         lambda: log.read_text().count(marker) >= 2, 95, guard, log)
                report = log.read_text()
                self.assertIn('no advancing', report)
                pids = ledger.read_text().splitlines()
                self.assertEqual(len(set(pids)), 2, report)
                self.assertEqual(len(pids), 2, report)
                groups = [int(p) for p in re.findall(r'started private camera launch PID (\d+)', report)]
                self.assertEqual(len(groups), 2)
                self.assertFalse(group_alive(groups[0]), 'old camera group still running')
                self.assertTrue(group_alive(groups[1]))
                self.assertIsNone(owner.poll())
                self.assertIsNone(unrelated.poll())
                owner.terminate()
                owner.wait(timeout=3)
                self.assertEqual(guard.wait(timeout=18), 0, log.read_text())
                self.assertFalse(group_alive(groups[1]), 'camera group survived owner shutdown')
                self.assertIsNone(unrelated.poll(), 'unrelated service received signal')
                print('PASS: both private camera groups gone; unrelated process still alive', flush=True)
            finally:
                # Signal only the test Popen handles, never names or production PIDs.
                for p in (owner, guard, unrelated):
                    if p is not None and p.poll() is None:
                        with contextlib.suppress(ProcessLookupError):
                            p.terminate()
                        try:
                            p.wait(timeout=14)
                        except subprocess.TimeoutExpired:
                            p.kill()
                            p.wait(timeout=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
