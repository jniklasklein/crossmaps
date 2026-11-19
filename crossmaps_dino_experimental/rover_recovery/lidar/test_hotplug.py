#!/usr/bin/env python3
"""Exercise the real ARM/ROS executable through two different PTYs, never hardware.

In a ROS Humble Linux shell:
  ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=73 python3 test_hotplug.py /path/to/ldlidar_node
"""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import pty
import signal
import struct
import subprocess
import tempfile
import time
import tty


def revolution():
    """36 valid 12-return LD19 packets at 10Hz, range 6m.

    Use far returns: the vendor NearFilter discards a perfectly circular, single
    near-range group on wrap-around. Flat-ring near returns are not a valid fixture.
    """
    packets = bytearray()
    for i in range(36):
        packet = (struct.pack("<BBHH", 0x54, 0x2C, 3600, i * 1000) +
                  struct.pack("<HB", 6000, 200) * 12 +
                  struct.pack("<HH", i * 1000 + 915, i * 3))
        crc = 0
        for byte in packet:
            crc ^= byte
            for _ in range(8):
                crc = ((crc << 1) ^ (0x4D if crc & 0x80 else 0)) & 255
        packets.extend(packet + bytes([crc]))
    return packets


def hotplug(binary):
    if os.environ.get("ROS_LOCALHOST_ONLY") != "1" or not 2 <= int(os.environ.get("ROS_DOMAIN_ID", "-1")) <= 100:
        raise SystemExit("Set ROS_LOCALHOST_ONLY=1 and an unused test ROS_DOMAIN_ID (2–100), never rover domain 0/1.")
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    with ExitStack() as cleanup:
        cleanup.callback(rclpy.shutdown)
        node = rclpy.create_node("ld19_probe_" + str(os.getpid()), enable_rosout=False, start_parameter_services=False)
        cleanup.callback(node.destroy_node)
        topic = "/ld19_recovery_test_" + str(os.getpid()) + "/scan"
        samples = []
        node.create_subscription(LaserScan, topic, lambda msg: samples.append((msg.header.stamp.sec, msg.header.stamp.nanosec, msg.scan_time)), qos_profile_sensor_data)
        tmp = Path(cleanup.enter_context(tempfile.TemporaryDirectory()))
        link = tmp / "lidar"
        # Allocate both ports now so their device numbers are guaranteed distinct.
        devices = []
        for _ in range(2):
            master, slave = pty.openpty()
            m = cleanup.enter_context(os.fdopen(master, "wb", buffering=0))
            s = cleanup.enter_context(os.fdopen(slave, "rb", buffering=0))
            tty.setraw(slave)
            os.set_blocking(master, False)
            devices.append((m, s, os.ttyname(s.fileno())))
        assert devices[0][2] != devices[1][2]
        cmd = [str(binary), "--ros-args", "-r", "__node:=ld19_pty_driver_" + str(os.getpid()),
               "-p", "port_name:=" + str(link), "-p", "port_baudrate:=230400",
               "-p", "product_name:=LDLiDAR_LD19", "-p", "frame_id:=test_frame", "-p", "topic_name:=" + topic]
        proc = subprocess.Popen(cmd)
        def stop():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(4)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        cleanup.callback(stop)
        cycle = revolution()
        def await_scans(port):
            samples.clear()
            n, deadline, next_write = 0, time.monotonic() + 30, time.monotonic()
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("LD19 process exited: " + str(proc.returncode))
                if time.monotonic() >= next_write:
                    next_write = time.monotonic() + .1
                    try:
                        n += os.write(port.fileno(), cycle)
                    except BlockingIOError:
                        pass
                rclpy.spin_once(node, timeout_sec=.03)
                if len({(s, ns) for s, ns, _ in samples}) >= 5:
                    if not all(0 < dt < 1.5 for _, _, dt in samples):
                        raise RuntimeError("Unexpected scan_time after connect: " + repr(samples))
                    return
            raise RuntimeError(f"No fresh scans after {n} bytes sent, samples={samples}")
        # Start with nothing connected, just like bringing up a rover without its hub.
        time.sleep(1.2)
        link.symlink_to(devices[0][2])
        await_scans(devices[0][0])
        link.unlink()
        devices[0][0].close()
        devices[0][1].close()
        # Flush any DDS samples from before the unplug. No new serial data is produced.
        until = time.monotonic() + 7
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.1)
        link.symlink_to(devices[1][2])
        await_scans(devices[1][0])
        # Distinguish a clean operator shutdown from a crash or a retry loop.
        proc.send_signal(signal.SIGINT)
        code = proc.wait(4)
        if code != 0:
            raise RuntimeError(f"Driver failed normal shutdown: {code}")
        print(f"PASS: PID {proc.pid} produced new scans on both {devices[0][2]} and {devices[1][2]}, then shut down cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    hotplug(parser.parse_args().binary.resolve(strict=True))
