#!/usr/bin/env python3
"""Stage the two verified Waveshare LD19 package revisions; never edit the live tree."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil

DEMO = Path("src/demo.cpp")
SERIAL = Path("ldlidar_driver/src/serialcom/serial_interface_linux.cpp")
LAUNCH = Path("launch/ld19.launch.py")
HASHES = {
    DEMO: {"9ee91c2005310b3ae31858bad885e982977364e4ded1eba81fc1c568a6ebfc13"},
    SERIAL: {"282e4499342a44a47ca1ad36cf3be9fde9c80cb075131fd86faa0b4678745889"},
    LAUNCH: {"c6208c6a2db8b4345c5895f6153bdf3ab227d1748fe32ccf10a90e3c0ebb1793",
             "58c24551deb3c632169bb908341a83d8f28dc776e5f74838b534d69e7db5a3b4"},
}


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Unexpected vendor source at " + repr(before[:70]))
    return text.replace(before, after)


def render_demo(text):
    text = once(text, '#include "ldlidar_driver.h"', '#include "ldlidar_driver.h"\n#include <algorithm>\n#include <chrono>')
    # Do not reuse the previous session's last timestamp after an outage.
    sig = "rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr& lidarpub)"
    if text.count(sig) != 2:
        raise ValueError("Unrecognized scan function signature")
    text = text.replace(sig, sig[:-1] + ", bool first_scan)")
    text = once(text, "  static bool first_scan = true;\n", "")
    text = once(text, "    first_scan = false;\n", "")
    a = text.index("  ldlidar::LDLidarDriver* ldlidarnode")
    b = text.index('  RCLCPP_INFO(node->get_logger(), "<product_name>:', a)
    text = text[:a] + text[b:]
    a = text.index("  ldlidarnode->RegisterGetTimestampFunctional")
    b = text.index('  RCLCPP_INFO(node->get_logger(), "ldlidar_published is end");', a)
    return text[:a] + Path(__file__).with_name("recovery_loop.cpp.inc").read_text() + "\n" + text[b:]


def render_serial(text):
    # Vendor C++14 atomics lacked initial values on open-failure paths.
    text = once(text, "rx_count_(0), read_callback_(nullptr)", "rx_count_(0), is_cmd_opened_(false), rx_thread_exit_flag_(true), read_callback_(nullptr)")
    # RX uses non-blocking read and <=100ms pselect. Join before closing the fd
    # so the reader cannot use fd=-1 or a descriptor recycled by ROS/DDS.
    frag = """  if (com_handle_ != -1) {
    close(com_handle_);
    com_handle_ = -1;
  }

"""
    a = text.index("bool SerialInterfaceLinux::Close()")
    b = text.index("bool SerialInterfaceLinux::ReadFromIO", a)
    close_fn = once(text[a:b], frag, "")
    close_fn = once(close_fn, "  is_cmd_opened_ = false;", frag + "  is_cmd_opened_ = false;")
    text = text[:a] + close_fn + text[b:]
    # Disconnected USB TTYs can return EOF immediately for every poll.
    text = once(text, "    }\n  }\n\n  delete[] rx_buf;", "    } else {\n      usleep(10000);\n    }\n  }\n\n  delete[] rx_buf;")
    return text


def prepare(source, out, serial_id):
    if not re.fullmatch(r"usb-1a86_USB_Single_Serial_[A-Za-z0-9]+-if00", serial_id):
        raise ValueError("Expected LD19 /dev/serial/by-id basename, not a tty number or glob")
    source = source.resolve(strict=True)
    if out.is_symlink():
        raise ValueError("Staging directory must not be a symlink")
    out = out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Source and staging trees must be independent")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Staging directory must be new or empty")
    for path, supported in HASHES.items():
        if hashlib.sha256((source / path).read_bytes()).hexdigest() not in supported:
            raise ValueError(f"{path} differs from the inspected stock version. No changes made; review before upgrading.")
    launch = (source / LAUNCH).read_text()
    launch, n = re.subn(r"\{'port_name': '[^']+'\}", "{'port_name': '/dev/serial/by-id/" + serial_id + "'}", launch)
    if n != 1:
        raise ValueError("Expected exactly one LD19 port parameter")
    launch = once(launch, "      name='LD19',", "      name='LD19',\n      respawn=True,\n      respawn_delay=5.0,")
    changes = {DEMO: render_demo((source / DEMO).read_text()), SERIAL: render_serial((source / SERIAL).read_text()), LAUNCH: launch}
    # Follow source symlinks into an independent tree; never modify through them.
    shutil.copytree(source, out, dirs_exist_ok=True, symlinks=False, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    for path, content in changes.items():
        (out / path).write_text(content)
        print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--serial-id", required=True)
    args = ap.parse_args()
    prepare(args.source, args.out, args.serial_id)
