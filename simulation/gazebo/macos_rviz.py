"""Give each native RViz window an identifiable, private macOS app bundle.

The executable is an unchanged copy of the installed RoboStack RViz binary.
Its original loader-relative RPATHs resolve through local directory symlinks;
no executable, library, signature, or system security setting is patched.
This module only prepares files. The navigation supervisor owns launching.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import plistlib
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE / ".runtime" / "rviz-apps"
ROBOTS = {"rover_a": "A", "rover_b": "B"}


def _directory(path):
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise RuntimeError(f"Refusing an unexpected path in the private RViz bundle: {path}")
    path.mkdir(exist_ok=True)


def _file_target(path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"Refusing to overwrite an unexpected RViz bundle entry: {path}")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _library_link(path, target):
    if path.is_symlink():
        if os.readlink(path) == str(target):
            return
        # Only this exact generated link is replaced, never its target.
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"Refusing to replace a real directory/file with a library link: {path}")
    path.symlink_to(target, target_is_directory=True)


def _write_plist(path, content):
    _file_target(path)
    encoded = plistlib.dumps(content, fmt=plistlib.FMT_XML, sort_keys=True)
    if path.exists() and path.read_bytes() == encoded:
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".Info-", suffix=".plist", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_rviz_app(python, robot):
    """Return a launchable RViz executable; never launch a GUI or a ROS node.

    On non-macOS platforms return the environment's original bin/rviz2. On
    macOS the only generated destinations are the two fixed simulator bundle
    names below .runtime/rviz-apps, with no user-controlled path components.
    """
    if robot not in ROBOTS:
        raise ValueError("Only rover_a and rover_b simulation RViz bundles are supported")
    python = Path(python).expanduser().resolve()
    prefix = python.parent.parent
    source = prefix / "bin" / "rviz2"
    if not source.is_file() or not os.access(source, os.X_OK):
        raise RuntimeError(f"Installed RViz executable is missing or not executable: {source}")
    if platform.system() != "Darwin":
        return str(source)
    for directory in (prefix / "lib", prefix / "opt" / "rviz_ogre_vendor" / "lib"):
        if not directory.is_dir():
            raise RuntimeError(f"RViz runtime library directory is missing: {directory}")

    name = f"CrossMaps Rover {ROBOTS[robot]} Navigation"
    bundle = APP_ROOT / (name + ".app")
    contents, executable = bundle / "Contents", bundle / "Contents" / "MacOS" / "RViz"
    # Check every generated ancestor before any write; a preexisting symlink
    # must not redirect our tiny wrapper into another application or prefix.
    for directory in (APP_ROOT.parent, APP_ROOT, bundle, contents, contents / "MacOS"):
        _directory(directory)
    _file_target(executable)
    source_hash = _sha256(source)
    if not executable.exists() or _sha256(executable) != source_hash:
        descriptor, temporary = tempfile.mkstemp(prefix=".RViz-", dir=executable.parent)
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            shutil.copy2(source, temporary)
            if _sha256(temporary) != source_hash:
                raise RuntimeError("RViz changed while copying; refusing an inconsistent bundle")
            temporary.replace(executable)
        finally:
            temporary.unlink(missing_ok=True)
    _library_link(contents / "lib", prefix / "lib")
    _library_link(contents / "opt", prefix / "opt")
    _write_plist(contents / "Info.plist", {
        "CFBundleName": name,
        "CFBundleDisplayName": name,
        "CFBundleIdentifier": "org.crossmaps.simulation.rviz." + robot.replace("_", "-"),
        "CFBundleExecutable": "RViz",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        "LSUIElement": False,
    })
    return str(executable)
