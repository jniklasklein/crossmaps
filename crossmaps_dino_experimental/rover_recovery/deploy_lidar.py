"""Stage and build in isolation, then atomically install the LD19 fix in a Jetson container."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_file(source, dest):
    """Keep the destination owner/mode and any install-space symlinks."""
    info = dest.stat()
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        shutil.copyfile(source, tmp)
        os.chown(tmp, info.st_uid, info.st_gid)
        os.chmod(tmp, info.st_mode & 0o7777)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/home/ws/ugv_ws'))
    p.add_argument('--stage', type=Path, required=True, help='New, nonexistent build and backup directory')
    p.add_argument('--serial-id', required=True)
    p.add_argument('--apply', action='store_true', help='Install verified build (otherwise only build)')
    a = p.parse_args()
    ws, stage = a.workspace.resolve(strict=True), a.stage.resolve()
    orig = ws / 'src/ugv_else/ldlidar'
    prefix = ws / 'install/ldlidar'
    launch = Path('launch/ld19.launch.py')
    driver = Path('src/demo.cpp')
    serial = Path('ldlidar_driver/src/serialcom/serial_interface_linux.cpp')
    binary = Path('lib/ldlidar/ldlidar_node')
    if not (Path('/dev/serial/by-id') / a.serial_id).is_char_device() or '/' in a.serial_id:
        p.error('Verify the actual rover-specific by-id device before installing.')
    stage.mkdir(mode=0o700, parents=False, exist_ok=False)
    pkg = stage / 'package'
    subprocess.run([sys.executable, str(Path(__file__).parent / 'lidar' / 'prepare.py'),
                    '--source', str(orig), '--out', str(pkg), '--serial-id', a.serial_id], check=True)
    # Use a separate build directory. Do not run colcon over live packages.
    subprocess.run(['cmake', '-S', str(pkg), '-B', str(stage/'build'),
                    '-DBUILD_TESTING=OFF', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_INSTALL_PREFIX=' + str(stage/'install')], check=True)
    subprocess.run(['cmake', '--build', str(stage/'build'), '--parallel', '2'], check=True)
    subprocess.run(['cmake', '--install', str(stage/'build')], check=True)
    exe = stage/'install'/binary
    if 'not found' in subprocess.check_output(['ldd', str(exe)], text=True):
        raise RuntimeError('Unresolved library in staged driver')
    compile((pkg/launch).read_text(), str(pkg/launch), 'exec')
    subprocess.run([sys.executable, str(Path(__file__).parent/'lidar/test_hotplug.py'), str(exe)],
                   env={**os.environ, 'ROS_LOCALHOST_ONLY':'1', 'ROS_DOMAIN_ID':'73'},
                   timeout=85, check=True)
    if not a.apply:
        print('Build validated; active files unchanged.')
        return
    targets = {dest.resolve(strict=True): source for dest, source in [
        (orig/driver, pkg/driver), (orig/serial, pkg/serial), (orig/launch, pkg/launch),
        (prefix/'share/ldlidar'/launch, pkg/launch), (prefix/binary, exe)]}
    backup = stage/'backup'
    backup.mkdir(mode=0o700)
    manifest = []
    for i,(dest, source) in enumerate(targets.items()):
        if ws not in dest.parents or not dest.is_file():
            raise RuntimeError('Unexpected install target: ' + str(dest))
        copy = backup/str(i)
        shutil.copy2(dest, copy)
        manifest.append(dict(path=str(dest), backup=str(copy), before=sha(copy), after=sha(source)))
    (backup/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    written = []
    try:
        for row, (dest, source) in zip(manifest, targets.items()):
            if sha(dest) != row['before']:
                raise RuntimeError('Concurrent edit: ' + str(dest))
            replace_file(source, dest)
            written.append((Path(row['backup']), dest))
            if sha(dest) != row['after']:
                raise RuntimeError('Install hash mismatch: ' + str(dest))
    except BaseException:
        for old, dest in reversed(written):
            replace_file(old, dest)
        raise
    print('Installed LD19; rollback manifest:', backup/'manifest.json')


if __name__ == '__main__':
    main()
