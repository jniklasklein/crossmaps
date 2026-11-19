"""Install the odometry module and (optionally) swap only the vendor lidar-bringup odom publisher.

Run INSIDE the sourced Humble container. No build, process changes, ROS calls or drive
actions. Without --apply this only prints the checked plan.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile


HERE = Path(__file__).resolve().parent
EXPECTED = 'c42aec356f96ff79c2c75e45549b55dd1a11bc18bbc3658711515b08c883e093'
VENDOR_BLOCK = """    base_node = Node(
        package='ugv_base_node',
        executable='base_node',
        parameters=[{'pub_odom_tf': LaunchConfiguration('pub_odom_tf')}]
    )"""
PATCH_BLOCK = """    # Only the odometry publisher changes. Keep hardware IO, RF2O and drive control.
    from launch.actions import ExecuteProcess
    import sys
    base_node = ExecuteProcess(
        cmd=[sys.executable, '-u', '-m', 'ugv_bringup.crossmaps_odometry_node',
             '--mode', 'live', '--owner-pid', str(os.getpid()),
             '--publish-tf', LaunchConfiguration('pub_odom_tf')],
        output='screen',
    )"""


def patch_vendor(existing):
    text = existing.decode()
    vendor = text.replace(PATCH_BLOCK, VENDOR_BLOCK).encode() if PATCH_BLOCK in text else existing
    if hashlib.sha256(vendor).hexdigest() != EXPECTED or vendor.count(VENDOR_BLOCK.encode()) != 1:
        raise ValueError('Unexpected vendor launch (refusing to overwrite local modifications)')
    return vendor.replace(VENDOR_BLOCK.encode(), PATCH_BLOCK.encode())


def plan(workspace, modules_only=False):
    import ugv_bringup
    ws = workspace.resolve(strict=True)
    src = ws/'src/ugv_main/ugv_bringup'
    active = ws/'install/ugv_bringup'
    source_launch = src/'launch/bringup_lidar.launch.py'
    install_launch = active/'share/ugv_bringup/launch/bringup_lidar.launch.py'
    active_module = Path(ugv_bringup.__file__).resolve(strict=True).parent
    if not active_module.is_relative_to(active) and active_module != (src/'ugv_bringup').resolve(strict=True):
        raise ValueError(f'Wrong ROS environment: {active_module} is not in this workspace')
    if not (active_module/'ugv_driver.py').is_file():
        raise ValueError(f'Wrong active ugv_bringup package {active_module}')
    actions = {}
    for module_dir in (src/'ugv_bringup',active_module):
        for filename,new_name in [('odometry_core.py','odometry_core.py'),('ros_node.py','crossmaps_odometry_node.py')]:
            path = (module_dir/new_name).resolve()
            if not path.is_relative_to(ws):
                raise ValueError(f'Destination escapes the workspace: {path}')
            actions[path]=(HERE/filename).read_bytes()
    for launch in (source_launch,install_launch):
        path = launch.resolve(strict=True)
        if not path.is_relative_to(ws):
            raise ValueError(f'Destination escapes the workspace: {path}')
        data = patch_vendor(path.read_bytes())
        if not modules_only:
            actions[path]=data
    return ws, {p:v for p,v in actions.items() if not p.exists() or p.read_bytes()!=v}


def replace(path,data):
    current=path.stat() if path.exists() else None
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary=Path(f.name)
        try:
            f.write(data); f.flush(); os.fsync(f.fileno())
            os.fchmod(f.fileno(),stat.S_IMODE(current.st_mode) if current else 0o644)
            if current and os.geteuid()==0:
                os.fchown(f.fileno(),current.st_uid,current.st_gid)
            os.replace(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--workspace',type=Path,default=Path('/home/ws/ugv_ws'))
    a.add_argument('--backup-dir',type=Path)
    a.add_argument('--modules-only', action='store_true', help='stage module but do not change normal launch')
    a.add_argument('--apply',action='store_true')
    opt=a.parse_args()
    ws, actions=plan(opt.workspace,opt.modules_only)
    if not actions:
        print('Odometry files already match')
        return
    for p in actions:
        print(('Install ' if opt.apply else 'Would install ')+str(p))
    if not opt.apply:
        return
    if not opt.backup_dir:
        a.error('--apply requires --backup-dir')
    for path in actions:
        if path.exists():
            backup = opt.backup_dir/path.relative_to(ws)
            backup.parent.mkdir(parents=True,exist_ok=True)
            if backup.exists() and backup.read_bytes()!=path.read_bytes():
                raise FileExistsError(f'Conflicting backup; choose a fresh directory: {backup}')
            if not backup.exists():
                shutil.copy2(path,backup)
    for path,data in actions.items():
        replace(path,data)
    print(f'Done; overwritten files backed up under {opt.backup_dir}')


if __name__=='__main__':
    main()
