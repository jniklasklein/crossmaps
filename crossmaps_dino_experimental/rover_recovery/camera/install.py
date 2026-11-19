"""Stage or install OAK recovery into the rover's symlink-installed ugv_vision.

Run inside the SLAM container. Does not invoke ros2, build, restart or signal anything.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile


VENDOR_SHA = 'f09c4df9c665518f811b33f943408769fa3ee8ca08b34950fac067dbce52e56a'
HERE = Path(__file__).resolve().parent


def plan(workspace):
    import yaml
    pkg = workspace / 'src/ugv_main/ugv_vision'
    install = workspace / 'install/ugv_vision/share/ugv_vision'
    launch = pkg / 'launch/oak_d_lite.launch.py'
    config = pkg / 'config/oak_d_lite.yaml'
    for path in (config, launch):
        if (install / path.relative_to(pkg)).resolve(strict=True) != path.resolve(strict=True):
            raise ValueError(f'Installed overlay is not symlinked to source: {path}')
    native = launch.with_name('crossmaps_oak_native.launch.py')
    guard = launch.with_name('crossmaps_oak_guard.py')
    wrapper = (HERE / 'oak_guard.launch.py').read_bytes()
    native_data = native.read_bytes() if native.exists() else launch.read_bytes()
    if hashlib.sha256(native_data).hexdigest() != VENDOR_SHA:
        raise ValueError(f'Unrecognized vendor launch; not overwriting {launch}')
    if launch.read_bytes() not in (native_data, wrapper):
        raise ValueError(f'Locally modified OAK launch; not overwriting {launch}')
    for path in (native, guard):
        if path.is_symlink():
            raise ValueError(f'Refusing symlink destination {path}')
    conf = yaml.safe_load(config.read_text())
    camera = conf['/oak']['ros__parameters']['camera']
    # Retain vendor settings and any local tuning. Avoid rewriting files already correct.
    data = config.read_bytes()
    if camera.get('i_restart_on_diagnostics_error') is not True:
        camera['i_restart_on_diagnostics_error'] = True
        data = yaml.safe_dump(conf, sort_keys=False).encode()
    # Wrapper last: neither a crash nor a failed dependency installation switches the
    # launch into supervisor mode until the native launch, guard and config exist.
    return pkg, [(native, native_data), (guard, (HERE / 'crossmaps_oak_guard.py').read_bytes()),
                 (config, data), (launch, wrapper)]


def write_atomically(path, data):
    existing = path.stat() if path.exists() else None
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            os.fchmod(f.fileno(), stat.S_IMODE(existing.st_mode) if existing else 0o644)
            if existing and os.geteuid() == 0:
                os.fchown(f.fileno(), existing.st_uid, existing.st_gid)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/home/ws/ugv_ws'))
    p.add_argument('--backup-dir', type=Path, help='dedicated directory for this deployment')
    p.add_argument('--apply', action='store_true', help='default: inspect and print plan only')
    args = p.parse_args()
    pkg, edits = plan(args.workspace)
    dirty = [(path, data) for path, data in edits if not path.exists() or path.read_bytes() != data]
    if not dirty:
        print('OAK recovery already installed')
        return
    for path, _ in dirty:
        print(f'{"Install" if args.apply else "Would install"} {path}')
    if args.apply:
        if not args.backup_dir:
            p.error('--apply requires --backup-dir')
        # Back up everything replaced, before the first operational file is written.
        for path, _ in dirty:
            if path.exists():
                backup = args.backup_dir / path.relative_to(pkg)
                backup.parent.mkdir(parents=True, exist_ok=True)
                if backup.exists() and backup.read_bytes() != path.read_bytes():
                    raise FileExistsError(f'Conflicting backup: {backup}. Choose a fresh directory.')
                if not backup.exists():
                    shutil.copy2(path, backup)
        for path, data in dirty:
            write_atomically(path, data)
        print(f'OAK configuration installed. Previous files: {args.backup_dir}')


if __name__ == '__main__':
    main()
