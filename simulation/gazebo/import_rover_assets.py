#!/usr/bin/env python3
"""Import the user's existing Waveshare CAD description, retaining provenance.

This does not modify the source workspace. Source declares BSD in package.xml
but contains no standalone license text; that limitation is recorded explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def import_assets(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    relative_files = [Path("urdf/ugv_rover.urdf"), Path("package.xml")]
    relative_files += sorted(Path("meshes/ugv_rover").joinpath(p.name)
                             for p in (source / "meshes/ugv_rover").glob("*.stl"))
    if len(relative_files) < 13 or any(not (source / p).is_file() for p in relative_files):
        raise ValueError("Expected ugv_description src with ugv_rover.urdf and all 11 STL assets")
    hashes = {}
    for relative in relative_files:
        origin, target = source / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(origin.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Refusing to replace changed vendor asset: {target}")
        shutil.copy2(origin, target)
        hashes[str(relative)] = digest
    manifest = {
        "source": str(source),
        "upstream_project": "https://github.com/waveshareteam/ugv_ws",
        "origin": "Existing local robot description workspace; no upstream revision available",
        "license_declared_by_package_xml": "BSD",
        "license_caveat": "Source snapshot has no standalone license/copyright file; preserve package.xml and obtain upstream terms before redistributing assets outside this project.",
        "modifications": "Imported files byte-for-byte. SDF conversion and collision simplification occur only in generated world.",
        "sha256": hashes,
    }
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent / "vendor/waveshare_ugv")
    args = parser.parse_args()
    import_assets(args.source, args.destination)
