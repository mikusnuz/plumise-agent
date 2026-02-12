#!/usr/bin/env python3
"""
Build helper script for plumise-agent Windows exe.

Usage:
    python scripts/build-exe.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    """Build plumise-agent.exe using PyInstaller."""
    # Get project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    spec_file = project_root / "plumise-agent.spec"

    if not spec_file.exists():
        print(f"Error: spec file not found at {spec_file}", file=sys.stderr)
        return 1

    # Change to project root
    os.chdir(project_root)

    print("=" * 70)
    print("Building plumise-agent.exe with PyInstaller")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Spec file: {spec_file}")
    print()

    # Clean previous build
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"

    if dist_dir.exists():
        print(f"Cleaning {dist_dir}...")
        shutil.rmtree(dist_dir)

    if build_dir.exists():
        print(f"Cleaning {build_dir}...")
        shutil.rmtree(build_dir)

    print()

    # Run PyInstaller
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec_file),
        "--clean",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode

    # Check output
    exe_path = dist_dir / "plumise-agent.exe"

    print()
    print("=" * 70)

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"Build successful!")
        print(f"Output: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
        print()
        print("Usage:")
        print(f"  {exe_path} --help")
        print(f"  {exe_path} start --private-key 0x...")
        print(f"  {exe_path} status")
    else:
        print("Build completed but exe not found!", file=sys.stderr)
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
