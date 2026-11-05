#!/usr/bin/env python3
"""
Build and publish maque-core to PyPI.

Usage:
    python publish.py           # Build + upload to PyPI
    python publish.py --test    # Build + upload to TestPyPI
    python publish.py --build   # Build only (no upload)
"""
import subprocess
import sys
import os
import shutil


def run(cmd, check=True):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def clean_dist():
    dist_dir = "dist"
    build_dir = "build"
    egg_dir = "maque_core.egg-info"
    for d in [dist_dir, build_dir, egg_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Cleaned {d}/")


def build():
    print("\n=== Building package ===")
    clean_dist()
    # Install/upgrade build tools
    run([sys.executable, "-m", "pip", "install", "--upgrade", "build", "twine"])
    # Build
    run([sys.executable, "-m", "build"])
    # Verify
    run([sys.executable, "-m", "twine", "check", "dist/*"])
    print("\n✅ Build successful!")
    # List artifacts
    for f in os.listdir("dist"):
        size = os.path.getsize(os.path.join("dist", f))
        print(f"   {f} ({size:,} bytes)")


def upload(repository="pypi"):
    print(f"\n=== Uploading to {repository} ===")
    if repository == "testpypi":
        run([sys.executable, "-m", "twine", "upload", "--repository", "testpypi", "dist/*"])
    else:
        run([sys.executable, "-m", "twine", "upload", "dist/*"])
    print(f"\n🎉 Published to {repository} successfully!")


def main():
    args = sys.argv[1:]

    if "--test" in args:
        build()
        upload("testpypi")
    elif "--build" in args:
        build()
    else:
        build()
        confirm = input("\nUpload to PyPI? [y/N]: ").strip().lower()
        if confirm == "y":
            upload("pypi")
        else:
            print("Skipped upload. Run 'twine upload dist/*' to publish later.")


if __name__ == "__main__":
    main()
