#!/usr/bin/env python3
"""Stage a versioned extension under extensions/; --publish commits and pushes it.

Bump the manifest and package version, commit source changes, then run this from a
TissueLab checkout. Old archives are immutable; only the latest version appears in the feed.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from tissuelab.extension_release import PUBLIC_REPOSITORY, FEED_URL, build_extension


def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def validate_release(existing, proposed):
    """Never silently replace an already published version or move the feed backward."""
    old = existing.get("data", [])
    new = proposed["data"][0]
    for entry in old:
        if entry["id"] != new["id"]:
            raise ValueError("unexpected extension in dedicated distribution repository")
        if tuple(map(int, new["version"].split("."))) < tuple(map(int, entry["version"].split("."))):
            raise ValueError("refusing to downgrade update feed")
        if entry["version"] == new["version"] and entry["archive_hash"] != new["archive_hash"]:
            raise ValueError("version already published with different contents; bump manifest version")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.publish:
        if run("git", "status", "--porcelain", cwd=root):
            raise SystemExit("Commit source changes before publishing")
        if run("git", "branch", "--show-current", cwd=root) != "main":
            raise SystemExit("Publish from main")
        remote = run("git", "remote", "get-url", "origin", cwd=root).removesuffix(".git")
        if remote not in (f"https://github.com/{PUBLIC_REPOSITORY}", f"git@github.com:{PUBLIC_REPOSITORY}"):
            raise SystemExit("Unexpected origin; refusing to publish")
    archive = build_extension(root)
    folder = root / "extensions"
    folder.mkdir(exist_ok=True)
    index = folder / "index.json"
    proposed = json.loads((archive.parent / "index.json").read_text())
    if index.exists():
        validate_release(json.loads(index.read_text()), proposed)
    target = folder / archive.name
    if target.exists() and target.read_bytes() != archive.read_bytes():
        raise SystemExit("Published archive is immutable; bump the extension version")
    names = [archive.name, "index.json", "README.md"]
    for name in names:
        shutil.copyfile(archive.parent / name, folder / name)
    if args.publish:
        subprocess.run(["git", "add", "--", *[f"extensions/{n}" for n in names]], cwd=root, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode:
            version = proposed["data"][0]["version"]
            subprocess.run(["git", "commit", "-m", f"Publish TissueLab {version}"], cwd=root, check=True)
        # Also retries a previous commit whose push failed.
        subprocess.run(["git", "push", "origin", "main"], cwd=root, check=True)
    print(f"Prepared {target}")
    print(f"Update feed: {FEED_URL}")


if __name__ == "__main__":
    main()
