"""Build an allowlisted Blender extension and static update index (no network calls).

The public distribution contains Python source inside the ZIP, licensed under GPL-3.0-or-later. Publishing is handled by scripts/publish_extension.py.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import tomllib
import zipfile

PUBLIC_REPOSITORY = "dylanrussellmd/TissueLab"
FEED_URL = f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/main/extensions/index.json"


def build_extension(root: Path | None = None, output: Path | None = None) -> Path:
    package = root / "tissuelab" if root is not None else Path(__file__).resolve().parent
    src = package / "blender_addon"
    manifest_bytes = (src / "blender_manifest.toml").read_bytes()
    manifest = tomllib.loads(manifest_bytes.decode())
    version = manifest["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("release version must be major.minor.patch")
    output = output or (root or Path.cwd()) / "build" / "blender_extension"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"tissuelab-{version}.zip"
    init = (src / "__init__.py").read_text()
    # Native extensions use the manifest; keep bl_info in the legacy distribution only.
    lines = init.splitlines(keepends=True)
    for node in ast.parse(init).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bl_info" for t in node.targets):
            lines[node.lineno-1:node.end_lineno] = [""] * (node.end_lineno - node.lineno + 1)
    # Explicit allowlist: never recurse through source, catalog, build, or patient files.
    files = {"__init__.py": "".join(lines).encode(),
             "engine.py": (src / "engine.py").read_bytes(),
             "tissue_pd.py": (package / "tissue_pd.py").read_bytes(),
             "blender_manifest.toml": manifest_bytes,
             "COPYING": (src / "COPYING").read_bytes(),
             "LICENSE-NOTICE.txt": (src / "LICENSE-NOTICE.txt").read_bytes()}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    blob = archive.read_bytes()
    entry = {**manifest, "archive_url": "./" + archive.name, "archive_size": len(blob),
             "archive_hash": "sha256:" + hashlib.sha256(blob).hexdigest()}
    (output / "index.json").write_text(json.dumps({"version": "v1", "blocklist": [], "data": [entry]}, indent=2) + "\n")
    (output / "README.md").write_text(f'''# TissueLab updates

Public distribution of the TissueLab Blender extension. The ZIP contains the add-on
Python source under GPL-3.0-or-later. No patient data, catalog assets or private pipeline
files are distributed here. Source code and tests are available in this repository. This is experimental tissue simulation, not a calibrated model.

## Install once

1. Save your work and disable/remove the old **Surgvid Tissue** add-on, then restart
   Blender. Do not enable both copies: they operate on the same body-wall scene objects.
2. Open **Edit → Preferences → Get Extensions** and allow online access.
3. Open **Repositories → + → Add Remote Repository** and paste:

   `{FEED_URL}`

4. Refresh the repository, find **TissueLab**, and click **Install**.
5. In the TissueLab sidebar, use **Install solver (SciPy)** once if needed. No CT data is uploaded.

## Update later

Use **Get Extensions → Check for Updates**, then install the available update. Enable
**Check for Updates on Startup** for this repository if your Blender version offers it.
Updates are installed through Blender; this feed does not silently replace code mid-session.
Save and stop Live simulation before updating; restart Blender and reload your wall afterward.

Body-wall NPZ files remain separate. Feature changes requiring new geometry (such as tethers)
still require rebuilding the NPZ with SurgVid. Extension updates do not install the CT pipeline.

## Version {version}

The manifest supports Blender 4.2 and later. Package validation uses Blender 5.2.3 tooling;
interactive compatibility has not been tested on every Blender version or platform.
''')
    return archive
