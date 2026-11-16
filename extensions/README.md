# TissueLab updates

Public distribution of the TissueLab Blender extension. The ZIP contains the add-on
Python source under GPL-3.0-or-later. No patient data, catalog assets or private pipeline
files are distributed here. Source code and tests are available in this repository. This is experimental tissue simulation, not a calibrated model.

## Install once

1. Save your work and disable/remove the old **Surgvid Tissue** add-on, then restart
   Blender. Do not enable both copies: they operate on the same body-wall scene objects.
2. Open **Edit → Preferences → Get Extensions** and allow online access.
3. Open **Repositories → + → Add Remote Repository** and paste:

   `https://raw.githubusercontent.com/dylanrussellmd/TissueLab/main/extensions/index.json`

4. Refresh the repository, find **TissueLab**, and click **Install**.
5. In the TissueLab sidebar, use **Install solver (SciPy)** once if needed. No CT data is uploaded.

## Update later

Use **Get Extensions → Check for Updates**, then install the available update. Enable
**Check for Updates on Startup** for this repository if your Blender version offers it.
Updates are installed through Blender; this feed does not silently replace code mid-session.
Save and stop Live simulation before updating; restart Blender and reload your wall afterward.

Body-wall NPZ files remain separate. Feature changes requiring new geometry (such as tethers)
still require rebuilding the NPZ with SurgVid. Extension updates do not install the CT pipeline.

## Version 0.2.4

The manifest supports Blender 4.2 and later. Package validation uses Blender 5.2.3 tooling;
interactive compatibility has not been tested on every Blender version or platform.
