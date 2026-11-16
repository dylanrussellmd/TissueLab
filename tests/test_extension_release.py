import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from tissuelab.extension_release import build_extension

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_is_allowlisted_deterministic_and_checksummed(tmp_path):
    src = tmp_path / 'tissuelab' / 'blender_addon'
    src.mkdir(parents=True)
    for name in ('__init__.py', 'engine.py', 'blender_manifest.toml', 'COPYING', 'LICENSE-NOTICE.txt'):
        shutil.copyfile(ROOT / 'tissuelab' / 'blender_addon' / name, src / name)
    shutil.copyfile(ROOT / 'tissuelab' / 'tissue_pd.py', tmp_path / 'tissuelab' / 'tissue_pd.py')
    (src / '.env').write_text('PRIVATE_SENTINEL')
    (src / 'patient.npz').write_bytes(b'PRIVATE_SENTINEL')
    archive = build_extension(tmp_path)
    first = archive.read_bytes()
    assert build_extension(tmp_path).read_bytes() == first
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {'__init__.py', 'engine.py', 'tissue_pd.py',
                                     'blender_manifest.toml', 'COPYING', 'LICENSE-NOTICE.txt'}
        assert b'bl_info =' not in z.read('__init__.py')
        assert all(b'PRIVATE_SENTINEL' not in z.read(n) for n in z.namelist())
        compile(z.read('__init__.py'), '__init__.py', 'exec')
    data = json.loads((archive.parent / 'index.json').read_text())['data'][0]
    assert data['archive_size'] == len(first)
    assert data['archive_hash'] == 'sha256:' + hashlib.sha256(first).hexdigest()
    assert data['archive_url'] == './' + archive.name


def test_publisher_rejects_version_replacement_and_downgrade():
    spec = importlib.util.spec_from_file_location('publisher', ROOT / 'scripts' / 'publish_extension.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = {'data': [{'id': 'tissuelab_tissue', 'version': '0.2.0', 'archive_hash': 'sha256:old'}]}
    module.validate_release(old, old)
    for version, digest in [('0.1.0', 'sha256:new'), ('0.2.0', 'sha256:new')]:
        with pytest.raises(ValueError):
            module.validate_release(old, {'data': [{'id': 'tissuelab_tissue', 'version': version, 'archive_hash': digest}]})


def test_solver_download_respects_blender_offline_mode(monkeypatch):
    from types import SimpleNamespace
    from tissuelab import blender_addon
    monkeypatch.setattr(blender_addon, 'bpy', SimpleNamespace(app=SimpleNamespace(online_access=False)))
    with pytest.raises(RuntimeError, match='Allow Online Access'):
        blender_addon.install_scipy('/unused')
