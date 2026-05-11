# -*- mode: python ; coding: utf-8 -*-
import plistlib
import subprocess
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata


APP_BUNDLE_ID = 'com.keyflow.studio'
APP_VERSION = '0.1.0'


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('app/assets', 'app/assets'), ('docs/cloud_aws_setup.ru.html', 'docs'), ('docs/cloud_aws_setup.en.html', 'docs')]
        + collect_data_files('imageio_ffmpeg')
        + collect_data_files('matanyone2')
        + collect_data_files('certifi')
        + copy_metadata('imageio')
        + copy_metadata('imageio-ffmpeg'),
    hiddenimports=['UI.main_ui', 'app.i18n', 'certifi'] + collect_submodules('segment_anything'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    exclude_binaries=False,
    name='KeyFlow Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    name='KeyFlow Studio.app',
    icon='MatAnyone2.icns',
    bundle_identifier=APP_BUNDLE_ID,
)

bundle_path = Path(DISTPATH) / 'KeyFlow Studio.app'
info_plist_path = bundle_path / 'Contents' / 'Info.plist'
if info_plist_path.exists():
    plist = plistlib.loads(info_plist_path.read_bytes())
    plist['CFBundleIdentifier'] = APP_BUNDLE_ID
    plist['CFBundleShortVersionString'] = APP_VERSION
    plist['CFBundleVersion'] = APP_VERSION
    info_plist_path.write_bytes(plistlib.dumps(plist))
    subprocess.run(['codesign', '--force', '--deep', '--sign', '-', str(bundle_path)], check=False)
