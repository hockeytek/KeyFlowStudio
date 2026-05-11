# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata


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
    bundle_identifier='com.keyflow.studio',
    info_plist={
        'CFBundleShortVersionString': '0.1.0',
        'CFBundleVersion': '0.1.0',
    },
)
