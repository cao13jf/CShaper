# -*- mode: python ; coding: utf-8 -*-
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

a = Analysis(
    ['CShaperApp.py'],
    pathex=[],
    binaries=[],
    datas=[('CShaperLogo.png', "."), ('/home/jeffery/ProjectCode/CShaperApp/libs/libcublas.so.11', "."), ('/home/jeffery/ProjectCode/CShaperApp/libs/libcublasLt.so.11', "."), ('/home/jeffery/ProjectCode/CShaperApp/libs/libcudnn.so.8', "."), ('/home/jeffery/ProjectCode/CShaperApp/libs/libcufft.so.10', "."), ('/home/jeffery/ProjectCode/CShaperApp/libs/libcusparse.so.11', "."), ('/home/jeffery/ProjectCode/CShaperApp/libs/libcudnn_ops_infer.so.8', "."), ('/home/jeffery/Software/miniconda/envs/cshapertf2/lib/libcudnn_cnn_infer.so.8', "."), ("/home/jeffery/Software/miniconda/envs/cshapertf2/lib/libcudnn_cnn_train.so.8", ".")],
    hiddenimports=['skimage.filters.edges'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CShaperApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['CShaperLogo.png'],
)