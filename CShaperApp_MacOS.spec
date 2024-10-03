import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
block_cipher = None


a = Analysis(
    ['CShaperApp.py'],
    pathex=['/opt/miniconda3/envs/cshapertf/lib', '/opt/miniconda3/envs/cshapertf/lib/python3.9/site-packages'],
    binaries=[],
    datas=[("CShaperLogo.ico", ".")],
    hiddenimports=["skimage.filters.edges"],
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
    [],
    exclude_binaries=True,
    name='CShaperApp',
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
    icon=['CShaperLogo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CShaperApp',
)
app = BUNDLE(
    coll,
    name='CShaperApp.app',
    icon='CShaperLogo.ico',
    bundle_identifier=None,
)
