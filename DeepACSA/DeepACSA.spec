# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['deep_acsa_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('gui_helpers/deep_acsa.jpg', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/ui_color_theme.json', 'gui_helpers/gui_files'), ('gui_helpers/icon.ico', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/gear.png', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/Cite.png', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/Info.png', 'gui_helpers/gui_files')],
    hiddenimports=['customtkinter'],
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
    [],
    exclude_binaries=True,
    name='deep_acsa_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['gui_helpers\\icon.ico'],
    onefile=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DeepACSA',
)