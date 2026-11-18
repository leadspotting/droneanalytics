# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[('dist', 'dist'), ('Drone_Forensics_Dataset.csv', '.')],
    hiddenimports=['api', 'backend.main', 'backend.routers.missions', 'backend.routers.planner', 'backend.routers.drones', 'backend.routers.image_missions', 'backend.routers.task4', 'uvicorn', 'fastapi', 'pydantic', 'starlette', 'email_validator', 'multipart', 'PIL'],
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
    name='DroneOversight',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DroneOversight',
)
