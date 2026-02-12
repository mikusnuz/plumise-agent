# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for plumise-agent Windows exe build.

Usage:
    pyinstaller plumise-agent.spec
"""
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules
import os

block_cipher = None

# Collect all dependencies for large packages
datas = []
binaries = []
hiddenimports = []

# Transformers
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')
datas += transformers_datas
binaries += transformers_binaries
hiddenimports += transformers_hiddenimports

# Torch
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
datas += torch_datas
binaries += torch_binaries
hiddenimports += torch_hiddenimports

# Tokenizers
tokenizers_datas, tokenizers_binaries, tokenizers_hiddenimports = collect_all('tokenizers')
datas += tokenizers_datas
binaries += tokenizers_binaries
hiddenimports += tokenizers_hiddenimports

# Accelerate
accelerate_datas, accelerate_binaries, accelerate_hiddenimports = collect_all('accelerate')
datas += accelerate_datas
binaries += accelerate_binaries
hiddenimports += accelerate_hiddenimports

# Safetensors
safetensors_datas, safetensors_binaries, safetensors_hiddenimports = collect_all('safetensors')
datas += safetensors_datas
binaries += safetensors_binaries
hiddenimports += safetensors_hiddenimports

# Sentencepiece
sentencepiece_datas, sentencepiece_binaries, sentencepiece_hiddenimports = collect_all('sentencepiece')
datas += sentencepiece_datas
binaries += sentencepiece_binaries
hiddenimports += sentencepiece_hiddenimports

# gRPC
grpcio_datas, grpcio_binaries, grpcio_hiddenimports = collect_all('grpcio')
datas += grpcio_datas
binaries += grpcio_binaries
hiddenimports += grpcio_hiddenimports

# Web3 and eth_account
web3_datas, web3_binaries, web3_hiddenimports = collect_all('web3')
datas += web3_datas
binaries += web3_binaries
hiddenimports += web3_hiddenimports

eth_account_datas, eth_account_binaries, eth_account_hiddenimports = collect_all('eth_account')
datas += eth_account_datas
binaries += eth_account_binaries
hiddenimports += eth_account_hiddenimports

# FastAPI and uvicorn
fastapi_datas, fastapi_binaries, fastapi_hiddenimports = collect_all('fastapi')
datas += fastapi_datas
binaries += fastapi_binaries
hiddenimports += fastapi_hiddenimports

uvicorn_datas, uvicorn_binaries, uvicorn_hiddenimports = collect_all('uvicorn')
datas += uvicorn_datas
binaries += uvicorn_binaries
hiddenimports += uvicorn_hiddenimports

# Pydantic
pydantic_datas, pydantic_binaries, pydantic_hiddenimports = collect_all('pydantic')
datas += pydantic_datas
binaries += pydantic_binaries
hiddenimports += pydantic_hiddenimports

# Project-specific data files
datas += [
    ('contracts/AgentRegistry.json', 'contracts'),
    ('contracts/RewardPool.json', 'contracts'),
    ('proto/inference.proto', 'proto'),
]

# Additional hidden imports
hiddenimports += [
    # gRPC generated
    'plumise_agent.grpc_.generated.inference_pb2',
    'plumise_agent.grpc_.generated.inference_pb2_grpc',

    # Plumise agent modules
    'plumise_agent.api',
    'plumise_agent.chain',
    'plumise_agent.cli',
    'plumise_agent.grpc_',
    'plumise_agent.model',
    'plumise_agent.node',
    'plumise_agent.pipeline',

    # Torch internals
    'torch._C',
    'torch._dynamo',
    'torch.distributed',

    # Transformers internals
    'transformers.models',
    'transformers.modeling_utils',
    'transformers.tokenization_utils_base',
    'transformers.configuration_utils',

    # Click
    'click',

    # Dotenv
    'dotenv',

    # Aiohttp
    'aiohttp',
    'aiohttp.web',

    # Protobuf
    'google.protobuf',

    # Numpy
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',

    # Other eth dependencies
    'eth_utils',
    'eth_keys',
    'eth_typing',
    'eth_abi',
    'eth_hash',
    'rlp',
    'cytoolz',
    'toolz',

    # HTTP/SSL
    'ssl',
    'certifi',

    # Uvicorn workers
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',

    # Pydantic internals
    'pydantic_core',
    'pydantic.deprecated',

    # JSON
    'json',

    # Asyncio
    'asyncio',
]

a = Analysis(
    ['src/plumise_agent/cli/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'PIL',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'test',
        'tests',
    ],
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
    name='plumise-agent',
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
    icon=None,
)
