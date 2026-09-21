param([string]$Python = 'python', [string]$IndexUrl = 'https://mirrors.aliyun.com/pypi/simple/')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw '创建环境失败' }
}
& .\.venv\Scripts\python.exe -c "import sys; assert sys.version_info[:2] == (3, 11), '本配置需要 Python 3.11 x64'"
if ($LASTEXITCODE -ne 0) { throw 'Python 版本不匹配，请指定 Python 3.11' }
& .\.venv\Scripts\python.exe -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw '初始化 pip 失败' }
& .\.venv\Scripts\python.exe -m scripts.download_torch
if ($LASTEXITCODE -ne 0) { throw '下载或校验 PyTorch 失败' }
& .\.venv\Scripts\python.exe -m scripts.install_dependencies '.uv-cache/downloads/torch-2.10.0+cu128-cp311-cp311-win_amd64.whl' --index-url $IndexUrl
if ($LASTEXITCODE -ne 0) { throw '安装 CUDA PyTorch 失败' }
& .\.venv\Scripts\python.exe -m scripts.install_dependencies -r requirements.txt --index-url $IndexUrl
if ($LASTEXITCODE -ne 0) { throw '安装训练依赖失败' }
& .\.venv\Scripts\python.exe -m pip freeze | ForEach-Object { $_ -replace '^torch @ .+$', 'torch==2.10.0+cu128' } | Set-Content -Encoding utf8 requirements.lock.txt
& .\.venv\Scripts\python.exe -m scripts.check_environment
if ($LASTEXITCODE -ne 0) { throw '环境检查失败，请查看输出' }
