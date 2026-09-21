param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & uv --cache-dir .uv-cache venv .venv --python $Python
    if ($LASTEXITCODE -ne 0) { throw '创建环境失败' }
}
& uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw '安装 CUDA PyTorch 失败' }
& uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw '安装训练依赖失败' }
& uv --cache-dir .uv-cache pip freeze --python .venv\Scripts\python.exe | Set-Content -Encoding utf8 requirements.lock.txt
& .\.venv\Scripts\python.exe -m scripts.check_environment
if ($LASTEXITCODE -ne 0) { throw '环境检查失败，请查看输出' }
