#requires -version 5.1
param(
    [string]$Python = '',
    [string]$IndexUrl = 'https://mirrors.aliyun.com/pypi/simple/',
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Torch = 'auto'
)
& (Join-Path $PSScriptRoot 'bootstrap.ps1') -NoLaunch -Python $Python -IndexUrl $IndexUrl -Torch $Torch
exit $LASTEXITCODE
