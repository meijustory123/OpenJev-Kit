#requires -version 5.1
param(
    [switch]$NoLaunch,
    [string]$Python = '',
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Torch = 'auto',
    [string]$IndexUrl = 'https://mirrors.aliyun.com/pypi/simple/',
    [switch]$IgnoreSystemPython
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$script:BootstrapErrorDisplayed = $false

function Assert-ProjectPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($ProjectRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "路径不在项目目录内：$absolute"
    }
    $cursor = $absolute
    while ($cursor -and $cursor -ne $ProjectRoot) {
        if ((Test-Path -LiteralPath $cursor) -and
            ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "安装目录不能使用符号链接或目录联接：$cursor"
        }
        $cursor = Split-Path $cursor -Parent
    }
    return $absolute
}

function Backup-ProjectPath([string]$Path) {
    $source = Assert-ProjectPath $Path
    if (Test-Path -LiteralPath $source) {
        $destination = Assert-ProjectPath ($source + '.backup-' + [Guid]::NewGuid().ToString('N'))
        Move-Item -LiteralPath $source -Destination $destination
        Write-Host "原目录已保留为：$destination"
    }
}

function Get-ArchiveHash([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}

function Find-CompatiblePython([string]$Executable, [string[]]$Prefix = @()) {
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $null }
    if ($Executable -match '\\WindowsApps\\') { return $null }
    $probe = "import sys,struct,json; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 and sys.implementation.name=='cpython'; print(json.dumps(sys.executable,ensure_ascii=True))"
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & $Executable @Prefix -I -c $probe 2>$null
        if ($LASTEXITCODE -eq 0 -and $output) {
            $path = ([string](@($output)[-1])).Trim() | ConvertFrom-Json
            if (Test-Path -LiteralPath $path -PathType Leaf) { return $path }
        }
    } catch { } finally { $ErrorActionPreference = $oldPreference }
    return $null
}

function Install-ProjectPython {
    $config = (Get-Content -LiteralPath (Join-Path $ProjectRoot 'configs/bootstrap.json') -Raw -Encoding UTF8 | ConvertFrom-Json).python
    $destination = Assert-ProjectPath (Join-Path $ProjectRoot ('.runtime/python-' + $config.version))
    $executable = Join-Path $destination 'python/python.exe'
    $found = Find-CompatiblePython $executable
    if ($found) { return $found }
    $tar = Get-Command tar.exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $tar) { throw '系统缺少 tar.exe。请使用带有系统解压工具的 Windows 10/11 x64。' }
    $downloads = Assert-ProjectPath (Join-Path $ProjectRoot '.runtime/downloads')
    New-Item -ItemType Directory -Path $downloads -Force | Out-Null
    $archive = Join-Path $downloads ('python-' + $config.version + '-' + $config.build + '.tar.gz')
    $verified = (Test-Path -LiteralPath $archive) -and ((Get-ArchiveHash $archive) -eq $config.sha256)
    if (-not $verified) {
        Write-Host "[2/5] 正在下载 Python $($config.version)（约 25 MB），请稍候……"
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $temporary = $archive + '.download'
        $downloaded = $false
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            try {
                $oldProgress = $ProgressPreference
                $ProgressPreference = 'SilentlyContinue'
                Invoke-WebRequest -UseBasicParsing -Uri $config.url -OutFile $temporary -TimeoutSec 300
                $downloaded = $true
                break
            } catch {
                if ($attempt -eq 3) { throw }
                Write-Host "下载中断，正在重试（$attempt/3）……"
                Start-Sleep -Seconds 2
            } finally { $ProgressPreference = $oldProgress }
        }
        if (-not $downloaded -or (Get-Item -LiteralPath $temporary).Length -ne $config.size -or
            (Get-ArchiveHash $temporary) -ne $config.sha256) {
            throw 'Python 下载文件校验失败，未解压或执行。请重新双击启动器。'
        }
        Move-Item -LiteralPath (Assert-ProjectPath $temporary) -Destination (Assert-ProjectPath $archive) -Force
    }
    Write-Host '[2/5] Python 下载已校验，正在安装到项目目录……'
    $stage = Assert-ProjectPath (Join-Path $ProjectRoot ('.runtime/python-stage-' + [Guid]::NewGuid().ToString('N')))
    New-Item -ItemType Directory -Path $stage | Out-Null
    & $tar.Source -xzf $archive -C $stage
    if ($LASTEXITCODE -ne 0) { throw 'Python 解压失败，请检查磁盘空间后重新启动。' }
    if (-not (Find-CompatiblePython (Join-Path $stage 'python/python.exe'))) { throw '新安装的 Python 未通过运行检查。' }
    Backup-ProjectPath $destination
    Move-Item -LiteralPath (Assert-ProjectPath $stage) -Destination (Assert-ProjectPath $destination)
    return $executable
}

function Ensure-ProjectEnvironment {
    Write-Host '[1/5] 正在检查 Python 3.11 与项目环境……'
    $environmentDir = Assert-ProjectPath (Join-Path $ProjectRoot '.venv')
    $environmentPython = Join-Path $environmentDir 'Scripts/python.exe'
    if ((Find-CompatiblePython $environmentPython) -and (Test-Path -LiteralPath (Join-Path $environmentDir 'Scripts/pythonw.exe'))) {
        Write-Host '[2/5] 已找到可用的项目 Python 3.11，继续检查依赖。'
        return $environmentPython
    }
    $basePython = $null
    if ($Python) {
        $command = Get-Command $Python -CommandType Application -ErrorAction SilentlyContinue
        if ($command) { $basePython = Find-CompatiblePython $command.Source }
        if (-not $basePython) { throw '指定的 Python 不是可运行的 Python 3.11 x64。' }
    } elseif (-not $IgnoreSystemPython) {
        $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue
        if ($launcher) { $basePython = Find-CompatiblePython $launcher.Source @('-3.11') }
        if (-not $basePython) {
            foreach ($name in @('python.exe', 'python3.11.exe')) {
                $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue
                if ($command) { $basePython = Find-CompatiblePython $command.Source }
                if ($basePython) { break }
            }
        }
    }
    if (-not $basePython) { $basePython = Install-ProjectPython }
    Write-Host '[3/5] 正在创建独立项目环境……'
    Backup-ProjectPath $environmentDir
    & $basePython -I -m venv $environmentDir
    if ($LASTEXITCODE -ne 0 -or -not (Find-CompatiblePython $environmentPython)) { throw '创建项目环境失败。' }
    return $environmentPython
}

function Start-OpenJevBootstrap {
    if (-not [Environment]::Is64BitOperatingSystem -or
        (($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') -and ($env:PROCESSOR_ARCHITEW6432 -ne 'AMD64'))) {
        throw '当前自动安装器支持 Windows 10/11 x64。'
    }
    Set-Location -LiteralPath $ProjectRoot
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONNOUSERSITE = '1'
    $env:HF_HUB_OFFLINE = '1'
    $env:TOKENIZERS_PARALLELISM = 'false'
    [Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
    $runtime = Assert-ProjectPath (Join-Path $ProjectRoot '.runtime')
    $logs = Assert-ProjectPath (Join-Path $ProjectRoot 'outputs/setup')
    New-Item -ItemType Directory -Path $runtime, $logs -Force | Out-Null
    $lock = $null
    $transcript = $false
    try {
        $announced = $false
        while (-not $lock) {
            try { $lock = [IO.File]::Open((Join-Path $runtime 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None') }
            catch [IO.IOException] {
                if (-not $announced) { Write-Host '另一个启动器正在准备环境，等待它完成……'; $announced = $true }
                Start-Sleep -Seconds 2
            }
        }
        $log = Join-Path $logs ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + $PID + '.log')
        Start-Transcript -Path $log | Out-Null
        $transcript = $true
        $executable = Ensure-ProjectEnvironment
        Write-Host '[4/5] 正在检查并安装缺失依赖，首次安装可能需要较长时间……'
        & $executable -m scripts.bootstrap_environment --torch $Torch --index-url $IndexUrl
        if ($LASTEXITCODE -ne 0) { throw '依赖安装或检查未完成。请查看上方错误；再次双击可重试。' }
        Write-Host '[5/5] 环境已就绪。'
        if (-not $NoLaunch) {
            Write-Host '正在启动模型网页……'
            Start-Process -FilePath (Join-Path $ProjectRoot '.venv/Scripts/pythonw.exe') -ArgumentList @('-m', 'scripts.launch_web') -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null
        }
    } catch {
        $script:BootstrapErrorDisplayed = $true
        Write-Host ("准备失败：" + $_.Exception.Message) -ForegroundColor Red
        Write-Host '日志目录：outputs\setup。联网后再次双击可继续；模型文件不会被改动。'
        throw
    } finally {
        if ($transcript) { Stop-Transcript | Out-Null }
        if ($lock) { $lock.Dispose() }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    try { Start-OpenJevBootstrap; exit 0 } catch {
        if (-not $script:BootstrapErrorDisplayed) {
            Write-Host ("启动准备失败：" + $_.Exception.Message) -ForegroundColor Red
        }
        exit 1
    }
}
