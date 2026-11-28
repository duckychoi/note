param(
    [Parameter(Mandatory = $true)] [string] $RunRoot,
    [int] $Seed = 4243,
    [int] $HypersimEpochs = 1,
    [switch] $IncludeSiLog
)

$ErrorActionPreference = 'Stop'
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$PythonPath = 'C:/ProgramData/miniforge3/envs/torch/python.exe'
$env:PYTHONPATH = Join-Path $Workspace 'research-sources/ml-depth-pro-2-deps'
$env:PYTHONIOENCODING = 'utf-8'
$Synthetic = Join-Path $Workspace 'research-sources/hypersim42_pilot'
$Real = Join-Path $Workspace 'research-sources/encoder42_eval/nyu-depthv2-wds'
$Archive = Join-Path $Workspace 'research-sources/encoder42_eval/DIODE.zip'
$Training = Join-Path $PSScriptRoot 'depth_full_train.py'
$Evaluator = Join-Path $PSScriptRoot 'depth_full_eval.py'

function Run-Training([string] $Arm) {
    $ArmRoot = Join-Path $RunRoot $Arm
    $arguments = @('--arm', $Arm, '--seed', $Seed, '--output', $ArmRoot)
    if ($Arm -eq 'real') {
        # Equal numbers of forwards with the two-view synthetic training stage.
        $samples = 2 * 59543 * $HypersimEpochs
        $arguments += @('--epochs', [math]::Ceiling($samples / 47584), '--steps', $samples)
    } else {
        $arguments += @('--epochs', $HypersimEpochs)
    }
    if (Test-Path -LiteralPath (Join-Path $ArmRoot 'latest.train.pt')) { $arguments += '--resume' }
    Write-Output "$(Get-Date -Format o) training $Arm"
    & $PythonPath -u $Training @arguments
    if ($LASTEXITCODE) { throw "Training failed for $Arm with exit $LASTEXITCODE" }
    New-Item -ItemType File -Force -Path (Join-Path $ArmRoot 'completed') | Out-Null
}

function Has-Line([string] $File, [string] $Pattern) {
    (Test-Path -LiteralPath $File) -and (Select-String -LiteralPath $File -Pattern $Pattern -Quiet)
}

$arms = @('real', 'd2', 'fixed', 'kf')
if ($IncludeSiLog) { $arms += 'silog' }
Write-Output "$(Get-Date -Format o) waiting for the verified download manifests"
while ($true) {
    $pending = @($arms | Where-Object { -not (Test-Path (Join-Path $RunRoot "$_/completed")) })
    if (-not $pending.Count) { break }
    $nyuShards = @(Get-ChildItem -LiteralPath $Real -File -Filter 'nyu-depth-train-*.tar' -ErrorAction SilentlyContinue)
    $nyuReady = $nyuShards.Count -eq 48 -and (Test-Path (Join-Path $Real 'nyu-depth-val-00000.tar'))
    $trainManifest = Join-Path $Synthetic 'train_full_acquisition.log'
    $trainReady = (Has-Line $trainManifest '^train_full: 59543/59543 RGB-depth pairs') -and (Has-Line $trainManifest '^train_full: all selected camera poses in 365/365 scenes$')
    $validationReady = Has-Line (Join-Path $Synthetic 'validation_acquisition.log') '^validation_24: 1104/1104 RGB-depth pairs'
    foreach ($arm in $pending) {
        if (($arm -eq 'real' -and $nyuReady) -or ($arm -ne 'real' -and $trainReady -and $validationReady)) {
            Run-Training $arm
        }
    }
    Start-Sleep -Seconds 45
}

while (-not (Test-Path -LiteralPath $Archive)) { Start-Sleep -Seconds 45 }
$scoreName = if ($IncludeSiLog) { 'diode_indoor_with_silog.json' } else { 'diode_indoor.json' }
$evalArms = @('frozen') + $arms
& $PythonPath -u $Evaluator --archive $Archive --runs $RunRoot --output (Join-Path $RunRoot $scoreName) --indoor --arms @evalArms
if ($LASTEXITCODE) { throw "Evaluation failed with exit $LASTEXITCODE" }
Write-Output "$(Get-Date -Format o) completed the depth comparison"
