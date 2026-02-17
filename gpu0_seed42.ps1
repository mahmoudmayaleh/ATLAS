# GPU 0 - Seed 42 - GPT-2 XL Experiments
# Run each method separately: atlas, fedavg_cluster, local_only
# Split into 3 sessions: 3 rounds, 4 rounds, 3 rounds

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('1','2','3')]
    [string]$Session = '1',
    
    [Parameter(Mandatory=$false)]
    [ValidateSet('atlas','fedavg_cluster','local_only')]
    [string]$Method = 'atlas'
)

$ErrorActionPreference = "Stop"

# Configuration
$SEED = 42
$MODEL = "gpt2-xl"
$TASKS = "sst2", "mrpc", "cola", "qnli"
$CLIENTS_PER_TASK = 3
$SAMPLES = 1000
$LOCAL_EPOCHS = 2
$BATCH_SIZE = 8
$FP_SAMPLES = 25
$FP_BATCHES = 20

# Session configurations (round ranges)
$SessionConfig = @{
    '1' = @{ Start = 0; Rounds = 3; TotalRounds = 3 }
    '2' = @{ Start = 3; Rounds = 4; TotalRounds = 7 }
    '3' = @{ Start = 7; Rounds = 3; TotalRounds = 10 }
}

$Config = $SessionConfig[$Session]
$CheckpointPath = "checkpoints/atlas_${Method}_seed${SEED}_round_$($Config.Start).pkl"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GPU 0 - Seed $SEED - Method: $Method" -ForegroundColor Cyan
Write-Host "Session $Session - Rounds $($Config.Start + 1) to $($Config.TotalRounds)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$BaseCmd = @(
    "python", "experiments/atlas_integrated.py",
    "--mode", "quick",
    "--ablation", $Method,
    "--model", $MODEL,
    "--tasks"
) + $TASKS + @(
    "--clients-per-task", $CLIENTS_PER_TASK,
    "--rounds", $Config.TotalRounds,
    "--samples", $SAMPLES,
    "--local-epochs", $LOCAL_EPOCHS,
    "--batch-size", $BATCH_SIZE,
    "--fingerprint-samples", $FP_SAMPLES,
    "--fingerprint-batches", $FP_BATCHES,
    "--seed", $SEED
)

# Add resume flag if not first session
if ($Session -ne '1' -and (Test-Path $CheckpointPath)) {
    Write-Host "[RESUME] Loading checkpoint: $CheckpointPath" -ForegroundColor Yellow
    $BaseCmd += @("--resume", $CheckpointPath)
} elseif ($Session -ne '1') {
    Write-Host "[ERROR] Checkpoint not found: $CheckpointPath" -ForegroundColor Red
    Write-Host "Run Session $($Session - 1) first!" -ForegroundColor Red
    exit 1
}

# Run experiment
Write-Host "`n[START] $($BaseCmd -join ' ')`n" -ForegroundColor Green
& $BaseCmd[0] $BaseCmd[1..($BaseCmd.Length-1)]

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Session $Session complete for $Method (seed $SEED)" -ForegroundColor Green
    Write-Host "Results: results/atlas_integrated_quick_${Method}_seed${SEED}.json" -ForegroundColor Green
    
    if ($Session -ne '3') {
        $NextCheckpoint = "checkpoints/atlas_${Method}_seed${SEED}_round_$($Config.TotalRounds).pkl"
        Write-Host "Checkpoint saved: $NextCheckpoint" -ForegroundColor Green
        Write-Host "`nNext: .\gpu0_seed42.ps1 -Session $($Session + 1) -Method $Method" -ForegroundColor Cyan
    } else {
        Write-Host "`n[COMPLETE] All sessions done for $Method!" -ForegroundColor Green
    }
} else {
    Write-Host "`n[FAILED] Session $Session failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
