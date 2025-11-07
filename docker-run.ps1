# PowerShell helper script for Docker commands

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("build", "prepare", "train", "evaluate", "visualize", "inference")]
    [string]$Command,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Build-Image {
    Write-Host "Building Docker image..." -ForegroundColor Green
    docker build -t vit-deepfake:latest .
}

function Prepare-Data {
    Write-Host "Running prepare_data.py..." -ForegroundColor Green
    docker run --rm `
        -v "${PWD}/data:/app/data" `
        vit-deepfake:latest `
        python scripts/prepare_data.py --train_ratio 0.9
}

function Train-Model {
    Write-Host "Running training..." -ForegroundColor Green
    $dockerArgs = $Args -join " "
    docker run --rm --gpus all `
        -v "${PWD}/data:/app/data" `
        -v "${PWD}/models:/app/models" `
        -v "${PWD}/results:/app/results" `
        vit-deepfake:latest `
        python scripts/train.py $dockerArgs
}

function Evaluate-Model {
    Write-Host "Running evaluation..." -ForegroundColor Green
    $dockerArgs = $Args -join " "
    docker run --rm --gpus all `
        -v "${PWD}/data:/app/data" `
        -v "${PWD}/models:/app/models" `
        -v "${PWD}/json:/app/json" `
        vit-deepfake:latest `
        python scripts/evaluate.py $dockerArgs
}

function Visualize-Training {
    Write-Host "Running visualization..." -ForegroundColor Green
    $dockerArgs = $Args -join " "
    docker run --rm `
        -v "${PWD}/results:/app/results" `
        -v "${PWD}/plots:/app/plots" `
        vit-deepfake:latest `
        python scripts/visualize_training.py $dockerArgs
}

function Run-Inference {
    Write-Host "Running inference..." -ForegroundColor Green
    $dockerArgs = $Args -join " "
    docker run --rm --gpus all `
        -v "${PWD}/data:/app/data" `
        -v "${PWD}/models:/app/models" `
        vit-deepfake:latest `
        python scripts/inference.py $dockerArgs
}

function Show-Usage {
    Write-Host "Usage: .\docker-run.ps1 <command> [args...]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Cyan
    Write-Host "  build              - Build Docker image"
    Write-Host "  prepare            - Run prepare_data.py"
    Write-Host "  train [args...]    - Run training (pass train.py args)"
    Write-Host "  evaluate [args...] - Run evaluation (pass evaluate.py args)"
    Write-Host "  visualize [args...]- Run visualization (pass visualize_training.py args)"
    Write-Host "  inference [args...]- Run inference (pass inference.py args)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\docker-run.ps1 build"
    Write-Host "  .\docker-run.ps1 prepare"
    Write-Host "  .\docker-run.ps1 train --pretrained --epochs 15 --batch_size 16"
    Write-Host "  .\docker-run.ps1 evaluate --model_name vit_base_patch16_224"
    Write-Host "  .\docker-run.ps1 visualize --model_name vit_base_patch16_224"
}

# Main
switch ($Command) {
    "build" {
        Build-Image
    }
    "prepare" {
        Prepare-Data
    }
    "train" {
        Train-Model
    }
    "evaluate" {
        Evaluate-Model
    }
    "visualize" {
        Visualize-Training
    }
    "inference" {
        Run-Inference
    }
    default {
        Show-Usage
        exit 1
    }
}

