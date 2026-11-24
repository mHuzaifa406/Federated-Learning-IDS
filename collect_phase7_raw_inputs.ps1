$ErrorActionPreference = 'Stop'

# Place this script in E:\IoMT_FL_IDS_TRFA and run it from PowerShell or CMD.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResultsRoot = Join-Path $ProjectRoot 'results'
$FusionFolders = @(Get-ChildItem -LiteralPath $ResultsRoot -Directory -Filter 'phase5_fusion_*' | Sort-Object Name)

if ($FusionFolders.Count -ne 12) {
    throw "Expected 12 phase5_fusion_* folders, but found $($FusionFolders.Count). No ZIP was created."
}

$ZipPath = Join-Path $ProjectRoot 'phase7_raw_round_inputs.zip'
if (Test-Path -LiteralPath $ZipPath) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $ZipPath = Join-Path $ProjectRoot "phase7_raw_round_inputs_$stamp.zip"
}

$StageRoot = Join-Path $env:TEMP ("trfa_phase7_raw_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $StageRoot | Out-Null
$CopiedCount = 0
$Missing = @()

try {
    foreach ($Folder in $FusionFolders) {
        $MetaPath = Join-Path $Folder.FullName 'phase5_fusion_metadata.json'
        if (!(Test-Path -LiteralPath $MetaPath)) {
            $Missing += "Metadata missing: $MetaPath"
            continue
        }

        $Meta = Get-Content -LiteralPath $MetaPath -Raw | ConvertFrom-Json
        $Inputs = $Meta.input_files
        if ($null -eq $Inputs) {
            $Missing += "input_files section missing in: $MetaPath"
            continue
        }

        $CaseDir = Join-Path $StageRoot $Folder.Name
        New-Item -ItemType Directory -Path $CaseDir | Out-Null
        Copy-Item -LiteralPath $MetaPath -Destination $CaseDir
        $CopiedCount++

        $RequiredInputs = @(
            @{ Key = 'historical_clean'; Name = 'historical_clean_rounds.csv' },
            @{ Key = 'historical_poisoned'; Name = 'historical_poisoned_rounds.csv' },
            @{ Key = 'similarity_clean'; Name = 'similarity_clean_rounds.csv' },
            @{ Key = 'similarity_poisoned'; Name = 'similarity_poisoned_rounds.csv' },
            @{ Key = 'cavc_v2_client_summary_min_support_100'; Name = 'cavc_client_summary_min_support_100.csv' }
        )

        foreach ($Item in $RequiredInputs) {
            $RawPath = [string]$Inputs.($Item.Key)
            if ([string]::IsNullOrWhiteSpace($RawPath)) {
                $Missing += "Input '$($Item.Key)' is empty in $MetaPath"
                continue
            }

            if ([System.IO.Path]::IsPathRooted($RawPath)) {
                $SourcePath = $RawPath
            } else {
                $SourcePath = Join-Path $ProjectRoot $RawPath
            }

            if (!(Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
                $Missing += "Input file missing: $SourcePath"
                continue
            }

            Copy-Item -LiteralPath $SourcePath -Destination (Join-Path $CaseDir $Item.Name)
            $CopiedCount++

            # For CAVC, also include the other CSV/JSON diagnostics from the same folder.
            if ($Item.Key -eq 'cavc_v2_client_summary_min_support_100') {
                $CavcDir = Split-Path -Parent $SourcePath
                $CavcExtraDir = Join-Path $CaseDir 'cavc_additional_diagnostics'
                New-Item -ItemType Directory -Path $CavcExtraDir | Out-Null
                Get-ChildItem -LiteralPath $CavcDir -File |
                    Where-Object { $_.Extension -in '.csv', '.json' -and $_.FullName -ne $SourcePath } |
                    ForEach-Object {
                        Copy-Item -LiteralPath $_.FullName -Destination $CavcExtraDir
                        $script:CopiedCount++
                    }
            }
        }
    }

    $PartitionMeta = Join-Path $ProjectRoot 'data\processed\federated_noniid_alpha05_seed43\metadata.json'
    if (Test-Path -LiteralPath $PartitionMeta -PathType Leaf) {
        Copy-Item -LiteralPath $PartitionMeta -Destination (Join-Path $StageRoot 'partition_metadata_seed43.json')
        $CopiedCount++
    }

    if ($Missing.Count -gt 0) {
        Write-Host ''
        Write-Host 'COLLECTION STOPPED. Some required inputs were missing:' -ForegroundColor Red
        $Missing | ForEach-Object { Write-Host (" - " + $_) }
        throw 'No ZIP was created because one or more referenced experiment files could not be found.'
    }

    Compress-Archive -Path (Join-Path $StageRoot '*') -DestinationPath $ZipPath -CompressionLevel Optimal
    $ZipInfo = Get-Item -LiteralPath $ZipPath
    Write-Host ''
    Write-Host 'PHASE 7 INPUT ZIP CREATED' -ForegroundColor Green
    Write-Host ("File: " + $ZipInfo.FullName)
    Write-Host ("Copied files: " + $CopiedCount)
    Write-Host ("ZIP size (MB): " + [math]::Round($ZipInfo.Length / 1MB, 2))
}
finally {
    if (Test-Path -LiteralPath $StageRoot) {
        Remove-Item -LiteralPath $StageRoot -Recurse -Force
    }
}
