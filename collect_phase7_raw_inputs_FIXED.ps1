$ErrorActionPreference = 'Stop'

# Collects the raw, round-level Phase 5/6 inputs for Phase 7.
# This version reads the actual metadata schema: .inputs
# It copies files only; original experiment outputs are never modified.

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResultsRoot = Join-Path $ProjectRoot 'results'
$FusionFolders = @(Get-ChildItem -LiteralPath $ResultsRoot -Directory -Filter 'phase5_fusion_*' | Sort-Object Name)

if ($FusionFolders.Count -ne 12) {
    Write-Host "Expected 12 phase5_fusion_* folders, but found $($FusionFolders.Count)." -ForegroundColor Red
    exit 1
}

$ZipPath = Join-Path $ProjectRoot 'phase7_raw_round_inputs.zip'
if (Test-Path -LiteralPath $ZipPath) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $ZipPath = Join-Path $ProjectRoot "phase7_raw_round_inputs_$stamp.zip"
}

$StageRoot = Join-Path $env:TEMP ("trfa_phase7_raw_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $StageRoot | Out-Null
$CopiedCount = 0
$Missing = New-Object System.Collections.Generic.List[string]

try {
    foreach ($Folder in $FusionFolders) {
        Write-Host ("Collecting: " + $Folder.Name)
        $MetaPath = Join-Path $Folder.FullName 'phase5_fusion_metadata.json'
        if (!(Test-Path -LiteralPath $MetaPath -PathType Leaf)) {
            $Missing.Add("Metadata missing: $MetaPath")
            continue
        }

        try {
            $Meta = Get-Content -LiteralPath $MetaPath -Raw | ConvertFrom-Json
        } catch {
            $Missing.Add("Could not parse metadata JSON: $MetaPath; $($_.Exception.Message)")
            continue
        }

        # IMPORTANT: Actual Phase 5 metadata stores paths under 'inputs'.
        $Inputs = $Meta.inputs
        if ($null -eq $Inputs) {
            $Missing.Add("'inputs' section missing in: $MetaPath")
            continue
        }

        $CaseDir = Join-Path $StageRoot $Folder.Name
        New-Item -ItemType Directory -Path $CaseDir | Out-Null
        Copy-Item -LiteralPath $MetaPath -Destination $CaseDir
        $CopiedCount++

        $RequiredInputs = @(
            @{ Key = 'hist_clean'; Name = 'historical_clean_rounds.csv' },
            @{ Key = 'hist_poison'; Name = 'historical_poison_rounds.csv' },
            @{ Key = 'sim_clean'; Name = 'similarity_clean_rounds.csv' },
            @{ Key = 'sim_poison'; Name = 'similarity_poison_rounds.csv' },
            @{ Key = 'cavc'; Name = 'cavc_client_summary_min_support_100.csv' }
        )

        $CavcSourcePath = $null
        foreach ($Item in $RequiredInputs) {
            $RawPath = [string]$Inputs.($Item.Key)
            if ([string]::IsNullOrWhiteSpace($RawPath)) {
                $Missing.Add("Input '$($Item.Key)' is empty in: $MetaPath")
                continue
            }

            if ([System.IO.Path]::IsPathRooted($RawPath)) {
                $SourcePath = $RawPath
            } else {
                $SourcePath = Join-Path $ProjectRoot $RawPath
            }

            if (!(Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
                $Missing.Add("Input file missing for '$($Item.Key)': $SourcePath")
                continue
            }

            if ($Item.Key -eq 'cavc') {
                $CavcSourcePath = $SourcePath
                continue
            }

            Copy-Item -LiteralPath $SourcePath -Destination (Join-Path $CaseDir $Item.Name)
            $CopiedCount++
        }

        # Include all CAVC CSV/JSON diagnostics from the same output folder.
        if ($null -ne $CavcSourcePath) {
            $CavcDir = Split-Path -Parent $CavcSourcePath
            $CavcDest = Join-Path $CaseDir 'cavc_diagnostics'
            New-Item -ItemType Directory -Path $CavcDest | Out-Null
            $CavcFiles = @(Get-ChildItem -LiteralPath $CavcDir -File | Where-Object { $_.Extension -in '.csv', '.json' })
            if ($CavcFiles.Count -eq 0) {
                $Missing.Add("No CAVC CSV/JSON diagnostics found in: $CavcDir")
            } else {
                foreach ($CavcFile in $CavcFiles) {
                    Copy-Item -LiteralPath $CavcFile.FullName -Destination $CavcDest
                    $CopiedCount++
                }
                if (!(Test-Path -LiteralPath (Join-Path $CavcDest 'cavc_v2_client_summary_min_support_100.csv'))) {
                    # The metadata may point to a summary with a different filename; copy it explicitly.
                    $summaryCopy = Join-Path $CavcDest 'cavc_client_summary_min_support_100.csv'
                    if (!(Test-Path -LiteralPath $summaryCopy)) {
                        Copy-Item -LiteralPath $CavcSourcePath -Destination $summaryCopy
                        $CopiedCount++
                    }
                }
            }
        }
    }

    $PartitionMeta = Join-Path $ProjectRoot 'data\processed\federated_noniid_alpha05_seed43\metadata.json'
    if (Test-Path -LiteralPath $PartitionMeta -PathType Leaf) {
        Copy-Item -LiteralPath $PartitionMeta -Destination (Join-Path $StageRoot 'partition_metadata_seed43.json')
        $CopiedCount++
    } else {
        $Missing.Add("Non-IID partition metadata missing: $PartitionMeta")
    }

    if ($Missing.Count -gt 0) {
        Write-Host ''
        Write-Host 'COLLECTION STOPPED. These inputs need attention:' -ForegroundColor Red
        foreach ($Item in $Missing) { Write-Host (" - " + $Item) }
        Write-Host 'No ZIP was created. Original results were not changed.' -ForegroundColor Yellow
        exit 1
    }

    Compress-Archive -Path (Join-Path $StageRoot '*') -DestinationPath $ZipPath -CompressionLevel Optimal
    $ZipInfo = Get-Item -LiteralPath $ZipPath
    $ActualFileCount = @(Get-ChildItem -LiteralPath $StageRoot -File -Recurse).Count

    Write-Host ''
    Write-Host 'PHASE 7 INPUT ZIP CREATED' -ForegroundColor Green
    Write-Host ("File: " + $ZipInfo.FullName)
    Write-Host ("Fusion scenarios: " + $FusionFolders.Count)
    Write-Host ("Copied files: " + $CopiedCount)
    Write-Host ("Files in staging folder: " + $ActualFileCount)
    Write-Host ("ZIP size (MB): " + [math]::Round($ZipInfo.Length / 1MB, 2))
    Write-Host 'Original experiment outputs were not modified.'
}
finally {
    if (Test-Path -LiteralPath $StageRoot) {
        Remove-Item -LiteralPath $StageRoot -Recurse -Force
    }
}
