param(
    [switch]$Replace
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Python = "python"
$Arguments = @(
    (Join-Path $Root "scripts/svtrv2/build_clean_dev_v2_m3_error_profile.py"),
    "--root", $Root
)
if ($Replace) {
    $Arguments += "--replace"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$Output = Join-Path $Root "05_evaluation/clean_dev_v2_m3_error_profile_v1"
Write-Host "CLEAN_DEV_V2_M3_ERROR_PROFILE_GENERATED"
Write-Host "Summary: $Output/error_profile_summary.json"
Write-Host "ZH review: $Output/review/zh_review.html"
Write-Host "UG review: $Output/review/ug_review.html"
Write-Host "KK review: $Output/review/kk_review.html"
