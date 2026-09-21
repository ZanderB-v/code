param(
  [string]$Root = "D:\text_renderer",
  [switch]$Execute
)

$ErrorActionPreference = "Stop"

$finalDir = Join-Path $Root "final_multilingual_meme_ocr_dataset"
$realDir = Join-Path $Root "Pedestrian image dataset"
$synthDir = Join-Path $Root "Synthetic datasets"
$prepDir = Join-Path $Root "experiments\multilingual_meme_ocr\svtrv2_line_recognition\01_data_preparation"
$corpusDir = Join-Path $Root "experiments\multilingual_meme_ocr\svtrv2_line_recognition\02_corpus_preparation"

function Ensure-Dir($path) {
  if ($Execute) { New-Item -ItemType Directory -Force -Path $path | Out-Null }
  Write-Output "DIR  $path"
}

function Move-Dir($src, $dst) {
  if (-not (Test-Path -LiteralPath $src)) { return }
  if (Test-Path -LiteralPath $dst) {
    Write-Output "SKIP target exists: $dst"
    return
  }
  Write-Output "MOVE $src -> $dst"
  if ($Execute) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Move-Item -LiteralPath $src -Destination $dst
  }
}

function Move-File($src, $dst) {
  if (-not (Test-Path -LiteralPath $src)) { return }
  if (Test-Path -LiteralPath $dst) {
    Write-Output "SKIP target exists: $dst"
    return
  }
  Write-Output "MOVE $src -> $dst"
  if ($Execute) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Move-Item -LiteralPath $src -Destination $dst
  }
}

Ensure-Dir $finalDir
Ensure-Dir $realDir
Ensure-Dir $synthDir

Move-Dir (Join-Path $prepDir "real_line_dataset_train_aligned_v2") (Join-Path $realDir "real_line_dataset_train_aligned_v2")
Move-Dir (Join-Path $prepDir "real_line_dataset") (Join-Path $realDir "real_line_dataset_eval_reviewed")
Move-Dir (Join-Path $prepDir "real_lines") (Join-Path $realDir "pilot_real_lines")
Move-Dir (Join-Path $prepDir "source_splits") (Join-Path $finalDir "source_splits")
Move-Dir (Join-Path $prepDir "synthetic_smoke") (Join-Path $synthDir "synthetic_smoke")
Move-Dir $corpusDir (Join-Path $synthDir "02_corpus_preparation")

$summaryPath = Join-Path $Root "data_layout_README.md"
$summary = @"
# Data Layout

Keep data in three directories:

1. final_multilingual_meme_ocr_dataset
   - final JSON indexes
   - consolidated full source/rendered images under full_images/
   - source_splits/

2. Pedestrian image dataset
   - cropped real line image datasets
   - dev/test reviewed annotation data
   - train aligned reviewed data

3. Synthetic datasets
   - synthetic text pools
   - synthetic smoke or large synthetic image datasets

Run this script without -Execute for dry run, then with -Execute to move directories.
"@
Write-Output "WRITE $summaryPath"
if ($Execute) { Set-Content -LiteralPath $summaryPath -Value $summary -Encoding UTF8 }
