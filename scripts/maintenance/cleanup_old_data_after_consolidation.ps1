param(
  [string]$Root = "D:\text_renderer",
  [switch]$Execute
)

$ErrorActionPreference = "Stop"

$targets = @(
  # Old large image/output data now consolidated into final_multilingual_meme_ocr_dataset/full_images.
  "Misogyny_Dataset_Project\images",
  "Misogyny_Dataset_Project\pipeline_outputs_local_full_translated",
  "Misogyny_Dataset_Project\manual_translation_review_all",
  "Misogyny_Dataset_Project\preview",
  "Meme_Dataset_Project\images",
  "Meme_Dataset_Project\pipeline_outputs_stage1",

  # Root-level historical scratch data.
  ".962_watermark_all_smoke",
  ".manual_delete_fix_smoke",
  ".tmp_doc_check",
  "__pycache__"
)

foreach ($rel in $targets) {
  $path = Join-Path $Root $rel
  if (Test-Path -LiteralPath $path) {
    Write-Output "DELETE $path"
    if ($Execute) {
      Remove-Item -LiteralPath $path -Recurse -Force
    }
  }
}

Write-Output "Done. Without -Execute this is only a dry run."
