# Phase 0-1 Plan

## Phase 0: Freeze Image-Level Split

目标：确认当前完整图片级 train/dev/test 划分可以作为所有后续行级识别实验的唯一来源。

必须检查：

- 同一个 `source_id` 不能同时出现在 train/dev/test 多个集合。
- UG 与 KK 记录在 image level 对齐。
- UG 与 KK 的文字行数量对齐。
- dev/test 原图不能进入训练背景库。
- dev/test 文本不能进入合成训练语料。

本阶段产物：

```text
01_data_preparation/source_splits/image_level_split_manifest.csv
01_data_preparation/source_splits/train_source_ids.txt
01_data_preparation/source_splits/dev_source_ids.txt
01_data_preparation/source_splits/test_source_ids.txt
01_data_preparation/source_splits/image_level_split_summary.json
01_data_preparation/source_splits/image_level_split_leaks.csv
```

## Phase 1: Real Line Dataset Pilot

目标：先裁剪 300-500 条真实模因文字行，用于统计真实行图像分布，再决定合成器参数。

建议分布：

```text
zh: 100-170 lines
ug: 100-170 lines
kk: 100-170 lines
```

每条样本记录：

```json
{
  "image": "real_lines/train/ug_0001.jpg",
  "text": "对应文字",
  "language": "ug",
  "source_image": "meme_001.jpg",
  "source_id": "meme:0001",
  "split": "train"
}
```

裁剪标准：

- 保留完整目标文字行。
- 保留少量上下左右背景。
- 可以保留轻微边缘噪声。
- 剔除完整邻行文字。
- 剔除目标文字被截断的样本。
- 两行无法明确分开时不进入 pilot。
- 标签无法确认时不进入 pilot。

pilot 统计项：

- 字符长度。
- 图片宽高比。
- 近似字体大小。
- 文字颜色。
- 描边与阴影。
- 背景复杂度。
- 长文本比例。
- RTL/LTR 与混排样本比例。

## Immediate Next Command

本地检查：

```powershell
python .\experiments\multilingual_meme_ocr\svtrv2_line_recognition\scripts\data\audit_image_level_split.py
```

服务器检查：

```bash
cd /data_home/wudayu
python experiments/multilingual_meme_ocr/svtrv2_line_recognition/scripts/data/audit_image_level_split.py \
  --input /data_home/wudayu/final_multilingual_meme_ocr_dataset/final_all.json \
  --output-dir /data_home/wudayu/experiments/multilingual_meme_ocr/svtrv2_line_recognition/01_data_preparation/source_splits
```

