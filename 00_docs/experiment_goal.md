# Experiment Goal

## Task Definition

本实验研究汉语、维吾尔语、西里尔字母哈萨克语模因文字行识别。

输入：

```text
已裁剪的单行模因文字图片
```

输出：

```text
对应的汉语、维吾尔语或哈萨克语字符序列
```

第一篇小论文只研究行级文字识别，不研究完整图片文字检测，也不研究有害性分类。

## Model Scope

主模型：

```text
SVTRv2
```

训练路线：

```text
合成行图预训练
-> 合成与真实行图混合训练
-> 真实模因行图小学习率微调
```

PaddleOCR-VL-1.5 仅作为 zero-shot whole-image OCR 补充基线，用来说明通用文档 VLM 在低资源多语言模因 OCR 上的局限，不作为本文主模型。

## Research Questions

RQ1. 大规模合成数据能否改善低资源维吾尔语和哈萨克语模因文字识别？

RQ2. 模因风格合成数据是否优于普通白底合成数据？

RQ3. 合成预训练加真实模因行图微调是否优于仅使用真实模因行图训练？

RQ4. SVTRv2 如何适应汉语、维吾尔语、哈萨克语三种文字体系，以及维吾尔语 RTL 与汉语/哈萨克语 LTR 的方向差异？

## Main Metrics

主指标：

```text
CER
```

辅助指标：

```text
WER
1-NED
Line Accuracy
Substitution / Deletion / Insertion rate
Length-bucket CER
Uyghur core-character statistics
Kazakh-specific Cyrillic character statistics
```

模型选择依据使用三种语言宏平均 CER，而不是全部样本混合后的总 CER。

