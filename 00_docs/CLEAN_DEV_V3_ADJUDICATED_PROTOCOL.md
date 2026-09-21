# Clean Dev V3 Adjudicated Protocol

## Purpose

Clean Dev V2 preserved the complete model-blind audit, but it did not inherit
the earlier 108-row error review. It also retained every `crop_issue` row. V3
merges both completed reviews by sample ID without modifying either source.

## Frozen rules

- Inherit every concrete GT correction from both completed reviews.
- A concrete correction supersedes a generic `gt_correct` decision.
- Resolve correction equality after normalization V2.
- Exclude the union of `crop_issue` and `image_or_crop_issue` decisions.
- Retain `ambiguous_image` rows with an audit flag.
- Preserve all 951 source rows in the manifest; excluded rows are omitted only
  from evaluation label files.
- Never use Corrupted Dev or Test for adjudication or checkpoint selection.

The protocol is not described as purely model-blind because one inherited
review was produced from model error candidates. This is recorded explicitly.
All models and all saved epochs must therefore be rescored with the same frozen
V3 labels before any result is reported.

## Build and verify

```bash
python scripts/protocol/build_clean_dev_v3_adjudicated.py --root . --replace
python scripts/protocol/test_clean_dev_v3_adjudicated.py
```

The previous V2 artifacts remain immutable historical evidence.
