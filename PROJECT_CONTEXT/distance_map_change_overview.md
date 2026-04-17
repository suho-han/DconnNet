# Distance-Map Change Overview (Current)

Last updated: 2026-04-17

## Goal

- Preserve upstream binary baseline behavior.
- Add distance-map training/evaluation as fork extension.
- Keep switching explicit and reversible via CLI/config.

## Active Design

### 1) Mode separation

- `label_mode` controls path:
  - `binary`
  - `dist`
  - `dist_inverted`

### 2) Supervision policy

- Final segmentation target remains binary-mask aligned.
- Distance-derived objectives are auxiliary supervision.

### 3) Data/loader behavior

- CHASE/DRIVE loaders support binary and distance-map labels.
- ISIC path includes precision/accuracy reporting in eval outputs.

### 4) Runtime/outputs

- Train/eval writes:
  - `results.csv` (epoch history)
  - `final_results.csv` (final summary)
  - `<split>_results.csv` (split summaries)

## What Is Stable Now

- Binary baseline path remains runnable.
- Dist path has dedicated control flags (`--dist_aux_loss`, `--dist_sf_l1_gamma`, etc.).
- `conn_num` support is aligned with current implementation (`8`, `24`).

## What Was Intentionally Deprecated In Context Docs

- Long chronological debug diary content.
- Historical intermediate hypotheses that are no longer action-driving.
- Old references to legacy aggregation script as primary workflow.

## Open Questions

1. Multi-class distance extension scope and priority.
2. Dist auxiliary loss/weight tuning policy across datasets.
3. Minimal CI-style checks for distance path regressions.

## Practical Next Steps

1. Maintain one canonical experiment/aggregation path in docs.
2. Keep this file as conceptual overview; put run-level details in `testing_notes.md`.
3. Update only when behavior/decisions change materially.
