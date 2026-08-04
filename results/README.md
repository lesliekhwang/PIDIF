# Results policy

`results/` is reserved for future standardized runs. Historical files and existing
result directories remain in their current locations.

- Baseline diffusion runs use `results/train_diffusion/<run_id>/`.
- Other protocols use their own protocol-specific directory under `results/`.
- A protocol directory does not require a `case_id` component; protocol-specific
  identifiers belong in the run manifest when needed.
- Run directories are never silently overwritten, deleted, or reused.
- Every run must contain a `manifest.json`.
- Directory names contain only limited retrieval fields; full parameters belong in the manifest.
- Do not use ambiguous names such as `latest`, `final`, `new`, or `updated`.
- Timestamps in run IDs use UTC `YYYYMMDDTHHMMSSZ`.
- A truth-selected result must be labeled `oracle_posthoc`.

The baseline diffusion layout is:

```text
results/
└── train_diffusion/
    └── <unique_utc_run_id>/
        ├── manifest.json
        ├── diffusion_best.pt
        ├── diffusion_latest.pt
        └── training_history.csv
```

Other protocols may add protocol-specific artifacts while retaining the unique
run-directory and manifest requirements. This phase creates no concrete run directory.
