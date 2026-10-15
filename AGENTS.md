# Repository instructions

- This is frozen H1 inference. No fitting, optimizer steps, or Test-driven model selection.
- `data` owns read-only acquisition and production-row validation; it does not build labels in inference.
- `features` owns causal numerical calculations. It must not read files, database tables, targets, or benchmark predictions.
- `models/archived` contains the reviewable forward dependency closure of the research implementation. `models/loader.py` reconstructs it explicitly from verified assets; no experiment imports or checkout path mutation.
- `state` owns warmup and atomic local history persistence; real state contains restricted market observations and stays out of Git and Releases.
- `inference` consumes only a loaded model/state and one current production row. No filesystem reads inside `predict_step`.
- `evaluation` alone accepts future outcomes. Needle/reversal measures are retrospective diagnostics.
- CLI modules are adapters. `tools/export_legacy.py` is an explicit one-time legacy-source migration boundary, never a runtime dependency.
- Preserve original experiment artifacts; never replace trained models with cached predictions.
- New clocks and log extensions must be flagged. Do not drop an observation because a feature/target is missing.
- Run `python -m pytest`; real-data tests must run with local authorized assets before a weight release. Source-only skips are not a release pass.
- Use `gh` for GitHub operations. Keep repository private. Do not commit source observations, history states, credentials, or generated weights. Weights belong in a private Release after audits; use relative names and SHA256 hashes.
- The user additionally authorized a small opening-clock training screen on 2026-10-15. `tools/opening_training_screen.py` is an offline Train-only Ridge experiment, not part of frozen inference. Never update production weights from this screen or choose features using Test.
