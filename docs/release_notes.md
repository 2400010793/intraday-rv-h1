H1 frozen research migration, no retraining.

- Shampoo epoch5 / Adam epoch6, six symbols, 20 monthly refits, five fixed seeds; 1200 weight files with SHA256, frozen transforms, Ridge and clock registries.
- 2026-09-16 row-by-row downstream replay passed all six symbols: futures243 / ETF240; original valid-clock maximum prediction difference 2.665e-15. New first clocks are explicitly mapped and flagged.
- All-month archival closure passed: 1200 weights, 5,507,900 prediction values, maximum absolute difference 5.552e-16. This checks archived tensor inputs, distinct from the raw-row replay.
- Model archive contains no raw market tables or historical streaming states.

**Research draft — production acceptance blocked.** Upstream point-in-time publication/revision records are unavailable; candidate log-extension noninferiority checks are incomplete. See docs/validation.md and manifest.acceptance. A one-day replay does not certify absence of all data leakage. Do not deploy this draft as an audited production model.
