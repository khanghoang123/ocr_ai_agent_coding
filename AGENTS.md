# Agent Guardrails

- Do not delete datasets, ground truth files, or model weights.
- Do not edit ground truth labels to improve metrics.
- Keep experiment artifacts under `experiments/runs/` and leaderboard artifacts under `experiments/leaderboard/`.
- Do not change FastAPI response formats unless a task explicitly requires it.
- Prefer wrapper/config/runner changes over rewriting the OCR pipeline.
- If an experiment option is not fully supported, record it in metrics/report instead of crashing.
