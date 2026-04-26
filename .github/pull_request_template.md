## Summary
- What changed:
- Why:

## Scope
- [ ] OCR pipeline (`src/ocr_pipeline/**`)
- [ ] API (`app/**`)
- [ ] Frontend (`frontend/**`)
- [ ] Tests (`tests/**`)
- [ ] CI/DevX (`.github/**`, tooling)

## Verification
- [ ] `pytest tests/test_pipeline.py`
- [ ] API smoke passed (`/health`, `/ocr`, `/ocr/export`)
- [ ] Frontend basic smoke passed (`python -m compileall frontend`)

## Safety checks
- [ ] No content changes in `models/`
- [ ] No content changes in `notebooks/`
- [ ] No destructive git operations (no history rewrite / force push)

## Risk / Rollback
- Risk:
- Rollback plan:
