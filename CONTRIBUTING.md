# Contributing to Proofbook

Proofbook is a research lab built to reject ideas honestly. Contributions are
welcome when they keep that property.

## Rules

1. Every idea needs a causal **why** and a **payer** (who is forced to trade) before code.
2. New hypotheses go through `ats lab propose` and an existing template. A mechanism
   that fits no template needs a new template in `src/ats/lab/templates.py` with a
   closed, bounded parameter schema, a baseline arm and tests — not a looser schema.
3. Freeze before results. Never retune or flip a REJECT. Never move split dates.
4. Do not unlock out-of-sample data except through `ats lab unlock` on a CANDIDATE.
5. Do not submit "profitable strategy" PRs, indicator combinations, or result-chasing parameter changes.
6. Do not commit secrets (`.env`), MT5 data (`data/` is gitignored), or `data/results/*`.
7. Keep commits small (code, then tests, then records).

## Checks

```powershell
pytest -q
python -m ats lab templates
python -m ats lab run --dry-run --no-sweep --no-propose
```
