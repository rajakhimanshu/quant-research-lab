# Contributing to Proofbook

This is a **personal research lab**, not an open strategy farm and not a company product.

## Rules

1. Every new idea needs a causal **why** before code.  
2. Freeze the spec in `config/hypotheses.yaml` before looking at results.  
3. After every test, follow `research/AFTER_EACH_TEST.md`.  
4. Prefer **many small commits** (ledger, then journal, then scoreboard).  
5. Never retune a REJECT. Never force-add Co-authored-by bots to history on purpose.  
6. Do not commit secrets (`.env`), raw MT5 dumps you do not need, or `data/results/*` noise.

## Useful commands

```powershell
python -m ats doctor
python -m ats test --id Hxxx
python -m ats ledger
python -m ats scoreboard
```
