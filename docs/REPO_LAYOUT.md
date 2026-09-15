# Repo layout

```
config/           Frozen hypotheses, ideas inbox, settings (costs, splits, gates)
src/ats/          Lab package: data, hypotheses, research pipeline, CLI
research/         Ledger, journal, scoreboard, how-we-test, briefs
tests/            Pytest coverage for handlers and wiring
data/             Raw/processed/results (results gitignored)
.cursor/rules/    Always-on research protocol for the agent
```

Canonical closed book: `research/ledger.yaml`  
Human scoreboard: `research/HYPOTHESIS_SCOREBOARD.md`  
CLI entry: `python -m ats`
