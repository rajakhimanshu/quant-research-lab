# Repo layout

```
config/           Frozen hypotheses, intake leads, mechanism library, settings (costs, splits, gates, lab budget)
src/ats/          Lab package: data (MT5), hypotheses, research pipeline, intake, CLI
src/ats/lab/      Automated loop: templates, spec validator, proposer, budget, book, human gates
research/         Ledger, journal, scoreboard, how-we-test, intake notes
research/lab/     Machine-written lab book, family runs, batch reports
examples/         Example hand-written proposal
tests/            Pytest suite (synthetic data; no MT5 needed)
data/             Raw/processed/results (gitignored; filled by `ats pull`)
```

Canonical closed book: `research/ledger.yaml` + `research/lab/book.yaml`  
Human scoreboard: `research/HYPOTHESIS_SCOREBOARD.md`  
CLI entry: `python -m ats`
