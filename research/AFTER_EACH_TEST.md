# After every hypothesis test — checklist

Do this **every time** an H finishes (REJECT / NEEDS_MORE_DATA / CANDIDATE / OOS).

1. **Ledger** — add or update the row in `research/ledger.yaml`  
   - `id`, `date`, `book`, `family`, `decision`, `mechanism`, `baseline`, `train`, `why_closed`  
   - If OOS unlocked: add `oos`  
2. **Journal** — one dated line in `research/journal.md`  
3. **Open notes** — update `open:` bullets in `ledger.yaml` if the family closed  
4. **Scoreboard** — run `python -m ats scoreboard`  
5. **Hypotheses yaml** — leave `enabled: false` after the one-shot (do not keep grinding)  
6. **Git** — prefer **tiny commits**, for example:  
   - `Record H196 REJECT in the ledger.`  
   - `Log H196 in the research journal.`  
   - `Refresh hypothesis scoreboard through H196.`  
7. **Do not** retune a REJECT. Open a **new** H id for a new why.

OOS stays locked unless you explicitly unlock after picking a survivor.
