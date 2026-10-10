---
description: Check the Proofbook lab, report, and collect the user's decisions
---

1. Check the runner: `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -match 'auto_lab' } | select ProcessId, CommandLine`. If none, start it with `Start-Process cmd -ArgumentList '/c', 'scripts\auto_lab.bat' -WindowStyle Hidden` from the repo root. Never start a second one.
2. `if (Test-Path research\lab\STOP) { Get-Content research\lab\STOP }`. If present, a candidate awaits the user's unlock/decide; show it.
3. `Get-Content research\lab\auto_lab.log -Tail 80` and `python -m ats lab status`.
4. If a family was tested since the last check, read the newest `research\lab\reports\*.md`.
5. `python -m ats intake list --status raw` and list the `groq_*` ideas awaiting approval with their why and payer; flag factual errors.
6. Report: leads, specs (queued / rejected_by_model / invalid / duplicate), families tested, each hypothesis's train vs validation, gap, cost-adjusted R, decision.
7. Ask the user for decisions. Do not approve, unlock, decide, force or change settings on your own.
