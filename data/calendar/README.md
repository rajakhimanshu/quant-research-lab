# Economic calendar CSV

H1 (news overreaction) needs this file.

Save as `high_impact.csv` in this folder.

Required columns:

```text
datetime_utc,currency,impact,event
2023-01-06 13:30:00,USD,high,Non-Farm Payrolls
```

- `datetime_utc` — event time in UTC (`YYYY-MM-DD HH:MM:SS`)
- `currency` — USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD
- `impact` — high / medium / low
- `event` — free text

Export from Forex Factory (or any calendar) and convert to this format.
Until the file exists, H1 is skipped with a clear reason. H2 does not need it.
