# Validation

Final validation of the size optimized repository.

## Oracle

```text
$ harbor run -p tasks/energy-storage/battery-health/lfp-soh-estimation -a oracle
  1/1 Mean: 1.000 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:04:09 0:00:00
adhoc • oracle
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃ Trials ┃ Exceptions ┃  Mean ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│      1 │          0 │ 1.000 │
└────────┴────────────┴───────┘

┏━━━━━━━━┳━━━━━━━┓
┃ Reward ┃ Count ┃
┡━━━━━━━━╇━━━━━━━┩
│ 1.0    │     1 │
└────────┴───────┘

Job Info
Total runtime: 4m 12s
Results written to jobs/2026-09-07__13-44-17/result.json
Inspect results by running `harbor view jobs`
Share results by running `harbor upload jobs/2026-09-07__13-44-17`

```

## NOP

```text
$ harbor run -p tasks/energy-storage/battery-health/lfp-soh-estimation -a nop
  1/1 Mean: 0.000 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:14 0:00:00
adhoc • nop
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃ Trials ┃ Exceptions ┃  Mean ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│      1 │          0 │ 0.000 │
└────────┴────────────┴───────┘

┏━━━━━━━━┳━━━━━━━┓
┃ Reward ┃ Count ┃
┡━━━━━━━━╇━━━━━━━┩
│ 0.0    │     1 │
└────────┴───────┘

Job Info
Total runtime: 17s
Results written to jobs/2026-09-07__13-48-33/result.json
Inspect results by running `harbor view jobs`
Share results by running `harbor upload jobs/2026-09-07__13-48-33`

```
