# Validation

## Oracle

```text
  1/1 Mean: 1.000 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:05:54 0:00:00
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
Total runtime: 6m 15s
Results written to jobs/2026-09-05__15-22-46/result.json
Inspect results by running `harbor view jobs`
Share results by running `harbor upload jobs/2026-09-05__15-22-46`

```

## NOP

```text
  1/1 Mean: 0.000 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:01:17 0:00:00
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
Total runtime: 1m 37s
Results written to jobs/2026-09-05__15-30-58/result.json
Inspect results by running `harbor view jobs`
Share results by running `harbor upload jobs/2026-09-05__15-30-58`

```


## Verifier-hardening validation

### Oracle

Trials: 1  
Exceptions: 0  
Mean: 1.000  
Reward: 1.0  
Total runtime: 5m 52s  
Results: jobs/2026-09-05__17-23-26/result.json

### NOP

Trials: 1  
Exceptions: 0  
Mean: 0.000  
Reward: 0.0  
Total runtime: 1m 34s  
Results: jobs/2026-09-05__17-30-58/result.json


## Final validation after verifier hardening and network isolation

### Oracle
Trials: 1
Exceptions: 0
Mean: 1.000
Results: jobs/2026-09-05__17-55-01/result.json

### NOP
Trials: 1
Exceptions: 0
Mean: 0.000
Results: jobs/2026-09-05__18-02-10/result.json
