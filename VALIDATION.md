# Validation

## Oracle

Command:

```text
harbor run -p tasks/energy-storage/battery-health/lfp-soh-estimation -a oracle
```

Terminal output:

```text
Mean: 1.000
```

## NOP

Command:

```text
harbor run -p tasks/energy-storage/battery-health/lfp-soh-estimation -a nop
```

Terminal output:

```text
Mean: 0.000
```
