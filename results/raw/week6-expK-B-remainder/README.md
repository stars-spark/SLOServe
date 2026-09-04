# week6-expK-B-remainder (not part of the formal expK-B result)

This directory holds a single bounded verification run of `clip512-s1`, executed immediately after
the machine's power profile was changed (turbo disabled, ACPI profile `balanced`) to confirm that
one full sweep point could complete inside the new thermal budget before committing to the rest of
the sweep. Its completeness report is marked `仅管线验证，不作策略性能比较` for that reason.

The formal expK-B result uses `results/raw/week6-expK-B/` only, where all twelve points were
executed one at a time behind a cooldown gate. The `clip512-s1` file here is a *different* run of
the same configuration and must not be merged with, or substituted for, the formal one. It is
retained rather than deleted so the thermal-recovery step stays auditable.
