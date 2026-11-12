# The buffer pool's design model

The model of `specs/buffer-pool.md` and what checks it. Nothing here needs a device.

| File | What it is |
| --- | --- |
| `KvCore.tla` | The design model: the buffer pool's handlers, the work it defers until a stamp is reached, the copies in flight, and its neighbours: an executor that meets A1 to A4 with batches of unbounded delay and a control plane that issues any request in any order. Each rule R1 to R10 is a guard in force when its name is in `Rules` |
| `MC_*.tla`, `MC_*.cfg` | The configurations TLC searches. `MC_Core`: one group, no copies. `MC_Copies`: copies run by the executor and the host tier |
| `predictions.md` | Every run with the prediction written before it and its outcome |
| `tlc.sh` | Runs one configuration and prints its verdict; `RULES=NoRn` leaves rule Rn out |
| `trace.py` | Prints a counterexample from a TLC log as its actions with the variables each changed |

The tools are TLC (`tla2tools.jar` 1.8.0) under a JDK 21, Apalache and TLAPS; `tlc.sh` reads their directory from `TLA_TOOLS`.

```bash
verification/buffer-pool/tlc.sh MC_Core                 # the design: no violation
RULES=NoR3 verification/buffer-pool/tlc.sh MC_Core      # a rule left out: a violation
```

Terms. Checked: no violation in a complete search of a stated configuration. Proved: safety of the design model, under the stated assumptions. Not proved: liveness; that the implementation refines the model. A run that times out or stops at a bound is INCONCLUSIVE.

The model covers the requests `Attach`, `Detach`, `Allocate`, `Pin`, `Release`, `Commit`, `Evict`, `Clone`, `Load`, `Store` to the host and `Finish`, and the rules R1 to R7. It does not yet cover tasks (`Transfer`, `Prefetch`, `Store` to the remote store; R9), the trial of an epoch (R8), regions with `Open`, `Close` and `Resize` (R10), peers, `Save` and `Restore`, or more than one worker group; no hazard of `specs/kv-cache-s0.md` §6 is modelled yet, and nothing is proved.
