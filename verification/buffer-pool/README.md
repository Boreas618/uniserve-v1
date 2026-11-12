# The buffer pool's design model

The model of `specs/buffer-pool.md` and what checks it. Nothing here needs a device.

| File | What it is |
| --- | --- |
| `KvCore.tla` | The design model: the buffer pool's handlers, the work it defers until a stamp is reached, copies and tasks in flight, tiers, regions, the trial of an epoch, the loss of an engine; and its neighbours: an executor that meets A1 to A4 with batches of unbounded delay and a control plane that issues any request in any order. Each rule R1 to R10 is a guard in force when its name is in `Rules`; each hazard of `specs/kv-cache-s0.md` §6 is behaviour switched on by its name in `Hazards` |
| `KvP1.tla`, `KvP2.tla`, `KvP3.tla`, `KvP5.tla`, `KvP6.tla`, `KvSafety.tla` | The safety properties as formulas over `KvCore`, one module per property, and their conjunction |
| `KvFinality.tla` | P4 as two temporal properties under stated fairness |
| `properties.toml`, `pins.sha256`, `pins.py` | The property list; each entry and its module are pinned by hash, and `pins.py check` fails when one differs |
| `KvInductive.tla` | `IndInv`, the inductive invariant under the design (every rule in force, no hazard) |
| `KvProof.tla` | The TLAPS proof, under the assumptions `Design` and `Sizes` |
| `MC_*.tla`, `MC_*.cfg` | The configurations TLC searches; `MC_Apalache.tla` fixes constants and generates the states for Apalache |
| `addresses.smt2` | The address arithmetic as thirteen goals for z3 |
| `predictions.md` | Every run with the prediction written before it and its outcome |
| `suite.sh` | The whole suite, with the verdict expected of every run |
| `tlc.sh`, `trace.py` | Run one configuration (`RULES=NoRn` leaves a rule out, `HAZARDS=Hn` switches a hazard on); print a counterexample from a TLC log |

The tools are TLC (`tla2tools.jar` 1.8.0) under a JDK 21, Apalache 0.63.0, TLAPS (`tlapm`) and z3 4.8.9; the scripts read their directory from `TLA_TOOLS`.

```bash
verification/buffer-pool/suite.sh                         # everything; about 50 minutes on 144 cores
verification/buffer-pool/suite.sh pins mutations hazards  # the quick parts
verification/buffer-pool/tlc.sh MC_Trial                  # one configuration
tlapm --threads 24 --stretch 6 KvProof.tla                # the proof, in this directory
apalache-mc check --init=Init --inv=IndInv --length=0 --cinit=ConstInit MC_Apalache.tla
apalache-mc check --init=IndInit --inv=Safety --length=0 --cinit=ConstInit MC_Apalache.tla
apalache-mc check --init=IndInit --inv=IndInv --length=1 --cinit=ConstInit \
    --tuning-options='search.invariantFilter=1->.*' MC_Apalache.tla
```

## What is established

Terms. Checked: no violation in a complete search of a stated configuration. Proved: safety of the design model, under the stated assumptions. Not proved: liveness; that the implementation refines the model. A run that times out or stops at a bound is INCONCLUSIVE.

- **Checked (TLC).** P1, P2, P3, P5 and P6 hold in the nine configurations of `suite.sh`; P4 holds in `MC_Finality`. Each rule R1 to R10 left out, and each hazard H1 to H13 but H6 and H13's safety side, violates the property `suite.sh` states for it; H6 and H13 violate P4. `IndInv` is an invariant of every configuration.
- **Proved (TLAPS), under `Design` and `Sizes`.** `Init => IndInv` and `IndInv => Safety` (164 obligations).
- **Checked (Apalache), for the constants of `MC_Apalache` and collections of at most three elements.** `Init => IndInv`; `IndInv => Safety`.
- **Proved (z3).** The thirteen goals of `addresses.smt2`, under A8.
- **Not established.** That `IndInv` is inductive: Apalache has given counterexamples to induction from states no behaviour reaches, each answered by strengthening `IndInv` (`predictions.md`, entries 28 to 30), and the step `IndInv /\ [Next]_vars => IndInv'` is not proved. So `Spec => []Safety` is not proved.

The model leaves out `Victims` and `List` (eviction takes any shared block without a pin, which includes every installed order), a peer's own pins on this host, and blocks of more than one unit.
