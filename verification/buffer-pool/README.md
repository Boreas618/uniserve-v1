# The buffer pool's design model

The model of `specs/buffer-pool.md` and what checks it. Nothing here needs a device.

| File | What it is |
| --- | --- |
| `KvCore.tla` | The design model: the buffer pool's handlers, the work it defers until a stamp is reached, copies and tasks in flight, tiers, regions, the trial of an epoch, the loss of an engine; and its neighbours: an executor that meets A1 to A4 with batches of unbounded delay and a control plane that issues any request in any order. Each rule R1 to R10 is a guard in force when its name is in `Rules`; each hazard of `specs/kv-cache-s0.md` §6 is behaviour switched on by its name in `Hazards` |
| `KvP1.tla`, `KvP2.tla`, `KvP3.tla`, `KvP5.tla`, `KvP6.tla`, `KvP7.tla`, `KvSafety.tla` | The safety properties as formulas over `KvCore`, one module per property; `KvSafety` is the conjunction of P1, P2, P3, P5 and P6 that `KvProof` proves `IndInv` implies |
| `KvFinality.tla` | P4 as two temporal properties under stated fairness |
| `properties.toml`, `pins.sha256`, `pins.py` | The property list; each entry and its module are pinned by hash, and `pins.py check` fails when one differs |
| `KvInductive.tla` | `IndInv`, the inductive invariant under the design (every rule in force, no hazard) |
| `KvProof.tla` | The TLAPS proof, under the assumptions `Design` and `Sizes` |
| `KvProofP7.tla` | `IndInv => P7`, under `Design`, `Sizes` and `Finite` |
| `MC_*.tla`, `MC_*.cfg` | The configurations TLC searches; `MC_Apalache.tla` fixes constants and generates the states for Apalache |
| `addresses.smt2` | The address arithmetic as thirteen goals for z3 |
| `predictions.md` | Every run with the prediction written before it and its outcome |
| `suite.sh` | The whole suite, with the verdict expected of every run |
| `tlc.sh`, `trace.py` | Run one configuration (`RULES=NoRn` leaves a rule out, `HAZARDS=Hn` switches a hazard on); print a counterexample from a TLC log |
| `MC_Pool.tla`, `MC_Pool{Core,Copies,Transfer,Tiers}.cfg` | The buffer pool alone: its handlers taken with requests of any stamps, with no batch, executor or scheduler; the relation the Rust buffer pool is compared with |
| `agreement.sh`, `agreement.py` | L1: the Rust buffer pool's enumeration of every request sequence (`crates/storage-server/src/buffer_pool/enumeration.rs`) against the states of `MC_Pool`, per configuration |
| `KvTrace.tla`, `steps.py` | Steps of the Rust buffer pool against `MC_Pool`'s steps: each window, a recorded state and the inputs and states that follow it, must be a path of `PoolNext` through those states; `steps.py` writes a run's windows and constants as `MC_Trace` and runs TLC |
| `agreement_steps.sh` | L1 step by step: every transition of the enumeration (a state, its input, the next state) of `core`, or of the configurations named, is a path of `MC_Pool`'s steps; a transition made by hand that the model does not allow is reported |
| `replay.sh` | The trace of a run of the Storage Server (`storage-server serve --buffer-pool-trace DIR`, `crates/storage-server/src/buffer_pool/trace.rs`): replayed on a new Rust buffer pool with every completion checked against the run's, and its states checked against `MC_Pool`'s steps in windows |
| `mutations.py` | L4: one replacement per rule R1 to R10 in the Rust buffer pool's source, each of which the crate's tests must catch |
| `apalache_step.sh`, `cti.py` | The inductive step under Apalache per group of actions; print a counterexample to induction |

The tools are TLC (`tla2tools.jar` 1.8.0) under a JDK 21, Apalache 0.63.0, TLAPS (`tlapm`) and z3 4.8.9; the scripts read their directory from `TLA_TOOLS`.

```bash
verification/buffer-pool/suite.sh                         # everything; about 50 minutes on 144 cores
verification/buffer-pool/suite.sh pins mutations hazards  # the quick parts
verification/buffer-pool/tlc.sh MC_Trial                  # one configuration
verification/buffer-pool/agreement.sh                     # L1; about 25 minutes
verification/buffer-pool/agreement_steps.sh               # L1 step by step, core; about 4 minutes
verification/buffer-pool/replay.sh TRACE_DIR              # a run's trace: its replay, and its states against the model
verification/buffer-pool/mutations.py                     # L4; about 5 minutes, edits the crate in place and restores it
tlapm --threads 24 --stretch 6 KvProof.tla                # the proof, in this directory
apalache-mc check --init=Init --inv=IndInv --length=0 --cinit=ConstInit MC_Apalache.tla
apalache-mc check --init=IndInit --inv=Safety --length=0 --cinit=ConstInit MC_Apalache.tla
apalache-mc check --init=IndInit --inv=IndInv --length=1 --cinit=ConstInit \
    --tuning-options='search.invariantFilter=1->.*' MC_Apalache.tla
```

## What is established

Terms. Checked: no violation in a complete search of a stated configuration. Proved: safety of the design model, under the stated assumptions. Not proved: liveness; that the implementation refines the model. A run that times out or stops at a bound is INCONCLUSIVE.

- **Checked (TLC).** P1, P2, P3, P5, P6 and P7 hold in the eleven configurations of `suite.sh`; P4 holds in `MC_Finality`. P7 is violated by the hazard H12 and by leaving out R2 (the part `validation`). Each rule R1 to R10 left out, and each hazard H1 to H13 but H6 and H13's safety side, violates the property `suite.sh` states for it; H6 and H13 violate P4. `IndInv` is an invariant of every configuration.
- **Proved (TLAPS), under `Design` and `Sizes`.** `Init => IndInv` and `IndInv => Safety` (164 obligations); with `Finite` as well, `IndInv => P7` (135 obligations).
- **Checked (Apalache), for the constants of `MC_Apalache` and collections of at most three elements.** `Init => IndInv`; `IndInv => Safety`.
- **Proved (z3).** The thirteen goals of `addresses.smt2`, under A8.
- **Not established.** That `IndInv` is inductive: Apalache has given counterexamples to induction from states no behaviour reaches, each answered by strengthening `IndInv` (`predictions.md`, entries 28 to 34); the step `IndInv /\ [Next]_vars => IndInv'` is being proved group by group (`KvStep*.tla`) and is not proved yet. So `Spec => []Safety` and `Spec => []P7` are not proved.

## The implementation against the model

- **L1, states (`agreement.sh`).** The Rust buffer pool is driven through every sequence of requests, `completed` values, task results and deadlines of four small configurations (`core`, `copies`, `transfer`, `tiers`). After every step the pool's own check holds: every unit is in exactly one place, a block's references equal its holders, a private block is in one place, a pin is of a cached block, an entry is valid in its owner's group, memory is in one region. Every state reached, written in the model's terms, is a state of `MC_Pool` under the same configuration: 1,281 states against the model's 3,746 projected states (core), 32,668 against 112,466 (copies), 86,680 against 86,308 (transfer; the enumeration tells apart states that differ only in what the model does not state, such as the order of freed slot rows), 1,204 against 3,476 (tiers). It compares states, not steps: that each step of the pool is a path of the model's steps is not checked.
- **L1, steps (`agreement_steps.sh`).** Every transition of `core` that the enumeration reaches, 5,553 of them, is a path of `MC_Pool`'s steps from the transition's state to the next (`KvTrace.tla`), with what the state leaves out (the stamps and keys of waiting work, the memories of a block only waiting work holds) chosen by TLC; a transition made by hand in which a commit takes effect before its stamp is reported. In `tiers`, 4,357 transitions are followed and the 1,714 from a state with a key cached as a stored trajectory are outside the model, which has one class: the enumeration names a trajectory and a KV block by one key, which A6 excludes. The Rust buffer pool's R1 for owners is stronger than the model's (a number below one taken on the queue pair is refused), so an owner the Rust pool states as gone matches a model owner never attached.
- **Traces (`replay.sh`).** With `--buffer-pool-trace DIR` the Storage Server's buffer pool thread records every input it acts on and every completion its pools produce (`specs/buffer-pool.md` §18). The replay drives a new pool with the inputs and stops at the first completion that differs from the run's; the states it writes are checked against `MC_Pool`'s steps in windows, each from a recorded state. The Storage Server's integration test of a session, recorded, replays with every completion equal and its 19 steps between the open and the close are followed. Outside the model: an open, a close, a pool's image, classes other than 0, prefetches from the peers, copies and tasks of several keys, states with no session or several. With `--buffer-pool-check` the thread runs the pool's check after every input, and logs and counts (`check_failures` of the pool's listing) a pool that turns inconsistent.
- **L4, mutations (`mutations.py`).** The guard of each rule R1 to R10 replaced in the Rust source is caught by the crate's tests.
- **Not established.** That the implementation refines the model: L1 is an enumeration of four bounded configurations, compared by states in all four and by steps in `core`, and a trace is checked in windows, each from a state with what it leaves out chosen, not as one behaviour. The step check of `copies` and `transfer` (177,982 and 501,924 transitions) has not been run.

The model leaves out `Victims` and `List` (eviction takes any shared block without a pin, which includes every installed order), a peer's own pins on this host, and blocks of more than one unit.
