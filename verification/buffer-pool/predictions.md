# Predictions

Each entry is written before the run it names and is not edited afterwards; the outcome is added beneath it. A run that times out or stops at a bound is INCONCLUSIVE.

## 1. The core of the design, one group (`MC_Core`), 2026-11-12

- **Model:** `KvCore` with `Attach`, `Allocate`, `Pin`, `Evict`, `Release`, `Detach`, `Commit`, the executor of A1 to A4 and a control plane that issues any request and hands on or drops any completion at any time. All rules in force. Two units, two slot rows, one key, two owners, one group, one position, two batches, two requests under way.
- **Prediction:** TLC reports a violation of `WriteIsolation` (P1). The expected behaviour: an owner is attached and allocated a block; its `Detach` is taken while no executor holds its entries, so it carries no stamp and its units are reclaimed at once; the completion of its `Allocate` is handed to the executor afterwards; the unit is allocated to a second owner; a batch writes it for the first. A1 to A4 as written allow it: the runs came from a completion and were written at the position the request named.
- **If it holds:** the assumptions do not carry "the scheduler's decisions cannot break them" (`specs/buffer-pool.md` §9), and A1 has to state when a completion may still be handed on.
- **Outcome:** as predicted. `P1` is violated in seven states: `Attach(o1)`, `Allocate(o1, p1)`, a `Release` of that position issued and taken with no stamp because no executor held the entry, the `Allocated` completion handed to the executor afterwards, and a batch launched that writes the reclaimed unit. Search stopped at the violation (34,199 states generated).
- **Consequence:** A1 states when a completion may be handed on: only before any `Release`, `Detach` or `Commit` that names its entry is handed to an executor; a completion of a request submitted before such a request is dropped. `KvCore` carries it as the guard of `ExecMap` and as `Kept` in the handlers of stamped requests.

## 2. The core with A1 as restated (`MC_Core`), 2026-11-12

- **Model:** as entry 1, with the restated A1.
- **Prediction:** the search completes with no violation of `TypeOK`, `P1`, `P2`, `P3`.
- **Outcome:** not as predicted. `P1` is violated in ten states: a `Release` of an owner's position and a `Detach` of the owner are both under way; the executor applies the `Release` and stamps it; the `Detach` then finds no entry in the executor's table, carries no stamp, is taken first and reclaims the unit while the batch that writes it is in flight. The model let a group stamp a request only while its table held the entries concerned, which is how `specs/buffer-pool.md` §9 words obligation 2.
- **Consequence:** obligation 2 is stated as `specs/scheduler-design.md` §3.1 states it: the groups that stamp a request are fixed by the owner (its group, and the groups that read the class), and each stamps whether or not its table still holds the entries. `KvCore` carries it as `StampGroups`.

## 3. The core with obligation 2 as restated (`MC_Core`), 2026-11-12

- **Model:** as entry 2, with `StampGroups`.
- **Prediction:** the search completes with no violation of `TypeOK`, `P1`, `P2`, `P3`.
- **Outcome:** INCONCLUSIVE. The search was stopped after 13 minutes at depth 25 with 139,710,503 distinct states found, 35,931,189 on the queue and no violation.
- **Consequence:** the model gains a reduction, `Eager`: a completion is handed to its executor when it is produced unless a stamped request under way names its entry. An executor's table then holds at every moment at least what it holds when completions are handed on later or dropped, so a batch addresses at least the same units and every violation with late hand-off remains one. Units, slot rows and owners are symmetric in `MC_Core`.

## 4. The core, reduced (`MC_Core` with `Eager` and symmetry), 2026-11-12

- **Model:** as entry 3, with `Eager = TRUE` and the symmetry of units, slot rows and owners.
- **Prediction:** the search completes with no violation of `TypeOK`, `P1`, `P2`, `P3`.
- **Outcome:** as predicted. The search completed: 113,917,775 states generated, 22,754,006 distinct, depth 46, no violation, in 1 min 54 s on 32 workers.

## 5. The rules of the core, each left out (`MC_Core`, `RULES=NoRn`), 2026-11-12

The invariants are checked by name; P1 is `FreeUnheld`, `Exclusive`, `WriteIsolation`, `ReadIsolation`; P2 is `KeyStable`, `PinsCached`, `OneKey`, `SharedUnwritten`; P3 is `NoLeak`, `Quiescent`. TLC reports the first violated invariant on a shortest behaviour, so where two are named either may be reported.

| Rule left out | Predicted violation | Expected behaviour |
| --- | --- | --- |
| R1 | P3 `NoLeak` | An owner that is attached is attached again; its first slot row is neither free nor held |
| R2 | P1 `FreeUnheld` | A block is committed and its owner's pin released; the unit returns to the region while the cache holds it |
| R3 | P1 `WriteIsolation` or `ReadIsolation` | A `Release` takes effect while the batch that addresses the block is in flight |
| R4 | P2 `KeyStable` | A second block is committed under a cached key and replaces the first, which is still pinned |
| R6 | P2 `SharedUnwritten` or P1 `WriteIsolation` | A commit waits; its position is released and allocated again; the commit comes due and shares the new block, which a batch writes |
| R7 | P2 `KeyStable` | `Evict` takes a block its owner pins: the pinned block has no key |

R5, R8, R9 and R10 guard requests the core does not have; they are predicted with the model that adds them.

- **Outcome:** R1 `NoLeak` in 3 states, R2 `FreeUnheld` in 9, R3 `WriteIsolation` in 8, R4 `KeyStable` in 13, R7 `KeyStable` in 9: as predicted. R6: not as predicted, `NoLeak` in 11 states. The mutation as modelled kept the waiting commit's reference to its block while choosing the block to share by position, so the held block was dropped without being reclaimed: an artefact of the mutation, not the behaviour of a commit that names its position alone.
- **Consequence:** the mutation of R6 is restated: without the rule a waiting commit holds no block. The prediction for R6 stands and is run again as entry 6.

## 6. R6 left out, restated (`MC_Core`, `RULES=NoR6`), 2026-11-12

- **Prediction:** P2 `SharedUnwritten` or P1 `WriteIsolation`, by the behaviour of entry 5.
- **Outcome:** as predicted: `WriteIsolation` in 12 states.

## 7. Copies and the host tier (`MC_Copies`), 2026-11-12

- **Model:** `KvCore` with `Clone`, `Load`, `Store`, `EvictHost`, `Finish` from the executor and as the scheduler's give-up, and R5. With no copies and no host units the model has the states of entry 4 (22,754,006 distinct, run again after the change). Two units, one slot row, one host unit, one key, two owners, one group, two batches, one request under way besides the executor's `Finish`, two copies.
- **Prediction:** the search completes with no violation.
- **Prediction, R5 left out:** the cached block becomes an owner's private block: P1 `Exclusive` and P2 `KeyStable` are false in the same state; TLC reports the first listed.
- **Outcome, R5 left out:** as predicted: `Exclusive` in 11 states.
- **Outcome, all rules:** not as predicted: `KeyStable` in 12 states, `Evict` taking a block a `Store` was reading. The cause was the model's text, not the design: the two conjuncts of `Evictable`'s guard for R2 were misaligned, so the second was read inside the first's quantifier and a copy's hold was not checked. The guard is corrected and the configuration run again as entry 8.

## 8. Copies and the host tier, guard corrected (`MC_Copies`), 2026-11-12

- **Prediction:** the search completes with no violation. `MC_Core` is unaffected by the correction (it has no copies) and is not run again for it.
- **Outcome:** as predicted. The search completed: 27,997,818 states generated, 6,648,697 distinct, depth 51, no violation, in 34 s.

## 9. Worker groups in the model; the earlier configurations again, 2026-11-12

- **Model:** `KvCore` gains the validity of a shared block per group, tasks as holders, `Transfer` with its task, and the rule that an owner whose transfer waits or runs takes only a `Detach`. No prediction was written before `MC_Copies` was run again after the change; it was expected to pass as in entry 8.
- **Outcome:** `MC_Copies` completed with no violation: 140,773,442 states generated, 33,499,148 distinct, depth 61, in 2 min 54 s. The state count rose because `Transfer` requests are now issued (and refused, with one group).

## 10. Transfers between two groups (`MC_Transfer`), 2026-11-12

- **Model:** two groups of one epoch with one slot row each, two units, one key, two owners, one position, one batch per group, one stamped request under way, no copies.
- **Prediction, all rules:** the search completes with no violation, and an owner attached to the group it was transferred to, with its block in that group's table, is reached.
- **Prediction, R9 left out:** P1 `TaskIsolation`: an owner is detached while its transfer's task runs, its block is reclaimed, and the task still writes it. `FreeUnheld` cannot report it, since without R9 a task is not a holder.
- **Prediction, R3 left out:** P1 `WriteIsolation` or `ReadIsolation` by a `Release`, as in entry 5; if the shortest behaviour uses the transfer instead, `TaskIsolation` cannot fail (the task holds what it writes), so one of the two is reported.
- **Outcome, all rules:** not as predicted: `ReadIsolation` in 13 states. A commit waits; its owner is transferred, and the task copies the block into the target group's memory; the commit comes due and the block is valid in the group that wrote it alone, while the owner, now in the target group, reads it there. The model recorded validity at the commit only for the writing group.
- **Outcome, R9 left out:** not as predicted: `NoLeak` in 7 states. The mutation as modelled made a task hold nothing, so the slot row it took was neither free nor held: an artefact of the mutation.
- **Outcome, witness:** an owner attached to the group it was transferred to is reached.
- **Consequence:** a block whose commit takes effect after its owner was transferred with it is valid in both groups (the transfer copied it, and nothing wrote it since the commit was stamped); `specs/buffer-pool.md` §13 states it. The mutation of R9 is restated: a task holds its blocks, and without the rule a `Detach` releases them while the task runs.

## 11. Transfers, restated (`MC_Transfer`), 2026-11-12

- **Prediction, all rules:** the search completes with no violation.
- **Prediction, R9 left out:** P1 `TaskIsolation`: the released block is free, or another owner's, while the task writes it.
- **Outcome, R9 left out:** as predicted: `TaskIsolation` in 10 states.
- **Outcome, all rules:** not as predicted: `ReadIsolation` in 16 states. As entry 10, but the owner's position is released after the transfer and before the commit comes due, so the rule of entry 10 (valid in the owner's group if the owner still holds the block) leaves the block valid in the writing group alone while a batch of the target group still reads it.
- **Consequence:** the groups in which a block is valid are the block's own, kept while it is private: the group of its owner at allocation; after a transfer the target group alone, or both groups when a commit of the block waits (the block is then read-only in both). A commit does not change them. `ReadIsolation` requires every device block a batch reads, private or shared, to be valid in the batch's group.

## 12. Transfers, validity kept per block (`MC_Transfer`), 2026-11-12

- **Prediction, all rules:** the search completes with no violation.
- **Outcome:** not as predicted: `TaskIsolation` in 16 states. A block whose commit waits is transferred to the second group and back; the second task copies it into the first group's memory, where it is still valid, and the commit comes due while the task writes it.
- **Consequence:** a task copies exactly the owner's blocks that are not valid in the target group, private or shared.

## 13. Transfers, a task copies what is not valid in the target (`MC_Transfer`), 2026-11-12

- **Prediction, all rules:** the search completes with no violation.
- **Outcome:** as predicted. The search completed: 186,819,804 states generated, 37,541,052 distinct, depth 86, no violation, in 3 min 32 s.

## 14. The earlier configurations on the model with worker groups, 2026-11-12

- **Prediction:** `MC_Core` and `MC_Copies` complete with no violation; `RULES=NoR9` on `MC_Transfer` reports `TaskIsolation`, and the mutations of R1 to R7 report a violation as before.
- **Outcome:** as predicted. `MC_Core`: complete, 773,021,171 states generated, 150,548,210 distinct, depth 52, no violation, 15 min 59 s on 48 workers. `MC_Copies`: complete, 94,751,160 generated, 22,596,078 distinct, depth 54, no violation, 2 min 35 s. Mutations: R1 `NoLeak`, R2 `FreeUnheld`, R3 `WriteIsolation`, R4 `KeyStable`, R6 `WriteIsolation`, R7 `KeyStable` on `MC_Core`; R5 `Exclusive` on `MC_Copies`; R9 `TaskIsolation` on `MC_Transfer`.

## 15. Two epochs of one layout and the trial (`MC_Trial`), 2026-11-12

- **Model:** validity is by device memory (`GroupMemory`): the groups of two epochs of one layout map the same memory, so a block one epoch committed is valid for the other. An epoch may be on trial: what it commits is visible to it alone (`vis`) until `CommitUpdate`; `Rollback` needs its owners gone and leaves its blocks visible to it alone; R1 refuses an `Attach` for a rolled-back epoch. Property P5 `TrialVisibility` is added: a block visible to one epoch is pinned and copied only for that epoch. Configuration: a serving epoch and a target on trial, one group and one owner each over the same memory, two units, one key, one batch per group, one stamped request under way.
- **Prediction, all rules:** the search completes with no violation; the serving epoch's owner pinning a block the target committed, after the update is committed, is reached.
- **Prediction, R8 left out:** P5 `TrialVisibility`: the serving epoch's owner pins a block the target committed while it is on trial. None of P1 to P3 is violated first on a shortest behaviour.
- **Outcome:** as predicted. All rules: complete, 36,713,471 states generated, 7,388,077 distinct, depth 48, no violation, 37 s; the witness is reached. R8 left out: `TrialVisibility`.

## 16. Regions of two layouts and `Resize` (`MC_Resize`), 2026-11-12

- **Model:** a device unit belongs to a region and occupies a chunk of device memory; units of two regions may occupy the same chunk. `member` is the units in their regions; `Shrink` takes a free unit out of its region and `Grow` puts a unit in, under R10 only when no unit of a region occupies its chunk. An owner allocates in the region of its scheduler. `NoLeak` requires a unit outside its region to be neither free nor held, and `OneRegion` (P3) that no two units in regions occupy one chunk. Configuration: a serving epoch with a region of two units and a target epoch of another layout whose region starts empty over the same two chunks; one group, one slot row and one owner each; one batch per group; one stamped request under way.
- **Prediction, all rules:** the search completes with no violation; both owners holding a private block at once is reached.
- **Prediction, R10 left out:** P3 `OneRegion`: a unit of the target's region grows over a chunk a unit of the serving region still occupies.
- **Outcome:** as predicted. All rules: complete, 73,399,823 states generated, 16,278,793 distinct, depth 56, no violation, 1 min 26 s; the witness is reached. R10 left out: `OneRegion`.

## 17. The tiers, and the loss of an engine (`MC_Tiers`, `MC_Loss`), 2026-11-12

- **Model:** tasks of the tiers: `Prefetch` from the remote store or a peer into a new host block, `Store` to the remote store by the connector's task, a deadline after which the request is answered, and the peer letting go of its pin; `Save` and `Restore` as copies between an owner's slot and a host block; the loss of an engine (A7): its batches and copies are over, its waiting commits and transfers are dropped, its owners are detached. `SourcesHeld` joins P2: no copy is cached from a source that was not held throughout its read. `Hazards` switches on behaviour of the system as built; H2 caches what a prefetch read whatever its peer did. `MC_Trial` was run after the change without a written prediction, as a check that the model still parses, and completed with no violation (8,552,255 distinct states; more than in entry 15 because `Shrink` and `Grow` now apply to it).
- **Prediction, `MC_Tiers`, all rules:** the search completes with no violation; a key acknowledged by the remote store, a prefetched host copy and a restore that came due are reached.
- **Prediction, `MC_Tiers`, R9 left out:** P1 `TaskIsolation`: a prefetch passes its deadline, its host block is released, and the task still writes it.
- **Prediction, `MC_Tiers`, hazard H2:** P2 `SourcesHeld`: a prefetch from a peer caches its block after the peer let go of its pin.
- **Prediction, `MC_Loss`, all rules:** the search completes with no violation; the surviving epoch's owner pinning a block the lost epoch committed is reached.
- **Outcome, `MC_Tiers`, R9 left out:** as predicted: `TaskIsolation`. **Hazard H2:** as predicted: `SourcesHeld`. **Witnesses:** all three of `MC_Tiers` and that of `MC_Loss` are reached.
- **Outcome, `MC_Tiers`, all rules:** not as predicted: `NoLeak` in 12 states. An owner saves its trajectory and is detached while the copy runs; when the `Finish` takes effect the slot row, which only the copy still held, is neither free nor held. The cause was the model's text: the branch that caches a stored or saved host block did not let go of what the copy read. Corrected; run again as entry 18.

## 18. The tiers, corrected, and the loss of an engine (`MC_Tiers`, `MC_Loss`), 2026-11-12

- **Prediction:** both searches complete with no violation.
- **Outcome, `MC_Tiers`:** as predicted: complete, 184,027,175 states generated, 32,587,250 distinct, depth 44, no violation, 3 min 11 s.

## 19. The whole suite on the model with tiers and losses (`suite.sh`), 2026-11-12

- **Prediction:** every configuration of the design (`MC_Core`, `MC_Copies`, `MC_Transfer`, `MC_Trial`, `MC_Resize`, `MC_Tiers`, `MC_Loss`) completes with no violation; each rule left out and hazard H2 report the invariants `suite.sh` states for them.
- **Outcome:** INCONCLUSIVE for four configurations. Every mutation and hazard H2 reported the stated invariant, and `MC_Loss` (3,205,242 distinct states), `MC_Trial` (8,552,255) and `MC_Resize` (16,278,793) completed with no violation. The run was stopped while `MC_Core`, `MC_Copies`, `MC_Transfer` and `MC_Tiers` were still searching, without a violation: `Shrink` and `Grow` apply to every unit of every configuration since entry 16 and multiply their states.
- **Consequence:** a constant `Movable` names the units `Resize` may move; it is empty except in `MC_Resize`.

## 20. The hazards of the system as built (`suite.sh hazards`), 2026-11-12

Each hazard of `specs/kv-cache-s0.md` §6 is a switch of `KvCore` that replaces the designed behaviour by the behaviour that document describes. H6 and H13 are violations of P4 and are not in this entry.

| Hazard | As modelled | Configuration | Predicted violation |
| --- | --- | --- | --- |
| H1 | A `Detach` releases what the owner's transfer holds while its task runs | `MC_Transfer` | P1 `TaskIsolation` |
| H2 | A prefetch caches what it read whatever its peer did | `MC_Tiers` | P2 `SourcesHeld` |
| H3 | The end of a failed copy takes effect without waiting for its stamp | `MC_Copies` | P1 `CopyIsolation` |
| H4 | A key is taken to be on the host when its store is asked | `MC_Tiers` | P6 `Exactness` |
| H5 | A `Detach` lets go of what the owner's copies hold while they run | `MC_Copies` | P1 `CopyIsolation` or `FreeUnheld` |
| H7 | The session of a request is not checked | `MC_Trial` | P1 `WriteIsolation` or `ReadIsolation` |
| H8 | The executor's rows are never cleared | `MC_Core` | P1 `WriteIsolation` or `ReadIsolation` |
| H9 | An epoch that adopted another's cache evicts what the other epoch pins | `MC_Trial` | P1 `Exclusive`, `WriteIsolation` or `ReadIsolation` |
| H10 | A block is shared while its writer still appends | `MC_Core` | P1 `WriteIsolation` or P2 `SharedUnwritten` |
| H11 | An `Evict` is answered as done and the block stays cached | `MC_Core` | P6 `Exactness` |
| H12 | State kept twice: the free list misses an evicted block | `MC_Core` | P3 `NoLeak` |

- **Outcome:** as predicted for all eleven: H1 `TaskIsolation`, H2 `SourcesHeld`, H3 `CopyIsolation`, H4 `Exactness`, H5 `CopyIsolation`, H7 `WriteIsolation`, H8 `WriteIsolation`, H9 `Exclusive`, H10 `WriteIsolation`, H11 `Exactness`, H12 `NoLeak`. The thirteen goals of `addresses.smt2` are `unsat` (z3 4.8.9).

## 21. The design and the mutations on the model with hazard switches (`suite.sh design mutations`), 2026-11-12

- **Prediction:** the seven configurations complete with no violation and the eleven mutations report the invariants `suite.sh` states.
- **Outcome:** INCONCLUSIVE, stopped on purpose after the eleven mutations reported the stated invariants and `MC_Loss` completed (2,754,061 distinct states): reading the model for its inductive invariant showed that one visibility per key is wrong once a key has copies on two tiers, and the model changes again (entry 22).

## 22. Visibility per copy (`MC_Trial`, `MC_TrialTiers`), 2026-11-12

- **Model:** a key's device copy and its host copy each have a visibility of their own, set when the copy is made by the epoch that makes it; a prefetched host copy is one every epoch may use; an epoch on trial fills no memory of a device block others may use (it loads into a block of its own, which stays private when the key already has a device block), and its host copies do not go to the remote store. `MC_TrialTiers` is `MC_Trial` with one host unit and two copies. The switches H6 (no deadline) and H13 (no give-up of a copy) are added for the liveness checks.
- **Prediction:** `MC_Trial` and `MC_TrialTiers` complete with no violation; with R8 left out both report `TrialVisibility`.

## 23. Finality as a checked property (`MC_Finality`), 2026-11-12

- **Model:** `KvFinality` adds fairness to `KvCore` (the scheduler gives a copy up, stamped requests are stamped and taken, a batch ends, a due `Finish` takes effect, a task's deadline passes) and two temporal properties: a copy in flight ends, and a request that waits on a prefetch or a remote put is answered. One unit, one slot row, one host unit, one key, one owner, one batch, one request, one copy, one task; no symmetry.
- **Prediction, design:** both properties hold.
- **Prediction, H6 (no deadline):** `TaskFinality` is violated by a task that never completes.
- **Prediction, H13 (no give-up):** `CopyFinality` is violated by a copy whose executor never returns its `Finish`.
- **Outcome:** as predicted. Design: complete, 845,591 states generated, 183,894 distinct, both properties hold. H6: `TaskFinality` violated. H13: `CopyFinality` violated.

## 24. `MC_TrialTiers`, smaller, 2026-11-12

- **Outcome of entry 22:** `MC_Trial`: as predicted, complete, 7,388,077 distinct states, no violation; R8 left out reports `TrialVisibility` on both configurations. `MC_TrialTiers` with two units and two copies: INCONCLUSIVE, stopped at 114 million distinct states with the queue still growing and no violation.
- **Model:** `MC_TrialTiers` with one unit and one copy.
- **Prediction:** the search completes with no violation; R8 left out reports `TrialVisibility`.
- **Outcome:** as predicted: complete, 88,133,355 states generated, 17,599,952 distinct, no violation; R8 left out reports `TrialVisibility`.

## 25. The whole suite on the model as it stands (`suite.sh`), 2026-11-12

- **Prediction:** the nine configurations of the design complete with no violation (`MC_Finality` with its two temporal properties); the twelve mutations and the thirteen hazards report the properties `suite.sh` states; the thirteen address goals are `unsat`.
- **Outcome:** INCONCLUSIVE, stopped on purpose: writing the inductive invariant showed two places where the model is looser than the control plane the documents describe (entry 26), and the suite is run once after them.

## 26. Stamping groups fixed at issue; a transfer hands on only unnamed entries, 2026-11-12

- **Model:** the groups that stamp a request are recorded in the request when it is issued (`sg`), as `specs/scheduler-design.md` §3.1 has the scheduler hand a request to its executors once; before, they were read from the owner's group at the moment of submission. A transfer that ends hands the target executor only the entries no stamped request under way names (A1), where before only a `Detach` under way was considered.
- **Prediction:** `MC_Transfer`, the configuration these two changes can affect most, completes with no violation; the other configurations and every mutation and hazard are run once by `suite.sh` after the inductive invariant is settled.
- **Note:** the first run after this entry searched the model before the change, because the edit had not been applied; its result (no violation) says nothing about the change and is not counted.
- **Outcome:** as predicted: `MC_Transfer` completes with no violation on the changed model: 182,674,180 states generated, 38,135,896 distinct.

## 27. The inductive invariant as an invariant (`IndInv` of `KvInductive`), 2026-11-12

- **Model:** `KvInductive` states `IndInv`: the allocation of every unit, what each deferred item, copy and task holds, what an executor's table and a completion not yet handed on may name, the stamps of requests, and what covers each unit a batch in flight addresses. Checked first on `MC_Loss` and `MC_Trial` without a written prediction: no violation (2,788,873 and 7,459,267 distinct states).
- **Prediction:** `IndInv` is an invariant of `MC_Core`, `MC_Copies`, `MC_Transfer`, `MC_Tiers`, `MC_TrialTiers` and `MC_Resize`. Whether it is inductive is a separate question, for Apalache and TLAPS.
- **Outcome so far:** `IndInv` is an invariant of `MC_Transfer` (38,135,896 distinct states), `MC_Tiers` (32,599,322), `MC_TrialTiers` (17,941,500) and `MC_Resize` (16,365,337); `MC_Core` and `MC_Copies` are still searching.

## 28. Apalache: types, the initial state, and what `IndInv` implies (`MC_Apalache`), 2026-11-12

- **Model:** `KvCore` and `KvInductive` carry Apalache's type annotations; the private holders of a unit are stated as two sets of one type each (`TableHolders`, `CopyWriters`), and one action's duplicate `UNCHANGED` is removed: neither changes the model's meaning. `IndInv` gains `TypeInv`, the values every variable takes. `MC_Apalache` fixes the constants (two epochs, three groups over two memories, two units, three slot rows, one host unit, one key, two owners, the second epoch on trial, either engine losable, both values of `Eager`) and generates any state of collections of at most three elements that satisfies `IndInv`.
- **Not predicted in writing before the runs:** the type checker passes; `Init => IndInv` holds (length 0); `IndInv => Safety` holds for the generated states (3 min 46 s); the generated predicate is satisfiable by a state with a batch, a deferred item, a request, a copy and a task (Apalache refutes `NoWork`).
- **Prediction for the inductive step** (`--init=IndInit --inv=IndInv --length=1`): I expect at least one counterexample to induction, most likely in `BatchesOk` or `RequestsOk`, whose clauses about stamps were written from the argument and never checked from an arbitrary state.
- **Outcome so far, entry 27:** `IndInv` is an invariant of `MC_Copies` (47,047,135 distinct states) and of `MC_Core` as well: of all six.
- **Outcome, entry 28, the inductive step:** as predicted, not inductive, though not where predicted: after 28 minutes Apalache reported `HandoffOk` violated by `Lose` with `Eager` false. The state before held a completion not yet handed on, for an owner of the epoch that is lost but a group of the other epoch, named by a request of the lost epoch; the loss removes the request and leaves the completion, which is then unnamed and names no live owner. `IndInv` said nothing about a named completion.
- **Consequence:** `HandoffOk` states for every completion not yet handed on that its group is of its owner's epoch, and `DeferredOk` and `TasksOk` state the same of a transfer's target.

## 29. The inductive step again (`MC_Apalache`), 2026-11-12

- **Prediction:** another counterexample to induction, in `BatchesOk`, `RequestsOk` or `TablesMapped`. The check now skips the invariants of the generated state (`search.invariantFilter=1->.*`).
- **Outcome:** as predicted, another counterexample, after 14 minutes, though again not in the conjuncts named: `CopiesOk` violated by `Attach`. The state before held a copy for an owner that was never attached; attaching it to another group breaks "an owner a copy writes for is in the copy's group". `IndInv` did not say that an owner a copy names was attached.
- **Consequence:** `AllocOwner` (a unit is allocated only to an owner that was attached) joins `IndInv`; `CopiesOk` and `DeferredOk` state it of the owner a copy or a waiting transfer names; a remote put's block is a host unit.

## 30. The inductive step, third run (`MC_Apalache`), 2026-11-12

- **Prediction:** another counterexample to induction from a state no behaviour reaches; where, I cannot say.

## 31. Outcome of entry 25: the whole suite, 2026-11-12

- **Outcome:** as predicted, on the model with stamping groups fixed at issue (entry 26) and Apalache's type annotations (entry 28): the nine configurations complete with no violation, the twelve mutations and the thirteen hazards report the stated properties, and the thirteen address goals are `unsat`. Distinct states: `MC_Core` 172,191,447; `MC_Copies` 47,047,135; `MC_Transfer` 38,135,896; `MC_Tiers` 32,599,322; `MC_TrialTiers` 17,941,500; `MC_Resize` 16,365,337; `MC_Trial` 7,459,267; `MC_Loss` 2,788,873; `MC_Finality` 188,424 with both temporal properties. The run's lines are kept in `artifacts/buffer-pool/suite-2026-11-12.txt`.
- **After the run, without a change of meaning:** `Reached` is stated without `CHOOSE` (a stamp is reached when every batch in flight of the group is later than it, or none is in flight and it is at most the last launched); the property formulas moved from `KvCore` into one module per property (`KvP1`, `KvP2`, `KvP3`, `KvP5`, `KvP6`, with `KvFinality` for P4) and are pinned (`pins.py`). Checked after each: `MC_Loss` and `MC_Finality` give the same state counts, a mutation and a hazard report the same invariants.
- **Outcome:** INCONCLUSIVE: the third run was stopped to check the step for ten groups of actions in ten processes at once (`apalache_step.sh`, `Next1` to `Next10` of `MC_Apalache`), which gives a counterexample per group and round instead of one.

## 32. The inductive step by groups of actions (`apalache_step.sh`), 2026-11-12

- **Prediction:** several groups report a counterexample to induction; at least the groups of the simplest actions (`Next3`, `Next5`) report none.
- **Outcome so far:** as predicted in kind. `Next10` (the loss of an engine): no error. Three counterexamples from states no behaviour reaches: `Next2`, a `Finish` under way names a copy that does not exist yet and `Clone` creates it; `Next8`, a waiting transfer that holds a unit is dropped by `TransferDue` and the unit leaks; `Next3`, `EvictHost` takes a host block that a waiting release holds. `Next3` was predicted to pass and did not. The other six groups are still running.
- **Consequence:** `DeferredOk` states what each kind of deferred item holds, `TasksOk` what each kind of task holds, and `RequestsOk` that a `Finish` names a copy that was asked for.
- **Further outcome:** `Next4`: a counterexample by `Issue`. The state before had a unit both private in its owner's table and held by a waiting release, with a batch of another group writing it; a `Commit` issued for the position is stamped by the owner's group, not by the batch's. `DeferredOk` now states that what a release holds is no owner's private block or slot row.
- **Further outcomes:** `Next7`: `CommitDue` shares a block that a copy also writes (a state no behaviour reaches); `DeferredOk` now states that no copy writes the block a commit waits to share. `Next3`, second round: `EvictHost` takes a host block that a transfer's task holds; `TasksOk` now states that a transfer holds device blocks and slot rows only.
- **Further outcome, `Next8`, second round:** a counterexample by `TransferEnd` that a behaviour can reach in kind. Two owners pin one shared block that is valid in the source group's memory only; both are transferred to groups of one other memory, and each task copies the block there. When the first task ends the block is valid in that memory, and the second task still writes it: `TaskIsolation`. `MC_Transfer` cannot show it, because its target group has one slot row and so one transfer at a time. Entry 33 takes it to TLC.

## 33. Two transfers that copy one shared block (`MC_TransferPair`), 2026-11-12

- **Model:** `MC_TransferPair` is `MC_Transfer` with two slot rows in each group, so two owners can be in the prefill group at once and in transfer to the decode group at once.
- **Prediction, on the model as it stands:** TLC reports `TaskIsolation` violated by a behaviour in which both owners pin the key's block, both transfers become due, and one task ends while the other is in flight.
- **Prediction, after the design change:** the design already makes a transfer wait while a `Load` fills the target group's memory of a block the owner holds. The same wait is stated for a transfer in flight that copies such a block into that memory: the second transfer becomes due only when the first has ended, and then copies nothing that is valid. With it `MC_TransferPair` completes with no violation, `MC_Transfer` keeps its state count (38,135,896 distinct: its target group never has two transfers), and the mutation `NoR9` and hazard `H1` on `MC_Transfer` still report `TaskIsolation`.
- **Outcome, on the model as it stood:** the first run was of a wrong configuration and is INCONCLUSIVE: it gave the target group two rows and left the source group one, so the two owners were never in the source group together; it was stopped at 140,118,753 distinct states with no violation. With two rows in each group TLC reports `TaskIsolation` violated as predicted, by a behaviour of 18 states found in 12 s: both owners pin the block, both transfers come due, and the first task's end makes the block valid in the decode group's memory while the second task writes it.
- **Outcome, after the design change:** as predicted. `MC_TransferPair` completes with no violation on the changed model: 2,614,092,964 states generated, 487,956,474 distinct, depth 94, 1 h 09 min.

## 34. The inductive step, rounds 3 and 4 (`apalache_step.sh`), 2026-11-12

- **Outcome of the rounds started before the transfer's wait:** `Next8` with the wait and its clause of `TasksOk`: no error. `Next10`: no error again. `Next3` on the invariant before that clause: no error. Two counterexamples: `Next2`, `Store` for a group whose epoch is lost makes a copy of a lost epoch; `Next7`, `CommitDue` shares on trial a block that a `Save` of another epoch's group reads.
- **What they show:** the first is the model's, not the invariant's: every other handler names an owner, which a lost epoch has none of, and `Store` names a group and had no guard, so a behaviour could store for a session that is closed. The buffer pool refuses it with the cause `group` (R1: its groups are gone with its session); the model now states that guard, and the mutation `NoR1` drops it. The second is a state no behaviour reaches: a `Save` reads its owner's slot row, and a copy's owner is of its group's session; `CopiesOk` now states both.
- **Prediction:** with these, `Next2` and `Next7` report no error; `IndInv` stays an invariant of `MC_Loss`, `MC_Tiers` and `MC_Copies`, whose state counts are those of entry 31 (no behaviour of theirs stored for a lost epoch: `MC_Loss` has no host unit).
- **Two configurations for the suite, predicted:** `MC_LossCopies` (`MC_Copies` with its epoch losable: the loss of an engine with copies in flight, which no configuration had) completes with no violation; `MC_TransferPair` with one unit instead of two (the block both owners pin) completes with no violation and is small enough for the suite.
- **Outcomes so far:** `MC_LossCopies` completes with no violation (245,379,791 states generated, 47,048,530 distinct, depth 53), and so with `IndInv` among its invariants. `MC_TransferPair` with one unit completes with no violation (53,095,803 generated, 11,593,052 distinct, depth 67). `IndInv` stays an invariant of `MC_Loss` (2,788,873 distinct, as in entry 31) and of `MC_Tiers` (32,599,322, as in entry 31). `Next5`: no error. `MC_Copies` with `IndInv`, the reruns of `Next2` and `Next7`, and the whole suite on the changed model were still running.
- **Further outcomes:** the whole suite on the changed model: every line as stated (eleven design configurations with no violation, among them `MC_LossCopies` and `MC_TransferPair`; the twelve mutations and thirteen hazards report the stated properties; the address goals `unsat`; the pins match). `IndInv` is an invariant of `MC_Copies` (47,047,135 distinct states), `MC_Transfer` (38,135,896) and `MC_TransferPair` as well. Apalache on the invariant of 15:30 (before the clauses of this entry): `Next1`, `Next4`, `Next6` and `Next9` report no error. On the invariant with this entry's clauses: `Next2` reports no error, as predicted; `Next7` does not: `FinishDue` of a clone makes its new block valid in the clone's group only, and a batch of another group in flight reads it, covered by a waiting release that holds the clone's new block. No behaviour reaches that state: a clone's or a load's new block is free when the copy takes it, and only the copy holds it until its `Finish`. `DeferredOk` now states that a release holds no new block a copy writes.
- **Prediction for the last round:** every group of `MC_Apalache` reports no error on the invariant with this clause. The round runs on r2, all ten groups at once.

## 35. P7 Capacity identity, through the five stages (L7), 2026-11-13

- **Specification.** The capacity identity of `specs/buffer-pool.md` §17.2, named by the owner's deliverable L7, is property P7: per region, the units that are free, private, shared and retired add up to the units in the region. It is stated in `KvP7.tla` over the vocabulary of §17.1 only (who holds each unit, the free list, the members of each region): no history variable, no change to `KvCore`, to the Rust buffer pool, to its interface, or to the entries and files of P1 to P6. A unit's observed state is the first that applies of shared (a cache entry on either tier), private (an owner's private entry or slot row, a waiting commit, a copy writing it for a target that exists, a transfer's new slot row for an owner still attached, a prefetch's host block) and retired (anything else that holds it), as `specs/buffer-pool.md` §8 states and `BufferPool::observed` computes.
- **Predictions, validation.** (a) Every design configuration of the suite completes with P7 among its invariants, with the state counts of entry 31 and of entry 34 (`MC_LossCopies`, `MC_TransferPair`): a property adds no state. (b) With only P7 checked, the hazard H12 (the free list misses an evicted block) violates it: the block's unit is neither free nor held, and the region's units add up to one less than its capacity. (c) With only P7 checked, the mutation `NoR2` (runs return to a region while other holders remain) violates it: a reclaimed unit is free and still held, and the region's units add up to one more.
- **Prediction, verification.** `IndInv => P7` is proved by TLAPS from `FreeClean`, `NoLeak` and `Regions` with the finite sets of the configuration, so that P7 is proved for the design model as soon as the step `IndInv /\ [Next]_vars => IndInv'` is.
- **Outcomes so far.** (b) and (c) as predicted: with only P7 checked, `MC_Core` with H12 reports `CapacityIdentity` violated after 11 states (an evicted block's unit is neither free nor held), and `MC_Core` with `NoR2` after 11 states (a release frees a unit a waiting commit still holds). Both runs are the suite's new part `validation`. The prediction for the proof holds: `KvProofP7.tla` proves `IndInv => P7` (135 obligations), under `Design`, `Sizes` and `Finite` (the configuration's sets of units are finite). (a) runs with the whole suite on r2.
- **Outcome (a).** As predicted: the whole suite with P7 among the invariants of every design configuration reports every line as stated (40 lines: eleven configurations with no violation, with the state counts of entries 31 and 34; twelve mutations; thirteen hazards; the two runs of the part `validation`; the address goals; the pins). P7 is `checked`.
- **`IndInv` as a TLC invariant**, on the invariant with entry 34's clauses: no violation in any of the ten safety configurations (`MC_Loss`, `MC_Trial`, `MC_TransferPair`, `MC_Resize`, `MC_TrialTiers`, `MC_Tiers`, `MC_Transfer`, `MC_Copies`, `MC_LossCopies`, `MC_Core`; 172,191,447 distinct states for `MC_Core`).

## 36. The step proof, first groups (TLAPS), 2026-11-13

- **Outcome.** `KvStep3.tla` (572 obligations), `KvStep10.tla` (405) and `KvStep8.tla` (1,145) prove `IndInv /\ NextN => IndInv'` for the groups `Next3`, `Next10` and `Next8` with every obligation proved by tlapm (shared lemmas in `KvFrame.tla`, 114). `Next5` cannot be proved as stated: `TypeInv` states each field of a copy but not that a copy is a record of those fields alone, so after `[c EXCEPT !.seq = n]` the other fields are not determined in TLA+'s logic; the same holds of tasks (`PeerLost`, `Deadline`) and requests (`ExecStamp`). No behaviour reaches such a state, and TLC, where every copy is a record, cannot show one.
- **Consequence.** `TypeInv` states that a copy, a task and a stamped request are records of their fields alone.
- **Prediction.** With the clause, `KvStep5` proves `IndInv /\ Next5 => IndInv'` as stated, `KvProof.tla` still proves its two theorems, and `IndInv` stays a TLC invariant of every configuration (a record of its fields is what every action makes).
- **Outcome.** As predicted: `KvStep5` proves with the clause (435 obligations), `KvProof` proves its two theorems (164), the record clauses also closed `PeerLost` and `Deadline` in `KvStep9`, and `IndInv` with them is a TLC invariant of every configuration (`artifacts/buffer-pool/indinv-check2` for eight; the two largest, rerun with entry 41's clause, in `indinv-check3`).

## 37. A recorded run of the buffer pool's integration test against the model's steps (`steps.py`, `KvTrace`), 2026-11-13

- **Run:** the Storage Server's integration test of a session (`an_engine_opens_a_session_and_its_schedulers_are_served_on_their_queue_pairs`) with the trace and the check on; its trace replayed on a new pool with the states written (22 inputs: the pool's image, the open, 19 inputs of the session among which 9 are the clock, the close); `steps.py --chain 40` on them. Constants from the run: 8 KV units, 4 slot rows of two groups on two memories, 4 host units, owners 1 and 2, positions 0 and 1, key 07, the values 0, 4, 5 of `completed` and of the stamps renumbered 0, 1, 2.
- **Prediction:** the image, the open and the close are outside the model; the 19 steps between the open and the close are one window from the state after the open, and TLC follows every one of them: the attach, the allocation of two blocks, the commit stamped with batch 5 waiting, `completed` at 4, the release of position 1 at once, `completed` at 5 making the commit due, the detach that leaves the shared block cached, the attach of the decode owner, a pin that finds no block valid in its group's memory, an unreadable record, and the clock as stutters. A few hundred distinct states, seconds.
- **Outcome:** as predicted: the 19 steps followed in one window, 188 distinct states, 1 s with 16 workers; the image, the open and the close outside. A copy of the replay's states with one state altered by hand (the release of position 1 leaves its unit out of the free units) is reported at that step (110 distinct states).

## 38. L1 step by step, `core` (`agreement_steps.sh`), 2026-11-13

- **Run:** the enumeration of the Rust buffer pool writes every transition of `core` (a state, the input taken in it, the next state; the 1,281 states of entry L1) with its input; each is a window of one step that TLC must follow with `MC_Pool`'s steps, from the source state with what it leaves out (the stamps of waiting work) chosen. Then the transition made by hand, a commit that takes effect while its stamp 1 is not reached and `completed` reads 0.
- **Prediction:** every transition of `core` is followed: a request that takes effect at once is its handler; a stamped one is its arrival, its taking and, when its stamps are reached, its work coming due; an advance of `completed` is `Advance` and the work it makes due. The transition made by hand is reported: no path reaches a cached key with `completed` at 0 by the commit's input, since `CommitDue` needs the stamp reached and only an advance moves `completed`. TLC's runs take minutes, not hours.
- **Outcome:** not as predicted. The transition made by hand is reported, as predicted, but 807 of the 5,553 transitions of `core` were not followed (3 TLC runs, 28 s). None is a difference between the Rust buffer pool and the model; each is the check's: (1) a window's first state took a stamp for a release that waits even when none waits, from a set that is empty once `completed` is at the largest value renumbered, so the window had no first state (most of them); (2) the renumbering had no value above the ones seen, so a stamp of waiting work, which no state states, could not be chosen above `completed`; (3) the Rust buffer pool takes owner numbers in increasing order on a queue pair, so attaching owner 2 first leaves number 1 gone though it was never attached, which the state it writes cannot tell apart, while the model keeps that owner `new` (20 transitions); (4) a state states the memories of the blocks an owner or the cache holds, not of a block only a waiting commit holds, which the first state left in no memory (412 advances of `completed` that make such a commit due). `KvTrace` and `steps.py` changed accordingly: the release's stamp is taken only when a release waits; one value above those seen stands for the stamps of waiting work; an owner the Rust buffer pool writes as gone matches a model owner that is gone or never attached; the memories of a held block the state does not state are chosen. The reruns after the changes were not predicted separately: every transition of `core` followed (5,553 windows, 3 TLC runs, 14,633 + 21,827 + 25,582 distinct states, 113 s with 16 workers), the transition made by hand reported, and the recorded run of entry 37 followed again (188 distinct states).
- **What it shows of the Rust buffer pool:** its rule R1 for owners is stronger than the model's: a number below one taken on the queue pair is refused, where the model refuses only an owner taken before. Every behaviour of the Rust rule is one of the model's.

## 39. L1 step by step, `tiers` (`steps.py` on the enumeration's `tiers.steps`), 2026-11-13

- **Run:** as entry 38 after its changes, on the 6,071 transitions of `tiers` (one owner, one KV unit, one slot row, one host unit; copies, a prefetch or a put in flight, the deadline, a task's key done and its end).
- **Prediction:** every transition is followed: a `Prefetch` or a put is the model's task, the tick past a deadline is `Deadline`, a key done is the task's end with what it found (the model's task is one key), the end of a task that found nothing is its end without it, and a `Finish` of a load, a store, a save or a restore is its arrival, its taking and its coming due. About two minutes.
- **Outcome:** not as predicted: 140 of the 6,071 transitions were not followed (4 TLC runs, 75 s), every one an advance of `completed` that makes a commit due while the key has a host copy that a `Save` made. The Rust buffer pool keeps a key's class: the key is cached as a stored trajectory, so the KV block committed under it is a duplicate and stays private (R4 with the entry's class); the model has one class and shares the block. The enumeration names a stored trajectory and a KV block by one key, which A6 excludes (a key names what its block holds), so these transitions are outside the model's terms and show no defect of either. The states for the step check now carry the class of each cached key in `#` lines (agreement.py's states do not), and a state with a key cached in a class other than 0 is outside the model.
- **Prediction for the rerun:** with those states outside, every remaining transition of `tiers` is followed, and `core`, which caches only KV blocks, is unchanged (5,553 transitions followed; the transition made by hand reported).
- **Outcome of the rerun:** as predicted. `core`: 5,553 transitions followed (3 TLC runs, 14,471 + 20,442 + 25,070 distinct states, 122 s); `tiers`: the 4,357 transitions from states with no key cached as a trajectory followed (39 s), 1,714 outside the model; the transition made by hand reported. The enumeration's states are those of L1 (1,281, 32,668, 86,680 and 1,204), its whole run about 1 minute in release.

## 40. `replay.sh` on a new recording of the integration test, 2026-11-13

- **Run:** the trace of entry 37 recorded again with the code as it stands (the states for the check now carry the class of each cached key), through `replay.sh` end to end.
- **Prediction:** as entry 37: 22 inputs replayed with every completion equal, 19 steps followed in one window, the image, the open and the close outside.
- **Outcome:** as predicted: 22 inputs replayed with every completion equal; 19 steps followed in one window (188 distinct states, 1 s); 33 s for the whole script, the release build of the replay included.

## 41. A counterexample to induction in `Next7` (TLAPS, TLC), 2026-11-13

- **Found.** The TLAPS step of `Next7` cannot prove `TasksOk'` after `FinishDue` of a load that does not fill: `IndInv` allows a gone owner whose transfer task holds and writes the unit its load (done, its `Finish` stamped) writes, and the `Finish` makes that unit valid in the task's target memory. TLC checks the state (`artifacts/buffer-pool/proof-work/cti/CTI7.tla`): `IndInv` holds in it and fails after one `FinishDue`, in `TasksOk` alone. It is a counterexample to induction, not to safety: a clone or a load takes a free or evictable unit, which no task holds, and a transfer takes only its owner's table units and slot.
- **Consequence.** `CopiesOk` states that no task holds the new block a clone or a load writes (`c.write \notin t.units`); `KvFrame.tla`'s `CopyOk` follows.
- **Prediction.** `IndInv` with the clause is a TLC invariant of every safety configuration (the ten of `indinv-check`); `KvStep7` proves `IndInv /\ Next7 => IndInv'`; the group modules whose steps make or end copies or tasks (`KvStep2`, `KvStep5`, `KvStep6`, `KvStep9`) still prove after the change; Apalache's `Next7` of round 6, started before the clause, reports this counterexample or one like it.
- **Outcome, Apalache.** As predicted: round 6 (on `IndInv` before the clause) reports no error for `Next1` to `Next6` and `Next8` to `Next10`, and for `Next7` a counterexample of the same shape (a gone owner `o2` whose transfer task holds and writes `u1`, a load of `o2` writing `u1` without filling, whose `FinishDue` makes `u1` valid in the task's target memory; state invariant 20, `TasksOk`). The clause excludes its first state. Apalache is not rerun on the strengthened invariant: the TLAPS step of every group is run on it instead.
- **Outcome, TLC.** As predicted: `IndInv` with the clause is a TLC invariant of all ten safety configurations (`artifacts/buffer-pool/indinv-check3`, `MC_Core` 172,191,447 distinct states; TLC's states on the host's local disk, after a first run failed writing them to `/workspace`, which is inconclusive and not counted).
- **Outcome, TLAPS.** As predicted: `KvStep7.tla` proves `IndInv /\ Next7 => IndInv'` (2,144 obligations), and every module rerun from a clean cache on the strengthened invariant proves (below, entry 42).

## 42. The proof (L3) and P7 (L7), 2026-11-13

- **Prediction.** With the ten group modules proved on one `IndInv`, `KvStep.tla` proves `IndInv /\ [Next]_vars => IndInv'` (a stuttering step by the frame lemmas, the hazard switches excluded by `Design`), `Spec => []IndInv` and `Spec => []Safety` by PTL with `KvProof`'s `InitInv` and `InvSafe`; and `KvSafeP7.tla` proves `Spec => []P7` from `Spec => []IndInv` and `KvProofP7`'s `IndInv => P7`.
- **Outcome.** As predicted, from clean caches on r2 (`tlapm --threads 48 --stretch 6`; logs in `artifacts/buffer-pool/proof-work/final/`): `KvProof` 164 obligations, `KvFrame` 374, `KvStep1` 897, `KvStep2` 735, `KvStep3` 591, `KvStep4` 627, `KvStep5` 435, `KvStep6` 1,604, `KvStep7` 2,144, `KvStep8` 1,151, `KvStep9` 780, `KvStep10` 405, `KvStep` 117 (10,024 in all, 7,046 s), `KvProofP7` 135, `KvSafeP7` 6; every obligation proved, no omitted step, no assumption beyond `Design`, `Sizes` and (for P7) `Finite`. P1, P2, P3, P5, P6 and P7 are `proved`.
- **Terms.** Proved: safety of the design model, under the stated assumptions. Not proved: liveness (P4 stays `checked`); that the implementation refines the model.


## 43. Products that cross processes, and host copies per host part (`MC_Cross`, `MC_HostParts`, `MC_HostSplit`, `suite.sh`), 2026-11-13

- **Model:** the extension of `artifacts/buffer-pool/p4-design.md` §4.1, from its checked scratch copy (`artifacts/buffer-pool/p4-model/`), with the design's decisions. Crossings: `TakeReplicate`, `ReplicateDue`, `ReplicateEnd`; every request of an owner is stamped by its group and by the groups that read its class (`Readers`); the hazard H14 (`MapEarly`). Host parts: `valid` over host units, R11 on `Load` and `Restore`. A `Store` whose host copy lacks some of the group's host parts fills those (decision 6): a copy states the host parts it fills (`parts`), and R5 makes them the parts that do not hold the block. A crossing into memory that holds its block already is answered at once and the reading group is handed the entry then (§2.2; the scratch copy refused it). P1's `TaskIsolation` and P5's `TrialVisibility` gain the crossing's and the fill's arms; P8 Placement (`KvP8.tla`: `CopiesPlaced`, `CrossingsPlaced`) is new; `Safety` is P1, P2, P3, P5, P6 and P8. The three new configurations are the scratch's searches with one owner, which complete (its two-owner searches stopped at depths 28 to 38). Every configuration's rules include R11.
- **Prediction, design part.** The eleven existing configurations complete with no violation and exactly the distinct and generated states of entry 35's suite (`MC_Core` 172,191,447 distinct of 883,432,459 generated; `MC_Copies` 47,047,135 of 198,329,421; `MC_LossCopies` 47,048,530; `MC_Transfer` 38,135,896; `MC_Tiers` 32,599,322; `MC_TrialTiers` 17,941,500; `MC_Resize` 16,365,337; `MC_TransferPair` 11,593,052; `MC_Trial` 7,459,267; `MC_Loss` 2,788,873; `MC_Finality` 188,424): no group reads in another memory, so no crossing is issued, and every group copies with the one host part, so no store fills, R11 never refuses and a copy's `parts` is always `{}`. `MC_HostParts` completes with no violation and the scratch's 51,006,680 distinct states (239,181,369 generated): its groups' parts are disjoint, so a fill writes all of its group's parts, as the scratch's did. `MC_Cross` completes with no violation and more distinct states than the scratch's 15,974,500 (the answer at once is new behaviour), fewer than 30 million. `MC_HostSplit` completes with no violation and more distinct states than the scratch's 46,356,213 (a store of `gb` into a host copy held on `pa` alone now fills `pb`), fewer than 100 million.
- **Prediction, mutations, hazards, validation.** `MC_Cross NoR5`: `TaskIsolation` (a crossing writes `gb`'s memory of a block valid there that a commit shares meanwhile, as X2's first run). `MC_HostParts NoR11` and `MC_HostSplit NoR11`: `CopiesPlaced` (a load of `gb` reads a host copy that does not hold `gb`'s parts; with one owner, after that owner is transferred from `ga`, which stored or saved it). `MC_Cross H14`: `ReadIsolation`. With one property alone: `MC_Cross NoR3` violates `CrossingsPlaced`, `MC_Cross NoR9` violates `CrossingTaskIsolation`, `MC_HostSplit NoR5` violates `CopiesPlaced` (a fill writes `pa`, which holds the copy). The runs of entry 35 report what it reports.
- **Prediction, `IndInv`.** `IndInv` extended as `p4-design.md` §4.2 states (`KvInductive.tla`: a reader's entries and completions, `ReaderRow`; covers of a reading group's batches; `FreezesRow` for a commit and a crossing; a store's fill among the copies that fill; the new conjunct `CrossingsOk` for crossings that wait and run; `CopiesPlaced` and `CrossingsPlaced` as conjuncts) is a TLC invariant of `MC_Cross`, `MC_HostParts` and `MC_HostSplit`.

- **Outcome, first run of the quick parts (not counted):** every line as stated, on the model before `TypeOK` stated that a device block's validity holds memories and a host block's host parts; the suite was run again whole on the final model.
- **Outcome, the suite on the final model:** as predicted: all 50 lines as stated (the inputs' digests are kept in `artifacts/buffer-pool/p4-u1/suite-inputs.md5`, its lines and logs in `artifacts/buffer-pool/p4-u1/logs/suite/`). The eleven existing configurations complete with no violation and exactly entry 35's distinct and generated states (`MC_Core` 172,191,447 of 883,432,459). `MC_HostParts` completes with the scratch's 51,006,680 distinct states (239,181,369 generated); `MC_Cross` with 16,292,780 (90,196,367 generated, depth 88); `MC_HostSplit` with 49,324,531 (229,593,974 generated, depth 80). `MC_Cross NoR5`: `TaskIsolation` in 15 states (an owner of `gb` commits, is transferred to `ga` while the commit waits, and a crossing into `gb` taken without R5 writes `gb`'s memory of the shared block valid there). `MC_HostParts NoR11` and `MC_HostSplit NoR11`: `CopiesPlaced` in 15 states, as predicted in kind with the groups the other way round: `gb` saves the key on `pb`, the owner is transferred to `ga`, and `ga`'s restore reads `pb`. `MC_Cross H14`: `ReadIsolation` in 5 states. Validation: `CrossingsPlaced` (9 states), `CrossingTaskIsolation` (10) and `CopiesPlaced` with `NoR5` on `MC_HostSplit` (19: `gb`'s store fills `pa`, which holds the copy) each violated.
- **Outcome, `IndInv`:** as predicted: a TLC invariant of `MC_Cross`, `MC_HostParts` and `MC_HostSplit`, complete searches on the final text (on r2; `artifacts/buffer-pool/p4-u1/logs/indinv-final/`, inputs' digests kept), with the counts above. Runs on the text before its last two clauses, kept beside them, gave the same.
## 44. The inductive step of the extended `IndInv` by groups (`apalache_step.sh`, `Next1` to `Next11`), 2026-11-13

- **Model:** entry 43's, with two clauses found by reading each action against `IndInv` before this run: a prefetch names a group (`PrefetchEnd` makes its block valid on that group's host parts), and no copy writes, as a new block, the block a waiting crossing holds. `MC_Apalache` gains host parts `pa` and `pb` (`g2` copies with both, `g1` and `g3` with `pa`), `g2` reading the class owners of `g1` hold, the new `Prefetch` in `Next9`, and `Next11` (`TakeReplicate`, `ReplicateDue`, `ReplicateEnd`). Apalache's type checker passes.
- **Prediction:** at least one group reports a counterexample to induction from a state no behaviour reaches, most likely `Next4` (a stamped request and a reader's entries) or `Next11` (the crossing's own steps, whose clauses were written from the argument alone); `Next3`, `Next5` and `Next10` report no error.

- **Outcome, a first round (not counted):** every group reported a deadlock at once: the generated initial predicate had no state, since `valid`, now a function over three units (two device units and the host unit), was generated with at most two elements. With `valid = Gen(3)` Apalache refutes `NoState` (the predicate has states).
- **Outcome:** not as predicted: no counterexample to induction in any group. `Next1` to `Next11` each report no error (on rank-1, paused twice for another unit's measurements; `Next7`, the largest, took about ten hours; `artifacts/buffer-pool/p4-u1/logs/apalache-step.txt`). The clauses written from the argument were inductive for these constants as they stood, which the TLAPS step confirms for all constants (entry 45).
## 45. The proof over the extended model: `Spec => []Safety`, `Spec => []P8`, `Spec => []P7` (TLAPS), 2026-11-13

- **Proof:** `KvProof` (`InitInv`; `InvSafe`, with P8 in `Safety`), `KvFrame` (the shared lemmas, with those that taking a request and retiring a deferred item share, moved from `KvStep6` and `KvStep7` and widened to crossings), `KvStep1` to `KvStep10` re-proved for the changed and new conjuncts, `KvStep11` for `Next11` (`TakeReplicate`, `ReplicateDue`, `ReplicateEnd`), `KvStep` (`Next11` in the assembly; H14's `MapEarly` excluded by `Design`; `SafeP8`), `KvProofP7` and `KvSafeP7`. Assumptions: `Design` (every rule R1 to R11 in force, no hazard), `Sizes` with three facts about the configuration added (each group's host parts are host parts; a memory is not a host part; a group reads only groups of its own epoch), and `Finite` for P7.
- **Prediction:** every module proves from a clean cache with every obligation proved and none omitted, about 13,000 obligations in all (entry 42's 10,165, the new step module about as many as `KvStep8`'s 1,151, and the new conjuncts in every module).

- **Outcome:** as predicted, with more obligations than estimated: every module proves from a clean cache on r2, every obligation proved, none omitted (logs in `artifacts/buffer-pool/p4-u1/proof/`): `KvProof` 217, `KvFrame` 841, `KvStep1` 1,065, `KvStep2` 968, `KvStep3` 816, `KvStep4` 693, `KvStep5` 573, `KvStep6` 1,721, `KvStep7` 2,441, `KvStep8` 1,460, `KvStep9` 924, `KvStep10` 529, `KvStep11` 1,784, `KvStep` 146, `KvProofP7` 135, `KvSafeP7` 6 (14,319 in all; one lemma `KvStep3` and `KvStep9` each proved, that tables and completions keep across a step that changes host validity alone, is proved once in `KvFrame`). No counterexample to induction was found: the proof needed no clause beyond those TLC checked (entry 43). One statement of `KvFrame` was unprovable as written (`CrossingsOkFrame` reads `trial` through `MadeBy`) and was corrected; steps that rested on one large expansion or on Zenon, which timed out in some clean runs on the busy host, were given structured proofs.
- **Terms.** Proved: safety of the design model, under the stated assumptions (`Design`, `Sizes`, and `Finite` for P7). Not proved: liveness; that the implementation refines the model.
## 46. L1 and L4 on the extended model (`agreement.sh`, `agreement_steps.sh`, `mutations.py`), 2026-11-13

- **Run:** the Rust buffer pool with the crossing and host parts (unit U2 of `p4-design.md`), against this model, in a private copy of the tree: the enumeration's six configurations (`core`, `copies`, `transfer`, `tiers`, and U2's `cross` and `parts` with `MC_PoolCross` and `MC_PoolParts`) by states, `core` step by step, and the mutations of R1 to R11.
- **Prediction:** as U2 reported against this model before its proof was settled: every state of every configuration is a state of `MC_Pool` (core 1,281, copies 31,207, transfer 86,680, tiers 1,204, cross 1,048, parts 1,672 Rust states); every transition of `core` followed; every rule R1 to R11, its guard replaced, caught by the crate's tests.
- **Outcome:** as predicted, twice: on U2's source and, after EN6's prefetch that evicts host copies (with `steps.py` following a prefetch's evictions as a store's), on HEAD's Rust buffer pool (`artifacts/buffer-pool/p4-u1/logs/l1-l4/`). Every state of the six configurations is a state of `MC_Pool` (Rust states 1,281, 31,207, 86,680, 1,204, 1,048 and 1,672 against 3,746, 112,466, 86,308, 3,476, 1,508 and 2,876 projected model states); all 5,553 transitions of `core` followed in 5,553 windows, the transition made by hand reported; each of R1 to R11, its guard replaced, caught by the crate's tests.

## 47. A serving run's trace through the model: stated stamps and keys, and windows of the part a transition touches (`steps.py`, `KvTrace`, `agreement.py`), 2026-11-14

- **Run:** the replay's states now state every field of waiting work, copies and tasks the model has (the stamps work still waits for, one `release` line per group and stamp, the keys of commits, copies and tasks, the memories of a block a waiting commit holds; keys in their 32 bytes), written by the Rust buffer pool and by `agreement.py` alike; a window's start is then fixed by its state. Every transition is checked from its recorded state on each part of the pool it touches, closed under what holds what, renamed in order, with one free unit of each region and one free slot row of each group kept free; windows of one layout and size share a TLC run, runs in parallel. In a private copy: L1 by states on the six configurations (`agreement.sh`), L1 step by step on all six (`agreement_steps.sh core copies transfer tiers cross parts`), the crate's tests; then the E2 smoke trace (`heavy/e2/smoke-478aaf612/trace/`, 14,813 inputs) and the failed control's (`heavy/e2/failed/control-478aaf612/trace/`) through `replay.sh` on rank-1 at nice 19. A probe of `core` alone came first: every transition followed (7,810 windows, 20 TLC runs, 14 s), the transition made by hand reported.
- **Prediction:** every state of the six configurations is a state of `MC_Pool` (the enumeration now reaches 1,825, 44,383, 136,092, 1,364, 1,348 and 1,936 states, more than before since per-record releases and stamps tell more states apart); every transition of the six configurations followed and the transition made by hand reported; the crate's tests pass. Both traces: every completion equal and every check clean, and every transition the model has terms for followed, the opens, closes and the pool's image outside; the model part within ten minutes for the smoke trace with 32 jobs. Risks named: a release of the Rust buffer pool that the model's single release record cannot split (now one record per line), and a store's or a save's group or owner chosen by TLC in `copies` and `transfer`.
- **Outcome:** as predicted, after three changes to the method that the first runs showed, none a difference between the Rust buffer pool and the model. (1) TLC read the table of windows on every state, so a run's cost grew with its windows (an allocation's window took 2 s alone and 7 times as long per state among 100): each window's input, next state and bound are variables set at its start. (2) A request's positions were tried in every order, with every unit at every position: a step now changes what the next state settles only toward it (`Toward`: it takes no unit the next state has free, writes an entry or caches a key only as the next state has it) and takes a request's positions in order; the first full run of the 478aaf612 smoke then failed at an allocation of 10 positions that evicted 3 cached blocks, which the Rust buffer pool evicts before it takes units and the model only from a region with no free unit, so an allocation takes the positions whose units are free first. (3) Parts of a step joined through an owner whose state did not change put 10 commits and 10 stores' ends into one window (2^20 orders): parts now join only through changed facts. Results: every state of the six configurations a state of `MC_Pool` (1,825 / 5,944; 44,383 / 247,976; 136,092 / 189,586; 1,364 / 4,738; 1,348 / 2,336; 1,936 / 3,988); every transition of the six followed (core 6,563 in 7,810 windows; copies 184,341 in 209,370; transfer 640,598 in 821,282; tiers 3,791; cross 4,490; parts 5,696; the stored-trajectory transitions outside), the transition made by hand reported; the crate's 55 tests pass and clippy is clean. Traces, rank-1 at nice 19, 32 jobs: the 478aaf612 smoke (14,813 inputs) every completion equal and every check clean in 91 s, every transition followed (14,810 steps, 1,676 that change nothing, 20,168 windows, 51 TLC runs, 20 s); the failed control (26,742 inputs, 259 s; 37,848 windows, 95 runs, 60 s); and E2's smoke on `da78a9be9` now at `heavy/e2/smoke/trace/` (27,073 inputs, 221 s; 23,894 windows, 61 runs, 41 s; `replay.sh` end to end 292 s). A forged state in a slice of the smoke (an allocated entry given another unit) was reported.
