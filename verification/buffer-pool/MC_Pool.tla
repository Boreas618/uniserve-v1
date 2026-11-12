------------------------------ MODULE MC_Pool ------------------------------
(* The buffer pool alone: its handlers, with stamped requests of any       *)
(* stamps, `launched` standing for the `completed` word of each group, and *)
(* no batch, executor or scheduler. The Rust buffer pool is driven by the  *)
(* same requests, and every state it reaches is compared with the states   *)
(* of this relation (L1 of specs/buffer-pool.md §19).                      *)
EXTENDS KvCore, TLC

CONSTANTS ga, gb, sa, sb, sc, sd, e1,
          Transfers,   \* whether `Transfer` requests arrive
          Stamping     \* the groups whose stamp a request may carry

(* A prefill group, and where the configuration has it a decode group, of  *)
(* one epoch over one unit space, each with a memory of its own. The slot  *)
(* rows sa and sb are the prefill group's, sc and sd the decode group's.   *)
MCSlotGroup == [s \in Slots |-> IF s \in {sa, sb} THEN ga ELSE gb]
MCGroupEpoch == [g \in Groups |-> e1]
MCOwnerEpoch == [o \in Owners |-> e1]
MCGroupMemory == [g \in Groups |-> g]
MCUnitRegion == [u \in Units |-> "kv"]
MCUnitChunk == [u \in Units |-> u]
MCGroupRegion == [g \in Groups |-> "kv"]
MCMovable == {}
MCInitialUnits == Units
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"}

(* A stamped request arrives on a queue pair with any stamps: the buffer   *)
(* pool cannot tell what its executors did.                                *)
Arrive ==
    /\ requests = {}
    /\ \E e \in Epochs \ lost,
          after \in {a \in [Groups -> 0 .. MaxBatch] : \A g \in Groups \ Stamping : a[g] = 0} :
         \/ \E o \in Owners :
              \/ \E ps \in (SUBSET Positions) \ {{}} :
                   requests' = {[Request("release", e, o, ps, NONE, NONE, 0, FALSE)
                                   EXCEPT !.after = after]}
              \/ requests' = {[Request("detach", e, o, {}, NONE, NONE, 0, FALSE)
                                 EXCEPT !.after = after]}
              \/ \E p \in Positions, k \in Keys :
                   requests' = {[Request("commit", e, o, {}, p, k, 0, FALSE)
                                   EXCEPT !.after = after]}
              \/ \E g \in Groups :
                   /\ Transfers
                   /\ requests' = {[Request("transfer", e, o, {}, NONE, NONE, 0, FALSE)
                                      EXCEPT !.after = after, !.to = g]}
         \/ \E c \in 1 .. ncopy, ok \in BOOLEAN :
              requests' = {[Request("finish", e, NONE, {}, NONE, NONE, c, ok)
                              EXCEPT !.after = after]}
    /\ UNCHANGED <<pool, dbt, launched, inflight, handoff, alloc, sharedKey>>

(* An executor writes a later value into its group's `completed` word.     *)
Advance(g) ==
    /\ launched[g] < MaxBatch
    /\ launched' = [launched EXCEPT ![g] = @ + 1]
    /\ UNCHANGED <<pool, dbt, inflight, handoff, requests, alloc, sharedKey>>

PoolServing ==
    \/ \E o \in Owners, g \in Groups : Attach(o, g)
    \/ \E o \in Owners, p \in Positions : Allocate(o, p)
    \/ \E o \in Owners, p \in Positions, k \in Keys : Pin(o, p, k) \/ Clone(o, p, k) \/ Load(o, p, k)
    \/ \E o \in Owners, k \in Keys : Save(o, k) \/ Restore(o, k)
    \/ \E g \in Groups, k \in Keys : Store(g, k)
    \/ \E k \in Keys : Evict(k) \/ EvictHost(k)
    \/ Arrive
    \/ \E g \in Groups : Advance(g)
    \/ \E r \in requests :
         TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r) \/ TakeTransfer(r)
    \/ \E d \in deferred : ReleaseDue(d) \/ CommitDue(d) \/ FinishDue(d) \/ TransferDue(d)
    \/ \E t \in tasks, ok \in BOOLEAN : TransferEnd(t, ok)
    \/ \E e \in Epochs : CommitUpdate(e) \/ Rollback(e)
    \/ \E u \in Units : Shrink(u) \/ Grow(u)

PoolNext ==
    \/ PoolServing /\ UNCHANGED rare
    \/ \E k \in Keys : Prefetch(k, "remote") \/ StoreRemote(k)
    \/ \E t \in tasks : Deadline(t)
    \/ \E t \in tasks, ok \in BOOLEAN : PrefetchEnd(t, ok) \/ PutEnd(t, ok)
    \/ \E e \in Epochs : Lose(e)

PoolSpec == Init /\ [][PoolNext]_vars

(* What the search tells states apart by: the buffer pool's own state. The *)
(* executors' tables and the history variables are written and never read  *)
(* by this relation, and neither is the validity of a unit that is free.   *)
View == <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, tasks,
          [u \in Units |-> IF u \in free THEN {} ELSE valid[u]],
          vis, trial, member, launched, requests, remote, lost>>
=============================================================================
