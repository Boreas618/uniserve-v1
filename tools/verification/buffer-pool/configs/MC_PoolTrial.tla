---------------------------- MODULE MC_PoolTrial ----------------------------
(* MC_Pool over an update: the serving epoch e1, its group ga in region A, *)
(* and the target e2, on trial, of another layout, its group gb in region  *)
(* B over the same device memory: one chunk of it, which A's unit a1 and   *)
(* B's unit b1 both occupy and which starts in A. The target's update is   *)
(* committed, or its engine is lost (`Lose`) and its update rolled back;   *)
(* the chunk moves between the regions by `Resize` (R10). The             *)
(* configuration `trial` of the Rust buffer pool's enumeration.            *)
EXTENDS MC_Pool

CONSTANTS e2, o1, o2, a1, b1

MCTrialGroupEpoch == [g \in Groups |-> IF g = ga THEN e1 ELSE e2]
MCTrialOwnerEpoch == [o \in Owners |-> IF o = o1 THEN e1 ELSE e2]
MCTrialUnitRegion == [u \in Units |-> IF u = a1 THEN "A" ELSE "B"]
MCTrialGroupRegion == [g \in Groups |-> IF g = ga THEN "A" ELSE "B"]
MCTrialUnitChunk == [u \in Units |-> 1]
MCTrialMovable == Units
MCTrialInitialUnits == {a1} \cup Slots
=============================================================================
