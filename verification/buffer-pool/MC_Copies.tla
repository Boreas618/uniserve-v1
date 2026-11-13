------------------------------ MODULE MC_Copies ----------------------------
(* Copies run by the executor and the host tier, on one worker group.      *)
EXTENDS KvInductive, KvP7, TLC

CONSTANTS g1, e1

MCSlotGroup == [s \in Slots |-> g1]
MCGroupEpoch == [g \in Groups |-> e1]
MCOwnerEpoch == [o \in Owners |-> e1]
MCGroupMemory == [g \in Groups |-> g]
(* One KV region; every unit has memory of its own and is in the region.     *)
MCUnitRegion == [u \in Units |-> "kv"]
MCUnitChunk == [u \in Units |-> u]
MCGroupRegion == [g \in Groups |-> "kv"]
MCMovable == {}
MCInitialUnits == Units
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"}
(* The rules in force; a mutation run overrides it with one rule left out. *)
Without(rule) == AllRules \ {rule}
NoR1 == Without("R1")
NoR2 == Without("R2")
NoR3 == Without("R3")
NoR4 == Without("R4")
NoR5 == Without("R5")
NoR6 == Without("R6")
NoR7 == Without("R7")
NoR8 == Without("R8")
NoR9 == Without("R9")
NoR10 == Without("R10")

(* Units and owners are interchangeable in this configuration.            *)
Symmetry == Permutations(Units) \cup Permutations(Owners)

(* Witnesses: each is false in some reachable state, which shows that the  *)
(* search reaches a stored host copy, and a clone and a load whose         *)
(* `Finish` came due with the copy done.                                   *)
NeverStored == \A k \in Keys : hcache[k] = NONE
NeverCloned == ~\E d \in deferred : \E c \in copies :
                   d.kind = "finish" /\ d.ok /\ d.c = c.id /\ c.kind = "clone" /\ Reached(d.after)
NeverLoaded == ~\E d \in deferred : \E c \in copies :
                   d.kind = "finish" /\ d.ok /\ d.c = c.id /\ c.kind = "load" /\ Reached(d.after)
=============================================================================
