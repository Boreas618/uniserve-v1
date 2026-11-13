------------------------------ MODULE MC_Loss -------------------------------
(* Two epochs of one layout over the same device memory, either of whose   *)
(* engines may be lost.                                                    *)
EXTENDS KvInductive, KvP7, TLC

CONSTANTS g1, g2, s1, s2, e1, e2, o1, o2, m

MCSlotGroup == [s \in Slots |-> IF s = s1 THEN g1 ELSE g2]
MCGroupEpoch == [g \in Groups |-> IF g = g1 THEN e1 ELSE e2]
MCOwnerEpoch == [o \in Owners |-> IF o = o1 THEN e1 ELSE e2]
MCGroupMemory == [g \in Groups |-> m]
(* One KV region; every unit has memory of its own and is in the region.     *)
MCUnitRegion == [u \in Units |-> "kv"]
MCUnitChunk == [u \in Units |-> u]
MCGroupRegion == [g \in Groups |-> "kv"]
MCMovable == {}
MCInitialUnits == Units
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"}
NoR8 == AllRules \ {"R8"}

Symmetry == Permutations(Units)

(* Witness: false in some reachable state, which shows that the search     *)
(* reaches the surviving epoch's owner pinning a block the lost epoch      *)
(* committed before its engine was lost.                                   *)
NeverInherited ==
    ~\E k \in Keys, p \in Positions :
        /\ e2 \in lost /\ cache[k] # "none" /\ table[o1][p] = Shared(cache[k])
        /\ ostate[o2] = "gone" /\ ostate[o1] = "live"
=============================================================================
