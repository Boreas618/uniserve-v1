------------------------------ MODULE MC_TrialTiers -------------------------
(* Two epochs of one layout: a serving epoch and a target on trial, each   *)
(* with one group over the same device memory, with a host tier and copies. *)
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
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
NoR8 == AllRules \ {"R8"}

Symmetry == Permutations(Units)

(* Witness: false in some reachable state, which shows that the search     *)
(* reaches the serving epoch's owner pinning a block the target committed  *)
(* once the target's update is committed.                                  *)
NeverSharedAcross ==
    ~\E k \in Keys, p \in Positions :
        /\ cache[k] # "none" /\ table[o1][p] = Shared(cache[k]) /\ sharedKey[cache[k]] = k
        /\ trial[e2] = "no" /\ alloc[cache[k]] = "shared" /\ ostate[o2] = "gone"
=============================================================================
