---------------------------- MODULE MC_HostParts ----------------------------
(* Host copies per host part: two replicas, each on its own host. ga       *)
(* copies with part pa of host A, gb with part pb of host B: the same part *)
(* index on two hosts. Two owners, each attached to either replica.        *)
EXTENDS KvInductive, KvP7, KvP8, TLC

CONSTANTS ga, gb, sa, sb, e1, pa, pb

MCSlotGroup == [s \in Slots |-> IF s = sa THEN ga ELSE gb]
MCGroupEpoch == [g \in Groups |-> e1]
MCOwnerEpoch == [o \in Owners |-> e1]
MCGroupMemory == [g \in Groups |-> g]
MCUnitRegion == [u \in Units |-> "kv"]
MCUnitChunk == [u \in Units |-> u]
MCGroupRegion == [g \in Groups |-> "kv"]
MCMovable == {}
MCInitialUnits == AllUnits
MCGroupParts == [g \in Groups |-> IF g = ga THEN {pa} ELSE {pb}]
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
Without(rule) == AllRules \ {rule}
NoR4 == Without("R4")
NoR5 == Without("R5")
NoR11 == Without("R11")

Symmetry == Permutations(Units) \cup Permutations(Owners)

(* Witnesses: a host copy held on both parts; a load by gb that started.   *)
NeverBothParts == ~\E k \in Keys : hcache[k] # NONE /\ valid[hcache[k]] = {pa, pb}
NeverLoadedOnB == ~\E c \in copies : c.kind = "load" /\ c.g = gb
=============================================================================
