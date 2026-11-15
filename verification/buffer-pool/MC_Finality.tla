----------------------------- MODULE MC_Finality ----------------------------
(* The smallest configuration with a copy and a task of the tiers.         *)
EXTENDS KvFinality, TLC

CONSTANTS g1, e1

MCSlotGroup == [s \in Slots |-> g1]
MCGroupEpoch == [g \in Groups |-> e1]
MCOwnerEpoch == [o \in Owners |-> e1]
MCGroupMemory == [g \in Groups |-> g]
MCUnitRegion == [u \in Units |-> "kv"]
MCUnitChunk == [u \in Units |-> u]
MCGroupRegion == [g \in Groups |-> "kv"]
MCMovable == {}
MCInitialUnits == Units \cup HostUnits
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
=============================================================================
