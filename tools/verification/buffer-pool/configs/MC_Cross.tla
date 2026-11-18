----------------------------- MODULE MC_Cross -----------------------------
(* A product that crosses processes: an owner attached to group ga, whose  *)
(* ranks lie in one process, holds a block that group gb, whose ranks lie  *)
(* in another (another memory), reads once a crossing brought it there.    *)
EXTENDS KvInductive, KvP7, KvP8, TLC

CONSTANTS ga, gb, sa, sb, e1

MCSlotGroup == [s \in Slots |-> IF s = sa THEN ga ELSE gb]
MCGroupEpoch == [g \in Groups |-> e1]
MCOwnerEpoch == [o \in Owners |-> e1]
MCGroupMemory == [g \in Groups |-> g]
(* One unit space; every unit is in the region.                            *)
MCUnitRegion == [u \in Units |-> "kv"]
MCUnitChunk == [u \in Units |-> u]
MCGroupRegion == [g \in Groups |-> "kv"]
MCMovable == {}
MCInitialUnits == AllUnits
(* gb reads, in its own memory, the class an owner of ga holds.            *)
MCReaders == [g \in Groups |-> IF g = ga THEN {gb} ELSE {}]
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
Without(rule) == AllRules \ {rule}
NoR1 == Without("R1")
NoR3 == Without("R3")
NoR5 == Without("R5")
NoR9 == Without("R9")

(* Units are interchangeable in this configuration.                       *)
Symmetry == Permutations(Units)

(* Witnesses, each false in some reachable state: a batch of gb reads a     *)
(* block of an owner of ga; a shared block is valid in both memories.       *)
NeverReadAcross ==
    ~\E b \in inflight : b.g = gb /\ \E r \in b.reads : r[2] \in Units /\ ogroup[r[1]] = ga
(* The crossing's arm of P1's TaskIsolation alone, so that a run checking  *)
(* it is not stopped first by the release's or the import's arm.            *)
CrossingTaskIsolation ==
    \A t \in tasks : t.kind = "replicate" =>
        \A u \in t.writes :
            /\ u \notin free
            /\ alloc[u] = t.o \/ (alloc[u] = "shared" /\ Mem(t.to) \notin valid[u])
NeverSharedAcross == ~\E k \in Keys : cache[k] # NONE /\ valid[cache[k]] = {ga, gb}
=============================================================================
