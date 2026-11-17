------------------------------ MODULE MC_Tiers ------------------------------
(* The tiers below the device on one worker group: prefetches from the     *)
(* remote store and from peers, puts to the remote store, a media          *)
(* request's trajectory saved and restored, and the host region resized    *)
(* by `Resize` beside them.                                                *)
EXTENDS KvInductive, KvP7, TLC

CONSTANTS g1, e1

MCSlotGroup == [s \in Slots |-> g1]
MCGroupEpoch == [g \in Groups |-> e1]
MCOwnerEpoch == [o \in Owners |-> e1]
MCGroupMemory == [g \in Groups |-> g]
MCUnitRegion == [u \in Units |-> "kv"]
MCUnitChunk == [u \in Units |-> u]
MCGroupRegion == [g \in Groups |-> "kv"]
MCMovable == HostUnits
MCInitialUnits == AllUnits
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
NoR9 == AllRules \ {"R9"}
NoR2 == AllRules \ {"R2"}

Symmetry == Permutations(HostUnits)

(* Witnesses: each is false in some reachable state.                       *)
NeverRemote == remote = {}
NeverPrefetched == ~\E k \in Keys : hcache[k] # "none" /\ alloc[hcache[k]] = "shared" /\ k \notin remote
                                     /\ cache[k] = "none" /\ ncopy = 0
NeverRestored == ~\E d \in deferred : \E c \in copies :
                     d.kind = "finish" /\ d.ok /\ d.c = c.id /\ c.kind = "restore" /\ Reached(d.after)
NeverShrunk == ~\E h \in HostUnits, k \in Keys : h \notin member /\ hcache[k] # "none"
=============================================================================
