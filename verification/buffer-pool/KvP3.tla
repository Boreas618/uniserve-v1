-------------------------------- MODULE KvP3 --------------------------------
(* P3 Conservation of specs/buffer-pool.md §17, as formulas over KvCore. *)
EXTENDS KvCore

(* P3 Conservation: every unit is free or held, and when no owner exists,  *)
(* nothing is deferred and no copy is in flight every block is free or     *)
(* shared without a pin.                                                   *)
NoLeak ==
    \A u \in AllUnits :
        IF InRegion(u) THEN u \in free \/ Held(u) ELSE u \notin free /\ ~Held(u)
(* Device memory is in one region.                                         *)
OneRegion == \A u, v \in member \cap Units : u # v => UnitChunk[u] # UnitChunk[v]
Quiescent ==
    (/\ \A o \in Owners : ostate[o] # "live"
     /\ deferred = {}
     /\ copies = {}
     /\ tasks = {})
        => \A u \in AllUnits :
               InRegion(u) => u \in free \/ (CacheKeys(u) # {} /\ ~Pinned(u))

P3 == NoLeak /\ Quiescent /\ OneRegion
=============================================================================
