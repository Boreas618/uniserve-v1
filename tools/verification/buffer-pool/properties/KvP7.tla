-------------------------------- MODULE KvP7 --------------------------------
(* P7 Capacity identity of specs/buffer-pool.md §17.2, as a formula over   *)
(* KvCore.                                                                 *)
EXTENDS KvCore, FiniteSets

(* The regions: each KV region of the device, the slot rows of each worker *)
(* group, and the host units.                                              *)
Capacities ==
    {<<"device", UnitRegion[u]>> : u \in Units}
        \cup {<<"slots", g>> : g \in Groups}
        \cup {<<"host", "host">>}

(* The units in a region: the members of a KV region, of a group's slot    *)
(* region or of the host region.                                           *)
\* @type: (<<Str, Str>>) => Set(Str);
UnitsIn(r) ==
    CASE r[1] = "device" -> {u \in member \cap Units : UnitRegion[u] = r[2]}
      [] r[1] = "slots" -> {s \in member \cap Slots : SlotGroup[s] = r[2]}
      [] OTHER -> member \cap HostUnits

(* What a unit that something holds is observed as (§8): the first that    *)
(* applies of shared, private and retired.                                 *)
ObservedShared(u) == CacheKeys(u) # {}
ObservedPrivate(u) ==
    \/ TableHolders(u) # {}
    \/ \E d \in deferred : d.kind = "commit" /\ u \in d.units
    \* A copy writes for a target that exists: a cache entry, or an owner
    \* that is attached.
    \/ \E c \in copies : c.write = u /\ (c.kind \in {"store", "save"} \/ ostate[c.o] = "live")
    \/ \E t \in tasks :
         /\ t.held
         /\ \/ t.kind = "transfer" /\ t.slot = u /\ ostate[t.o] = "live"
            \/ t.kind = "prefetch" /\ t.h = u

FreeIn(r) == free \cap UnitsIn(r)
SharedIn(r) == {u \in UnitsIn(r) : ObservedShared(u)}
PrivateIn(r) == {u \in UnitsIn(r) : ~ObservedShared(u) /\ ObservedPrivate(u)}
RetiredIn(r) == {u \in UnitsIn(r) : ~ObservedShared(u) /\ ~ObservedPrivate(u) /\ Held(u)}

(* P7 Capacity identity: per region, the free, private, shared and retired *)
(* units add up to the units in the region.                                *)
CapacityIdentity ==
    \A r \in Capacities :
        Cardinality(FreeIn(r)) + Cardinality(PrivateIn(r))
            + Cardinality(SharedIn(r)) + Cardinality(RetiredIn(r))
            = Cardinality(UnitsIn(r))

P7 == CapacityIdentity
=============================================================================
