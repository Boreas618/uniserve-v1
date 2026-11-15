------------------------------- MODULE KvStep3 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for eviction, the end of a trial and       *)
(* resizing: Evict, EvictHost, CommitUpdate, Rollback, Shrink and Grow.    *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, KvFrame, TLAPS

USE Sizes, Design

-----------------------------------------------------------------------------
(* Evict                                                                   *)

LEMMA Evict_Effect ==
  ASSUME IndInv, NEW k \in Keys, Evict(k)
  PROVE  LET u == cache[k] IN
         /\ u \in Units /\ u \in member /\ u \notin free /\ Held(u)
         /\ alloc[u] = "shared" /\ sharedKey[u] = k
         /\ \A o \in Owners, p \in Positions : table[o][p].u # u
         /\ \A o \in Owners : oslot[o] # u
         /\ \A d \in deferred : u \notin d.units
         /\ \A c \in copies : c.read # u /\ c.write # u
         /\ \A t \in tasks : u \notin t.units /\ t.slot # u /\ t.h # u
         /\ \A j \in Keys : cache[j] = u => j = k
         /\ \A j \in Keys : hcache[j] # u
         /\ \A v \in AllUnits : alloc'[v] = IF v = u THEN NONE ELSE alloc[v]
         /\ \A v \in AllUnits : sharedKey'[v] = IF v = u THEN NONE ELSE sharedKey[v]
         /\ \A j \in Keys : cache'[j] = IF j = k THEN NONE ELSE cache[j]
         /\ cache' = [cache EXCEPT ![k] = NONE]
         /\ free' = free \cup {u}
         /\ alloc' = [alloc EXCEPT ![u] = NONE]
         /\ sharedKey' = [sharedKey EXCEPT ![u] = NONE]
         /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, copies, ncopy, tasks, trial,
                        member, valid, vis, dbt, launched, inflight, handoff, requests>>
  <1> DEFINE u == cache[k]
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, WorkTypes
  <1>1. /\ u # NONE /\ Evictable(u)
        /\ cache' = [cache EXCEPT ![k] = NONE]
        /\ Drop({u}, table, oslot, cache', hcache, deferred, copies)
        /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, copies, ncopy, tasks, trial,
                       member, valid, vis, dbt, launched, inflight, handoff, requests>>
    BY DEF Evict, plane
  <1>2. u \in Units /\ alloc[u] = "shared" /\ sharedKey[u] = k
    BY <1>0, <1>1 DEF IndInv, CacheShared, SharedIff, UnitTyped, AllUnits
  <1>3. /\ \A o \in Owners, p \in Positions : table[o][p] # Shared(u)
        /\ \A d \in deferred : u \notin d.units
        /\ \A c \in copies : c.read # u /\ c.write # u
        /\ \A t \in tasks : t.held => u \notin t.units
    BY <1>1 DEF Evictable, EvictableFor, PinnedFor, Pinned, On, AllRules
  <1>4. \A o \in Owners, p \in Positions : table[o][p].u # u
    BY <1>0, <1>2, <1>3 DEF IndInv, Tables, OwnerTyped, Empty, Private, Shared
  <1>5. \A t \in tasks : u \notin t.units /\ t.slot # u /\ t.h # u
    BY <1>0, <1>2, <1>3 DEF IndInv, TasksOk, WorkTyped
  <1>6. \A j \in Keys : cache[j] = u => j = k
    BY <1>2 DEF IndInv, CacheShared
  <1>7. /\ \A o \in Owners : oslot[o] # u
        /\ \A j \in Keys : hcache[j] # u
    BY <1>0, <1>2 DEF OwnerTyped, UnitTyped
  <1>8. Held(u) /\ u \in member /\ u \notin free
    BY <1>2 DEF IndInv, NoLeak, FreeClean, Held, HeldIn, InRegion, AllUnits
  <1>9. ~HeldIn(u, table, oslot, cache', hcache, deferred, copies, tasks)
    BY <1>0, <1>1, <1>3, <1>4, <1>5, <1>6, <1>7 DEF HeldIn, UnitTyped
  <1>10. ReclaimedWith({u}, table, oslot, cache', hcache, deferred, copies, tasks) = {u}
    BY <1>9, ReclaimedHeld DEF On, AllRules
  <1>11. /\ free' = free \cup {u}
         /\ alloc' = [alloc EXCEPT ![u] = NONE]
         /\ sharedKey' = [sharedKey EXCEPT ![u] = NONE]
    BY <1>0, <1>1, <1>2, <1>10 DEF Drop, DropWith, Reset, UnitTyped, AllUnits
  <1>12. /\ \A v \in AllUnits : alloc'[v] = IF v = u THEN NONE ELSE alloc[v]
         /\ \A v \in AllUnits : sharedKey'[v] = IF v = u THEN NONE ELSE sharedKey[v]
         /\ \A j \in Keys : cache'[j] = IF j = k THEN NONE ELSE cache[j]
    BY <1>0, <1>1, <1>11 DEF UnitTyped
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>11, <1>12

LEMMA Evict_Held ==
  ASSUME IndInv, NEW k \in Keys, Evict(k)
  PROVE  \A v \in AllUnits : Held(v)' = (Held(v) /\ v # cache[k])
  <1> SUFFICES ASSUME NEW v \in AllUnits
               PROVE  Held(v)' = (Held(v) /\ v # cache[k])
    OBVIOUS
  <1>1. CASE v # cache[k]
    BY <1>1, Evict_Effect DEF Held, HeldIn, AllUnits
  <1>2. CASE v = cache[k]
    BY <1>2, Evict_Effect DEF Held, HeldIn
  <1> QED
    BY <1>1, <1>2

LEMMA Evict_Typed ==
  ASSUME IndInv, NEW k \in Keys, Evict(k), UNCHANGED rare
  PROVE  TypeInv'
  <1>1. UnitTyped'
    BY UnitTypes, Evict_Effect DEF UnitTyped, AllUnits
  <1>2. OwnerTyped' /\ PlaneTyped' /\ WorkTyped' /\ RequestTyped'
    BY OwnerTypes, PlaneTypes, WorkTypes, RequestTypes, OwnerTypedFrame, PlaneTypedFrame,
       WorkTypedFrame, RequestTypedFrame, Evict_Effect DEF rare
  <1> QED
    BY <1>1, <1>2, TypeInvNext

LEMMA Evict_Free ==
  ASSUME IndInv, NEW k \in Keys, Evict(k)
  PROVE  FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
  <1> DEFINE u == cache[k]
  <1>1. /\ \A v \in AllUnits : alloc'[v] = IF v \in {u} THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v \in {u} THEN NONE ELSE sharedKey[v]
        /\ {u} \subseteq AllUnits
    BY Evict_Effect DEF AllUnits
  <1>2. /\ \A v \in AllUnits : Held(v)' => Held(v)
        /\ \A v \in AllUnits : Held(v) => Held(v)' \/ v \in {u}
        /\ \A v \in {u} : Held(v) /\ ~Held(v)'
    BY Evict_Held, Evict_Effect DEF AllUnits
  <1> QED
    BY <1>1, <1>2, Evict_Effect, UnitTypes, FreeReclaim DEF IndInv

LEMMA Evict_Shared ==
  ASSUME IndInv, NEW k \in Keys, Evict(k), UNCHANGED rare
  PROVE  AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared' /\ TrialVisibility'
  <1> DEFINE u == cache[k]
  <1>0. UnitTyped
    BY UnitTypes
  <1>1. AllocOwner' /\ SharedIff'
    BY Evict_Effect DEF IndInv, AllocOwner, SharedIff, AllUnits, NONE
  <1>2. \A v \in AllUnits : alloc[v] = "shared" => sharedKey[v] \in Keys
    BY <1>0 DEF IndInv, SharedIff, UnitTyped
  <1>3. SharedCached'
    BY <1>2, Evict_Effect DEF IndInv, SharedCached, AllUnits, NONE
  <1>4. CacheShared'
    BY <1>0, Evict_Effect DEF IndInv, CacheShared, UnitTyped, AllUnits
  <1>5. \A t \in tasks : NONE \notin t.units
    BY WorkTypes DEF WorkTyped, AllUnits
  <1>6. TrialVisibility'
    BY <1>5, Evict_Effect DEF IndInv, TrialVisibility, MadeBy
  <1> QED
    BY <1>1, <1>3, <1>4, <1>6

LEMMA Evict_Owners ==
  ASSUME IndInv, NEW k \in Keys, Evict(k), UNCHANGED rare
  PROVE  Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk' /\ TasksOk'
  <1> DEFINE u == cache[k]
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, WorkTypes
  <1>1. Tables'
    BY <1>0, Evict_Effect DEF IndInv, Tables, Live, Mem, OwnerTyped, AllUnits
  <1>2. OwnersOk'
    BY <1>0, Evict_Effect DEF IndInv, OwnersOk, Live, AllUnits, rare
  <1>3. DeferredOk'
    BY <1>0, Evict_Effect DEF IndInv, DeferredOk, WorkTyped, AllUnits, rare
  <1>4. CopiesOk'
    <2>1. \A c \in copies : (c.fill /\ c.kind = "load") => c.k # k
      BY Evict_Effect DEF IndInv, CopiesOk
    <2>2. \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits /\ c.k \in Keys
      BY <1>0 DEF WorkTyped
    <2> QED
      BY <2>1, <2>2, Evict_Effect DEF IndInv, CopiesOk, Live, Moving, Mem, rare
  <1>5. TasksOk'
    <2>1. \A t \in tasks : t.units \subseteq AllUnits /\ t.writes \subseteq AllUnits
                           /\ (t.slot # NONE => t.slot \in AllUnits)
                           /\ (t.h # NONE => t.h \in AllUnits)
      BY <1>0 DEF WorkTyped, AllUnits
    <2>2. \A t \in tasks : \A u \in t.units \cup {t.slot, t.h} : u # NONE => alloc'[u] = alloc[u]
      BY <2>1, Evict_Effect
    <2>3. \A t \in tasks :
            /\ t.kind = "transfer" =>
                 /\ alloc'[t.slot] = alloc[t.slot]
                 /\ \A u \in t.writes : alloc'[u] = alloc[u] /\ valid'[u] = valid[u]
                 /\ ostate[t.o] = "live" =>
                      /\ TableUnits(t.o)' = TableUnits(t.o)
                      /\ \A u \in TableUnits(t.o) : valid'[u] = valid[u]
            /\ t.kind = "prefetch" => alloc'[t.h] = alloc[t.h]
      <3>1. \A t \in tasks : /\ t.kind = "transfer" => t.writes \subseteq t.units /\ t.slot # NONE
                             /\ t.kind = "prefetch" => t.h # NONE
        BY DEF IndInv, TasksOk
      <3> QED
        BY <2>1, <2>2, <3>1, Evict_Effect DEF TableUnits, AllUnits
    <2> QED
      BY <2>3, Evict_Effect, TasksKeep DEF IndInv
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5

LEMMA Evict_Plane ==
  ASSUME IndInv, NEW k \in Keys, Evict(k), UNCHANGED rare
  PROVE  /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
         /\ BatchesOk'
  <1> DEFINE u == cache[k]
  <1>1. UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, copies, ncopy, tasks, trial,
                    member, valid, vis, dbt, launched, inflight, handoff, requests, remote, lost, bad>>
    BY Evict_Effect DEF rare
  <1>2. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
    BY <1>1, MovingUnmappedFrame, LostOkFrame, TablesMappedFrame, HandoffOkFrame, RequestsOkFrame,
       NoBadFrame, TransfersFrame DEF IndInv
  <1>3. \A b \in inflight : \A x \in b.reads : x[2] \notin {u} /\ alloc'[x[2]] = alloc[x[2]]
    <2>1. \A b \in inflight : \A x \in b.reads : x[2] \in AllUnits
      BY RequestTypes DEF RequestTyped
    <2>2. \A b \in inflight : \A x \in b.reads : x[2] # u
      BY ReadHeld, Evict_Effect, RequestTypes DEF RequestTyped
    <2> QED
      BY <2>1, <2>2, Evict_Effect
  <1> QED
    BY <1>1, <1>2, <1>3, Evict_Effect, BatchesKeep DEF IndInv

(* The crossings and P8: the evicted block is held by no crossing and no  *)
(* copy, and no validity changes.                                          *)
LEMMA Evict_Cross ==
  ASSUME IndInv, NEW k \in Keys, Evict(k), UNCHANGED rare
  PROVE  CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
  <1> DEFINE u == cache[k]
  <1>0. WorkTyped
    BY WorkTypes
  <1>1. UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, copies, ncopy, tasks, trial,
                    member, valid, vis, dbt, launched, inflight, handoff, requests, remote, lost, bad>>
    BY Evict_Effect DEF rare
  <1>2. CopiesPlaced' /\ CrossingsPlaced'
    BY <1>1, TransfersFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
  <1>3. \A o \in Owners : Moving(o)' = Moving(o)
    BY <1>1 DEF Moving
  <1>4. \A x \in AllUnits \ {u} : alloc'[x] = alloc[x]
    BY Evict_Effect
  <1>5. \A j \in Keys : cache'[j] # NONE => cache'[j] = cache[j]
    BY Evict_Effect
  <1>6. \A d \in deferred : CrossWaitOk(d)'
    <2> SUFFICES ASSUME NEW d \in deferred PROVE CrossWaitOk(d)'
      OBVIOUS
    <2>1. CrossWaitOk(d) /\ u \notin d.units /\ d.units \subseteq AllUnits
      BY <1>0, Evict_Effect, CrossingsOkEach DEF IndInv, WorkTyped
    <2>2. \A j \in Keys : cache'[j] \in d.units => cache[j] = cache'[j]
      BY <1>5, <2>1 DEF AllUnits
    <2> QED
      BY <1>1, <1>3, <1>4, <2>1, <2>2 DEF CrossWaitOk, ReadersOf, MadeBy
  <1>7. \A t \in tasks : CrossTaskOk(t)'
    <2> SUFFICES ASSUME NEW t \in tasks PROVE CrossTaskOk(t)'
      OBVIOUS
    <2>1. CrossTaskOk(t) /\ u \notin t.units /\ t.units \subseteq AllUnits
      BY <1>0, Evict_Effect, CrossingsOkEach DEF IndInv, WorkTyped
    <2> QED
      BY <1>1, <1>3, <1>4, <2>1 DEF CrossTaskOk, ReadersOf, Mem
  <1> QED
    BY <1>1, <1>2, <1>6, <1>7, CrossingsOkEach

LEMMA Evict_Inv ==
  ASSUME IndInv, NEW k \in Keys, Evict(k), UNCHANGED rare
  PROVE  IndInv'
  BY Evict_Typed, Evict_Free, Evict_Shared, Evict_Owners, Evict_Plane, Evict_Cross DEF IndInv

-----------------------------------------------------------------------------
(* EvictHost                                                               *)

(* BatchesKeep, for a step that changes the validity of host units alone.  *)
LEMMA HostValidBatches ==
  ASSUME BatchesOk, NEW G,
         \A b \in inflight : \A x \in b.reads : x[2] \notin G /\ alloc'[x[2]] = alloc[x[2]],
         free' \subseteq free \cup G,
         \A v \in Units : valid'[v] = valid[v],
         UNCHANGED <<inflight, launched, ostate, ogroup, oslot, table, requests, deferred,
                     tasks, copies, lost>>
  PROVE  BatchesOk'
  <1>1. \A b, o, u : Covered(b, o, u)' = Covered(b, o, u)
    BY CoveredFrame, TransfersFrame
  <1>2. \A b \in inflight : b.writes \subseteq b.reads
    BY DEF BatchesOk
  <1> QED
    BY <1>1, <1>2 DEF BatchesOk, Mem

LEMMA EvictHost_Effect ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k)
  PROVE  LET h == hcache[k] IN
         /\ h \in HostUnits /\ h \notin free /\ Held(h)
         /\ alloc[h] = "shared" /\ sharedKey[h] = k
         /\ \A o \in Owners, p \in Positions : table[o][p].u # h
         /\ \A o \in Owners : oslot[o] # h
         /\ \A d \in deferred : h \notin d.units
         /\ \A c \in copies : c.read # h /\ c.write # h
         /\ \A t \in tasks : h \notin t.units /\ t.slot # h /\ t.h # h
         /\ \A j \in Keys : hcache[j] = h => j = k
         /\ \A j \in Keys : cache[j] # h
         /\ \A v \in AllUnits : alloc'[v] = IF v = h THEN NONE ELSE alloc[v]
         /\ \A v \in AllUnits : sharedKey'[v] = IF v = h THEN NONE ELSE sharedKey[v]
         /\ \A j \in Keys : hcache'[j] = IF j = k THEN NONE ELSE hcache[j]
         /\ hcache' = [hcache EXCEPT ![k] = NONE]
         /\ free' = free \cup {h}
         /\ alloc' = [alloc EXCEPT ![h] = NONE]
         /\ sharedKey' = [sharedKey EXCEPT ![h] = NONE]
         /\ valid' = [valid EXCEPT ![h] = {}]
         /\ \A v \in Units \cup HostUnits : valid'[v] = IF v = h THEN {} ELSE valid[v]
         /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, deferred, copies, ncopy, tasks, trial,
                        member, vis, dbt, launched, inflight, handoff, requests>>
  <1> DEFINE h == hcache[k]
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, WorkTypes
  <1>1. /\ h # NONE
        /\ \A c \in copies : c.read # h /\ c.write # h
        /\ \A t \in tasks : t.held => t.h # h
        /\ hcache' = [hcache EXCEPT ![k] = NONE]
        /\ valid' = [valid EXCEPT ![h] = {}]
        /\ Drop({h}, table, oslot, cache, hcache', deferred, copies)
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, deferred, copies, ncopy, tasks, trial,
                       member, vis, dbt, launched, inflight, handoff, requests>>
    BY DEF EvictHost, plane, On, AllRules
  <1>2. h \in HostUnits /\ alloc[h] = "shared" /\ sharedKey[h] = k
    BY <1>0, <1>1 DEF IndInv, CacheShared, SharedIff, UnitTyped, AllUnits
  <1>3. \A d \in deferred : h \notin d.units
    BY <1>2 DEF IndInv, DeferredOk, CrossingsOk
  <1>4. \A t \in tasks : h \notin t.units /\ t.slot # h /\ t.h # h
    BY <1>0, <1>1, <1>2 DEF IndInv, TasksOk, CrossingsOk, WorkTyped
  <1>5. \A j \in Keys : hcache[j] = h => j = k
    BY <1>2 DEF IndInv, CacheShared
  <1>6. /\ \A o \in Owners, p \in Positions : table[o][p].u # h
        /\ \A o \in Owners : oslot[o] # h
        /\ \A j \in Keys : cache[j] # h
    BY <1>0, <1>2 DEF OwnerTyped, UnitTyped, Empty, Private, Shared
  <1>7. Held(h) /\ h \notin free
    BY <1>2 DEF IndInv, FreeClean, Held, HeldIn
  <1>8. ~HeldIn(h, table, oslot, cache, hcache', deferred, copies, tasks)
    BY <1>0, <1>1, <1>3, <1>4, <1>5, <1>6 DEF HeldIn, UnitTyped
  <1>9. ReclaimedWith({h}, table, oslot, cache, hcache', deferred, copies, tasks) = {h}
    BY <1>8, ReclaimedHeld DEF On, AllRules
  <1>10. /\ free' = free \cup {h}
         /\ alloc' = [alloc EXCEPT ![h] = NONE]
         /\ sharedKey' = [sharedKey EXCEPT ![h] = NONE]
    BY <1>0, <1>1, <1>2, <1>9 DEF Drop, DropWith, Reset, UnitTyped, AllUnits
  <1>11. /\ \A v \in AllUnits : alloc'[v] = IF v = h THEN NONE ELSE alloc[v]
         /\ \A v \in AllUnits : sharedKey'[v] = IF v = h THEN NONE ELSE sharedKey[v]
         /\ \A j \in Keys : hcache'[j] = IF j = k THEN NONE ELSE hcache[j]
    BY <1>0, <1>1, <1>10 DEF UnitTyped
  <1>12. \A v \in Units \cup HostUnits : valid'[v] = IF v = h THEN {} ELSE valid[v]
    BY <1>0, <1>1, <1>2 DEF UnitTyped
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>10, <1>11, <1>12

LEMMA EvictHost_Held ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k)
  PROVE  \A v \in AllUnits : Held(v)' = (Held(v) /\ v # hcache[k])
  <1> SUFFICES ASSUME NEW v \in AllUnits
               PROVE  Held(v)' = (Held(v) /\ v # hcache[k])
    OBVIOUS
  <1>1. CASE v # hcache[k]
    BY <1>1, EvictHost_Effect DEF Held, HeldIn, AllUnits
  <1>2. CASE v = hcache[k]
    BY <1>2, EvictHost_Effect DEF Held, HeldIn
  <1> QED
    BY <1>1, <1>2

(* The validity of every device unit stays; the evicted host block holds  *)
(* nothing.                                                                *)
LEMMA EvictHost_Valid ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k)
  PROVE  /\ \A v \in Units : valid'[v] = valid[v]
         /\ \A v \in HostUnits : v # hcache[k] => valid'[v] = valid[v]
         /\ ValidTyped'
  <1>1. hcache[k] \in HostUnits /\ \A v \in Units : v # hcache[k]
    BY EvictHost_Effect, Sizes
  <1>2. ValidTyped
    BY UnitTypes DEF UnitTyped, ValidTyped
  <1>3. valid' = [valid EXCEPT ![hcache[k]] = {}]
    BY EvictHost_Effect
  <1> QED
    BY <1>1, <1>2, <1>3 DEF ValidTyped

LEMMA EvictHost_Typed ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k), UNCHANGED rare
  PROVE  TypeInv'
  <1>1. UnitTyped'
    BY UnitTypes, EvictHost_Effect, EvictHost_Valid DEF UnitTyped, ValidTyped, AllUnits
  <1>2. OwnerTyped' /\ PlaneTyped' /\ WorkTyped' /\ RequestTyped'
    BY OwnerTypes, PlaneTypes, WorkTypes, RequestTypes, OwnerTypedFrame, PlaneTypedFrame,
       WorkTypedFrame, RequestTypedFrame, EvictHost_Effect DEF rare
  <1> QED
    BY <1>1, <1>2, TypeInvNext

LEMMA EvictHost_Free ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k)
  PROVE  FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
  <1> DEFINE h == hcache[k]
  <1>1. /\ \A v \in AllUnits : alloc'[v] = IF v \in {h} THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v \in {h} THEN NONE ELSE sharedKey[v]
        /\ {h} \subseteq AllUnits
    BY EvictHost_Effect DEF AllUnits
  <1>2. /\ \A v \in AllUnits : Held(v)' => Held(v)
        /\ \A v \in AllUnits : Held(v) => Held(v)' \/ v \in {h}
        /\ \A v \in {h} : Held(v) /\ ~Held(v)'
    BY EvictHost_Held, EvictHost_Effect DEF AllUnits
  <1> QED
    BY <1>1, <1>2, EvictHost_Effect, UnitTypes, FreeReclaim DEF IndInv

LEMMA EvictHost_Shared ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k), UNCHANGED rare
  PROVE  AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared' /\ TrialVisibility'
  <1>0. UnitTyped
    BY UnitTypes
  <1>1. AllocOwner' /\ SharedIff'
    BY EvictHost_Effect DEF IndInv, AllocOwner, SharedIff, AllUnits, NONE
  <1>2. \A v \in AllUnits : alloc[v] = "shared" => sharedKey[v] \in Keys
    BY <1>0 DEF IndInv, SharedIff, UnitTyped
  <1>3. SharedCached'
    BY <1>2, EvictHost_Effect DEF IndInv, SharedCached, AllUnits, NONE
  <1>4. CacheShared'
    BY <1>0, EvictHost_Effect DEF IndInv, CacheShared, UnitTyped, AllUnits
  <1>5. TrialVisibility'
    BY EvictHost_Effect DEF IndInv, TrialVisibility, MadeBy
  <1> QED
    BY <1>1, <1>3, <1>4, <1>5

LEMMA EvictHost_Owners ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k), UNCHANGED rare
  PROVE  Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk' /\ TasksOk'
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, WorkTypes
  <1>1. Tables'
    <2>1. \A o \in Owners, p \in Positions : table[o][p].u # NONE => table[o][p].u \in Units
      BY <1>0 DEF OwnerTyped, Empty, Private, Shared
    <2> QED
      BY <1>0, <2>1, EvictHost_Effect, EvictHost_Valid DEF IndInv, Tables, Live, Mem, OwnerTyped,
                                                           AllUnits
  <1>2. OwnersOk'
    BY <1>0, EvictHost_Effect DEF IndInv, OwnersOk, Live, AllUnits, rare
  <1>3. DeferredOk'
    BY <1>0, EvictHost_Effect DEF IndInv, DeferredOk, WorkTyped, AllUnits, rare
  <1>4. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY EvictHost_Effect, CopiesOkEach
    <2>2. CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits /\ c.k \in Keys
      BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. /\ c.read # hcache[k] /\ c.write # hcache[k]
          /\ alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
      BY <2>2, EvictHost_Effect
    <2>4. (c.fill /\ c.kind = "load") => c.write \in Units /\ valid'[c.write] = valid[c.write]
      BY <2>2, <1>0, EvictHost_Valid DEF CopyOk, UnitTyped, AllUnits
    <2>5. (c.fill /\ c.kind = "store") => hcache'[c.k] = hcache[c.k]
      BY <2>2, <2>3, EvictHost_Effect DEF CopyOk
    <2>6. \A q \in Owners : Moving(q)' = Moving(q)
      BY EvictHost_Effect DEF Moving
    <2> QED
      BY <2>2, <2>3, <2>4, <2>5, <2>6, EvictHost_Effect DEF CopyOk, Live, Mem, rare
  <1>5. TasksOk'
    <2>1. \A t \in tasks : t.units \subseteq AllUnits /\ t.writes \subseteq AllUnits
                           /\ (t.slot # NONE => t.slot \in AllUnits)
                           /\ (t.h # NONE => t.h \in AllUnits) /\ t.k \in Keys \cup {NONE}
      BY <1>0 DEF WorkTyped, AllUnits
    <2>2. \A t \in tasks : t.kind = "put" => t.k # k /\ t.k \in Keys
      BY <1>0, EvictHost_Effect DEF IndInv, TasksOk, WorkTyped
    <2>3. \A t \in tasks : \A u \in t.units \cup {t.slot, t.h} : u # NONE => alloc'[u] = alloc[u]
      BY <2>1, EvictHost_Effect
    <2>4. \A t \in tasks :
            /\ t.kind = "transfer" =>
                 /\ alloc'[t.slot] = alloc[t.slot]
                 /\ \A u \in t.writes : alloc'[u] = alloc[u] /\ valid'[u] = valid[u]
                 /\ ostate[t.o] = "live" =>
                      /\ TableUnits(t.o)' = TableUnits(t.o)
                      /\ \A u \in TableUnits(t.o) : valid'[u] = valid[u]
            /\ t.kind = "prefetch" => alloc'[t.h] = alloc[t.h]
      <3> SUFFICES ASSUME NEW t \in tasks
                   PROVE  /\ t.kind = "transfer" =>
                               /\ alloc'[t.slot] = alloc[t.slot]
                               /\ \A u \in t.writes : alloc'[u] = alloc[u] /\ valid'[u] = valid[u]
                               /\ ostate[t.o] = "live" =>
                                    /\ TableUnits(t.o)' = TableUnits(t.o)
                                    /\ \A u \in TableUnits(t.o) : valid'[u] = valid[u]
                          /\ t.kind = "prefetch" => alloc'[t.h] = alloc[t.h]
        OBVIOUS
      <3>1. /\ t.kind = "transfer" =>
                 t.writes \subseteq t.units /\ t.slot # NONE /\ t.writes \subseteq Units
                 /\ t.o \in Owners
            /\ t.kind = "prefetch" => t.h # NONE
        BY DEF IndInv, TasksOk
      <3>2. t.o \in Owners => TableUnits(t.o) \subseteq Units /\ TableUnits(t.o)' = TableUnits(t.o)
        BY <1>0, EvictHost_Effect DEF TableUnits, OwnerTyped, Empty, Private, Shared
      <3>3. \A u \in Units : valid'[u] = valid[u]
        BY EvictHost_Valid
      <3>4. \A u \in t.units \cup {t.slot, t.h} : u # NONE => alloc'[u] = alloc[u]
        BY <2>3
      <3> QED
        BY <3>1, <3>2, <3>3, <3>4
    <2>5. \A t \in tasks : t.kind = "put" => hcache'[t.k] = hcache[t.k]
      BY <2>2, EvictHost_Effect
    <2> QED
      BY <2>4, <2>5, EvictHost_Effect, TasksKeep DEF IndInv
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5

LEMMA EvictHost_Plane ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k), UNCHANGED rare
  PROVE  /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
         /\ BatchesOk'
  <1> DEFINE h == hcache[k]
  <1>1. UNCHANGED <<ostate, ogroup, oslot, table, cache, deferred, copies, ncopy, tasks, trial,
                    member, vis, dbt, launched, inflight, handoff, requests, remote, lost, bad>>
    BY EvictHost_Effect DEF rare
  <1>2. MovingUnmapped' /\ LostOk' /\ RequestsOk' /\ NoBad'
    BY <1>1, MovingUnmappedFrame, LostOkFrame, RequestsOkFrame, NoBadFrame, TransfersFrame
       DEF IndInv
  <1>3. TablesMapped' /\ HandoffOk'
    BY <1>1, TransfersFrame, EvictHost_Valid, HostValidMapped
  <1>4. \A b \in inflight : \A x \in b.reads : x[2] \notin {h} /\ alloc'[x[2]] = alloc[x[2]]
    <2>1. \A b \in inflight : \A x \in b.reads : x[2] \in AllUnits
      BY RequestTypes DEF RequestTyped
    <2>2. \A b \in inflight : \A x \in b.reads : x[2] # h
      BY ReadHeld, EvictHost_Effect, RequestTypes DEF RequestTyped
    <2> QED
      BY <2>1, <2>2, EvictHost_Effect
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, EvictHost_Effect, EvictHost_Valid, HostValidBatches DEF IndInv

(* The crossings and P8: the evicted host block is held by no crossing and *)
(* no copy, and the validity of every block they hold stays.               *)
LEMMA EvictHost_Cross ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k), UNCHANGED rare
  PROVE  CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
  <1> DEFINE h == hcache[k]
  <1>0. WorkTyped /\ h \in HostUnits
    BY WorkTypes, EvictHost_Effect
  <1>1. UNCHANGED <<ostate, ogroup, oslot, table, cache, deferred, copies, ncopy, tasks, trial,
                    member, vis, dbt, launched, inflight, handoff, requests, remote, lost, bad>>
    BY EvictHost_Effect DEF rare
  <1>2. \A o \in Owners : Moving(o)' = Moving(o)
    BY <1>1 DEF Moving
  <1>3. \A x \in AllUnits \ {h} : alloc'[x] = alloc[x]
    BY EvictHost_Effect
  <1>4. CopiesPlaced'
    <2> SUFFICES ASSUME NEW c \in copies PROVE CopyPlaced(c)'
      BY <1>1, PlacedEach
    <2>1. CopyPlaced(c) /\ c.read # h /\ c.write # h
      BY EvictHost_Effect, PlacedEach DEF IndInv
    <2> QED
      BY <2>1, EvictHost_Effect DEF CopyPlaced, Mem
  <1>5. CrossingsPlaced'
    <2> SUFFICES ASSUME NEW t \in tasks PROVE CrossingPlaced(t)'
      BY <1>1, PlacedEach
    <2>1. CrossingPlaced(t) /\ CrossTaskOk(t)
      BY PlacedEach, CrossingsOkEach DEF IndInv
    <2>2. t.kind = "replicate" => t.units \subseteq Units
      BY <2>1 DEF CrossTaskOk
    <2> QED
      BY <1>1, <2>1, <2>2, EvictHost_Valid DEF CrossingPlaced, Mem
  <1>6. \A d \in deferred : CrossWaitOk(d)'
    <2> SUFFICES ASSUME NEW d \in deferred PROVE CrossWaitOk(d)'
      OBVIOUS
    <2>1. CrossWaitOk(d) /\ h \notin d.units /\ d.units \subseteq AllUnits
      BY <1>0, EvictHost_Effect, CrossingsOkEach DEF IndInv, WorkTyped
    <2> QED
      BY <1>1, <1>2, <1>3, <2>1 DEF CrossWaitOk, ReadersOf, MadeBy
  <1>7. \A t \in tasks : CrossTaskOk(t)'
    <2> SUFFICES ASSUME NEW t \in tasks PROVE CrossTaskOk(t)'
      OBVIOUS
    <2>1. CrossTaskOk(t) /\ h \notin t.units /\ t.units \subseteq AllUnits
      BY <1>0, EvictHost_Effect, CrossingsOkEach DEF IndInv, WorkTyped
    <2>2. t.kind = "replicate" => t.writes \subseteq Units
      BY <2>1 DEF CrossTaskOk
    <2> QED
      BY <1>1, <1>2, <1>3, <2>1, <2>2, EvictHost_Valid DEF CrossTaskOk, ReadersOf, Mem
  <1> QED
    BY <1>1, <1>4, <1>5, <1>6, <1>7, CrossingsOkEach

LEMMA EvictHost_Inv ==
  ASSUME IndInv, NEW k \in Keys, EvictHost(k), UNCHANGED rare
  PROVE  IndInv'
  BY EvictHost_Typed, EvictHost_Free, EvictHost_Shared, EvictHost_Owners, EvictHost_Plane,
     EvictHost_Cross DEF IndInv

-----------------------------------------------------------------------------
(* CommitUpdate and Rollback                                               *)

(* Every conjunct but TypeInv, OwnersOk, TasksOk, CrossingsOk and          *)
(* TrialVisibility reads neither trial nor vis.                            *)
LEMMA TrialFrame ==
  ASSUME IndInv,
         UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                     tasks, valid, member, dbt, launched, inflight, handoff, requests, alloc,
                     sharedKey, remote, lost, bad>>
  PROVE  /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
         /\ SharedCached' /\ CacheShared' /\ Tables' /\ DeferredOk' /\ CopiesOk'
         /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
         /\ BatchesOk' /\ NoBad' /\ CopiesPlaced' /\ CrossingsPlaced'
  BY FreeCleanFrame, NoLeakFrame, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
     SharedCachedFrame, CacheSharedFrame, TablesFrame, DeferredOkFrame, CopiesOkFrame,
     MovingUnmappedFrame, LostOkFrame, TablesMappedFrame, HandoffOkFrame, RequestsOkFrame,
     BatchesOkFrame, NoBadFrame, CopiesPlacedFrame, CrossingsPlacedFrame, TransfersFrame DEF IndInv

LEMMA CommitUpdate_Inv ==
  ASSUME IndInv, NEW e \in Epochs, CommitUpdate(e), UNCHANGED rare
  PROVE  IndInv'
  <1>0. PlaneTyped
    BY PlaneTypes
  <1>1. /\ trial[e] = "yes"
        /\ trial' = [trial EXCEPT ![e] = "no"]
        /\ vis' = [k \in Keys |-> [d |-> IF vis[k].d = e THEN NONE ELSE vis[k].d,
                                   h |-> IF vis[k].h = e THEN NONE ELSE vis[k].h]]
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       tasks, valid, member, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF CommitUpdate, plane, rare
  <1>2. /\ \A k \in Keys : /\ vis'[k].d = IF vis[k].d = e THEN NONE ELSE vis[k].d
                           /\ vis'[k].h = IF vis[k].h = e THEN NONE ELSE vis[k].h
        /\ \A x \in Epochs : trial'[x] = IF x = e THEN "no" ELSE trial[x]
    BY <1>0, <1>1 DEF PlaneTyped
  <1>3. TypeInv'
    <2>1. PlaneTyped'
      BY <1>0, <1>1, <1>2 DEF PlaneTyped
    <2> QED
      BY <1>1, <2>1, UnitTypes, OwnerTypes, WorkTypes, RequestTypes, UnitTypedFrame,
         OwnerTypedFrame, WorkTypedFrame, RequestTypedFrame, TypeInvNext
  <1>4. OwnersOk'
    BY <1>1, <1>2 DEF IndInv, OwnersOk, Live
  <1>5. TasksOk'
    <2>1. \A t \in tasks : t.kind = "put" => t.k \in Keys
      BY WorkTypes DEF IndInv, TasksOk, WorkTyped
    <2> QED
      BY <1>1, <1>2, <2>1 DEF IndInv, TasksOk, Live, TableUnits, Mem, NONE
  <1>6. \A x \in Epochs : MadeBy(x)' = IF x = e THEN NONE ELSE MadeBy(x)
    BY <1>1, <1>2 DEF MadeBy
  (* A copy of a key's visibility that MadeBy of an epoch states keeps it.  *)
  <1>7. \A k \in Keys, x \in Epochs :
          /\ vis[k].d = MadeBy(x) => vis'[k].d = MadeBy(x)'
          /\ vis[k].h = MadeBy(x) => vis'[k].h = MadeBy(x)'
    <2> SUFFICES ASSUME NEW k \in Keys, NEW x \in Epochs
                 PROVE  /\ vis[k].d = MadeBy(x) => vis'[k].d = MadeBy(x)'
                        /\ vis[k].h = MadeBy(x) => vis'[k].h = MadeBy(x)'
      OBVIOUS
    <2>1. MadeBy(x) \in {NONE, x} /\ (x = e => MadeBy(x) = e)
      BY <1>1 DEF MadeBy
    <2> QED
      BY <1>2, <1>6, <2>1, Sizes DEF NONE
  <1>8. TrialVisibility'
    <2>1. \A c \in copies : c.g \in Groups /\ GroupEpoch[c.g] \in Epochs
      BY DEF IndInv, CopiesOk
    <2>2. \A t \in tasks : t.kind = "replicate" => OwnerEpoch[t.o] \in Epochs
      BY CrossingsOkEach DEF IndInv, CrossTaskOk
    <2>3. /\ \A k \in Keys :
               \A c \in copies :
                 (c.kind = "load" /\ c.fill /\ c.k = k) => vis'[k].d = MadeBy(GroupEpoch[c.g])'
          /\ \A k \in Keys :
               \A c \in copies :
                 (c.kind = "store" /\ c.fill /\ c.k = k) => vis'[k].h = MadeBy(GroupEpoch[c.g])'
          /\ \A k \in Keys :
               \A t \in tasks :
                 (t.kind = "replicate" /\ cache[k] \in t.units) =>
                     vis'[k].d = MadeBy(OwnerEpoch[t.o])'
      BY <1>7, <2>1, <2>2 DEF IndInv, TrialVisibility
    <2>4. \A k \in Keys :
            /\ (cache[k] # NONE /\ vis'[k].d # NONE) =>
                 /\ \A o \in Owners, p \in Positions :
                      table[o][p] = Shared(cache[k]) => OwnerEpoch[o] = vis'[k].d
                 /\ \A c \in copies : c.read = cache[k] => GroupEpoch[c.g] = vis'[k].d
            /\ (hcache[k] # NONE /\ vis'[k].h # NONE) =>
                 /\ \A c \in copies : c.read = hcache[k] => GroupEpoch[c.g] = vis'[k].h
                 /\ \A t \in tasks : t.kind = "put" => t.h # hcache[k]
      BY <1>1, <1>2 DEF IndInv, TrialVisibility, NONE
    <2> QED
      BY <1>1, <2>3, <2>4 DEF TrialVisibility
  <1>9. CrossingsOk'
    <2>1. \A d \in deferred : CrossWaitOk(d)'
      <3> SUFFICES ASSUME NEW d \in deferred PROVE CrossWaitOk(d)'
        OBVIOUS
      <3>1. CrossWaitOk(d)
        BY CrossingsOkEach DEF IndInv
      <3>2. d.kind = "replicate" =>
              \A k \in Keys : cache[k] \in d.units => vis'[k].d = MadeBy(OwnerEpoch[d.o])'
        BY <1>7, <3>1 DEF CrossWaitOk
      <3> QED
        BY <1>1, <3>1, <3>2 DEF CrossWaitOk, Moving, ReadersOf
    <2>2. \A t \in tasks : CrossTaskOk(t)'
      BY <1>1, CrossingsOkEach DEF IndInv, CrossTaskOk, Moving, ReadersOf, Mem
    <2> QED
      BY <1>1, <2>1, <2>2, CrossingsOkEach
  <1> QED
    BY <1>1, <1>3, <1>4, <1>5, <1>8, <1>9, TrialFrame DEF IndInv

LEMMA Rollback_Inv ==
  ASSUME IndInv, NEW e \in Epochs, Rollback(e), UNCHANGED rare
  PROVE  IndInv'
  <1>0. PlaneTyped
    BY PlaneTypes
  <1>1. /\ trial[e] = "yes"
        /\ \A o \in Owners : OwnerEpoch[o] = e => ostate[o] # "live"
        /\ trial' = [trial EXCEPT ![e] = "rolled"]
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       tasks, valid, vis, member, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF Rollback, plane, rare
  <1>2. \A x \in Epochs : trial'[x] = IF x = e THEN "rolled" ELSE trial[x]
    BY <1>0, <1>1 DEF PlaneTyped
  <1>3. TypeInv'
    <2>1. PlaneTyped'
      BY <1>0, <1>1, <1>2 DEF PlaneTyped
    <2> QED
      BY <1>1, <2>1, UnitTypes, OwnerTypes, WorkTypes, RequestTypes, UnitTypedFrame,
         OwnerTypedFrame, WorkTypedFrame, RequestTypedFrame, TypeInvNext
  <1>4. OwnersOk'
    BY <1>1, <1>2 DEF IndInv, OwnersOk, Live
  <1>5. TasksOk'
    BY <1>1, TasksOkFrame DEF IndInv
  <1>6. \A x \in Epochs : MadeBy(x)' = MadeBy(x)
    BY <1>1, <1>2 DEF MadeBy
  <1>7. TrialVisibility'
    <2>1. \A c \in copies : c.g \in Groups /\ GroupEpoch[c.g] \in Epochs
      BY DEF IndInv, CopiesOk
    <2>2. \A t \in tasks : t.kind = "replicate" => OwnerEpoch[t.o] \in Epochs
      BY CrossingsOkEach DEF IndInv, CrossTaskOk
    <2> QED
      BY <1>1, <1>6, <2>1, <2>2 DEF IndInv, TrialVisibility
  <1>8. CrossingsOk'
    <2>1. \A d \in deferred : d.kind = "replicate" => OwnerEpoch[d.o] \in Epochs
      BY CrossingsOkEach DEF IndInv, CrossWaitOk
    <2> QED
      BY <1>1, <1>6, <2>1 DEF IndInv, CrossingsOk, Moving, ReadersOf, Mem
  <1> QED
    BY <1>1, <1>3, <1>4, <1>5, <1>7, <1>8, TrialFrame DEF IndInv

-----------------------------------------------------------------------------
(* Shrink and Grow                                                         *)

(* Every conjunct but TypeInv, FreeClean, NoLeak, Outside, Regions and     *)
(* BatchesOk reads neither free nor member.                                *)
LEMMA ResizeFrame ==
  ASSUME IndInv,
         UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, tasks,
                     valid, vis, trial, dbt, launched, inflight, handoff, requests, alloc,
                     sharedKey, remote, lost, bad>>
  PROVE  /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk'
         /\ DeferredOk' /\ CopiesOk' /\ TasksOk' /\ MovingUnmapped' /\ LostOk' /\ TablesMapped'
         /\ HandoffOk' /\ RequestsOk' /\ TrialVisibility' /\ NoBad' /\ CrossingsOk'
         /\ CopiesPlaced' /\ CrossingsPlaced'
  <1>1. CrossingsOk'
    BY DEF IndInv, CrossingsOk, Moving, ReadersOf, Mem, MadeBy
  <1> QED
    BY <1>1, AllocOwnerFrame, SharedIffFrame, SharedCachedFrame, CacheSharedFrame, TablesFrame,
       OwnersOkFrame, DeferredOkFrame, CopiesOkFrame, TasksOkFrame, MovingUnmappedFrame, LostOkFrame,
       TablesMappedFrame, HandoffOkFrame, RequestsOkFrame, TrialVisibilityFrame, NoBadFrame,
       CopiesPlacedFrame, CrossingsPlacedFrame, TransfersFrame DEF IndInv

LEMMA Shrink_Inv ==
  ASSUME IndInv, NEW u \in Units, Shrink(u), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ u \in free /\ u \in Movable
        /\ free' = free \ {u}
        /\ member' = member \ {u}
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, tasks,
                       valid, vis, trial, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF Shrink, plane, rare
  <1>2. alloc[u] = NONE /\ sharedKey[u] = NONE /\ ~Held(u)
    BY <1>1 DEF IndInv, FreeClean
  <1>3. \A v : Held(v)' = Held(v)
    BY <1>1 DEF Held, HeldIn
  <1>4. TypeInv'
    <2>1. UnitTyped'
      BY <1>1, UnitTypes DEF UnitTyped
    <2> QED
      BY <1>1, <2>1, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes, OwnerTypedFrame,
         PlaneTypedFrame, WorkTypedFrame, RequestTypedFrame, TypeInvNext
  <1>5. FreeClean' /\ NoLeak'
    BY <1>1, <1>2, <1>3 DEF IndInv, FreeClean, NoLeak, InRegion
  <1>6. Outside' /\ Regions'
    BY <1>1, <1>2 DEF IndInv, Outside, Regions, OneRegion
  <1>7. BatchesOk'
    <2>1. /\ \A b \in inflight : \A x \in b.reads : x[2] \notin {} /\ alloc'[x[2]] = alloc[x[2]]
          /\ free' \subseteq free \cup {}
      BY <1>1
    <2> QED
      BY <1>1, <2>1, BatchesKeep DEF IndInv
  <1> QED
    BY <1>1, <1>4, <1>5, <1>6, <1>7, ResizeFrame DEF IndInv

LEMMA Grow_Inv ==
  ASSUME IndInv, NEW u \in Units, Grow(u), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ u \in Movable /\ u \notin member
        /\ \A v \in member : UnitChunk[v] # UnitChunk[u]
        /\ free' = free \cup {u}
        /\ member' = member \cup {u}
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, tasks,
                       valid, vis, trial, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF Grow, plane, rare, On, AllRules
  <1>2. alloc[u] = NONE /\ sharedKey[u] = NONE /\ ~Held(u) /\ u \notin free
    BY <1>1 DEF IndInv, Outside, NoLeak, InRegion, AllUnits
  <1>3. \A v : Held(v)' = Held(v)
    BY <1>1 DEF Held, HeldIn
  <1>4. TypeInv'
    <2>1. UnitTyped'
      BY <1>1, UnitTypes DEF UnitTyped, AllUnits
    <2> QED
      BY <1>1, <2>1, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes, OwnerTypedFrame,
         PlaneTypedFrame, WorkTypedFrame, RequestTypedFrame, TypeInvNext
  <1>5. FreeClean' /\ NoLeak'
    BY <1>1, <1>2, <1>3 DEF IndInv, FreeClean, NoLeak, InRegion
  <1>6. Outside' /\ Regions'
    BY <1>1, <1>2 DEF IndInv, Outside, Regions, OneRegion
  <1>7. BatchesOk'
    <2>1. \A b \in inflight : \A x \in b.reads : x[2] \notin {u}
      BY <1>2, ReadHeld, RequestTypes DEF Held, HeldIn, RequestTyped
    <2> QED
      BY <1>1, <2>1, BatchesKeep DEF IndInv
  <1> QED
    BY <1>1, <1>4, <1>5, <1>6, <1>7, ResizeFrame DEF IndInv

-----------------------------------------------------------------------------

THEOREM Step3 ==
    IndInv /\ (/\ \/ \E k \in Keys : Evict(k) \/ EvictHost(k)
                  \/ \E e \in Epochs : CommitUpdate(e) \/ Rollback(e)
                  \/ \E u \in Units : Shrink(u) \/ Grow(u)
               /\ UNCHANGED rare)
    => IndInv'
  BY Evict_Inv, EvictHost_Inv, CommitUpdate_Inv, Rollback_Inv, Shrink_Inv, Grow_Inv
=============================================================================
