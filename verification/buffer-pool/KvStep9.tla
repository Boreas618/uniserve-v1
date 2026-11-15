------------------------------- MODULE KvStep9 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for the tasks of the host tier: Prefetch,  *)
(* StoreRemote, PeerLost, Deadline, PrefetchEnd and PutEnd.                *)
(***************************************************************************)
EXTENDS KvFrame

USE Sizes, Design DEF On, AllRules

-----------------------------------------------------------------------------
(* Task t is a record of the fields of Task alone.                         *)
TaskRecord(t) ==
  t = [kind |-> t.kind, k |-> t.k, h |-> t.h, src |-> t.src, ok |-> t.ok,
       expired |-> t.expired, p |-> t.p, o |-> t.o, to |-> t.to, slot |-> t.slot,
       held |-> t.held, units |-> t.units, writes |-> t.writes]

(* The prefetch a group's admission asks for: a task that fills that       *)
(* group's host parts.                                                     *)
PrefetchTask(k, h, src, g) == [Task("prefetch", k, h, src) EXCEPT !.to = g]

-----------------------------------------------------------------------------
(* The crossings, and where copies and crossings read, across a step that  *)
(* changes no deferred work, copy, owner or batch, no transfer or crossing *)
(* task, and the allocation, validity and visibility of no device unit and *)
(* of no host unit a copy touches.                                         *)
LEMMA TierCrossings ==
  ASSUME CrossingsOk, CrossingsPlaced, CopiesPlaced,
         UNCHANGED <<deferred, copies, ostate, ogroup, lost, cache, inflight, trial>>,
         Transfers' = Transfers, Crossings' = Crossings,
         \A u \in Units : alloc'[u] = alloc[u] /\ valid'[u] = valid[u],
         \A k \in Keys : vis'[k].d = vis[k].d,
         \A c \in copies : /\ c.read \in HostUnits => valid'[c.read] = valid[c.read]
                           /\ c.write \in HostUnits => valid'[c.write] = valid[c.write]
  PROVE  CrossingsOk' /\ CrossingsPlaced' /\ CopiesPlaced'
  <1>1. \A t : (t \in tasks' /\ t.kind = "replicate") <=> (t \in tasks /\ t.kind = "replicate")
    BY DEF Crossings
  <1>2. \A t : (t \in tasks' /\ t.kind = "transfer") <=> (t \in tasks /\ t.kind = "transfer")
    BY DEF Transfers
  <1>3. \A o : Moving(o)' = Moving(o)
    BY <1>2 DEF Moving
  <1>4. \A d \in deferred : CrossWaitOk(d) => CrossWaitOk(d)'
    <2> SUFFICES ASSUME NEW d \in deferred, CrossWaitOk(d), d.kind = "replicate"
                 PROVE  CrossWaitOk(d)'
      BY DEF CrossWaitOk
    <2>1. PICK u \in Units : d.units = {u} /\ (alloc[u] = d.o \/ alloc[u] = "shared")
      BY DEF CrossWaitOk
    <2> QED
      BY <1>3, <2>1 DEF CrossWaitOk, ReadersOf, MadeBy
  <1>5. \A t \in tasks : (t.kind = "replicate" /\ CrossTaskOk(t)) => CrossTaskOk(t)'
    <2> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate", CrossTaskOk(t)
                 PROVE  CrossTaskOk(t)'
      OBVIOUS
    <2>1. PICK u \in Units : t.units = {u} /\ t.writes = {u}
      BY DEF CrossTaskOk
    <2>2. \A x \in tasks' : x.kind \in {"transfer", "replicate"} => x \in tasks
      BY <1>1, <1>2
    <2> QED
      BY <1>3, <2>1, <2>2 DEF CrossTaskOk, ReadersOf, Mem
  <1>6. CrossingsOk'
    <2>1. \A d \in deferred' : CrossWaitOk(d)'
      BY <1>4, CrossingsOkEach
    <2>2. \A t \in tasks' : CrossTaskOk(t)'
      <3> SUFFICES ASSUME NEW t \in tasks', t.kind = "replicate" PROVE CrossTaskOk(t)'
        BY DEF CrossTaskOk
      <3>1. t \in tasks /\ CrossTaskOk(t)
        BY <1>1, CrossingsOkEach
      <3> QED
        BY <3>1, <1>5
    <2> QED
      BY <2>1, <2>2, CrossingsOkEach
  <1>7. CrossingsPlaced'
    <2> SUFFICES ASSUME NEW t \in tasks', t.kind = "replicate" PROVE CrossingPlaced(t)'
      BY PlacedEach DEF CrossingPlaced
    <2>1. t \in tasks /\ CrossingPlaced(t)
      BY <1>1, PlacedEach
    <2>2. t.units \subseteq Units
      BY <2>1, CrossingsOkEach DEF CrossTaskOk
    <2> QED
      BY <2>1, <2>2 DEF CrossingPlaced, Mem
  <1>8. CopiesPlaced'
    <2> SUFFICES ASSUME NEW c \in copies PROVE CopyPlaced(c)'
      BY PlacedEach
    <2>1. CopyPlaced(c)
      BY PlacedEach
    <2> QED
      BY <2>1 DEF CopyPlaced, Mem
  <1> QED
    BY <1>6, <1>7, <1>8

-----------------------------------------------------------------------------
(* A step that changes only the prefetches and remote stores and leaves    *)
(* every unit held as it was keeps every conjunct but the types, the       *)
(* tasks and trial visibility.                                             *)

LEMMA TierFrame ==
  ASSUME IndInv,
         UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                     trial, member, valid, vis, dbt, launched, inflight, handoff, requests, alloc,
                     sharedKey, lost, bad>>,
         Transfers' = Transfers, Crossings' = Crossings, TaskUnits' \subseteq TaskUnits,
         \A v \in AllUnits : Held(v)' <=> Held(v)
  PROVE  /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
         /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
         /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
         /\ BatchesOk' /\ NoBad' /\ CrossingsOk' /\ CrossingsPlaced' /\ CopiesPlaced'
  <1>1. free \subseteq AllUnits
    BY DEF IndInv, TypeInv, TypeOK
  <1>2. FreeClean'
    BY <1>1 DEF IndInv, FreeClean
  <1>3. NoLeak'
    BY DEF IndInv, NoLeak, InRegion
  <1>4. /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
        /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk' /\ MovingUnmapped' /\ LostOk'
        /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ BatchesOk' /\ NoBad'
    BY OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame, SharedCachedFrame,
       CacheSharedFrame, TablesFrame, OwnersOkFrame, DeferredOkFrame, CopiesOkFrame,
       MovingUnmappedFrame, LostOkFrame, TablesMappedFrame, HandoffOkFrame, RequestsOkFrame,
       BatchesOkFrame, NoBadFrame DEF IndInv
  <1>5. CrossingsOk' /\ CrossingsPlaced' /\ CopiesPlaced'
    BY TierCrossings DEF IndInv
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5

(* The types of every variable but the tasks and the remote store's keys,  *)
(* after a step that changes only those.                                   *)
LEMMA TierTypes ==
  ASSUME IndInv,
         UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                     trial, member, valid, vis, dbt, launched, inflight, handoff, requests, alloc,
                     sharedKey, lost, bad>>,
         remote' \subseteq Keys,
         \A t \in tasks' :
           /\ t.units \subseteq AllUnits /\ t.writes \subseteq AllUnits
           /\ t.k \in Keys \cup {NONE} /\ t.h \in HostUnits \cup {NONE}
           /\ t.slot \in Slots \cup {NONE} /\ t.o \in Owners \cup {NONE} /\ t.to \in Groups \cup {NONE}
           /\ t.p \in Positions \cup {NONE} /\ t.src \in {NONE, "remote", "peer"}
           /\ t.kind \in {"transfer", "replicate"} <=> t.k = NONE
           /\ t.kind = "prefetch" => t.to \in Groups
           /\ TaskRecord(t)
  PROVE  TypeInv'
  BY DEF IndInv, TypeInv, TypeOK, Stamps, TaskRecord

-----------------------------------------------------------------------------
(* Prefetch                                                                *)

(* Prefetch(k, src, g) taking host unit h, which is free.                  *)
PrefetchAt(k, src, g, h) ==
  /\ hcache[k] = NONE
  /\ h \in free /\ alloc[h] = NONE /\ sharedKey[h] = NONE /\ Unheld(h)
  /\ \A j \in Keys : cache[j] # h
  /\ free' = free \ {h}
  /\ alloc' = [alloc EXCEPT ![h] = "task"]
  /\ tasks' = tasks \cup {PrefetchTask(k, h, src, g)}
  /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, trial,
                 member, valid, vis, dbt, launched, inflight, handoff, requests, sharedKey, remote,
                 lost, bad>>

LEMMA PrefetchPick ==
  ASSUME IndInv, NEW k \in Keys, NEW src \in {"remote", "peer"}, NEW g \in Groups,
         Prefetch(k, src, g)
  PROVE  \E h \in HostUnits : PrefetchAt(k, src, g, h)
  <1>1. PICK h \in free \cap HostUnits :
          /\ free' = free \ {h}
          /\ alloc' = [alloc EXCEPT ![h] = "task"]
          /\ tasks' = tasks \cup {PrefetchTask(k, h, src, g)}
    BY DEF Prefetch, PrefetchTask
  <1>2. /\ hcache[k] = NONE
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, trial,
                       member, valid, vis, dbt, launched, inflight, handoff, requests, sharedKey,
                       remote, lost, bad>>
    BY DEF Prefetch, plane, rare
  <1>3. alloc[h] = NONE /\ sharedKey[h] = NONE /\ ~Held(h)
    BY <1>1 DEF IndInv, FreeClean
  <1>4. Unheld(h) /\ \A j \in Keys : cache[j] # h
    BY <1>3, TasksHeld DEF Held, HeldIn, Unheld
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4 DEF PrefetchAt

LEMMA PrefetchInv ==
  ASSUME IndInv, NEW k \in Keys, NEW src \in {"remote", "peer"}, NEW g \in Groups,
         NEW h \in HostUnits, PrefetchAt(k, src, g, h)
  PROVE  IndInv'
  <1> DEFINE t0 == PrefetchTask(k, h, src, g)
  <1>1. TypeOK
    BY TypeFacts
  <1>2. /\ t0.kind = "prefetch" /\ t0.k = k /\ t0.h = h /\ t0.src = src /\ t0.held
        /\ t0.units = {} /\ t0.writes = {} /\ t0.slot = NONE /\ t0.o = NONE /\ t0.to = g
        /\ t0.p = NONE
        /\ TaskRecord(t0)
    BY DEF PrefetchTask, Task, TaskRecord
  <1>3. h \in AllUnits /\ h \notin Units /\ h \notin Slots /\ h # NONE
    BY DEF AllUnits
  <1>4. \A v \in AllUnits : v # h => alloc'[v] = alloc[v]
    BY <1>1 DEF PrefetchAt, TypeOK
  <1>5. alloc'[h] = "task"
    BY <1>1, <1>3 DEF PrefetchAt, TypeOK
  <1>6. Transfers' = Transfers /\ Crossings' = Crossings
    BY <1>2 DEF PrefetchAt, Transfers, Crossings
  <1>7. \A v \in AllUnits : Held(v)' <=> Held(v) \/ v = h
    BY <1>2, <1>3 DEF PrefetchAt, Held, HeldIn
  <1>8. TypeInv'
    <2>1. TypeOK'
      BY <1>1, <1>3, <1>5 DEF PrefetchAt, TypeOK
    <2>2. UnitTyped'
      BY <2>1 DEF TypeOK, UnitTyped
    <2>3. OwnerTyped' /\ PlaneTyped' /\ RequestTyped'
      BY OwnerTypes, PlaneTypes, RequestTypes, OwnerTypedFrame, PlaneTypedFrame, RequestTypedFrame
         DEF PrefetchAt
    <2>4. WorkTyped'
      BY <1>2, <1>3, WorkTypes DEF PrefetchAt, WorkTyped, TaskRecord
    <2> QED
      BY <2>2, <2>3, <2>4, TypeInvNext
  <1>9. FreeClean'
    <2> SUFFICES ASSUME NEW v \in free' PROVE alloc'[v] = NONE /\ sharedKey'[v] = NONE /\ ~Held(v)'
      BY DEF FreeClean
    <2>1. v \in free /\ v # h /\ v \in AllUnits
      BY <1>1 DEF PrefetchAt, TypeOK
    <2>2. alloc[v] = NONE /\ sharedKey[v] = NONE /\ ~Held(v)
      BY <2>1 DEF IndInv, FreeClean
    <2> QED
      BY <1>4, <1>7, <2>1, <2>2 DEF PrefetchAt
  <1>10. NoLeak'
    <2> SUFFICES ASSUME NEW v \in AllUnits
                 PROVE  IF InRegion(v)' THEN v \in free' \/ Held(v)' ELSE v \notin free' /\ ~Held(v)'
      BY DEF NoLeak
    <2>1. CASE v = h
      BY <2>1, <1>3, <1>7 DEF InRegion, PrefetchAt
    <2>2. CASE v # h
      BY <2>2, <1>7 DEF InRegion, PrefetchAt, IndInv, NoLeak
    <2> QED
      BY <2>1, <2>2
  <1>11. Outside' /\ Regions' /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. \A v \in Units : alloc'[v] = alloc[v]
      BY <1>3, <1>4 DEF AllUnits
    <2>2. Outside' /\ Regions'
      BY <2>1 DEF PrefetchAt, IndInv, Outside, Regions, OneRegion
    <2>3. AllocOwner' /\ SharedIff' /\ SharedCached'
      BY <1>4, <1>5 DEF PrefetchAt, IndInv, AllocOwner, SharedIff, SharedCached
    <2>4. CacheShared'
      BY CacheSharedFrame DEF PrefetchAt, IndInv
    <2> QED
      BY <2>2, <2>3, <2>4
  <1>12. Tables' /\ OwnersOk' /\ DeferredOk'
    <2>1. \A v \in Units \cup Slots : alloc'[v] = alloc[v]
      BY <1>3, <1>4 DEF AllUnits
    <2>2. \A q \in Owners, p \in Positions : table[q][p].u \in Units \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>3. \A q \in Owners : oslot[q] \in Slots \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>4. \A d \in deferred : d.kind = "commit" => d.units \subseteq Units
      BY DEF IndInv, DeferredOk
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4 DEF PrefetchAt, IndInv, Tables, OwnersOk, DeferredOk, Live, Mem
  <1>13. CopiesOk'
    <2>1. \A c \in copies : c.read # h /\ c.write # h
      BY DEF PrefetchAt, Unheld
    <2>2. \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
      BY DEF IndInv, TypeInv
    <2>3. \A q : Moving(q)' = Moving(q)
      BY <1>6 DEF PrefetchAt, Moving, Transfers
    <2> QED
      BY <1>2, <1>4, <1>6, <2>1, <2>2, <2>3 DEF PrefetchAt, IndInv, CopiesOk, Live, Mem, Transfers
  <1>14. TasksOk'
    <2> SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
      BY TasksOkEach
    <2>1. \A x \in tasks : x.kind = "prefetch" => x.h # h
      BY DEF PrefetchAt, Unheld
    <2>2. CASE t = t0
      BY <2>2, <1>2, <1>5, <2>1 DEF PrefetchAt, TaskOk
    <2>3. CASE t # t0
      <3>1. t \in tasks
        BY <2>3 DEF PrefetchAt
      <3>2. t.slot # h /\ t.h # h /\ h \notin t.units
        BY <3>1 DEF PrefetchAt, Unheld
      <3>3. TaskOk(t)
        BY <3>1, TasksOkEach DEF IndInv
      <3> QED
        BY <3>1, <3>2, <3>3, <1>2, <1>3, <1>4, <1>6
           DEF PrefetchAt, TaskOk, Live, TableUnits, Mem, Transfers, AllUnits
    <2> QED
      BY <2>2, <2>3
  <1>15. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
    <2>1. UNCHANGED <<deferred, dbt, handoff, lost, inflight, requests, ostate, ogroup, oslot,
                      table, launched, ncopy, copies, bad, valid>>
      BY DEF PrefetchAt
    <2> QED
      BY <1>6, <2>1, MovingUnmappedFrame, LostOkFrame, TablesMappedFrame, HandoffOkFrame,
         RequestsOkFrame, NoBadFrame DEF IndInv
  <1>16. BatchesOk'
    <2>1. \A b \in inflight : \A x \in b.reads : x[2] # h
      BY DEF IndInv, BatchesOk, PrefetchAt
    <2>2. \A b \in inflight : b.writes \subseteq b.reads
      BY DEF IndInv, BatchesOk
    <2>3. \A b \in inflight : \A x \in b.reads : x[2] \in AllUnits
      BY RequestTypes DEF RequestTyped
    <2>4. \A b, o, u : Covered(b, o, u)' = Covered(b, o, u)
      BY <1>6, CoveredFrame DEF PrefetchAt
    <2> QED
      BY <1>4, <2>1, <2>2, <2>3, <2>4 DEF PrefetchAt, IndInv, BatchesOk, Mem
  <1>17. TrialVisibility'
    BY <1>2 DEF PrefetchAt, IndInv, TrialVisibility, MadeBy
  <1>18. CrossingsOk' /\ CrossingsPlaced' /\ CopiesPlaced'
    <2>1. \A u \in Units : alloc'[u] = alloc[u]
      BY <1>3, <1>4 DEF AllUnits
    <2> QED
      BY <1>6, <2>1, TierCrossings DEF PrefetchAt, IndInv
  <1> QED
    BY <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18 DEF IndInv

-----------------------------------------------------------------------------
(* StoreRemote                                                             *)

(* StoreRemote(k), with what IndInv says of the host block it reads.       *)
StoreRemoteAt(k) ==
  /\ hcache[k] \in HostUnits /\ alloc[hcache[k]] = "shared" /\ sharedKey[hcache[k]] = k
  /\ \A j \in Keys : hcache[j] = hcache[k] => j = k
  /\ vis[k].h = NONE
  /\ tasks' = tasks \cup {Task("put", k, hcache[k], "remote")}
  /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, trial,
                 member, valid, vis, dbt, launched, inflight, handoff, requests, alloc, sharedKey,
                 remote, lost, bad>>

LEMMA StoreRemotePick ==
  ASSUME IndInv, NEW k \in Keys, StoreRemote(k)
  PROVE  StoreRemoteAt(k)
  <1>1. /\ hcache[k] # NONE /\ vis[k].h = NONE
        /\ tasks' = tasks \cup {Task("put", k, hcache[k], "remote")}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       trial, member, valid, vis, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF StoreRemote, plane, rare
  <1>2. /\ hcache[k] \in HostUnits /\ alloc[hcache[k]] = "shared" /\ sharedKey[hcache[k]] = k
        /\ \A j \in Keys : hcache[j] = hcache[k] => j = k
    BY <1>1, HostCached
  <1> QED
    BY <1>1, <1>2 DEF StoreRemoteAt

LEMMA StoreRemoteInv ==
  ASSUME IndInv, NEW k \in Keys, StoreRemoteAt(k)
  PROVE  IndInv'
  <1> DEFINE t0 == Task("put", k, hcache[k], "remote")
  <1>1. /\ t0.kind = "put" /\ t0.k = k /\ t0.h = hcache[k] /\ t0.src = "remote" /\ t0.held
        /\ t0.units = {} /\ t0.writes = {} /\ t0.slot = NONE /\ t0.o = NONE /\ t0.to = NONE
    BY DEF Task
  <1>2. Transfers' = Transfers /\ Crossings' = Crossings /\ TaskUnits' \subseteq TaskUnits
    BY <1>1 DEF StoreRemoteAt, Transfers, Crossings, TaskUnits
  <1>3. \A v \in AllUnits : Held(v)' <=> Held(v)
    BY <1>1 DEF StoreRemoteAt, Held, HeldIn, AllUnits
  <1>4. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
        /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
        /\ BatchesOk' /\ NoBad' /\ CrossingsOk' /\ CrossingsPlaced' /\ CopiesPlaced'
    BY <1>2, <1>3, TierFrame DEF StoreRemoteAt
  <1>5. TypeInv'
    <2>1. remote' \subseteq Keys
      BY DEF StoreRemoteAt, IndInv, TypeInv
    <2>2. \A t \in tasks' :
            /\ t.units \subseteq AllUnits /\ t.writes \subseteq AllUnits
            /\ t.k \in Keys \cup {NONE} /\ t.h \in HostUnits \cup {NONE}
            /\ t.slot \in Slots \cup {NONE} /\ t.o \in Owners \cup {NONE} /\ t.to \in Groups \cup {NONE}
            /\ t.p \in Positions \cup {NONE} /\ t.src \in {NONE, "remote", "peer"}
            /\ t.kind \in {"transfer", "replicate"} <=> t.k = NONE
            /\ t.kind = "prefetch" => t.to \in Groups
            /\ TaskRecord(t)
      BY <1>1, WorkTypes DEF StoreRemoteAt, WorkTyped, Task, TaskRecord
    <2> QED
      BY <2>1, <2>2, TierTypes DEF StoreRemoteAt
  <1>6. TasksOk'
    <2> SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
      BY TasksOkEach
    <2>1. CASE t = t0
      BY <2>1, <1>1 DEF StoreRemoteAt, TaskOk
    <2>2. CASE t # t0
      <3>1. t \in tasks
        BY <2>2 DEF StoreRemoteAt
      <3>2. TaskOk(t)
        BY <3>1, TasksOkEach DEF IndInv
      <3> QED
        BY <3>1, <3>2, <1>1 DEF StoreRemoteAt, TaskOk, Live, TableUnits, Mem
    <2> QED
      BY <2>1, <2>2
  <1>7. TrialVisibility'
    <2>1. \A j \in Keys : (hcache[j] # NONE /\ vis[j].h # NONE) => hcache[j] # hcache[k]
      BY DEF StoreRemoteAt
    <2> QED
      BY <1>1, <2>1 DEF StoreRemoteAt, IndInv, TrialVisibility, MadeBy
  <1> QED
    BY <1>4, <1>5, <1>6, <1>7 DEF IndInv

-----------------------------------------------------------------------------
(* PeerLost and Deadline: a prefetch or a remote store is replaced by one  *)
(* that differs from it only in whether its source held the block and      *)
(* whether its deadline passed.                                            *)

LEMMA RelabelInv ==
  ASSUME IndInv, NEW t \in tasks, t.kind \in {"prefetch", "put"}, NEW x,
         x.kind = t.kind, x.k = t.k, x.h = t.h, x.src = t.src, x.o = t.o, x.to = t.to,
         x.p = t.p, x.slot = t.slot, x.held = t.held, x.units = t.units, x.writes = t.writes,
         TaskRecord(x),
         tasks' = (tasks \ {t}) \cup {x},
         UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                     trial, member, valid, vis, dbt, launched, inflight, handoff, requests, alloc,
                     sharedKey, remote, lost, bad>>
  PROVE  IndInv'
  <1>1. Transfers' = Transfers /\ Crossings' = Crossings /\ TaskUnits' \subseteq TaskUnits
    BY DEF Transfers, Crossings, TaskUnits
  <1>2. \A v \in AllUnits : Held(v)' <=> Held(v)
    BY DEF Held, HeldIn
  <1>3. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
        /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
        /\ BatchesOk' /\ NoBad' /\ CrossingsOk' /\ CrossingsPlaced' /\ CopiesPlaced'
    BY <1>1, <1>2, TierFrame
  <1>4. TypeInv'
    <2>1. \A y \in tasks' :
            /\ y.units \subseteq AllUnits /\ y.writes \subseteq AllUnits
            /\ y.k \in Keys \cup {NONE} /\ y.h \in HostUnits \cup {NONE}
            /\ y.slot \in Slots \cup {NONE} /\ y.o \in Owners \cup {NONE} /\ y.to \in Groups \cup {NONE}
            /\ y.p \in Positions \cup {NONE} /\ y.src \in {NONE, "remote", "peer"}
            /\ y.kind \in {"transfer", "replicate"} <=> y.k = NONE
            /\ y.kind = "prefetch" => y.to \in Groups
            /\ TaskRecord(y)
      BY DEF IndInv, TypeInv, TaskRecord
    <2>2. remote' \subseteq Keys
      BY DEF IndInv, TypeInv
    <2> QED
      BY <2>1, <2>2, TierTypes
  <1>5. TasksOk'
    <2>1. /\ t.held
          /\ t.kind = "prefetch" =>
               /\ t.h \in HostUnits /\ alloc[t.h] = "task"
               /\ \A y \in tasks : (y.kind = "prefetch" /\ y.h = t.h) => y = t
          /\ t.kind = "put" => t.h \in HostUnits /\ hcache[t.k] = t.h /\ vis[t.k].h = NONE
          /\ t.units = {} /\ t.writes = {} /\ t.slot = NONE
      BY DEF IndInv, TasksOk
    <2>2. \A y \in tasks : (y # t /\ y.kind = "prefetch") => y.h # t.h \/ t.kind # "prefetch"
      BY DEF IndInv, TasksOk
    <2> SUFFICES ASSUME NEW y \in tasks' PROVE TaskOk(y)'
      BY TasksOkEach
    <2>3. CASE y = x
      BY <2>3, <2>1, <2>2 DEF TaskOk
    <2>4. CASE y # x
      <3>1. y \in tasks /\ y # t
        BY <2>4
      <3>2. y.kind = "prefetch" => t.h # y.h \/ t.kind # "prefetch"
        BY <3>1 DEF IndInv, TasksOk
      <3>3. TaskOk(y)
        BY <3>1, TasksOkEach DEF IndInv
      <3> QED
        BY <3>1, <3>2, <3>3, <2>1 DEF TaskOk, Live, TableUnits, Mem
    <2> QED
      BY <2>3, <2>4
  <1>6. TrialVisibility'
    BY DEF IndInv, TrialVisibility, MadeBy
  <1> QED
    BY <1>3, <1>4, <1>5, <1>6 DEF IndInv

LEMMA PeerLostInv ==
  ASSUME IndInv, NEW t \in tasks, PeerLost(t)
  PROVE  IndInv'
  <1> DEFINE x == [t EXCEPT !.ok = FALSE]
  <1>1. /\ t.kind = "prefetch"
        /\ tasks' = (tasks \ {t}) \cup {x}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       trial, member, valid, vis, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF PeerLost, plane, rare
  <1>2. TaskRecord(t)
    BY DEF IndInv, TypeInv, TaskRecord
  <1>3. /\ x.kind = t.kind /\ x.k = t.k /\ x.h = t.h /\ x.src = t.src /\ x.o = t.o /\ x.to = t.to
        /\ x.p = t.p /\ x.slot = t.slot /\ x.held = t.held /\ x.units = t.units
        /\ x.writes = t.writes
        /\ TaskRecord(x)
    BY <1>2 DEF TaskRecord
  <1> QED
    BY <1>1, <1>3, RelabelInv

LEMMA DeadlineInv ==
  ASSUME IndInv, NEW t \in tasks, Deadline(t)
  PROVE  IndInv'
  <1> DEFINE x == [t EXCEPT !.expired = TRUE, !.held = On("R9")]
  <1>1. /\ t.kind \in {"prefetch", "put"}
        /\ tasks' = (tasks \ {t}) \cup {x}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       trial, member, valid, vis, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF Deadline, TierTasks, plane, rare
  <1>2. t.kind \in {"prefetch", "put"} /\ t.held
    BY <1>1 DEF IndInv, TasksOk
  <1>3. TaskRecord(t)
    BY DEF IndInv, TypeInv, TaskRecord
  <1>4. /\ x.kind = t.kind /\ x.k = t.k /\ x.h = t.h /\ x.src = t.src /\ x.o = t.o /\ x.to = t.to
        /\ x.p = t.p /\ x.slot = t.slot /\ x.held = t.held /\ x.units = t.units
        /\ x.writes = t.writes
        /\ TaskRecord(x)
    BY <1>2, <1>3 DEF TaskRecord
  <1> QED
    BY <1>1, <1>2, <1>4, RelabelInv

-----------------------------------------------------------------------------
(* PrefetchEnd                                                             *)

(* The host block of a prefetch is the prefetch's alone.                   *)
LEMMA PrefetchSole ==
  ASSUME IndInv, NEW t \in tasks, t.kind = "prefetch"
  PROVE  /\ t.h \in HostUnits /\ alloc[t.h] = "task" /\ t.held /\ t.h \notin free
         /\ t.units = {} /\ t.slot = NONE
         /\ \A q \in Owners, r \in Positions : table[q][r].u # t.h
         /\ \A q \in Owners : oslot[q] # t.h
         /\ \A j \in Keys : cache[j] # t.h /\ hcache[j] # t.h
         /\ \A d \in deferred : t.h \notin d.units
         /\ \A c \in copies : c.read # t.h /\ c.write # t.h
         /\ \A x \in tasks : x # t => t.h \notin x.units /\ t.h # x.slot /\ t.h # x.h
         /\ \A b \in inflight : \A y \in b.reads : y[2] # t.h
  <1>1. TypeOK
    BY TypeFacts
  <1>2. /\ t.h \in HostUnits /\ alloc[t.h] = "task" /\ t.held
        /\ t.units = {} /\ t.slot = NONE
        /\ \A x \in tasks : (x.kind = "prefetch" /\ x.h = t.h) => x = t
    BY DEF IndInv, TasksOk
  <1>3. t.h \notin Units /\ t.h \notin Slots /\ t.h # NONE /\ t.h \in AllUnits
    BY <1>2 DEF AllUnits
  <1>4. t.h \notin free
    <2>1. Held(t.h)
      BY <1>2 DEF Held, HeldIn
    <2> QED
      BY <2>1 DEF IndInv, FreeClean
  <1>5. \A j \in Keys : hcache[j] # t.h
    <2> SUFFICES ASSUME NEW j \in Keys, hcache[j] = t.h PROVE FALSE
      OBVIOUS
    <2>1. sharedKey[t.h] = j
      BY <1>3 DEF IndInv, CacheShared
    <2> QED
      BY <2>1, <1>2, <1>3 DEF IndInv, SharedIff
  <1>6. \A d \in deferred : t.h \notin d.units
    <2> SUFFICES ASSUME NEW d \in deferred PROVE t.h \notin d.units
      OBVIOUS
    <2>1. /\ d.kind \in {"release", "commit", "finish", "transfer", "replicate"}
          /\ d.kind = "release" => d.units \subseteq Units \cup Slots
          /\ d.kind \in {"finish", "transfer"} => d.units = {}
          /\ d.kind = "commit" => d.units \subseteq Units
      BY DEF IndInv, DeferredOk
    <2>2. d.kind = "replicate" => d.units \subseteq Units
      BY DEF IndInv, CrossingsOk
    <2> QED
      BY <2>1, <2>2, <1>3
  <1>7. \A c \in copies : c.read # t.h /\ c.write # t.h
    <2> SUFFICES ASSUME NEW c \in copies PROVE c.read # t.h /\ c.write # t.h
      OBVIOUS
    <2>1. /\ IF c.kind = "save" THEN c.read \in Slots ELSE alloc[c.read] = "shared"
          /\ ~c.fill =>
               /\ c.kind \in {"clone", "load", "store", "save"}
               /\ c.kind \in {"clone", "load"} => c.write \in Units
               /\ c.kind \in {"store", "save"} => alloc[c.write] = "copy"
          /\ c.fill =>
               \/ c.kind = "load" /\ cache[c.k] = c.write
               \/ c.kind = "restore" /\ c.write \in Slots
               \/ c.kind = "store" /\ hcache[c.k] = c.write
      BY DEF IndInv, CopiesOk
    <2>2. c.k \in Keys
      BY DEF IndInv, TypeInv
    <2>3. cache[c.k] \in Units \cup {NONE}
      BY <1>1, <2>2 DEF TypeOK
    <2> QED
      BY <2>1, <2>2, <2>3, <1>2, <1>3, <1>5
  <1>8. \A x \in tasks : x # t => t.h \notin x.units /\ t.h # x.slot /\ t.h # x.h
    <2> SUFFICES ASSUME NEW x \in tasks, x # t PROVE t.h \notin x.units /\ t.h # x.slot /\ t.h # x.h
      OBVIOUS
    <2>1. /\ x.kind \in {"transfer", "prefetch", "put", "replicate"}
          /\ x.kind = "transfer" => x.units \subseteq Units \cup Slots /\ x.slot \in Slots /\ x.h = NONE
          /\ x.kind \in {"prefetch", "put"} => x.units = {} /\ x.slot = NONE
          /\ x.kind = "put" => hcache[x.k] = x.h
      BY DEF IndInv, TasksOk
    <2>2. x.kind = "put" => x.k \in Keys
      BY DEF IndInv, TypeInv
    <2>3. x.kind = "replicate" => x.units \subseteq Units /\ x.slot = NONE /\ x.h = NONE
      BY DEF IndInv, CrossingsOk
    <2> QED
      BY <2>1, <2>2, <2>3, <1>2, <1>3, <1>5
  <1>9. \A b \in inflight : \A y \in b.reads : y[2] # t.h
    <2> SUFFICES ASSUME NEW b \in inflight, NEW y \in b.reads PROVE y[2] # t.h
      OBVIOUS
    <2>1. y[1] \in Owners /\ alloc[y[2]] \in {y[1], "shared"}
      BY DEF IndInv, BatchesOk
    <2> QED
      BY <2>1, <1>2
  <1>10. /\ \A q \in Owners, r \in Positions : table[q][r].u # t.h
         /\ \A q \in Owners : oslot[q] # t.h
         /\ \A j \in Keys : cache[j] # t.h
    BY <1>1, <1>3 DEF TypeOK
  <1> QED
    BY <1>2, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10

(* The tasks other than a prefetch t keep the properties TasksOk states    *)
(* when t leaves and its host block changes allocation or is cached under  *)
(* t's key, which no remote store reads.                                   *)
LEMMA TasksWithout ==
  ASSUME IndInv, NEW t \in tasks, t.kind = "prefetch",
         tasks' = tasks \ {t},
         UNCHANGED <<deferred, copies, oslot, ostate, table>>,
         \A u \in Units : valid'[u] = valid[u],
         \A v \in AllUnits : v # t.h => alloc'[v] = alloc[v],
         \A j \in Keys : hcache[j] # NONE => hcache'[j] = hcache[j],
         \A j \in Keys : hcache[j] # NONE => vis'[j].h = vis[j].h
  PROVE  TasksOk'
  <1>1. \A x \in tasks : x # t => t.h \notin x.units /\ t.h # x.slot /\ t.h # x.h
    BY PrefetchSole
  <1>2. t.h \in HostUnits
    BY PrefetchSole
  <1> SUFFICES ASSUME NEW y \in tasks' PROVE TaskOk(y)'
    BY TasksOkEach
  <1>3. y \in tasks /\ y # t
    OBVIOUS
  <1>4. y.kind = "transfer" =>
          /\ y.slot \in AllUnits /\ y.writes \subseteq AllUnits /\ y.writes \subseteq Units
          /\ TableUnits(y.o) \subseteq Units
    <2>1. \A q \in Owners, r \in Positions : table[q][r].u \in Units \cup {NONE}
      BY OwnerTypes DEF OwnerTyped, Empty, Private, Shared
    <2>2. y.kind = "transfer" => y.o \in Owners
      BY <1>3 DEF IndInv, TasksOk
    <2> QED
      BY <1>3, <2>1, <2>2 DEF IndInv, TasksOk, AllUnits, TableUnits
  <1>5. y.kind = "prefetch" => y.h \in AllUnits
    BY <1>3 DEF IndInv, TasksOk, AllUnits
  <1>6. y.kind = "put" => y.k \in Keys /\ hcache[y.k] # NONE
    <2>1. y.k \in Keys \cup {NONE} /\ (y.kind \in {"transfer", "replicate"} <=> y.k = NONE)
      BY <1>3, WorkTypes DEF WorkTyped
    <2>2. y.kind = "put" => hcache[y.k] = y.h /\ y.h \in HostUnits
      BY <1>3 DEF IndInv, TasksOk
    <2> QED
      BY <2>1, <2>2
  <1>7. TaskOk(y)
    BY <1>3, TasksOkEach DEF IndInv
  <1> QED
    BY <1>1, <1>3, <1>4, <1>5, <1>6, <1>7 DEF TaskOk, Live, TableUnits, Mem

(* PrefetchEnd(t, found) when it caches the host block, valid on the host  *)
(* parts of the group that asked for it.                                   *)
LEMMA PrefetchCachedInv ==
  ASSUME IndInv, NEW t \in tasks, t.kind = "prefetch", hcache[t.k] = NONE,
         tasks' = tasks \ {t},
         hcache' = [hcache EXCEPT ![t.k] = t.h],
         free' = free,
         alloc' = [alloc EXCEPT ![t.h] = "shared"],
         sharedKey' = [sharedKey EXCEPT ![t.h] = t.k],
         bad' = bad,
         vis' = [vis EXCEPT ![t.k].h = NONE],
         valid' = [valid EXCEPT ![t.h] = GroupParts[t.to]],
         UNCHANGED <<ostate, ogroup, oslot, table, cache, deferred, copies, ncopy, trial, member,
                     dbt, launched, inflight, handoff, requests, remote, lost>>
  PROVE  IndInv'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. /\ t.h \in HostUnits /\ alloc[t.h] = "task" /\ t.held /\ t.h \notin free
        /\ t.units = {} /\ t.slot = NONE
        /\ \A q \in Owners, r \in Positions : table[q][r].u # t.h
        /\ \A q \in Owners : oslot[q] # t.h
        /\ \A j \in Keys : cache[j] # t.h /\ hcache[j] # t.h
        /\ \A d \in deferred : t.h \notin d.units
        /\ \A c \in copies : c.read # t.h /\ c.write # t.h
        /\ \A x \in tasks : x # t => t.h \notin x.units /\ t.h # x.slot /\ t.h # x.h
        /\ \A b \in inflight : \A y \in b.reads : y[2] # t.h
    BY PrefetchSole
  <1>3. t.k \in Keys /\ t.h \in AllUnits /\ t.h \notin Units /\ t.h \notin Slots
    BY <1>2 DEF IndInv, TypeInv, AllUnits
  <1>4. /\ \A v \in AllUnits : v # t.h => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
        /\ alloc'[t.h] = "shared" /\ sharedKey'[t.h] = t.k
    BY <1>1, <1>3 DEF TypeOK
  <1>5. /\ \A j \in Keys : j # t.k => hcache'[j] = hcache[j]
        /\ hcache'[t.k] = t.h
    BY <1>1, <1>3 DEF TypeOK
  <1>6. /\ \A j \in Keys : j # t.k => vis'[j] = vis[j]
        /\ \A j \in Keys : vis'[j].d = vis[j].d
        /\ vis'[t.k].h = NONE
        /\ vis' \in [Keys -> [d : Epochs \cup {NONE}, h : Epochs \cup {NONE}]]
    <2>1. vis \in [Keys -> [d : Epochs \cup {NONE}, h : Epochs \cup {NONE}]]
      BY DEF IndInv, TypeInv
    <2> QED
      BY <2>1, <1>3
  <1>7. Transfers' = Transfers /\ Crossings' = Crossings /\ TaskUnits' \subseteq TaskUnits
    BY DEF Transfers, Crossings, TaskUnits
  (* Validity changes on the prefetch's host block alone, which no copy    *)
  (* touches.                                                              *)
  <1>20. /\ t.to \in Groups /\ GroupParts[t.to] \subseteq HostParts
         /\ \A v \in Units \cup HostUnits : v # t.h => valid'[v] = valid[v]
         /\ \A v \in Units : valid'[v] = valid[v]
         /\ valid'[t.h] = GroupParts[t.to]
         /\ \A c \in copies : /\ c.read \in HostUnits => valid'[c.read] = valid[c.read]
                              /\ c.write \in HostUnits => valid'[c.write] = valid[c.write]
    <2>1. t.to \in Groups
      BY WorkTypes DEF WorkTyped
    <2>2. valid \in [Units \cup HostUnits -> SUBSET (Memories \cup HostParts)]
      BY <1>1 DEF TypeOK
    <2> QED
      BY <1>2, <1>3, <2>1, <2>2
  <1>8. \A v \in AllUnits : Held(v)' <=> Held(v)
    <2>1. \A v \in AllUnits : v # t.h =>
            ((\E j \in Keys : cache'[j] = v \/ hcache'[j] = v) <=> (\E j \in Keys : cache[j] = v \/ hcache[j] = v))
      BY <1>5, <1>3 DEF AllUnits
    <2>2. \A v \in AllUnits : v # t.h =>
            ((\E x \in tasks' : x.held /\ (v \in x.units \/ v = x.slot \/ v = x.h))
               <=> (\E x \in tasks : x.held /\ (v \in x.units \/ v = x.slot \/ v = x.h)))
      BY <1>2 DEF AllUnits
    <2>3. Held(t.h)' /\ Held(t.h)
      BY <1>5, <1>2, <1>3 DEF Held, HeldIn
    <2> QED
      BY <2>1, <2>2, <2>3 DEF Held, HeldIn
  <1>9. TypeInv'
    <2>1. hcache' \in [Keys -> HostUnits \cup {NONE}]
      BY <1>1, <1>2, <1>3 DEF TypeOK
    <2>2. /\ alloc' \in [AllUnits -> Owners \cup {NONE, "shared", "copy", "task"}]
          /\ sharedKey' \in [AllUnits -> Keys \cup {NONE}]
      BY <1>1, <1>3 DEF TypeOK
    <2>3. \A j \in Keys : vis'[j].d \in Epochs \cup {NONE} /\ vis'[j].h \in Epochs \cup {NONE}
      BY <1>6
    <2>5. /\ valid' \in [Units \cup HostUnits -> SUBSET (Memories \cup HostParts)]
          /\ \A u \in Units : valid'[u] \subseteq Memories
          /\ \A h \in HostUnits : valid'[h] \subseteq HostParts
      BY <1>1, <1>2, <1>20 DEF TypeOK
    <2>4. TypeOK'
      BY <1>1, <2>1, <2>2, <2>3, <2>5 DEF TypeOK
    <2> QED
      BY <2>4, <1>6 DEF IndInv, TypeInv, Stamps
  <1>10. FreeClean' /\ NoLeak'
    <2>1. free \subseteq AllUnits
      BY <1>1 DEF TypeOK
    <2>2. \A v \in free : v # t.h
      BY <1>2
    <2> QED
      BY <1>4, <1>8, <2>1, <2>2 DEF IndInv, FreeClean, NoLeak, InRegion
  <1>11. Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
    BY <1>3, <1>4 DEF IndInv, Outside, Regions, OneRegion, AllocOwner, SharedIff, AllUnits
  <1>12. SharedCached'
    <2> SUFFICES ASSUME NEW v \in AllUnits, alloc'[v] = "shared"
                 PROVE  cache'[sharedKey'[v]] = v \/ hcache'[sharedKey'[v]] = v
      BY DEF SharedCached
    <2>1. CASE v = t.h
      BY <2>1, <1>4, <1>5
    <2>2. CASE v # t.h
      <3>1. alloc[v] = "shared" /\ sharedKey'[v] = sharedKey[v]
        BY <2>2, <1>4
      <3>2. cache[sharedKey[v]] = v \/ hcache[sharedKey[v]] = v
        BY <3>1 DEF IndInv, SharedCached
      <3>3. sharedKey[v] \in Keys
        BY <3>1, <1>1 DEF IndInv, SharedIff, TypeOK
      <3>4. hcache[sharedKey[v]] = v => sharedKey[v] # t.k
        BY <3>1 DEF AllUnits
      <3> QED
        BY <3>1, <3>2, <3>3, <3>4, <1>5
    <2> QED
      BY <2>1, <2>2
  <1>13. CacheShared'
    <2> SUFFICES ASSUME NEW j \in Keys
                 PROVE  /\ cache'[j] # NONE => cache'[j] \in Units /\ sharedKey'[cache'[j]] = j
                        /\ hcache'[j] # NONE => hcache'[j] \in HostUnits /\ sharedKey'[hcache'[j]] = j
      BY DEF CacheShared
    <2>1. /\ cache[j] # NONE => cache[j] \in Units /\ sharedKey[cache[j]] = j
          /\ hcache[j] # NONE => hcache[j] \in HostUnits /\ sharedKey[hcache[j]] = j
      BY DEF IndInv, CacheShared
    <2>2. CASE j = t.k
      BY <2>1, <2>2, <1>2, <1>3, <1>4, <1>5 DEF AllUnits
    <2>3. CASE j # t.k
      BY <2>1, <2>3, <1>2, <1>3, <1>4, <1>5 DEF AllUnits
    <2> QED
      BY <2>2, <2>3
  <1>14. Tables' /\ OwnersOk' /\ DeferredOk'
    <2>1. \A v \in Units \cup Slots : alloc'[v] = alloc[v]
      BY <1>3, <1>4 DEF AllUnits
    <2>2. \A q \in Owners, p \in Positions : table[q][p].u \in Units \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>3. \A q \in Owners : oslot[q] \in Slots \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>4. \A d \in deferred : d.kind = "commit" => d.units \subseteq Units
      BY DEF IndInv, DeferredOk
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <1>20 DEF IndInv, Tables, OwnersOk, DeferredOk, Live, Mem
  <1>15. CopiesOk'
    <2>1. \A c \in copies : alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
      <3>1. \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
        BY DEF IndInv, TypeInv
      <3> QED
        BY <3>1, <1>2, <1>4
    <2>2. \A c \in copies : ~c.fill => \A q \in Owners, r \in Positions : table'[q][r].u # c.write
      BY DEF IndInv, CopiesOk
    <2>3. /\ \A c \in copies : c.kind \in {"clone", "load"} => table'[c.o][c.p] = table[c.o][c.p]
          /\ \A c \in copies : c.fill /\ c.kind = "load" =>
               cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write]
      <3>1. \A c \in copies : c.fill /\ c.kind = "load" => c.write \in Units
        <4> SUFFICES ASSUME NEW c \in copies, c.fill, c.kind = "load" PROVE c.write \in Units
          OBVIOUS
        <4>1. cache[c.k] = c.write /\ c.write \in AllUnits /\ c.k \in Keys
          BY DEF IndInv, CopiesOk, TypeInv
        <4> QED
          BY <4>1, <1>1 DEF TypeOK, AllUnits
      <3> QED
        BY <3>1, <1>20
    (* A store fills the host block of a key that has one, not t.k.        *)
    <2>4. \A c \in copies : c.fill /\ c.kind = "store" => hcache'[c.k] = hcache[c.k]
      <3> SUFFICES ASSUME NEW c \in copies, c.fill, c.kind = "store"
                   PROVE  hcache'[c.k] = hcache[c.k]
        OBVIOUS
      <3>1. hcache[c.k] = c.write /\ c.write \in HostUnits /\ c.k \in Keys
        BY DEF IndInv, CopiesOk, TypeInv
      <3>2. c.k # t.k
        BY <3>1 DEF NONE
      <3> QED
        BY <3>1, <3>2, <1>5
    <2> QED
      BY <1>7, <2>1, <2>2, <2>3, <2>4, CopiesStay DEF IndInv
  <1>16. TasksOk'
    <2>1. \A v \in AllUnits : v # t.h => alloc'[v] = alloc[v]
      BY <1>4
    <2>2. \A j \in Keys : hcache[j] # NONE => hcache'[j] = hcache[j] /\ vis'[j].h = vis[j].h
      BY <1>5, <1>6
    <2> QED
      BY <2>1, <2>2, <1>20, TasksWithout
  <1>17. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
    BY <1>7, <1>20, MovingUnmappedFrame, LostOkFrame, HostValidMapped, RequestsOkFrame, NoBadFrame
       DEF IndInv
  <1>18. BatchesOk'
    <2>1. \A b, o, u : Covered(b, o, u)' = Covered(b, o, u)
      BY <1>7, CoveredFrame
    <2>2. \A b \in inflight : \A y \in b.reads : y[2] \in AllUnits /\ y[2] # t.h
      BY <1>2, RequestTypes DEF RequestTyped
    <2>3. \A b \in inflight : b.writes \subseteq b.reads
      BY DEF IndInv, BatchesOk
    <2> QED
      BY <2>1, <2>2, <2>3, <1>4, <1>20 DEF IndInv, BatchesOk, Mem
  <1>19. TrialVisibility'
    <2>1. \A j \in Keys : hcache'[j] # NONE /\ vis'[j].h # NONE => hcache'[j] = hcache[j] /\ vis'[j].h = vis[j].h
      BY <1>5, <1>6
    <2>2. \A j \in Keys : hcache[j] # t.h
      BY <1>2
    <2>3. \A c \in copies : (c.kind = "store" /\ c.fill) => c.k # t.k
      <3> SUFFICES ASSUME NEW c \in copies, c.kind = "store", c.fill PROVE c.k # t.k
        OBVIOUS
      <3>1. hcache[c.k] = c.write /\ c.write \in HostUnits
        BY DEF IndInv, CopiesOk
      <3> QED
        BY <3>1 DEF NONE
    <2>4. \A j \in Keys : j # t.k => vis'[j].h = vis[j].h
      BY <1>6
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <1>6 DEF IndInv, TrialVisibility, MadeBy
  <1>21. CrossingsOk' /\ CrossingsPlaced' /\ CopiesPlaced'
    <2>1. \A u \in Units : alloc'[u] = alloc[u] /\ valid'[u] = valid[u]
      BY <1>3, <1>4, <1>20 DEF AllUnits
    <2>2. \A j \in Keys : vis'[j].d = vis[j].d
      BY <1>6
    <2>3. \A c \in copies : /\ c.read \in HostUnits => valid'[c.read] = valid[c.read]
                            /\ c.write \in HostUnits => valid'[c.write] = valid[c.write]
      BY <1>20
    <2>4. CrossingsOk /\ CrossingsPlaced /\ CopiesPlaced
      BY DEF IndInv
    <2>5. /\ Transfers' = Transfers /\ Crossings' = Crossings
          /\ UNCHANGED <<deferred, copies, ostate, ogroup, lost, cache, inflight, trial>>
      BY <1>7
    <2> QED
      BY ONLY <2>1, <2>2, <2>3, <2>4, <2>5, TierCrossings
  <1> QED
    BY <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18, <1>19, <1>21
       DEF IndInv

(* PrefetchEnd(t, found) when it lets go of the host block.                *)
LEMMA PrefetchDroppedInv ==
  ASSUME IndInv, NEW t \in tasks, t.kind = "prefetch",
         tasks' = tasks \ {t},
         free' = free \cup {t.h},
         alloc' = [u \in AllUnits |-> IF u = t.h THEN NONE ELSE alloc[u]],
         sharedKey' = [u \in AllUnits |-> IF u = t.h THEN NONE ELSE sharedKey[u]],
         UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, trial,
                     member, valid, vis, dbt, launched, inflight, handoff, requests, remote, lost,
                     bad>>
  PROVE  IndInv'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. /\ t.h \in HostUnits /\ alloc[t.h] = "task" /\ t.held /\ t.h \notin free
        /\ t.units = {} /\ t.slot = NONE
        /\ \A q \in Owners, r \in Positions : table[q][r].u # t.h
        /\ \A q \in Owners : oslot[q] # t.h
        /\ \A j \in Keys : cache[j] # t.h /\ hcache[j] # t.h
        /\ \A d \in deferred : t.h \notin d.units
        /\ \A c \in copies : c.read # t.h /\ c.write # t.h
        /\ \A x \in tasks : x # t => t.h \notin x.units /\ t.h # x.slot /\ t.h # x.h
        /\ \A b \in inflight : \A y \in b.reads : y[2] # t.h
    BY PrefetchSole
  <1>3. t.h \in AllUnits /\ t.h \notin Units /\ t.h \notin Slots
    BY <1>2 DEF AllUnits
  <1>4. /\ \A v \in AllUnits : v # t.h => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
        /\ alloc'[t.h] = NONE /\ sharedKey'[t.h] = NONE
    BY <1>3
  <1>5. Transfers' = Transfers /\ Crossings' = Crossings /\ TaskUnits' \subseteq TaskUnits
    BY DEF Transfers, Crossings, TaskUnits
  <1>6. \A v \in AllUnits : Held(v)' <=> Held(v) /\ v # t.h
    BY <1>2 DEF Held, HeldIn, AllUnits
  <1>7. TypeInv'
    <2>1. TypeOK'
      BY <1>1, <1>3 DEF TypeOK
    <2> QED
      BY <2>1 DEF IndInv, TypeInv, Stamps
  <1>8. FreeClean' /\ NoLeak'
    <2>1. free \subseteq AllUnits
      BY <1>1 DEF TypeOK
    <2>2. FreeClean'
      BY <1>3, <1>4, <1>6, <2>1 DEF IndInv, FreeClean
    <2>3. NoLeak'
      BY <1>3, <1>6 DEF IndInv, NoLeak, InRegion
    <2> QED
      BY <2>2, <2>3
  <1>9. Outside' /\ Regions' /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. \A v \in Units : alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
      BY <1>3, <1>4 DEF AllUnits
    <2>4. Outside'
      BY <2>1 DEF IndInv, Outside
    <2>5. Regions'
      BY <1>3 DEF IndInv, Regions, OneRegion
    <2>6. AllocOwner' /\ SharedIff'
      BY <1>4 DEF IndInv, AllocOwner, SharedIff, NONE
    <2>2. SharedCached'
      <3> SUFFICES ASSUME NEW v \in AllUnits, alloc'[v] = "shared"
                   PROVE  cache'[sharedKey'[v]] = v \/ hcache'[sharedKey'[v]] = v
        BY DEF SharedCached
      <3>1. v # t.h /\ alloc[v] = "shared" /\ sharedKey'[v] = sharedKey[v]
        BY <1>4 DEF NONE
      <3> QED
        BY <3>1 DEF IndInv, SharedCached
    <2>3. CacheShared'
      BY <1>2, <1>4 DEF IndInv, CacheShared, AllUnits
    <2> QED
      BY <2>2, <2>3, <2>4, <2>5, <2>6
  <1>10. Tables' /\ OwnersOk' /\ DeferredOk'
    <2>1. \A v \in Units \cup Slots : alloc'[v] = alloc[v]
      BY <1>3, <1>4 DEF AllUnits
    <2>2. \A q \in Owners, p \in Positions : table[q][p].u \in Units \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>3. \A q \in Owners : oslot[q] \in Slots \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>4. \A d \in deferred : d.kind = "commit" => d.units \subseteq Units
      BY DEF IndInv, DeferredOk
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4 DEF IndInv, Tables, OwnersOk, DeferredOk, Live, Mem
  <1>11. CopiesOk'
    <2>1. \A c \in copies : alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
      <3>1. \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
        BY DEF IndInv, TypeInv
      <3> QED
        BY <3>1, <1>2, <1>4
    <2>2. \A c \in copies : ~c.fill => \A q \in Owners, r \in Positions : table'[q][r].u # c.write
      BY DEF IndInv, CopiesOk
    <2>3. /\ \A c \in copies : c.kind \in {"clone", "load"} => table'[c.o][c.p] = table[c.o][c.p]
          /\ \A c \in copies : c.fill /\ c.kind = "load" =>
               cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write]
      OBVIOUS
    <2> QED
      BY <1>5, <2>1, <2>2, <2>3, CopiesStay DEF IndInv
  <1>12. TasksOk'
    BY <1>4, TasksWithout
  <1>13. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
    BY <1>5, MovingUnmappedFrame, LostOkFrame, TablesMappedFrame, HandoffOkFrame, RequestsOkFrame,
       NoBadFrame DEF IndInv
  <1>14. BatchesOk'
    <2>1. \A b, o, u : Covered(b, o, u)' = Covered(b, o, u)
      BY <1>5, CoveredFrame
    <2>2. \A b \in inflight : \A y \in b.reads : y[2] \in AllUnits /\ y[2] # t.h
      BY <1>2, RequestTypes DEF RequestTyped
    <2>3. \A b \in inflight : b.writes \subseteq b.reads
      BY DEF IndInv, BatchesOk
    <2> QED
      BY <2>1, <2>2, <2>3, <1>4 DEF IndInv, BatchesOk, Mem
  <1>15. TrialVisibility'
    BY DEF IndInv, TrialVisibility, MadeBy
  <1>16. CrossingsOk' /\ CrossingsPlaced' /\ CopiesPlaced'
    <2>1. \A u \in Units : alloc'[u] = alloc[u]
      BY <1>3, <1>4 DEF AllUnits
    <2> QED
      BY <1>5, <2>1, TierCrossings DEF IndInv
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16 DEF IndInv

LEMMA PrefetchEndInv ==
  ASSUME IndInv, NEW t \in tasks, NEW found \in BOOLEAN, PrefetchEnd(t, found)
  PROVE  IndInv'
  <1>1. /\ t.kind = "prefetch" /\ tasks' = tasks \ {t}
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, deferred, copies, ncopy, trial, member,
                       dbt, launched, inflight, handoff, requests, remote, lost>>
    BY DEF PrefetchEnd, plane
  <1>2. t.held /\ t.h \in HostUnits
    BY <1>1, PrefetchSole
  <1>3. CASE found /\ ~t.expired /\ t.held /\ hcache[t.k] = NONE /\ t.ok
    <2>1. /\ hcache' = [hcache EXCEPT ![t.k] = t.h]
          /\ free' = free
          /\ alloc' = [alloc EXCEPT ![t.h] = "shared"]
          /\ sharedKey' = [sharedKey EXCEPT ![t.h] = t.k]
          /\ bad' = bad
          /\ vis' = [vis EXCEPT ![t.k].h = NONE]
          /\ valid' = [valid EXCEPT ![t.h] = GroupParts[t.to]]
      BY <1>3 DEF PrefetchEnd
    <2> QED
      BY <1>1, <1>3, <2>1, PrefetchCachedInv
  <1>4. CASE ~(found /\ ~t.expired /\ t.held /\ hcache[t.k] = NONE /\ t.ok)
    <2>1. /\ DropWith({t.h}, table, oslot, cache, hcache, deferred, copies, tasks')
          /\ UNCHANGED <<hcache, bad, vis, valid>>
      BY <1>2, <1>4 DEF PrefetchEnd
    <2>2. ~HeldIn(t.h, table, oslot, cache, hcache, deferred, copies, tasks')
      BY <1>1, PrefetchSole DEF HeldIn
    <2>3. ReclaimedWith({t.h}, table, oslot, cache, hcache, deferred, copies, tasks') = {t.h}
      BY <2>2, ReclaimedHeld
    <2>4. /\ free' = free \cup {t.h}
          /\ alloc' = [u \in AllUnits |-> IF u = t.h THEN NONE ELSE alloc[u]]
          /\ sharedKey' = [u \in AllUnits |-> IF u = t.h THEN NONE ELSE sharedKey[u]]
      BY <2>1, <2>3 DEF DropWith, Reset
    <2> QED
      BY <1>1, <2>1, <2>4, PrefetchDroppedInv
  <1> QED
    BY <1>3, <1>4

-----------------------------------------------------------------------------
(* PutEnd                                                                  *)

LEMMA PutEndInv ==
  ASSUME IndInv, NEW t \in tasks, NEW stored \in BOOLEAN, PutEnd(t, stored)
  PROVE  IndInv'
  <1>1. /\ t.kind = "put" /\ tasks' = tasks \ {t}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       trial, member, valid, vis, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, lost>>
    BY DEF PutEnd, plane
  <1>2. /\ t.held /\ t.h \in HostUnits /\ hcache[t.k] = t.h /\ t.units = {} /\ t.slot = NONE
        /\ t.k \in Keys
    BY <1>1 DEF IndInv, TasksOk, TypeInv
  <1>3. /\ remote' \subseteq Keys
        /\ bad' = bad
    <2>1. CASE stored /\ ~t.expired
      BY <2>1, <1>2 DEF PutEnd, IndInv, TypeInv
    <2>2. CASE ~(stored /\ ~t.expired)
      BY <2>2 DEF PutEnd, IndInv, TypeInv
    <2> QED
      BY <2>1, <2>2
  <1>4. Transfers' = Transfers /\ Crossings' = Crossings /\ TaskUnits' \subseteq TaskUnits
    BY <1>1 DEF Transfers, Crossings, TaskUnits
  <1>5. \A v \in AllUnits : Held(v)' <=> Held(v)
    BY <1>1, <1>2 DEF Held, HeldIn, AllUnits
  <1>6. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
        /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
        /\ BatchesOk' /\ NoBad' /\ CrossingsOk' /\ CrossingsPlaced' /\ CopiesPlaced'
    BY <1>1, <1>3, <1>4, <1>5, TierFrame
  <1>7. TypeInv'
    <2>1. \A y \in tasks' :
            /\ y.units \subseteq AllUnits /\ y.writes \subseteq AllUnits
            /\ y.k \in Keys \cup {NONE} /\ y.h \in HostUnits \cup {NONE}
            /\ y.slot \in Slots \cup {NONE} /\ y.o \in Owners \cup {NONE} /\ y.to \in Groups \cup {NONE}
            /\ y.p \in Positions \cup {NONE} /\ y.src \in {NONE, "remote", "peer"}
            /\ y.kind \in {"transfer", "replicate"} <=> y.k = NONE
            /\ y.kind = "prefetch" => y.to \in Groups
            /\ TaskRecord(y)
      BY <1>1 DEF IndInv, TypeInv, TaskRecord
    <2> QED
      BY <1>1, <1>3, <2>1, TierTypes
  <1>8. TasksOk'
    BY <1>1 DEF IndInv, TasksOk, Live, TableUnits, Mem
  <1>9. TrialVisibility'
    BY <1>1 DEF IndInv, TrialVisibility, MadeBy
  <1> QED
    BY <1>6, <1>7, <1>8, <1>9 DEF IndInv

-----------------------------------------------------------------------------
(* The group                                                               *)

Next9 ==
  \/ \E k \in Keys, g \in Groups : Prefetch(k, "remote", g) \/ Prefetch(k, "peer", g)
  \/ \E k \in Keys : StoreRemote(k)
  \/ \E t \in tasks : PeerLost(t) \/ Deadline(t)
  \/ \E t \in tasks, ok \in BOOLEAN : PrefetchEnd(t, ok) \/ PutEnd(t, ok)

THEOREM Step9 == IndInv /\ Next9 => IndInv'
  <1> SUFFICES ASSUME IndInv, Next9 PROVE IndInv'
    OBVIOUS
  <1>1. CASE \E k \in Keys, g \in Groups : Prefetch(k, "remote", g) \/ Prefetch(k, "peer", g)
    <2>1. PICK k \in Keys, g \in Groups, src \in {"remote", "peer"} : Prefetch(k, src, g)
      BY <1>1
    <2>2. PICK h \in HostUnits : PrefetchAt(k, src, g, h)
      BY <2>1, PrefetchPick
    <2> QED
      BY <2>2, PrefetchInv
  <1>2. CASE \E k \in Keys : StoreRemote(k)
    BY <1>2, StoreRemotePick, StoreRemoteInv
  <1>3. CASE \E t \in tasks : PeerLost(t) \/ Deadline(t)
    BY <1>3, PeerLostInv, DeadlineInv
  <1>4. CASE \E t \in tasks, ok \in BOOLEAN : PrefetchEnd(t, ok) \/ PutEnd(t, ok)
    BY <1>4, PrefetchEndInv, PutEndInv
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4 DEF Next9
=============================================================================
