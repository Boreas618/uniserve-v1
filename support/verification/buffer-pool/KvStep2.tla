------------------------------- MODULE KvStep2 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for the handlers that ask for a copy:      *)
(* Clone, Load, Save, Restore and Store. Each adds one copy with a fresh   *)
(* identifier, in one of four forms: a new private device block that the   *)
(* copy writes (Clone, and Load of a key with no device block for the      *)
(* owner), a new host block (Store, Save), no new block (Load that fills a *)
(* group's memory of a cached block, Restore), or the host parts a key's   *)
(* host block lacks (Store of a key whose host block does not hold every  *)
(* host part of the group).                                                *)
(***************************************************************************)
EXTENDS KvFrame

USE Sizes, Design DEF On, AllRules

-----------------------------------------------------------------------------
(* What adding copy c0 with a fresh identifier keeps.                      *)

(* Copy c is a record of the fields of Copy alone.                         *)
CopyRecord(c) ==
  c = [id |-> c.id, kind |-> c.kind, o |-> c.o, p |-> c.p, k |-> c.k, g |-> c.g,
       read |-> c.read, write |-> c.write, fill |-> c.fill, parts |-> c.parts, seq |-> c.seq,
       done |-> c.done, stopped |-> c.stopped]

LEMMA CopiesAdded ==
  ASSUME CopiesOk, NEW c0,
         copies' = copies \cup {c0}, ncopy \in Nat, ncopy' = ncopy + 1, c0.id = ncopy + 1,
         CopyOk(c0)',
         UNCHANGED <<lost, launched, ostate, ogroup, deferred, table, tasks>>,
         Transfers' = Transfers,
         \A c \in copies : alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write],
         \A c \in copies : c.fill /\ c.kind = "load" =>
                             /\ cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write]
                             /\ ~(c0.write = c.write /\ Mem(c0.g) = Mem(c.g)),
         \A c \in copies : ~c.fill => c0.write # c.write,
         \A c \in copies : c.fill /\ c.kind = "store" =>
                             hcache'[c.k] = hcache[c.k] /\ c0.write # c.write,
         \A c \in copies : c.kind \in {"clone", "load"} => ~(c0.o = c.o /\ c0.p = c.p)
  PROVE  CopiesOk'
  <1>1. \A q : Moving(q)' = Moving(q)
    BY DEF Moving, Transfers
  <1> SUFFICES ASSUME NEW c \in copies' PROVE CopyOk(c)'
    BY CopiesOkEach
  <1>2. CASE c = c0
    BY <1>2
  <1>3. CASE c # c0
    <2>1. c \in copies
      BY <1>3
    <2>2. CopyOk(c)
      BY <2>1, CopiesOkEach
    <2>3. c.id # c0.id
      BY <2>2 DEF CopyOk
    <2>4. \A x : (\A t \in tasks' : (t.kind = "transfer" /\ Mem(t.to) = Mem(c.g)) => x \notin t.writes)
                   <=> (\A t \in tasks : (t.kind = "transfer" /\ Mem(t.to) = Mem(c.g)) => x \notin t.writes)
      BY DEF Transfers
    <2> QED
      BY <1>1, <2>1, <2>2, <2>3, <2>4 DEF CopyOk, Live
  <1> QED
    BY <1>2, <1>3

LEMMA RequestsAdded ==
  ASSUME RequestsOk, NEW c0, copies' = copies \cup {c0}, ncopy \in Nat, ncopy' = ncopy + 1,
         c0.id = ncopy + 1, UNCHANGED <<requests, launched, dbt>>
  PROVE  RequestsOk'
  <1>1. \A r \in requests : r.kind = "finish" => r.c \in 1 .. ncopy /\ r.c # c0.id
    BY DEF RequestsOk
  <1> QED
    BY <1>1 DEF RequestsOk

LEMMA DeferredAdded ==
  ASSUME DeferredOk, NEW c0, copies' = copies \cup {c0},
         UNCHANGED <<deferred, oslot, table, lost, ostate>>,
         \A d \in deferred : d.kind = "commit" => \A u \in d.units : alloc'[u] = alloc[u],
         \A d \in deferred : (d.kind = "release" /\ ~c0.fill) => c0.write \notin d.units,
         \A d \in deferred : d.kind = "commit" => c0.write \notin d.units
  PROVE  DeferredOk'
  BY DEF DeferredOk

LEMMA TasksAdded ==
  ASSUME TasksOk, NEW c0, copies' = copies \cup {c0},
         UNCHANGED <<tasks, deferred, oslot, ostate, hcache, vis, valid, table>>,
         \A t \in tasks : t.kind = "transfer" =>
             /\ alloc'[t.slot] = alloc[t.slot]
             /\ \A u \in t.writes : alloc'[u] = alloc[u]
             /\ c0.read # t.slot /\ c0.write # t.slot
             /\ ostate[t.o] = "live" => c0.o # t.o,
         \A t \in tasks : t.kind = "prefetch" => alloc'[t.h] = alloc[t.h]
  PROVE  TasksOk'
  <1> SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
    BY TasksOkEach
  <1>1. TaskOk(t)
    BY TasksOkEach
  <1> QED
    BY <1>1 DEF TaskOk, Live, TableUnits

(* P8 after a copy c0 that reads where its block is valid is added.        *)
LEMMA PlacedAdded ==
  ASSUME CopiesPlaced, CrossingsPlaced, NEW c0, copies' = copies \cup {c0}, CopyPlaced(c0),
         UNCHANGED <<valid, ogroup, inflight>>, Crossings' = Crossings
  PROVE  CopiesPlaced' /\ CrossingsPlaced'
  <1>1. CopiesPlaced'
    <2> SUFFICES ASSUME NEW c \in copies' PROVE CopyPlaced(c)'
      BY PlacedEach
    <2>1. CopyPlaced(c)
      BY PlacedEach
    <2> QED
      BY <2>1 DEF CopyPlaced, Mem
  <1>2. CrossingsPlaced'
    BY CrossingsPlacedFrame
  <1> QED
    BY <1>1, <1>2

(* CrossingsOk after copy c0 is added, when no unit a crossing holds       *)
(* changes its allocation or the device block cached under a key, c0 does *)
(* not write such a unit as a new block, and c0 fills no memory a          *)
(* crossing's task fills.                                                  *)
LEMMA CrossingsAdded ==
  ASSUME CrossingsOk, NEW c0, copies' = copies \cup {c0},
         UNCHANGED <<deferred, tasks, ostate, ogroup, lost, valid, vis, trial>>,
         \A d \in deferred : d.kind = "replicate" =>
            /\ \A u \in d.units : alloc'[u] = alloc[u]
            /\ ~c0.fill => c0.write \notin d.units
            /\ \A j \in Keys : cache'[j] \in d.units => cache'[j] = cache[j],
         \A t \in tasks : t.kind = "replicate" =>
            /\ \A u \in t.writes : alloc'[u] = alloc[u]
            /\ (c0.fill /\ c0.kind = "load" /\ Mem(c0.g) = Mem(t.to)) => c0.write \notin t.writes
  PROVE  CrossingsOk'
  <1>1. \A q \in Owners : Moving(q)' = Moving(q)
    BY DEF Moving
  <1>2. \A d \in deferred : CrossWaitOk(d)'
    <2> SUFFICES ASSUME NEW d \in deferred PROVE CrossWaitOk(d)'
      OBVIOUS
    <2>1. CrossWaitOk(d)
      BY CrossingsOkEach
    <2> QED
      BY <1>1, <2>1 DEF CrossWaitOk, ReadersOf, MadeBy
  <1>3. \A t \in tasks : CrossTaskOk(t)'
    <2> SUFFICES ASSUME NEW t \in tasks PROVE CrossTaskOk(t)'
      OBVIOUS
    <2>1. CrossTaskOk(t)
      BY CrossingsOkEach
    <2> QED
      BY <1>1, <2>1 DEF CrossTaskOk, ReadersOf, Mem
  <1> QED
    BY <1>2, <1>3, CrossingsOkEach

LEMMA BatchesAdded ==
  ASSUME BatchesOk, copies \subseteq copies',
         UNCHANGED <<inflight, launched, lost, valid, ostate, ogroup, oslot, table, requests,
                     deferred>>,
         Transfers' = Transfers,
         free' \subseteq free,
         \A b \in inflight : \A x \in b.reads : alloc'[x[2]] = alloc[x[2]]
  PROVE  BatchesOk'
  <1>1. \A b, o, u : Covered(b, o, u)' = Covered(b, o, u)
    BY CoveredFrame
  <1>2. \A b \in inflight : b.writes \subseteq b.reads
    BY DEF BatchesOk
  <1> QED
    BY <1>1, <1>2 DEF BatchesOk, Mem

-----------------------------------------------------------------------------
(* A new private device block w for position p of owner o, which copy c0   *)
(* fills from a shared block: Clone, and Load of a key whose device block, *)
(* if any, is valid in the owner's memory but not visible to it.           *)

NewBlockAt(o, p, w, c0) ==
  /\ CopyRecord(c0)
  /\ ostate[o] = "live" /\ table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
  /\ ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
  /\ ncopy < MaxCopies
  /\ c0.kind \in {"clone", "load"} /\ c0.id = ncopy + 1 /\ c0.o = o /\ c0.p = p /\ c0.k \in Keys
  /\ c0.g = ogroup[o] /\ c0.write = w /\ c0.seq = 0 /\ ~c0.done /\ ~c0.fill /\ c0.parts = {}
  /\ c0.read \in Units \cup HostUnits /\ c0.read # w /\ alloc[c0.read] = "shared"
  /\ CopyPlaced(c0)
  /\ \E j \in Keys : cache[j] = c0.read \/ hcache[j] = c0.read
  /\ \A j \in Keys :
       /\ (cache[j] = c0.read /\ vis[j].d # NONE) => vis[j].d = OwnerEpoch[o]
       /\ (hcache[j] = c0.read /\ vis[j].h # NONE) => vis[j].h = OwnerEpoch[o]
  /\ w \in member /\ Unheld(w)
  /\ \A b \in inflight : \A x \in b.reads : x[2] # w
  /\ free' = free \ {w}
  /\ alloc' = [alloc EXCEPT ![w] = o]
  /\ sharedKey' = [sharedKey EXCEPT ![w] = NONE]
  /\ cache' = [j \in Keys |-> IF cache[j] = w THEN NONE ELSE cache[j]]
  /\ copies' = copies \cup {c0}
  /\ ncopy' = ncopy + 1
  /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, tasks, trial, member, valid, vis,
                 dbt, launched, inflight, handoff, requests, remote, lost, bad>>

LEMMA NewBlockInv ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW w \in Units, NEW c0,
         NewBlockAt(o, p, w, c0)
  PROVE  IndInv'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. /\ \A v \in AllUnits : v # w => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
        /\ alloc'[w] = o /\ sharedKey'[w] = NONE
        /\ \A j \in Keys : cache'[j] = IF cache[j] = w THEN NONE ELSE cache[j]
    BY <1>1 DEF NewBlockAt, TypeOK, AllUnits
  <1>3. /\ w \in AllUnits /\ w # NONE /\ w \notin Slots /\ w \notin HostUnits
        /\ c0.read \in AllUnits /\ c0.read \notin Slots
    BY DEF NewBlockAt, AllUnits
  <1>4. ncopy \in Nat /\ ncopy' \in 0 .. MaxCopies
    BY <1>1 DEF NewBlockAt, TypeOK
  <1>5. Transfers' = Transfers
    BY DEF NewBlockAt, Transfers
  <1>6. \A v \in AllUnits : Held(v)' <=> Held(v) \/ v = w
    <2>1. Held(c0.read)
      BY DEF NewBlockAt, Held, HeldIn
    <2>2. \A v \in AllUnits : v # w =>
            ((\E j \in Keys : cache'[j] = v \/ hcache'[j] = v)
               <=> (\E j \in Keys : cache[j] = v \/ hcache[j] = v))
      BY <1>2 DEF NewBlockAt, AllUnits
    <2> QED
      BY <2>1, <2>2 DEF NewBlockAt, Held, HeldIn
  <1>7. TypeInv'
    <2>1. TypeOK'
      BY <1>1, <1>2, <1>3, <1>4 DEF NewBlockAt, TypeOK, AllUnits
    <2>2. \A c \in copies' :
            /\ c.seq \in 0 .. MaxBatch /\ c.k \in Keys
            /\ c.o \in Owners \cup {NONE} /\ c.p \in Positions \cup {NONE}
            /\ c.read \in AllUnits /\ c.write \in AllUnits /\ c.parts \subseteq HostParts
      BY <1>3, WorkTypes DEF NewBlockAt, WorkTyped, AllUnits
    <2> QED
      BY <2>1, <2>2 DEF NewBlockAt, IndInv, TypeInv, Stamps, CopyRecord
  <1>8. FreeClean'
    <2> SUFFICES ASSUME NEW v \in free' PROVE alloc'[v] = NONE /\ sharedKey'[v] = NONE /\ ~Held(v)'
      BY DEF FreeClean
    <2>1. v \in free /\ v # w /\ v \in AllUnits
      BY <1>1 DEF NewBlockAt, TypeOK
    <2>2. alloc[v] = NONE /\ sharedKey[v] = NONE /\ ~Held(v)
      BY <2>1 DEF IndInv, FreeClean
    <2> QED
      BY <1>2, <1>6, <2>1, <2>2
  <1>9. NoLeak'
    <2> SUFFICES ASSUME NEW v \in AllUnits
                 PROVE  IF InRegion(v)' THEN v \in free' \/ Held(v)' ELSE v \notin free' /\ ~Held(v)'
      BY DEF NoLeak
    <2>1. CASE v = w
      BY <2>1, <1>6 DEF InRegion, NewBlockAt
    <2>2. CASE v # w
      BY <2>2, <1>6 DEF InRegion, NewBlockAt, IndInv, NoLeak
    <2> QED
      BY <2>1, <2>2
  <1>10. Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
    BY <1>2 DEF NewBlockAt, IndInv, Outside, Regions, OneRegion, AllocOwner, SharedIff, AllUnits
  <1>11. SharedCached'
    <2> SUFFICES ASSUME NEW v \in AllUnits, alloc'[v] = "shared"
                 PROVE  cache'[sharedKey'[v]] = v \/ hcache'[sharedKey'[v]] = v
      BY DEF SharedCached
    <2>1. v # w /\ alloc[v] = "shared" /\ sharedKey'[v] = sharedKey[v]
      BY <1>2 DEF NewBlockAt
    <2>2. cache[sharedKey[v]] = v \/ hcache[sharedKey[v]] = v
      BY <2>1 DEF IndInv, SharedCached
    <2>3. sharedKey[v] \in Keys
      BY <2>1, <1>1 DEF IndInv, SharedIff, TypeOK
    <2> QED
      BY <2>1, <2>2, <2>3, <1>2 DEF NewBlockAt
  <1>12. CacheShared'
    <2> SUFFICES ASSUME NEW j \in Keys
                 PROVE  /\ cache'[j] # NONE => cache'[j] \in Units /\ sharedKey'[cache'[j]] = j
                        /\ hcache'[j] # NONE => hcache'[j] \in HostUnits /\ sharedKey'[hcache'[j]] = j
      BY DEF CacheShared
    <2>1. hcache[j] # w
      BY DEF NewBlockAt, Unheld
    <2>2. /\ cache[j] # NONE => cache[j] \in Units /\ sharedKey[cache[j]] = j
          /\ hcache[j] # NONE => hcache[j] \in HostUnits /\ sharedKey[hcache[j]] = j
      BY DEF IndInv, CacheShared
    <2> QED
      BY <1>2, <2>1, <2>2 DEF NewBlockAt, AllUnits
  <1>13. Tables' /\ OwnersOk'
    <2>1. \A q \in Owners, r \in Positions : table[q][r].u # w
      BY DEF NewBlockAt, Unheld
    <2>2. \A q \in Owners, r \in Positions : table[q][r].u \in Units \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>3. \A q \in Owners : oslot[q] \in Slots \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>4. \A v \in Units \cup Slots : v # w => alloc'[v] = alloc[v]
      BY <1>2 DEF AllUnits
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4 DEF NewBlockAt, IndInv, Tables, OwnersOk, Live, Mem
  <1>14. DeferredOk'
    <2>1. \A d \in deferred : w \notin d.units
      BY DEF NewBlockAt, Unheld
    <2>2. \A d \in deferred : d.kind = "commit" => d.units \subseteq AllUnits
      BY WorkTypes DEF WorkTyped
    <2> QED
      BY <1>2, <2>1, <2>2, DeferredAdded DEF NewBlockAt, IndInv
  <1>15. CopiesOk'
    <2>1. CopyOk(c0)'
      <3>1. \A x \in copies : x.id \in 1 .. ncopy /\ x.write # w /\ ~(x.o = o /\ x.p = p /\ x.kind \in {"clone", "load"})
        BY DEF IndInv, CopiesOk, NewBlockAt, Unheld, CopyFor
      <3>2. \A x \in copies : x.kind \in {"clone", "load"} => x.o \in Owners /\ x.p \in Positions
        BY DEF IndInv, CopiesOk
      <3>3. \A x \in copies : (x.o = o /\ x.p = p) => FALSE
        BY DEF NewBlockAt, CopyFor
      <3>4. launched[ogroup[o]] \in 0 .. MaxBatch
        BY <1>1 DEF NewBlockAt, TypeOK
      <3>5. alloc'[c0.read] = "shared"
        BY <1>2, <1>3 DEF NewBlockAt
      <3>6. \A q : Moving(q)' = Moving(q)
        BY <1>5 DEF NewBlockAt, Moving, Transfers
      <3> QED
        BY <1>2, <1>4, <3>1, <3>3, <3>4, <3>5, <3>6 DEF NewBlockAt, CopyOk, Live, Unheld
    <2>2. \A c \in copies : c.read # w /\ c.write # w
      BY DEF NewBlockAt, Unheld
    <2>3. \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
      BY WorkTypes DEF WorkTyped
    <2>4. \A c \in copies : c.fill /\ c.kind = "load" =>
            /\ cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write]
            /\ ~(c0.write = c.write /\ Mem(c0.g) = Mem(c.g))
      <3> SUFFICES ASSUME NEW c \in copies, c.fill, c.kind = "load"
                   PROVE  /\ cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write]
                          /\ ~(c0.write = c.write /\ Mem(c0.g) = Mem(c.g))
        OBVIOUS
      <3>1. cache[c.k] = c.write /\ c.k \in Keys
        BY DEF IndInv, CopiesOk, TypeInv
      <3> QED
        BY <3>1, <2>2, <1>2 DEF NewBlockAt
    <2>5. \A c \in copies : c.kind \in {"clone", "load"} => ~(c0.o = c.o /\ c0.p = c.p)
      BY DEF NewBlockAt, CopyFor
    <2>6. \A c \in copies : c.fill /\ c.kind = "store" =>
            hcache'[c.k] = hcache[c.k] /\ c0.write # c.write
      BY <2>2 DEF NewBlockAt
    <2> QED
      BY <1>2, <1>4, <1>5, <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, CopiesAdded DEF NewBlockAt, IndInv
  <1>16. TasksOk'
    <2>1. \A t \in tasks : t.slot # w /\ t.h # w /\ w \notin t.units
      BY DEF NewBlockAt, Unheld
    <2>2. \A t \in tasks : t.kind = "transfer" =>
            /\ t.o # o /\ t.slot \in Slots /\ t.writes \subseteq t.units
      BY DEF NewBlockAt, Moving, IndInv, TasksOk
    <2>3. \A t \in tasks : t.kind = "transfer" => t.units \subseteq AllUnits
      BY WorkTypes DEF WorkTyped
    <2>4. \A t \in tasks : t.kind = "prefetch" => t.h \in HostUnits
      BY DEF IndInv, TasksOk
    <2> QED
      BY <1>2, <1>3, <2>1, <2>2, <2>3, <2>4, TasksAdded DEF NewBlockAt, IndInv, AllUnits
  <1>17. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ NoBad'
    BY <1>5, TransfersFrame, MovingUnmappedFrame, LostOkFrame, TablesMappedFrame, HandoffOkFrame,
       NoBadFrame DEF NewBlockAt, IndInv
  <1>18. RequestsOk'
    BY <1>4, RequestsAdded DEF NewBlockAt, IndInv
  <1>19. BatchesOk'
    <2>1. \A b \in inflight : \A x \in b.reads : alloc'[x[2]] = alloc[x[2]]
      <3>1. \A b \in inflight : \A x \in b.reads : x[2] \in AllUnits
        BY RequestTypes DEF RequestTyped
      <3> QED
        BY <3>1, <1>2 DEF NewBlockAt
    <2> QED
      BY <1>5, <2>1, BatchesAdded DEF NewBlockAt, IndInv
  <1>20. TrialVisibility'
    <2>1. \A j \in Keys : cache'[j] # NONE => cache'[j] = cache[j]
      BY <1>2
    <2>2. \A j \in Keys : (cache[j] = c0.read /\ vis[j].d # NONE) => GroupEpoch[c0.g] = vis[j].d
      BY DEF NewBlockAt
    <2>3. \A j \in Keys : (hcache[j] = c0.read /\ vis[j].h # NONE) => GroupEpoch[c0.g] = vis[j].h
      BY DEF NewBlockAt
    <2>4. \A t \in tasks : NONE \notin t.units
      BY WorkTypes DEF WorkTyped, AllUnits
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4 DEF NewBlockAt, IndInv, TrialVisibility, MadeBy
  <1>21. CrossingsOk'
    <2>1. \A d \in deferred : d.kind = "replicate" =>
            /\ \A u \in d.units : alloc'[u] = alloc[u]
            /\ ~c0.fill => c0.write \notin d.units
            /\ \A j \in Keys : cache'[j] \in d.units => cache'[j] = cache[j]
      <3>1. \A d \in deferred : d.units \subseteq AllUnits /\ w \notin d.units
        BY WorkTypes DEF WorkTyped, NewBlockAt, Unheld
      <3> QED
        BY <1>2, <3>1 DEF NewBlockAt, AllUnits
    <2>2. \A t \in tasks : t.kind = "replicate" => \A u \in t.writes : alloc'[u] = alloc[u]
      <3>1. \A t \in tasks : t.kind = "replicate" =>
              t.writes \subseteq t.units /\ t.units \subseteq AllUnits /\ w \notin t.units
        BY WorkTypes, CrossingsOkEach DEF IndInv, CrossTaskOk, WorkTyped, NewBlockAt, Unheld
      <3> QED
        BY <1>2, <3>1
    <2> QED
      BY <2>1, <2>2, CrossingsAdded DEF NewBlockAt, IndInv
  <1>22. CopiesPlaced' /\ CrossingsPlaced'
    BY TransfersFrame, PlacedAdded DEF NewBlockAt, IndInv
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18,
       <1>19, <1>20, <1>21, <1>22 DEF IndInv

-----------------------------------------------------------------------------
(* A new host block h that copy c0 fills from a cached device block        *)
(* (Store) or from an owner's slot (Save).                                 *)

HostCopyAt(h, c0) ==
  /\ CopyRecord(c0)
  /\ ncopy < MaxCopies
  /\ c0.kind \in {"store", "save"} /\ c0.id = ncopy + 1 /\ c0.k \in Keys /\ c0.p = NONE
  /\ c0.g \in Groups /\ GroupEpoch[c0.g] \notin lost
  /\ c0.write = h /\ c0.seq = 0 /\ ~c0.done /\ ~c0.fill /\ c0.parts = {}
  /\ CopyPlaced(c0)
  /\ c0.kind = "store" =>
       /\ c0.o = NONE /\ c0.read \in Units /\ alloc[c0.read] = "shared"
       /\ \E j \in Keys : cache[j] = c0.read
  /\ c0.kind = "save" =>
       /\ c0.o \in Owners /\ ostate[c0.o] = "live" /\ ogroup[c0.o] = c0.g /\ ~Moving(c0.o)
       /\ c0.read = oslot[c0.o]
  /\ \A j \in Keys :
       /\ (cache[j] = c0.read /\ vis[j].d # NONE) => vis[j].d = GroupEpoch[c0.g]
       /\ (hcache[j] = c0.read /\ vis[j].h # NONE) => vis[j].h = GroupEpoch[c0.g]
  /\ h \in free /\ alloc[h] = NONE /\ sharedKey[h] = NONE /\ Unheld(h)
  /\ \A j \in Keys : cache[j] # h
  /\ free' = free \ {h}
  /\ alloc' = [alloc EXCEPT ![h] = "copy"]
  /\ copies' = copies \cup {c0}
  /\ ncopy' = ncopy + 1
  /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, tasks, trial, member, valid,
                 vis, dbt, launched, inflight, handoff, requests, sharedKey, remote, lost, bad>>

LEMMA HostCopyInv ==
  ASSUME IndInv, NEW h \in HostUnits, NEW c0, HostCopyAt(h, c0)
  PROVE  IndInv'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. /\ \A v \in AllUnits : v # h => alloc'[v] = alloc[v]
        /\ alloc'[h] = "copy"
    BY <1>1 DEF HostCopyAt, TypeOK, AllUnits
  <1>3. /\ h \in AllUnits /\ h # NONE /\ h \notin Units /\ h \notin Slots
        /\ h \in member
    BY DEF AllUnits, HostCopyAt, IndInv, Regions
  <1>4. ncopy \in Nat /\ ncopy' \in 0 .. MaxCopies
    BY <1>1 DEF HostCopyAt, TypeOK
  <1>5. Transfers' = Transfers
    BY DEF HostCopyAt, Transfers
  <1>6. c0.kind = "save" =>
          /\ c0.read \in Slots /\ alloc[c0.read] = c0.o /\ OwnerEpoch[c0.o] = GroupEpoch[c0.g]
          /\ \A t \in tasks : t.kind = "transfer" => c0.read # t.slot
    <2> SUFFICES ASSUME c0.kind = "save"
                 PROVE  /\ c0.read \in Slots /\ alloc[c0.read] = c0.o
                        /\ OwnerEpoch[c0.o] = GroupEpoch[c0.g]
                        /\ \A t \in tasks : t.kind = "transfer" => c0.read # t.slot
      OBVIOUS
    <2>1. c0.o \in Owners /\ ostate[c0.o] = "live" /\ ogroup[c0.o] = c0.g /\ c0.read = oslot[c0.o]
      BY DEF HostCopyAt
    <2>2. oslot[c0.o] \in Slots /\ alloc[oslot[c0.o]] = c0.o
          /\ GroupEpoch[ogroup[c0.o]] = OwnerEpoch[c0.o]
      BY <2>1 DEF IndInv, OwnersOk, Live
    <2>3. \A t \in tasks : t.kind = "transfer" => oslot[c0.o] # t.slot
      BY <2>1 DEF IndInv, TasksOk
    <2> QED
      BY <2>1, <2>2, <2>3
  <1>7. c0.read \in AllUnits /\ c0.read # h /\ c0.read \notin HostUnits
    BY <1>6 DEF HostCopyAt, AllUnits
  <1>8. \A v \in AllUnits : Held(v)' <=> Held(v) \/ v = h
    <2>1. Held(c0.read)
      <3>1. CASE c0.kind = "store"
        BY <3>1 DEF HostCopyAt, Held, HeldIn
      <3>2. CASE c0.kind = "save"
        BY <3>2 DEF HostCopyAt, Held, HeldIn
      <3> QED
        BY <3>1, <3>2 DEF HostCopyAt
    <2> QED
      BY <2>1 DEF HostCopyAt, Held, HeldIn
  <1>9. TypeInv'
    <2>1. TypeOK'
      BY <1>1, <1>2, <1>3, <1>4 DEF HostCopyAt, TypeOK, AllUnits
    <2>2. \A c \in copies' :
            /\ c.seq \in 0 .. MaxBatch /\ c.k \in Keys
            /\ c.o \in Owners \cup {NONE} /\ c.p \in Positions \cup {NONE}
            /\ c.read \in AllUnits /\ c.write \in AllUnits /\ c.parts \subseteq HostParts
      BY <1>3, <1>7, WorkTypes DEF HostCopyAt, WorkTyped
    <2> QED
      BY <2>1, <2>2 DEF HostCopyAt, IndInv, TypeInv, Stamps, CopyRecord
  <1>10. FreeClean'
    <2> SUFFICES ASSUME NEW v \in free' PROVE alloc'[v] = NONE /\ sharedKey'[v] = NONE /\ ~Held(v)'
      BY DEF FreeClean
    <2>1. v \in free /\ v # h /\ v \in AllUnits
      BY <1>1 DEF HostCopyAt, TypeOK
    <2>2. alloc[v] = NONE /\ sharedKey[v] = NONE /\ ~Held(v)
      BY <2>1 DEF IndInv, FreeClean
    <2> QED
      BY <1>2, <1>8, <2>1, <2>2 DEF HostCopyAt
  <1>11. NoLeak'
    <2> SUFFICES ASSUME NEW v \in AllUnits
                 PROVE  IF InRegion(v)' THEN v \in free' \/ Held(v)' ELSE v \notin free' /\ ~Held(v)'
      BY DEF NoLeak
    <2>1. CASE v = h
      BY <2>1, <1>3, <1>8 DEF InRegion, HostCopyAt
    <2>2. CASE v # h
      BY <2>2, <1>8 DEF InRegion, HostCopyAt, IndInv, NoLeak
    <2> QED
      BY <2>1, <2>2
  <1>12. Outside' /\ Regions' /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. Outside' /\ Regions'
      BY <1>2, <1>3 DEF HostCopyAt, IndInv, Outside, Regions, OneRegion, AllUnits
    <2>2. AllocOwner' /\ SharedIff' /\ SharedCached'
      BY <1>2 DEF HostCopyAt, IndInv, AllocOwner, SharedIff, SharedCached
    <2>3. CacheShared'
      BY CacheSharedFrame DEF HostCopyAt, IndInv
    <2> QED
      BY <2>1, <2>2, <2>3
  <1>13. Tables' /\ OwnersOk'
    <2>1. \A q \in Owners, r \in Positions : table[q][r].u \in Units \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>2. \A q \in Owners : oslot[q] \in Slots \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>3. \A v \in Units \cup Slots : alloc'[v] = alloc[v]
      BY <1>2, <1>3 DEF AllUnits
    <2> QED
      BY <2>1, <2>2, <2>3 DEF HostCopyAt, IndInv, Tables, OwnersOk, Live, Mem
  <1>14. DeferredOk'
    <2>1. \A d \in deferred : h \notin d.units
      BY DEF HostCopyAt, Unheld
    <2>2. \A d \in deferred : d.kind = "commit" => d.units \subseteq Units
      BY DEF IndInv, DeferredOk
    <2> QED
      BY <1>2, <1>3, <2>1, <2>2, DeferredAdded DEF HostCopyAt, IndInv, AllUnits
  <1>15. CopiesOk'
    <2>1. CopyOk(c0)'
      <3>1. \A x \in copies : x.id \in 1 .. ncopy /\ x.write # h
        BY DEF IndInv, CopiesOk, HostCopyAt, Unheld
      <3>2. launched[c0.g] \in 0 .. MaxBatch
        BY <1>1 DEF HostCopyAt, TypeOK
      <3>3. \A q \in Owners, r \in Positions : table[q][r].u # h
        BY DEF HostCopyAt, Unheld
      <3>4. c0.kind = "store" => alloc'[c0.read] = "shared"
        BY <1>2, <1>7 DEF HostCopyAt
      <3>5. c0.kind = "save" => alloc'[c0.read] = c0.o /\ c0.read \in Slots
        BY <1>2, <1>6, <1>7
      <3>6. \A q : Moving(q)' = Moving(q)
        BY <1>5 DEF HostCopyAt, Moving, Transfers
      <3>7. c0.o \in Owners => c0.kind = "save"
        BY DEF HostCopyAt
      <3> QED
        BY <1>2, <1>4, <1>6, <3>1, <3>2, <3>3, <3>4, <3>5, <3>6, <3>7 DEF HostCopyAt, CopyOk, Live
    <2>2. \A c \in copies : c.read # h /\ c.write # h
      BY DEF HostCopyAt, Unheld
    <2>3. \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
      BY WorkTypes DEF WorkTyped
    <2>4. \A c \in copies : c.fill /\ c.kind = "load" =>
            /\ cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write]
            /\ ~(c0.write = c.write /\ Mem(c0.g) = Mem(c.g))
      BY <2>2 DEF HostCopyAt
    <2>5. \A c \in copies : c.kind \in {"clone", "load"} => ~(c0.o = c.o /\ c0.p = c.p)
      BY DEF HostCopyAt, IndInv, CopiesOk
    <2>6. \A c \in copies : c.fill /\ c.kind = "store" =>
            hcache'[c.k] = hcache[c.k] /\ c0.write # c.write
      BY <2>2 DEF HostCopyAt
    <2> QED
      BY <1>2, <1>4, <1>5, <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, CopiesAdded DEF HostCopyAt, IndInv
  <1>16. TasksOk'
    <2>1. \A t \in tasks : t.slot # h /\ t.h # h /\ h \notin t.units
      BY DEF HostCopyAt, Unheld
    <2>2. \A t \in tasks : t.kind = "transfer" =>
            /\ t.slot \in Slots /\ t.writes \subseteq t.units /\ t.o \in Owners
            /\ ostate[t.o] = "live" => ~\E c \in copies : c.o = t.o
      BY DEF IndInv, TasksOk, Live
    <2>3. \A t \in tasks : t.kind = "transfer" => t.units \subseteq AllUnits
      BY WorkTypes DEF WorkTyped
    <2>4. \A t \in tasks : t.kind = "prefetch" => t.h \in HostUnits
      BY DEF IndInv, TasksOk
    <2>5. \A t \in tasks : t.kind = "transfer" => c0.read # t.slot
      <3>1. CASE c0.kind = "store"
        BY <3>1, <2>2 DEF HostCopyAt
      <3>2. CASE c0.kind = "save"
        BY <3>2, <1>6
      <3> QED
        BY <3>1, <3>2 DEF HostCopyAt
    <2>6. \A t \in tasks : (t.kind = "transfer" /\ ostate[t.o] = "live") => c0.o # t.o
      <3>1. CASE c0.kind = "store"
        BY <3>1, <2>2 DEF HostCopyAt
      <3>2. CASE c0.kind = "save"
        BY <3>2 DEF HostCopyAt, Moving
      <3> QED
        BY <3>1, <3>2 DEF HostCopyAt
    <2> QED
      BY <1>2, <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, TasksAdded DEF HostCopyAt, IndInv, AllUnits
  <1>17. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ NoBad'
    BY <1>5, TransfersFrame, MovingUnmappedFrame, LostOkFrame, TablesMappedFrame, HandoffOkFrame,
       NoBadFrame DEF HostCopyAt, IndInv
  <1>18. RequestsOk'
    BY <1>4, RequestsAdded DEF HostCopyAt, IndInv
  <1>19. BatchesOk'
    <2>1. \A b \in inflight : \A x \in b.reads : alloc'[x[2]] = alloc[x[2]]
      <3>1. \A b \in inflight : \A x \in b.reads : x[2] \in AllUnits /\ x[2] \notin free
        BY RequestTypes DEF IndInv, RequestTyped, BatchesOk
      <3> QED
        BY <3>1, <1>2 DEF HostCopyAt
    <2> QED
      BY <1>5, <2>1, BatchesAdded DEF HostCopyAt, IndInv
  <1>20. TrialVisibility'
    BY DEF HostCopyAt, IndInv, TrialVisibility, MadeBy
  <1>21. CrossingsOk'
    <2>1. \A d \in deferred : d.units \subseteq AllUnits /\ h \notin d.units
      BY WorkTypes DEF WorkTyped, HostCopyAt, Unheld
    <2>2. \A t \in tasks : t.kind = "replicate" =>
            t.writes \subseteq t.units /\ t.units \subseteq AllUnits /\ h \notin t.units
      BY WorkTypes, CrossingsOkEach DEF IndInv, CrossTaskOk, WorkTyped, HostCopyAt, Unheld
    <2> QED
      BY <1>2, <2>1, <2>2, CrossingsAdded DEF HostCopyAt, IndInv
  <1>22. CopiesPlaced' /\ CrossingsPlaced'
    BY TransfersFrame, PlacedAdded DEF HostCopyAt, IndInv
  <1> QED
    BY <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18, <1>19, <1>20,
       <1>21, <1>22 DEF IndInv

-----------------------------------------------------------------------------
(* A copy c0 that fills memory already held: the owner's group's memory of *)
(* a cached device block (Load), or the owner's slot (Restore).            *)

FillCopyAt(o, c0) ==
  /\ CopyRecord(c0)
  /\ ostate[o] = "live" /\ ~Moving(o)
  /\ ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
  /\ ncopy < MaxCopies
  /\ c0.kind \in {"load", "restore"} /\ c0.id = ncopy + 1 /\ c0.o = o /\ c0.k \in Keys
  /\ c0.g = ogroup[o] /\ c0.seq = 0 /\ ~c0.done /\ c0.fill /\ c0.parts = {}
  /\ c0.read = hcache[c0.k] /\ c0.read \in HostUnits /\ alloc[c0.read] = "shared"
  /\ CopyPlaced(c0)
  /\ \A j \in Keys : (hcache[j] = c0.read /\ vis[j].h # NONE) => vis[j].h = OwnerEpoch[o]
  /\ c0.kind = "load" =>
       /\ c0.p \in Positions /\ table[o][c0.p] = Empty /\ ~CopyFor(o, c0.p)
       /\ c0.write = cache[c0.k] /\ c0.write \in Units /\ alloc[c0.write] = "shared"
       /\ Mem(c0.g) \notin valid[c0.write]
       /\ vis[c0.k].d = MadeBy(OwnerEpoch[o])
       /\ ~\E c \in copies : c.write = c0.write /\ Mem(c.g) = Mem(c0.g)
       /\ ~\E t \in tasks :
             t.kind \in {"transfer", "replicate"} /\ Mem(t.to) = Mem(c0.g) /\ c0.write \in t.writes
  /\ c0.kind = "restore" => c0.p = NONE /\ c0.write = oslot[o]
  /\ copies' = copies \cup {c0}
  /\ ncopy' = ncopy + 1
  /\ UNCHANGED <<free, alloc, sharedKey, cache, ostate, ogroup, oslot, table, hcache, deferred,
                 tasks, trial, member, valid, vis, dbt, launched, inflight, handoff, requests,
                 remote, lost, bad>>

LEMMA FillCopyInv ==
  ASSUME IndInv, NEW o \in Owners, NEW c0, FillCopyAt(o, c0)
  PROVE  IndInv'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. ncopy \in Nat /\ ncopy' \in 0 .. MaxCopies
    BY <1>1 DEF FillCopyAt, TypeOK
  <1>3. /\ oslot[o] \in Slots /\ alloc[oslot[o]] = o
        /\ \A t \in tasks : t.kind = "transfer" => oslot[o] # t.slot
    <2>1. oslot[o] \in Slots /\ alloc[oslot[o]] = o
      BY DEF FillCopyAt, IndInv, OwnersOk, Live
    <2>2. \A t \in tasks : t.kind = "transfer" => oslot[o] # t.slot
      BY DEF IndInv, TasksOk
    <2> QED
      BY <2>1, <2>2
  <1>4. /\ c0.write \in AllUnits /\ c0.read \in AllUnits /\ c0.read \notin Units
        /\ c0.read \notin Slots /\ c0.write \notin HostUnits
        /\ c0.kind = "restore" => c0.write \in Slots /\ alloc[c0.write] = o
    BY <1>3 DEF FillCopyAt, AllUnits
  <1>5. Transfers' = Transfers
    BY DEF FillCopyAt, Transfers
  <1>6. \A v \in AllUnits : Held(v)' <=> Held(v)
    <2>1. Held(c0.read) /\ Held(c0.write)
      BY DEF FillCopyAt, Held, HeldIn
    <2> QED
      BY <2>1 DEF FillCopyAt, Held, HeldIn
  <1>7. TypeInv'
    <2>1. TypeOK'
      BY <1>1, <1>2 DEF FillCopyAt, TypeOK
    <2>2. \A c \in copies' :
            /\ c.seq \in 0 .. MaxBatch /\ c.k \in Keys
            /\ c.o \in Owners \cup {NONE} /\ c.p \in Positions \cup {NONE}
            /\ c.read \in AllUnits /\ c.write \in AllUnits /\ c.parts \subseteq HostParts
      BY <1>4, WorkTypes DEF FillCopyAt, WorkTyped
    <2> QED
      BY <2>1, <2>2 DEF FillCopyAt, IndInv, TypeInv, Stamps, CopyRecord
  <1>8. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk'
        /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ NoBad'
    <2>1. free \subseteq AllUnits
      BY <1>1 DEF TypeOK
    <2>2. FreeClean' /\ NoLeak'
      BY <1>6, <2>1 DEF FillCopyAt, IndInv, FreeClean, NoLeak, InRegion
    <2> QED
      BY <1>5, <2>2, TransfersFrame, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
         SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame, MovingUnmappedFrame,
         LostOkFrame, TablesMappedFrame, HandoffOkFrame, NoBadFrame DEF FillCopyAt, IndInv
  <1>9. DeferredOk'
    <2>1. \A d \in deferred : d.kind = "commit" => c0.write \notin d.units
      <3> SUFFICES ASSUME NEW d \in deferred, d.kind = "commit" PROVE c0.write \notin d.units
        OBVIOUS
      <3>1. \E u \in Units : d.units = {u} /\ alloc[u] = d.o /\ d.o \in Owners
        BY DEF IndInv, DeferredOk
      <3>2. CASE c0.kind = "load"
        BY <3>1, <3>2 DEF FillCopyAt
      <3>3. CASE c0.kind = "restore"
        BY <3>1, <3>3, <1>4 DEF AllUnits
      <3> QED
        BY <3>2, <3>3 DEF FillCopyAt
    <2> QED
      BY <2>1, DeferredAdded DEF FillCopyAt, IndInv
  <1>10. CopiesOk'
    <2>1. CopyOk(c0)'
      <3>1. \A x \in copies : x.id \in 1 .. ncopy
        BY DEF IndInv, CopiesOk
      <3>2. launched[c0.g] \in 0 .. MaxBatch
        BY <1>1 DEF FillCopyAt, TypeOK
      <3>3. c0.kind = "load" => \A x \in copies : ~(x.o = o /\ x.p = c0.p)
        BY DEF FillCopyAt, CopyFor
      <3>4. \A q : Moving(q)' = Moving(q)
        BY <1>5 DEF FillCopyAt, Moving, Transfers
      <3>5. c0.kind = "load" =>
              \A t \in tasks' : (t.kind = "transfer" /\ Mem(t.to) = Mem(c0.g)) => c0.write \notin t.writes
        BY DEF FillCopyAt
      <3> QED
        BY <1>2, <1>4, <3>1, <3>2, <3>3, <3>4, <3>5 DEF FillCopyAt, CopyOk, Live
    <2>2. \A c \in copies : c.fill /\ c.kind = "load" => ~(c0.write = c.write /\ Mem(c0.g) = Mem(c.g))
      <3> SUFFICES ASSUME NEW c \in copies, c.fill, c.kind = "load"
                   PROVE  ~(c0.write = c.write /\ Mem(c0.g) = Mem(c.g))
        OBVIOUS
      <3>1. cache[c.k] = c.write /\ c.k \in Keys
        BY DEF IndInv, CopiesOk, TypeInv
      <3>2. c.write \in Units
        BY <3>1, <1>1, WorkTypes DEF TypeOK, AllUnits, WorkTyped
      <3>3. CASE c0.kind = "load"
        BY <3>3 DEF FillCopyAt
      <3>4. CASE c0.kind = "restore"
        BY <3>4, <3>2, <1>4
      <3> QED
        BY <3>3, <3>4 DEF FillCopyAt
    <2>3. \A c \in copies : ~c.fill => c0.write # c.write
      <3> SUFFICES ASSUME NEW c \in copies, ~c.fill PROVE c0.write # c.write
        OBVIOUS
      <3>1. /\ c.kind \in {"clone", "load", "store", "save"}
            /\ c.kind \in {"clone", "load"} => c.write \in Units /\ alloc[c.write] = c.o
            /\ c.kind \in {"store", "save"} => c.write \in HostUnits
            /\ c.o \in Owners \cup {NONE}
        BY DEF IndInv, CopiesOk, TypeInv
      <3>2. CASE c0.kind = "load"
        BY <3>1, <3>2 DEF FillCopyAt, NONE
      <3>3. CASE c0.kind = "restore"
        BY <3>1, <3>3, <1>4 DEF AllUnits
      <3> QED
        BY <3>2, <3>3 DEF FillCopyAt
    <2>4. \A c \in copies : c.kind \in {"clone", "load"} => ~(c0.o = c.o /\ c0.p = c.p)
      <3> SUFFICES ASSUME NEW c \in copies, c.kind \in {"clone", "load"}
                   PROVE  ~(c0.o = c.o /\ c0.p = c.p)
        OBVIOUS
      <3>1. c.p \in Positions
        BY DEF IndInv, CopiesOk
      <3>2. CASE c0.kind = "load"
        BY <3>2 DEF FillCopyAt, CopyFor
      <3>3. CASE c0.kind = "restore"
        BY <3>1, <3>3 DEF FillCopyAt
      <3> QED
        BY <3>2, <3>3 DEF FillCopyAt
    (* A store's fill writes a host block, which c0 does not write.        *)
    <2>5. \A c \in copies : c.fill /\ c.kind = "store" =>
            hcache'[c.k] = hcache[c.k] /\ c0.write # c.write
      <3> SUFFICES ASSUME NEW c \in copies, c.fill, c.kind = "store"
                   PROVE  c0.write # c.write
        BY DEF FillCopyAt
      <3>1. c.write \in HostUnits
        BY DEF IndInv, CopiesOk
      <3> QED
        BY <3>1, <1>4
    <2> QED
      BY <1>2, <1>5, <2>1, <2>2, <2>3, <2>4, <2>5, CopiesAdded DEF FillCopyAt, IndInv
  <1>11. TasksOk'
    <2>1. \A t \in tasks : t.kind = "transfer" =>
            /\ t.o \in Owners /\ t.o # o /\ t.slot \in Slots
      BY DEF FillCopyAt, Moving, IndInv, TasksOk
    <2>2. \A t \in tasks : t.kind = "transfer" => c0.write # t.slot
      <3> SUFFICES ASSUME NEW t \in tasks, t.kind = "transfer" PROVE c0.write # t.slot
        OBVIOUS
      <3>1. CASE c0.kind = "load"
        BY <3>1, <2>1 DEF FillCopyAt
      <3>2. CASE c0.kind = "restore"
        BY <3>2, <1>3 DEF FillCopyAt
      <3> QED
        BY <3>1, <3>2 DEF FillCopyAt
    <2> QED
      BY <1>4, <2>1, <2>2, TasksAdded DEF FillCopyAt, IndInv
  <1>12. RequestsOk'
    BY <1>2, RequestsAdded DEF FillCopyAt, IndInv
  <1>13. BatchesOk'
    BY <1>5, BatchesAdded DEF FillCopyAt, IndInv
  <1>14. TrialVisibility'
    <2>1. \A j \in Keys : cache[j] # c0.read
      BY <1>1, <1>4 DEF TypeOK, AllUnits
    <2>2. c0.kind = "load" => vis[c0.k].d = MadeBy(GroupEpoch[c0.g])
      BY DEF FillCopyAt
    <2> QED
      BY <2>1, <2>2 DEF FillCopyAt, IndInv, TrialVisibility, MadeBy
  <1>15. CrossingsOk'
    <2>1. \A t \in tasks : t.kind = "replicate" =>
            ((c0.fill /\ c0.kind = "load" /\ Mem(c0.g) = Mem(t.to)) => c0.write \notin t.writes)
      BY DEF FillCopyAt
    <2> QED
      BY <2>1, CrossingsAdded DEF FillCopyAt, IndInv
  <1>16. CopiesPlaced' /\ CrossingsPlaced'
    BY TransfersFrame, PlacedAdded DEF FillCopyAt, IndInv
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16 DEF IndInv

-----------------------------------------------------------------------------
(* A copy c0 that fills the host parts of a key's host block that do not  *)
(* hold it (Store): it reads the key's device block, valid in its group's *)
(* memory, and writes the key's host block, both cached and both already  *)
(* held; nothing else fills that host block.                               *)

StoreFillAt(g, k, c0) ==
  /\ CopyRecord(c0)
  /\ ncopy < MaxCopies
  /\ GroupEpoch[g] \notin lost
  /\ c0.kind = "store" /\ c0.id = ncopy + 1 /\ c0.o = NONE /\ c0.p = NONE /\ c0.k = k
  /\ c0.g = g /\ c0.seq = 0 /\ ~c0.done /\ ~c0.stopped /\ c0.fill
  /\ c0.read = cache[k] /\ c0.write = hcache[k]
  /\ c0.read \in Units /\ c0.write \in HostUnits
  /\ alloc[c0.read] = "shared" /\ alloc[c0.write] = "shared"
  /\ \A j \in Keys : cache[j] = c0.read => j = k
  /\ c0.parts \subseteq HostParts
  /\ CopyPlaced(c0)
  /\ vis[k].d \in {NONE, GroupEpoch[g]}
  /\ vis[k].h = MadeBy(GroupEpoch[g])
  /\ ~\E c \in copies : c.write = c0.write
  /\ copies' = copies \cup {c0}
  /\ ncopy' = ncopy + 1
  /\ UNCHANGED <<free, alloc, sharedKey, cache, ostate, ogroup, oslot, table, hcache, deferred,
                 tasks, trial, member, valid, vis, dbt, launched, inflight, handoff, requests,
                 remote, lost, bad>>

LEMMA StoreFillInv ==
  ASSUME IndInv, NEW g \in Groups, NEW k \in Keys, NEW c0, StoreFillAt(g, k, c0)
  PROVE  IndInv'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. ncopy \in Nat /\ ncopy' \in 0 .. MaxCopies
    BY <1>1 DEF StoreFillAt, TypeOK
  <1>3. /\ c0.read \in AllUnits /\ c0.write \in AllUnits
        /\ c0.read \notin Slots /\ c0.write \notin Slots
        /\ c0.read \notin HostUnits /\ c0.write \notin Units
    BY DEF StoreFillAt, AllUnits
  <1>4. Transfers' = Transfers /\ Crossings' = Crossings /\ TaskUnits' \subseteq TaskUnits
    BY TransfersFrame DEF StoreFillAt
  <1>5. \A v \in AllUnits : Held(v)' <=> Held(v)
    <2>1. Held(c0.read) /\ Held(c0.write)
      BY DEF StoreFillAt, Held, HeldIn
    <2> QED
      BY <2>1 DEF StoreFillAt, Held, HeldIn
  <1>6. TypeInv'
    <2>1. TypeOK'
      BY <1>1, <1>2 DEF StoreFillAt, TypeOK
    <2>2. \A c \in copies' :
            /\ c.seq \in 0 .. MaxBatch /\ c.k \in Keys
            /\ c.o \in Owners \cup {NONE} /\ c.p \in Positions \cup {NONE}
            /\ c.read \in AllUnits /\ c.write \in AllUnits /\ c.parts \subseteq HostParts
      BY <1>3, WorkTypes DEF StoreFillAt, WorkTyped
    <2> QED
      BY <2>1, <2>2 DEF StoreFillAt, IndInv, TypeInv, Stamps, CopyRecord
  <1>7. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk'
        /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ NoBad'
    <2>1. free \subseteq AllUnits
      BY <1>1 DEF TypeOK
    <2>2. FreeClean' /\ NoLeak'
      BY <1>5, <2>1 DEF StoreFillAt, IndInv, FreeClean, NoLeak, InRegion
    <2> QED
      BY <1>4, <2>2, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
         SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame, MovingUnmappedFrame,
         LostOkFrame, TablesMappedFrame, HandoffOkFrame, NoBadFrame DEF StoreFillAt, IndInv
  <1>8. DeferredOk'
    <2>1. \A d \in deferred : d.kind = "commit" => c0.write \notin d.units
      <3> SUFFICES ASSUME NEW d \in deferred, d.kind = "commit" PROVE c0.write \notin d.units
        OBVIOUS
      <3>1. \E u \in Units : d.units = {u}
        BY DEF IndInv, DeferredOk
      <3> QED
        BY <3>1, <1>3
    <2> QED
      BY <2>1, DeferredAdded DEF StoreFillAt, IndInv
  <1>9. CopiesOk'
    <2>1. CopyOk(c0)'
      <3>1. \A x \in copies : x.id \in 1 .. ncopy /\ x.write # c0.write
        BY DEF IndInv, CopiesOk, StoreFillAt
      <3>2. launched[g] \in 0 .. MaxBatch
        BY <1>1 DEF StoreFillAt, TypeOK
      <3>3. NONE \notin Owners
        OBVIOUS
      <3> QED
        BY <1>2, <3>1, <3>2, <3>3 DEF StoreFillAt, CopyOk
    <2>2. \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
      BY WorkTypes DEF WorkTyped
    <2>3. \A c \in copies : c.fill /\ c.kind = "load" =>
            /\ cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write]
            /\ ~(c0.write = c.write /\ Mem(c0.g) = Mem(c.g))
      <3> SUFFICES ASSUME NEW c \in copies, c.fill, c.kind = "load"
                   PROVE  ~(c0.write = c.write /\ Mem(c0.g) = Mem(c.g))
        BY DEF StoreFillAt
      <3>1. cache[c.k] = c.write /\ c.k \in Keys
        BY DEF IndInv, CopiesOk, TypeInv
      <3>2. c.write \in Units
        BY <3>1, <1>1, WorkTypes DEF TypeOK, AllUnits, WorkTyped, IndInv
      <3> QED
        BY <3>2, <1>3
    <2>4. \A c \in copies : ~c.fill => c0.write # c.write
      BY DEF StoreFillAt
    <2>5. \A c \in copies : c.fill /\ c.kind = "store" =>
            hcache'[c.k] = hcache[c.k] /\ c0.write # c.write
      BY DEF StoreFillAt
    <2>6. \A c \in copies : c.kind \in {"clone", "load"} => ~(c0.o = c.o /\ c0.p = c.p)
      BY DEF StoreFillAt, IndInv, CopiesOk
    <2> QED
      BY <1>2, <1>4, <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, CopiesAdded DEF StoreFillAt, IndInv
  <1>10. TasksOk'
    <2>1. \A t \in tasks : t.kind = "transfer" => t.slot \in Slots /\ t.o \in Owners
      BY DEF IndInv, TasksOk
    <2> QED
      BY <1>3, <2>1, TasksAdded DEF StoreFillAt, IndInv
  <1>11. RequestsOk'
    BY <1>2, RequestsAdded DEF StoreFillAt, IndInv
  <1>12. BatchesOk'
    BY <1>4, BatchesAdded DEF StoreFillAt, IndInv
  <1>13. TrialVisibility'
    <2>1. \A j \in Keys : hcache[j] # c0.read
      BY <1>1, <1>3 DEF TypeOK, AllUnits
    <2>2. \A j \in Keys : (cache[j] = c0.read /\ vis[j].d # NONE) => GroupEpoch[c0.g] = vis[j].d
      BY DEF StoreFillAt
    <2>3. vis[c0.k].h = MadeBy(GroupEpoch[c0.g])
      BY DEF StoreFillAt
    <2> QED
      BY <2>1, <2>2, <2>3 DEF StoreFillAt, IndInv, TrialVisibility, MadeBy
  <1>14. CrossingsOk'
    BY CrossingsAdded DEF StoreFillAt, IndInv
  <1>15. CopiesPlaced' /\ CrossingsPlaced'
    BY <1>4, PlacedAdded DEF StoreFillAt, IndInv
  <1> QED
    BY <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15 DEF IndInv

-----------------------------------------------------------------------------
(* Clone                                                                   *)

LEMMA CloneInv ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, Clone(o, p, k),
         UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ ncopy < MaxCopies /\ ostate[o] = "live" /\ cache[k] # NONE /\ VisibleDevice(k, o)
        /\ Mem(ogroup[o]) \in valid[cache[k]]
        /\ table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
    BY DEF Clone
  <1>2. PICK w \in RegionUnits(o) \ {cache[k]} :
          /\ Obtain(w, RegionUnits(o), OwnerEpoch[o])
          /\ free' = free \ {w}
          /\ alloc' = [alloc EXCEPT ![w] = o]
          /\ sharedKey' = [sharedKey EXCEPT ![w] = NONE]
          /\ copies' = copies \cup {Copy("clone", o, p, k, ogroup[o], cache[k], w)}
    BY DEF Clone
  <1>3. /\ ncopy' = ncopy + 1
        /\ UNCHANGED <<table, dbt, handoff, ostate, ogroup, oslot, hcache, deferred, tasks, trial,
                       member, valid, vis, launched, inflight, requests, remote, lost, bad>>
    BY DEF Clone, rare
  <1>4. ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
    BY <1>1 DEF IndInv, OwnersOk, Live
  <1>5. /\ cache[k] \in Units /\ alloc[cache[k]] = "shared"
        /\ \A j \in Keys : cache[j] = cache[k] => j = k
    BY <1>1, DeviceCached
  <1>6. /\ w \in Units /\ w \in member /\ Unheld(w)
        /\ cache' = [j \in Keys |-> IF cache[j] = w THEN NONE ELSE cache[j]]
        /\ \A b \in inflight : \A x \in b.reads : x[2] # w
    BY <1>1, <1>2, ObtainFacts DEF Live
  <1>7. \A j \in Keys :
          /\ (cache[j] = cache[k] /\ vis[j].d # NONE) => vis[j].d = OwnerEpoch[o]
          /\ (hcache[j] = cache[k] /\ vis[j].h # NONE) => vis[j].h = OwnerEpoch[o]
    <2>1. hcache \in [Keys -> HostUnits \cup {NONE}]
      BY TypeFacts DEF TypeOK
    <2> QED
      BY <1>1, <1>5, <2>1 DEF VisibleDevice
  <1>8. NewBlockAt(o, p, w, Copy("clone", o, p, k, ogroup[o], cache[k], w))
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF NewBlockAt, Copy, CopyRecord, CopyPlaced
  <1> QED
    BY <1>6, <1>8, NewBlockInv

-----------------------------------------------------------------------------
(* Load                                                                    *)

LEMMA LoadInv ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, Load(o, p, k),
         UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ ncopy < MaxCopies /\ ostate[o] = "live" /\ hcache[k] # NONE /\ VisibleHost(k, o)
        /\ GroupParts[ogroup[o]] \subseteq valid[hcache[k]]
        /\ table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
        /\ ncopy' = ncopy + 1
        /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, tasks, trial, member, valid,
                       vis, dbt, launched, inflight, handoff, requests, remote, lost, bad>>
    BY DEF Load, plane, rare
  <1>2. ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
    BY <1>1 DEF IndInv, OwnersOk, Live
  <1>3. /\ hcache[k] \in HostUnits /\ alloc[hcache[k]] = "shared"
        /\ \A j \in Keys : hcache[j] = hcache[k] => j = k
    BY <1>1, HostCached
  <1>4. \A j \in Keys : (hcache[j] = hcache[k] /\ vis[j].h # NONE) => vis[j].h = OwnerEpoch[o]
    BY <1>1, <1>3 DEF VisibleHost
  <1>5. \A j \in Keys : cache[j] # hcache[k]
    <2>1. cache \in [Keys -> Units \cup {NONE}]
      BY TypeFacts DEF TypeOK
    <2> QED
      BY <1>3, <2>1
  <1>6. CASE /\ cache[k] # NONE
             /\ Mem(ogroup[o]) \notin valid[cache[k]]
             /\ vis[k].d = MadeBy(OwnerEpoch[o])
    <2> DEFINE c0 == [Copy("load", o, p, k, ogroup[o], hcache[k], cache[k]) EXCEPT !.fill = TRUE]
    <2>1. /\ ~\E c \in copies : c.write = cache[k] /\ Mem(c.g) = Mem(ogroup[o])
          /\ ~\E t \in tasks :
                /\ t.kind \in {"transfer", "replicate"}
                /\ Mem(t.to) = Mem(ogroup[o]) /\ cache[k] \in t.writes
          /\ copies' = copies \cup {c0}
          /\ UNCHANGED <<free, cache, alloc, sharedKey>>
      BY <1>6 DEF Load
    <2>2. cache[k] \in Units /\ alloc[cache[k]] = "shared"
      BY <1>6, DeviceCached
    <2>3. FillCopyAt(o, c0)
      BY <1>1, <1>2, <1>3, <1>4, <1>6, <2>1, <2>2 DEF FillCopyAt, Copy, CopyRecord, CopyPlaced
    <2> QED
      BY <2>3, FillCopyInv
  <1>7. CASE ~(/\ cache[k] # NONE
               /\ Mem(ogroup[o]) \notin valid[cache[k]]
               /\ vis[k].d = MadeBy(OwnerEpoch[o]))
    <2>1. PICK w \in RegionUnits(o) \ {cache[k]} :
            /\ Obtain(w, RegionUnits(o), OwnerEpoch[o])
            /\ free' = free \ {w}
            /\ alloc' = [alloc EXCEPT ![w] = o]
            /\ sharedKey' = [sharedKey EXCEPT ![w] = NONE]
            /\ copies' = copies \cup {Copy("load", o, p, k, ogroup[o], hcache[k], w)}
      BY <1>7 DEF Load
    <2>2. /\ w \in Units /\ w \in member /\ Unheld(w)
          /\ cache' = [j \in Keys |-> IF cache[j] = w THEN NONE ELSE cache[j]]
          /\ \A b \in inflight : \A x \in b.reads : x[2] # w
      BY <1>1, <2>1, ObtainFacts DEF Live
    <2>3. NewBlockAt(o, p, w, Copy("load", o, p, k, ogroup[o], hcache[k], w))
      BY <1>1, <1>2, <1>3, <1>4, <1>5, <2>1, <2>2 DEF NewBlockAt, Copy, AllUnits, CopyRecord,
                                                    CopyPlaced
    <2> QED
      BY <2>2, <2>3, NewBlockInv
  <1> QED
    BY <1>6, <1>7

-----------------------------------------------------------------------------
(* Save and Store                                                          *)

LEMMA SaveInv ==
  ASSUME IndInv, NEW o \in Owners, NEW k \in Keys, Save(o, k), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ ncopy < MaxCopies /\ ostate[o] = "live" /\ ~Moving(o)
        /\ ncopy' = ncopy + 1
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, tasks, trial, member,
                       valid, vis, dbt, launched, inflight, handoff, requests, sharedKey, remote,
                       lost, bad>>
    BY DEF Save, plane, rare
  <1>2. PICK h \in free \cap HostUnits :
          /\ free' = free \ {h}
          /\ alloc' = [alloc EXCEPT ![h] = "copy"]
          /\ copies' = copies \cup {Copy("save", o, NONE, k, ogroup[o], oslot[o], h)}
    BY DEF Save
  <1>3. /\ ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
        /\ oslot[o] \in Slots
    BY <1>1 DEF IndInv, OwnersOk, Live
  <1>4. alloc[h] = NONE /\ sharedKey[h] = NONE /\ ~Held(h)
    BY <1>2 DEF IndInv, FreeClean
  <1>5. Unheld(h) /\ \A j \in Keys : cache[j] # h
    BY <1>4, TasksHeld DEF Held, HeldIn, Unheld
  <1>6. \A j \in Keys : cache[j] # oslot[o] /\ hcache[j] # oslot[o]
    <2>1. cache \in [Keys -> Units \cup {NONE}] /\ hcache \in [Keys -> HostUnits \cup {NONE}]
      BY TypeFacts DEF TypeOK
    <2> QED
      BY <1>3, <2>1
  <1>7. HostCopyAt(h, Copy("save", o, NONE, k, ogroup[o], oslot[o], h))
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6 DEF HostCopyAt, Copy, CopyRecord, CopyPlaced
  <1> QED
    BY <1>2, <1>7, HostCopyInv

LEMMA StoreInv ==
  ASSUME IndInv, NEW g \in Groups, NEW k \in Keys, Store(g, k), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ ncopy < MaxCopies /\ GroupEpoch[g] \notin lost /\ cache[k] # NONE
        /\ Mem(g) \in valid[cache[k]]
        /\ vis[k].d \in {NONE, GroupEpoch[g]}
        /\ ncopy' = ncopy + 1
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, tasks, trial, member,
                       valid, vis, dbt, launched, inflight, handoff, requests, sharedKey, remote,
                       lost, bad>>
    BY DEF Store, plane, rare
  <1>2. /\ cache[k] \in Units /\ alloc[cache[k]] = "shared"
        /\ \A j \in Keys : cache[j] = cache[k] => j = k
    BY <1>1, DeviceCached
  <1>3. CASE hcache[k] = NONE
    <2>1. PICK h \in free \cap HostUnits :
            /\ free' = free \ {h}
            /\ alloc' = [alloc EXCEPT ![h] = "copy"]
            /\ copies' = copies \cup {Copy("store", NONE, NONE, k, g, cache[k], h)}
      BY <1>3 DEF Store
    <2>2. alloc[h] = NONE /\ sharedKey[h] = NONE /\ ~Held(h)
      BY <2>1 DEF IndInv, FreeClean
    <2>3. Unheld(h) /\ \A j \in Keys : cache[j] # h
      BY <2>2, TasksHeld DEF Held, HeldIn, Unheld
    <2>4. \A j \in Keys : hcache[j] # cache[k]
      <3>1. hcache \in [Keys -> HostUnits \cup {NONE}]
        BY TypeFacts DEF TypeOK
      <3> QED
        BY <1>2, <3>1
    <2>5. HostCopyAt(h, Copy("store", NONE, NONE, k, g, cache[k], h))
      BY <1>1, <1>2, <2>1, <2>2, <2>3, <2>4 DEF HostCopyAt, Copy, CopyRecord, CopyPlaced
    <2> QED
      BY <2>1, <2>5, HostCopyInv
  <1>4. CASE hcache[k] # NONE
    <2> DEFINE c0 == [Copy("store", NONE, NONE, k, g, cache[k], hcache[k])
                        EXCEPT !.fill = TRUE, !.parts = GroupParts[g] \ valid[hcache[k]]]
    <2>1. /\ vis[k].h = MadeBy(GroupEpoch[g])
          /\ ~\E c \in copies : c.write = hcache[k]
          /\ copies' = copies \cup {c0}
          /\ UNCHANGED <<free, alloc>>
      BY <1>4 DEF Store
    <2>2. hcache[k] \in HostUnits /\ alloc[hcache[k]] = "shared"
      BY <1>4, HostCached
    <2>3. StoreFillAt(g, k, c0)
      BY <1>1, <1>2, <2>1, <2>2 DEF StoreFillAt, Copy, CopyRecord, CopyPlaced
    <2> QED
      BY <2>3, StoreFillInv
  <1> QED
    BY <1>3, <1>4

-----------------------------------------------------------------------------
(* Restore                                                                 *)

LEMMA RestoreInv ==
  ASSUME IndInv, NEW o \in Owners, NEW k \in Keys, Restore(o, k), UNCHANGED rare
  PROVE  IndInv'
  <1> DEFINE c0 == [Copy("restore", o, NONE, k, ogroup[o], hcache[k], oslot[o]) EXCEPT !.fill = TRUE]
  <1>1. /\ ncopy < MaxCopies /\ ostate[o] = "live" /\ hcache[k] # NONE /\ VisibleHost(k, o)
        /\ GroupParts[ogroup[o]] \subseteq valid[hcache[k]]
        /\ ~Moving(o)
        /\ copies' = copies \cup {c0}
        /\ ncopy' = ncopy + 1
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, tasks, trial,
                       member, valid, vis, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF Restore, plane, rare
  <1>2. ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
    BY <1>1 DEF IndInv, OwnersOk, Live
  <1>3. /\ hcache[k] \in HostUnits /\ alloc[hcache[k]] = "shared"
        /\ \A j \in Keys : hcache[j] = hcache[k] => j = k
    BY <1>1, HostCached
  <1>4. \A j \in Keys : (hcache[j] = hcache[k] /\ vis[j].h # NONE) => vis[j].h = OwnerEpoch[o]
    BY <1>1, <1>3 DEF VisibleHost
  <1>5. FillCopyAt(o, c0)
    BY <1>1, <1>2, <1>3, <1>4 DEF FillCopyAt, Copy, CopyRecord, CopyPlaced
  <1> QED
    BY <1>5, FillCopyInv

-----------------------------------------------------------------------------
(* The group                                                               *)

Next2 ==
  /\ \/ \E o \in Owners, p \in Positions, k \in Keys : Clone(o, p, k) \/ Load(o, p, k)
     \/ \E o \in Owners, k \in Keys : Save(o, k) \/ Restore(o, k)
     \/ \E g \in Groups, k \in Keys : Store(g, k)
  /\ UNCHANGED rare

THEOREM Step2 == IndInv /\ Next2 => IndInv'
  <1> SUFFICES ASSUME IndInv, Next2 PROVE IndInv'
    OBVIOUS
  <1>1. UNCHANGED rare
    BY DEF Next2
  <1>2. CASE \E o \in Owners, p \in Positions, k \in Keys : Clone(o, p, k) \/ Load(o, p, k)
    BY <1>1, <1>2, CloneInv, LoadInv
  <1>3. CASE \E o \in Owners, k \in Keys : Save(o, k) \/ Restore(o, k)
    BY <1>1, <1>3, SaveInv, RestoreInv
  <1>4. CASE \E g \in Groups, k \in Keys : Store(g, k)
    BY <1>1, <1>4, StoreInv
  <1> QED
    BY <1>2, <1>3, <1>4 DEF Next2
=============================================================================
