------------------------------ MODULE KvStep10 ------------------------------
(***************************************************************************)
(* The inductive step of IndInv for the loss of an engine: Lose.           *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, KvFrame, TLAPS

USE Sizes, Design

(* The values Lose(e) computes, named as in its definition.                *)
Mine(e, g) == GroupEpoch[g] = e
GoneOwners(e) == {o \in Owners : OwnerEpoch[o] = e /\ ostate[o] = "live"}
KeptCopies(e) == {c \in copies : ~Mine(e, c.g)}
KeptDeferred(e) ==
    {d \in deferred :
       /\ ~(d.kind = "finish" /\ \E c \in copies \ KeptCopies(e) : c.id = d.c)
       /\ ~(d.kind \in {"commit", "lend", "replicate"} /\ OwnerEpoch[d.o] = e)}
KeptTasks(e) == {t \in tasks : ~(t.kind = "lend" /\ OwnerEpoch[t.o] = e)}
KeptTable(e) ==
    [o \in Owners |-> IF o \in GoneOwners(e) THEN [p \in Positions |-> Empty] ELSE table[o]]
KeptSlot(e) == [o \in Owners |-> IF o \in GoneOwners(e) THEN NONE ELSE oslot[o]]
Dropped(e) ==
    (UNION {{table[o][p].u : p \in Positions} \cup {oslot[o]} : o \in GoneOwners(e)})
        \cup (UNION {{c.read, c.write} : c \in copies \ KeptCopies(e)})
        \cup (UNION {d.units : d \in deferred \ KeptDeferred(e)})
Reclaimed(e) ==
    ReclaimedWith(Dropped(e) \ {NONE}, KeptTable(e), KeptSlot(e), cache, hcache, KeptDeferred(e),
                  KeptCopies(e), KeptTasks(e))

LEMMA Lose_Effect ==
  ASSUME NEW e \in Epochs, Lose(e)
  PROVE  /\ e \in Losable /\ e \notin lost
         /\ lost' = lost \cup {e}
         /\ inflight' = {b \in inflight : ~Mine(e, b.g)}
         /\ dbt' = [g \in Groups |->
                      IF Mine(e, g) THEN [o \in Owners |-> [p \in Rows |-> NoEntry]] ELSE dbt[g]]
         /\ handoff' = {h \in handoff : ~Mine(e, h.g)}
         /\ requests' = {r \in requests : r.e # e}
         /\ copies' = KeptCopies(e)
         /\ deferred' = KeptDeferred(e)
         /\ table' = KeptTable(e)
         /\ oslot' = KeptSlot(e)
         /\ ostate' = [o \in Owners |-> IF o \in GoneOwners(e) THEN "gone" ELSE ostate[o]]
         /\ tasks' = KeptTasks(e)
         /\ free' = free \cup Reclaimed(e)
         /\ alloc' = Reset(alloc, Reclaimed(e))
         /\ sharedKey' = Reset(sharedKey, Reclaimed(e))
         /\ UNCHANGED <<ogroup, cache, hcache, ncopy, trial, member, valid, vis, launched,
                        remote, bad>>
  \* Lose tests each owner by its epoch and state, as GoneOwners(e) states it.
  <1>0. /\ [o \in Owners |-> IF OwnerEpoch[o] = e /\ ostate[o] = "live"
                             THEN [p \in Positions |-> Empty] ELSE table[o]] = KeptTable(e)
        /\ [o \in Owners |-> IF OwnerEpoch[o] = e /\ ostate[o] = "live" THEN NONE ELSE oslot[o]]
             = KeptSlot(e)
        /\ [o \in Owners |-> IF OwnerEpoch[o] = e /\ ostate[o] = "live" THEN "gone" ELSE ostate[o]]
             = [o \in Owners |-> IF o \in GoneOwners(e) THEN "gone" ELSE ostate[o]]
    BY DEF KeptTable, KeptSlot, GoneOwners
  <1>1. /\ e \in Losable /\ e \notin lost
        /\ lost' = lost \cup {e}
        /\ inflight' = {b \in inflight : ~Mine(e, b.g)}
        /\ dbt' = [g \in Groups |->
                     IF Mine(e, g) THEN [o \in Owners |-> [p \in Rows |-> NoEntry]] ELSE dbt[g]]
        /\ handoff' = {h \in handoff : ~Mine(e, h.g)}
        /\ requests' = {r \in requests : r.e # e}
        /\ copies' = KeptCopies(e)
        /\ deferred' = KeptDeferred(e)
        /\ table' = KeptTable(e)
        /\ oslot' = KeptSlot(e)
        /\ ostate' = [o \in Owners |-> IF o \in GoneOwners(e) THEN "gone" ELSE ostate[o]]
        /\ tasks' = KeptTasks(e)
        /\ DropWith(Dropped(e) \ {NONE}, KeptTable(e), KeptSlot(e), cache, hcache, KeptDeferred(e),
                    KeptCopies(e), KeptTasks(e))
        /\ UNCHANGED <<ogroup, cache, hcache, ncopy, trial, member, valid, vis, launched,
                       remote, bad>>
    BY <1>0 DEF Lose, Mine, GoneOwners, KeptCopies, KeptDeferred, KeptTasks, Dropped
  <1> QED
    BY <1>1 DEF DropWith, Reclaimed

(* The owners, tables and slots after the step.                            *)
LEMMA Lose_Owners ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  /\ \A o \in Owners : ostate'[o] = IF o \in GoneOwners(e) THEN "gone" ELSE ostate[o]
         /\ \A o \in Owners : o \in GoneOwners(e) <=> OwnerEpoch[o] = e /\ Live(o)
         /\ \A o \in Owners : Live(o)' <=> Live(o) /\ OwnerEpoch[o] # e
         /\ \A o \in Owners : o \notin GoneOwners(e) => table'[o] = table[o] /\ oslot'[o] = oslot[o]
         /\ \A o \in Owners : o \in GoneOwners(e) =>
              oslot'[o] = NONE /\ \A p \in Positions : table'[o][p] = Empty
         /\ \A o \in Owners, p \in Positions :
              table'[o][p] = table[o][p] \/ table'[o][p] = Empty
         /\ \A o \in Owners : oslot'[o] = oslot[o] \/ oslot'[o] = NONE
         /\ \A o \in Owners : Live(o) /\ ~Mine(e, ogroup[o]) => o \notin GoneOwners(e)
  <1>1. \A o \in Owners : Live(o) /\ ~Mine(e, ogroup[o]) => o \notin GoneOwners(e)
    BY DEF IndInv, OwnersOk, Mine, GoneOwners, Live
  <1> QED
    BY <1>1, Lose_Effect DEF GoneOwners, Live, KeptTable, KeptSlot, NONE

(* The tasks kept are the tasks but the lends of the lost epoch's owners,  *)
(* which hold nothing: every task kept is one of the tasks, and a unit is  *)
(* held by a kept task exactly when it is held by a task.                   *)
LEMMA Lose_KeptTasks ==
  ASSUME IndInv, NEW e \in Epochs
  PROVE  /\ \A t \in KeptTasks(e) : t \in tasks
         /\ \A t \in tasks : t.kind # "lend" => t \in KeptTasks(e)
         /\ \A t \in tasks : OwnerEpoch[t.o] # e => t \in KeptTasks(e)
         /\ \A u \in AllUnits :
              (\E t \in KeptTasks(e) : t.held /\ (u \in t.units \/ u = t.h))
                <=> (\E t \in tasks : t.held /\ (u \in t.units \/ u = t.h))
  <1>1. \A t \in tasks : t.kind = "lend" => t.units = {} /\ t.h = NONE
    BY DEF IndInv, TasksOk
  <1> QED
    BY <1>1 DEF KeptTasks, AllUnits

(* What the step leaves held: the kept tables, slots, deferred work and    *)
(* copies hold less than before, and what lost a holder was dropped.       *)
LEMMA Lose_Held ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  /\ \A u \in AllUnits : Held(u)' => Held(u)
         /\ \A u \in AllUnits : Held(u) => Held(u)' \/ u \in Dropped(e)
         /\ \A u \in Dropped(e) \ {NONE} : Held(u) /\ u \in AllUnits
         /\ Reclaimed(e) = {u \in Dropped(e) \ {NONE} : ~Held(u)'}
  <1>0. OwnerTyped /\ WorkTyped
    BY OwnerTypes, WorkTypes
  <1>1. \A u : Held(u)' = HeldIn(u, KeptTable(e), KeptSlot(e), cache, hcache, KeptDeferred(e),
                                  KeptCopies(e), KeptTasks(e))
    BY Lose_Effect DEF Held
  <1>9. \A u \in AllUnits :
          (\E t \in KeptTasks(e) : t.held /\ (u \in t.units \/ u = t.h))
            <=> (\E t \in tasks : t.held /\ (u \in t.units \/ u = t.h))
    BY Lose_KeptTasks
  <1>2. \A u \in AllUnits : Held(u)' => Held(u)
    <2> SUFFICES ASSUME NEW u \in AllUnits, Held(u)' PROVE Held(u)
      OBVIOUS
    <2>1. HeldIn(u, KeptTable(e), KeptSlot(e), cache, hcache, KeptDeferred(e), KeptCopies(e),
                 KeptTasks(e))
      BY <1>1
    <2>2. CASE \E o \in Owners, p \in Positions : KeptTable(e)[o][p].u = u
      <3>1. PICK o \in Owners, p \in Positions : KeptTable(e)[o][p].u = u
        BY <2>2
      <3>2. table[o][p].u = u
        BY <3>1 DEF KeptTable, Empty, AllUnits
      <3> QED
        BY <3>2 DEF Held, HeldIn
    <2>3. CASE \E o \in Owners : KeptSlot(e)[o] = u
      <3>1. PICK o \in Owners : KeptSlot(e)[o] = u
        BY <2>3
      <3>2. oslot[o] = u
        BY <3>1 DEF KeptSlot, AllUnits
      <3> QED
        BY <3>2 DEF Held, HeldIn
    <2>4. CASE \E d \in KeptDeferred(e) : u \in d.units
      BY <2>4 DEF KeptDeferred, Held, HeldIn
    <2>5. CASE \E c \in KeptCopies(e) : u = c.read \/ u = c.write
      BY <2>5 DEF KeptCopies, Held, HeldIn
    <2>6. CASE \/ \E k \in Keys : cache[k] = u \/ hcache[k] = u
               \/ \E t \in KeptTasks(e) : t.held /\ (u \in t.units \/ u = t.h)
      BY <2>6, <1>9 DEF Held, HeldIn
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>6 DEF HeldIn
  <1>3. \A u \in AllUnits : Held(u) => Held(u)' \/ u \in Dropped(e)
    <2> SUFFICES ASSUME NEW u \in AllUnits, Held(u), u \notin Dropped(e) PROVE Held(u)'
      OBVIOUS
    <2>1. Held(u)' = HeldIn(u, KeptTable(e), KeptSlot(e), cache, hcache, KeptDeferred(e),
                            KeptCopies(e), KeptTasks(e))
      BY <1>1
    <2>2. CASE \E o \in Owners, p \in Positions : table[o][p].u = u
      <3>1. PICK o \in Owners, p \in Positions : table[o][p].u = u
        BY <2>2
      <3>2. o \notin GoneOwners(e)
        <4> SUFFICES ASSUME o \in GoneOwners(e) PROVE FALSE
          OBVIOUS
        <4>1. u \in {table[o][q].u : q \in Positions} \cup {oslot[o]}
          BY <3>1
        <4> QED
          BY <4>1 DEF Dropped
      <3>3. KeptTable(e)[o][p].u = u
        BY <3>1, <3>2 DEF KeptTable
      <3> QED
        BY <2>1, <3>3 DEF HeldIn
    <2>3. CASE \E o \in Owners : oslot[o] = u
      <3>1. PICK o \in Owners : oslot[o] = u
        BY <2>3
      <3>2. KeptSlot(e)[o] = u
        BY <3>1 DEF KeptSlot, Dropped
      <3> QED
        BY <2>1, <3>2 DEF HeldIn
    <2>4. CASE \E d \in deferred : u \in d.units
      <3>1. PICK d \in deferred : u \in d.units
        BY <2>4
      <3>2. d \in KeptDeferred(e)
        BY <3>1 DEF Dropped
      <3> QED
        BY <2>1, <3>1, <3>2 DEF HeldIn
    <2>5. CASE \E c \in copies : u = c.read \/ u = c.write
      <3>1. PICK c \in copies : u = c.read \/ u = c.write
        BY <2>5
      <3>2. c \in KeptCopies(e)
        BY <3>1 DEF Dropped
      <3> QED
        BY <2>1, <3>1, <3>2 DEF HeldIn
    <2>6. CASE \/ \E k \in Keys : cache[k] = u \/ hcache[k] = u
               \/ \E t \in tasks : t.held /\ (u \in t.units \/ u = t.h)
      BY <2>1, <2>6, <1>9 DEF HeldIn
    <2> QED
      BY <2>2, <2>3, <2>4, <2>5, <2>6 DEF Held, HeldIn
  <1>4. \A u \in Dropped(e) \ {NONE} : Held(u) /\ u \in AllUnits
    <2>1. \A u \in Dropped(e) :
            \/ \E o \in Owners, p \in Positions : table[o][p].u = u
            \/ \E o \in Owners : oslot[o] = u
            \/ \E c \in copies : u = c.read \/ u = c.write
            \/ \E d \in deferred : u \in d.units
      BY DEF Dropped, GoneOwners
    <2>2. \A u \in Dropped(e) \ {NONE} : u \in AllUnits
      BY <1>0, <2>1 DEF OwnerTyped, WorkTyped, AllUnits, Empty, Private, Shared
    <2> QED
      BY <2>1, <2>2 DEF Held, HeldIn
  <1>5. Reclaimed(e) = {u \in Dropped(e) \ {NONE} : ~Held(u)'}
    BY <1>1, ReclaimedHeld DEF Reclaimed, On, AllRules
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5

(* The allocation after the step.                                          *)
LEMMA Lose_Alloc ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  /\ \A u \in AllUnits : alloc'[u] = IF u \in Reclaimed(e) THEN NONE ELSE alloc[u]
         /\ \A u \in AllUnits : sharedKey'[u] = IF u \in Reclaimed(e) THEN NONE ELSE sharedKey[u]
         /\ \A u \in AllUnits : Held(u)' => alloc'[u] = alloc[u] /\ sharedKey'[u] = sharedKey[u]
         /\ Reclaimed(e) \subseteq AllUnits
         /\ \A u \in Reclaimed(e) : Held(u) /\ ~Held(u)' /\ u \notin free
  <1>1. Reclaimed(e) \subseteq AllUnits /\ \A u \in Reclaimed(e) : Held(u) /\ ~Held(u)'
    BY Lose_Held
  <1>2. \A u \in Reclaimed(e) : u \notin free
    BY <1>1 DEF IndInv, FreeClean
  <1>3. /\ \A u \in AllUnits : alloc'[u] = IF u \in Reclaimed(e) THEN NONE ELSE alloc[u]
        /\ \A u \in AllUnits : sharedKey'[u] = IF u \in Reclaimed(e) THEN NONE ELSE sharedKey[u]
    BY Lose_Effect DEF Reset
  <1>4. \A u \in AllUnits : Held(u)' => u \notin Reclaimed(e)
    BY Lose_Held
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4

(* What the kept tables, slots, deferred work, copies and tasks hold is    *)
(* held after the step, so it keeps its allocation.                        *)
LEMMA Lose_Kept ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  /\ \A o \in Owners, p \in Positions :
              table'[o][p].u # NONE => alloc'[table'[o][p].u] = alloc[table'[o][p].u]
         /\ \A o \in Owners : oslot'[o] # NONE => alloc'[oslot'[o]] = alloc[oslot'[o]]
         /\ \A d \in deferred' : \A u \in d.units : alloc'[u] = alloc[u]
         /\ \A c \in copies' : alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
         /\ \A t \in tasks : \A u \in t.units \cup {t.h} : u # NONE => alloc'[u] = alloc[u]
  <1>0. OwnerTyped /\ WorkTyped
    BY OwnerTypes, WorkTypes
  <1>1. \A o \in Owners, p \in Positions :
          table'[o][p].u # NONE =>
              table'[o][p].u \in AllUnits /\ \A u \in AllUnits : u = table'[o][p].u => Held(u)'
    <2>1. \A o \in Owners, p \in Positions : table'[o][p].u \in Units \cup {NONE}
      BY <1>0, Lose_Owners DEF OwnerTyped, Empty, Private, Shared
    <2> QED
      BY <2>1 DEF Held, HeldIn, AllUnits
  <1>2. \A o \in Owners : oslot'[o] # NONE =>
          oslot'[o] \in AllUnits /\ \A u \in AllUnits : u = oslot'[o] => Held(u)'
    <2>1. \A o \in Owners : oslot'[o] \in Slots \cup {NONE}
      BY <1>0, Lose_Owners DEF OwnerTyped
    <2> QED
      BY <2>1 DEF Held, HeldIn, AllUnits
  <1>3. \A d \in deferred' : \A u \in d.units : Held(u)' /\ u \in AllUnits
    <2>1. \A d \in deferred' : d \in deferred
      BY Lose_Effect DEF KeptDeferred
    <2> QED
      BY <1>0, <2>1 DEF Held, HeldIn, WorkTyped
  <1>4. \A c \in copies' : Held(c.read)' /\ Held(c.write)' /\ c.read \in AllUnits /\ c.write \in AllUnits
    <2>1. \A c \in copies' : c \in copies
      BY Lose_Effect DEF KeptCopies
    <2> QED
      BY <1>0, <2>1 DEF Held, HeldIn, WorkTyped
  <1>5. \A t \in tasks : \A u \in t.units \cup {t.h} : u # NONE => Held(u)' /\ u \in AllUnits
    <2>1. \A t \in tasks : t.held
      BY DEF IndInv, TasksOk
    <2>2. \A t \in tasks : \A u \in t.units \cup {t.h} : u # NONE => u \in AllUnits
      BY <1>0 DEF WorkTyped, AllUnits
    <2>3. \A u \in AllUnits :
            (\E t \in KeptTasks(e) : t.held /\ (u \in t.units \/ u = t.h))
              <=> (\E t \in tasks : t.held /\ (u \in t.units \/ u = t.h))
      BY Lose_KeptTasks
    <2>4. \A u : Held(u)' = HeldIn(u, KeptTable(e), KeptSlot(e), cache, hcache, KeptDeferred(e),
                                   KeptCopies(e), KeptTasks(e))
      BY Lose_Effect DEF Held
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4 DEF HeldIn
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, Lose_Alloc

LEMMA Lose_Typed ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  TypeInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. UnitTyped'
    <2>1. alloc' \in [AllUnits -> Owners \cup {NONE, "shared", "copy", "task"}]
          /\ sharedKey' \in [AllUnits -> Keys \cup {NONE}]
      BY <1>0, Lose_Effect DEF UnitTyped, Reset
    <2> QED
      BY <1>0, <2>1, Lose_Effect, Lose_Alloc DEF UnitTyped
  <1>2. OwnerTyped'
    BY <1>0, Lose_Effect, Lose_Owners DEF OwnerTyped, KeptTable, KeptSlot, Empty
  <1>3. PlaneTyped'
    BY <1>0, Lose_Effect DEF PlaneTyped, NoEntry
  <1>4. WorkTyped'
    BY <1>0, Lose_Effect DEF WorkTyped, KeptDeferred, KeptCopies, KeptTasks
  <1>5. RequestTyped'
    BY <1>0, Lose_Effect DEF RequestTyped
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, TypeInvNext

LEMMA Lose_Free ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
  <1>1. /\ \A v \in AllUnits : Held(v)' => Held(v)
        /\ \A v \in AllUnits : Held(v) => Held(v)' \/ v \in Reclaimed(e)
    BY Lose_Held DEF AllUnits
  <1> QED
    BY <1>1, Lose_Alloc, Lose_Effect, UnitTypes, FreeReclaim DEF IndInv

LEMMA Lose_Shared ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
  <1>0. UnitTyped
    BY UnitTypes
  <1>1. AllocOwner'
    BY Lose_Alloc, Lose_Owners DEF IndInv, AllocOwner, NONE
  <1>2. SharedIff'
    BY Lose_Alloc DEF IndInv, SharedIff, NONE
  <1>3. \A k \in Keys : \A u \in AllUnits : (cache[k] = u \/ hcache[k] = u) => Held(u)'
    BY <1>0, Lose_Effect DEF Held, HeldIn, UnitTyped, AllUnits
  <1>4. SharedCached'
    BY <1>3, Lose_Alloc, Lose_Effect DEF IndInv, SharedCached, NONE
  <1>5. CacheShared'
    BY <1>0, <1>3, Lose_Alloc, Lose_Effect DEF IndInv, CacheShared, UnitTyped, AllUnits
  <1> QED
    BY <1>1, <1>2, <1>4, <1>5

LEMMA Lose_Tables ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  Tables' /\ OwnersOk'
  <1>1. Tables'
    <2>1. UNCHANGED <<ogroup, valid>>
      BY Lose_Effect
    <2> QED
      BY <2>1, Lose_Kept, Lose_Owners DEF IndInv, Tables, Mem, Empty
  <1>2. OwnersOk'
    <2>1. UNCHANGED <<ogroup, trial>> /\ lost' = lost \cup {e}
      BY Lose_Effect
    <2> QED
      BY <2>1, Lose_Kept, Lose_Owners DEF IndInv, OwnersOk, Empty, Live
  <1> QED
    BY <1>1, <1>2

LEMMA Lose_Deferred ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  DeferredOk'
  <1>1. SUFFICES ASSUME NEW d \in deferred' PROVE DeferOk(d)'
    BY DeferredOkEach
  <1>2. d \in deferred /\ DeferOk(d)
    BY Lose_Effect, DeferredOkEach DEF KeptDeferred, IndInv
  <1>3. /\ d.kind \in {"commit", "lend", "replicate"} => OwnerEpoch[d.o] # e
        /\ d.kind = "finish" => \A c \in copies : c.id = d.c => c \in copies'
    BY Lose_Effect DEF KeptDeferred, KeptCopies
  <1>4. /\ \A x \in deferred' : x \in deferred
        /\ \A c \in copies' : c \in copies
        /\ lost' = lost \cup {e}
    BY Lose_Effect DEF KeptDeferred, KeptCopies
  <1>5. d.kind = "release" =>
          /\ \A o \in Owners : oslot'[o] \notin d.units
          /\ \A o \in Owners, p \in Positions :
               table'[o][p].k = "private" => table'[o][p].u \notin d.units
    BY <1>2, Lose_Owners DEF DeferOk, Empty, NONE
  <1>6. d.kind = "commit" =>
          /\ \E u \in Units : d.units = {u} /\ alloc'[u] = d.o
          /\ \A o \in Owners, p \in Positions :
               table'[o][p].u \in d.units => o = d.o /\ p = d.p /\ table'[o][p].k = "private"
    <2>1. d.kind = "commit" => \A u \in d.units : u # NONE
      BY <1>2 DEF DeferOk
    <2> QED
      BY <1>2, <2>1, Lose_Kept, Lose_Owners DEF DeferOk, Empty
  <1>7. d.kind = "lend" => ostate'[d.o] # "new" /\ ~Importing(d.o)'
    <2>1. \A t \in tasks' : t \in tasks
      BY Lose_Effect, Lose_KeptTasks
    <2> QED
      BY <1>2, <2>1, Lose_Owners DEF DeferOk, Importing
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF DeferOk

LEMMA Lose_Copies ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  CopiesOk'
  <1>1. SUFFICES ASSUME NEW c \in copies' PROVE CopyOk(c)'
    BY CopiesOkEach
  <1>2. c \in copies /\ GroupEpoch[c.g] # e /\ CopyOk(c)
    BY Lose_Effect, CopiesOkEach DEF KeptCopies, Mine, IndInv
  <1>3. /\ \A x \in copies' : x \in copies
        /\ lost' = lost \cup {e}
        /\ UNCHANGED <<ncopy, launched, cache, hcache, valid, ogroup>>
        /\ \A t \in tasks' : t \in tasks
    BY Lose_Effect, Lose_KeptTasks DEF KeptCopies
  <1>4. c.write \in AllUnits /\ c.read \in AllUnits
    BY <1>2, WorkTypes DEF WorkTyped
  <1>5. \A o \in Owners, p \in Positions : table'[o][p].u = c.write => table[o][p].u = c.write
    BY <1>4, Lose_Owners DEF Empty, AllUnits
  <1>6. c.kind \in {"clone", "load"} => table'[c.o][c.p] = Empty
    BY <1>2, Lose_Owners DEF CopyOk
  <1>7. \A o \in Owners : Lending(o)' => Lending(o)
    BY <1>3, Lose_Effect DEF Lending, KeptDeferred
  <1>8. c.o \in Owners => ostate'[c.o] # "new" /\ (Live(c.o)' => Live(c.o))
    BY <1>2, Lose_Owners DEF CopyOk
  <1> QED
    BY <1>2, <1>3, <1>5, <1>6, <1>7, <1>8, Lose_Kept DEF CopyOk, Mem

LEMMA Lose_Tasks ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  TasksOk'
  <1>1. SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
    BY TasksOkEach
  <1>2. /\ t \in tasks /\ TaskOk(t) /\ UNCHANGED <<valid, hcache, vis, cache>>
        /\ ~(t.kind = "lend" /\ OwnerEpoch[t.o] = e)
    BY Lose_Effect, TasksOkEach DEF IndInv, KeptTasks
  <1>3. /\ \A x \in deferred' : x \in deferred
        /\ \A c \in copies' : c \in copies
        /\ \A x \in tasks' : x \in tasks
        /\ \A x \in tasks : OwnerEpoch[x.o] # e => x \in tasks'
    BY Lose_Effect, Lose_KeptTasks DEF KeptDeferred, KeptCopies
  <1>4. /\ \A q \in Owners : Live(q)' <=> Live(q) /\ OwnerEpoch[q] # e
        /\ \A q \in Owners : Live(q)' => TableUnits(q)' = TableUnits(q) /\ table'[q] = table[q]
        /\ \A q \in Owners : ostate[q] # "new" => ostate'[q] # "new"
        /\ \A q \in Owners, p \in Positions : table'[q][p] = table[q][p] \/ table'[q][p] = Empty
    BY Lose_Owners DEF TableUnits
  <1>5. \A u \in t.units \cup {t.h} : u # NONE => alloc'[u] = alloc[u]
    BY <1>2, Lose_Kept
  <1>6. /\ t.units \subseteq AllUnits
        /\ \A u \in t.units : alloc'[u] = alloc[u]
        /\ Reads(t) \subseteq t.units
    <2>1. t.units \subseteq AllUnits
      BY <1>2, WorkTypes DEF WorkTyped
    <2> QED
      BY <1>5, <2>1 DEF Reads, AllUnits
  <1>7. t.kind = "prefetch" => t.h # NONE
    BY <1>2 DEF TaskOk
  <1>8. /\ \A q : Lending(q)' => Lending(q)
        /\ \A q : Importing(q)' => Importing(q)
        /\ \A q \in Owners, p \in Positions : CopyFor(q, p)' => CopyFor(q, p)
    BY <1>3 DEF Lending, Importing, CopyFor
  (* A lend of a live owner kept is of an epoch that is not lost: kept.   *)
  <1>9. \A q \in Owners : Live(q)' =>
          ((\E x \in tasks : x.kind = "lend" /\ x.o = q) => (\E x \in tasks' : x.kind = "lend" /\ x.o = q))
    BY <1>3, <1>4
  <1>10. t.kind = "import" => \A u \in t.writes : \A q \in Owners, p \in Positions : table'[q][p].u # u
    <2>1. t.kind = "import" => \A u \in t.writes : \A q \in Owners, p \in Positions : table[q][p].u # u
      BY <1>2 DEF TaskOk
    <2>2. t.kind = "import" => t.writes \subseteq Units
      BY <1>2 DEF TaskOk
    <2> QED
      BY <2>1, <2>2, <1>4 DEF Empty
  <1>11. /\ t.held
         /\ t.kind \in {"import", "lend", "borrow", "prefetch", "put", "replicate"}
         /\ t.kind \in {"import", "lend", "borrow"} =>
              /\ t.h = NONE /\ t.p = NONE /\ t.to = NONE
              /\ t.o \in Owners /\ ostate'[t.o] # "new"
              /\ t.units \subseteq Units
         /\ t.kind \in {"lend", "borrow"} => t.writes = {} /\ t.places = {}
    BY <1>2, <1>4 DEF TaskOk
  <1>12. t.kind = "lend" =>
           /\ Live(t.o)' /\ t.units = {}
           /\ \A x \in tasks' : (x.kind = "lend" /\ x.o = t.o) => x = t
           /\ ~\E d \in deferred' : d.kind = "lend" /\ d.o = t.o
           /\ ~Importing(t.o)'
    <2> SUFFICES ASSUME t.kind = "lend"
                 PROVE  /\ Live(t.o)' /\ t.units = {}
                        /\ \A x \in tasks' : (x.kind = "lend" /\ x.o = t.o) => x = t
                        /\ ~\E d \in deferred' : d.kind = "lend" /\ d.o = t.o
                        /\ ~Importing(t.o)'
      OBVIOUS
    <2>1. /\ Live(t.o) /\ t.units = {} /\ t.o \in Owners
          /\ \A x \in tasks : (x.kind = "lend" /\ x.o = t.o) => x = t
          /\ ~\E d \in deferred : d.kind = "lend" /\ d.o = t.o
          /\ ~Importing(t.o)
          /\ OwnerEpoch[t.o] # e
      BY <1>2 DEF TaskOk
    <2> QED
      BY <2>1, <1>3, <1>4, <1>8
  <1>13. (t.kind = "borrow" /\ Live(t.o)') =>
           /\ \E x \in tasks' : x.kind = "lend" /\ x.o = t.o
           /\ t.units = TableUnits(t.o)'
    <2> SUFFICES ASSUME t.kind = "borrow", Live(t.o)'
                 PROVE  /\ \E x \in tasks' : x.kind = "lend" /\ x.o = t.o
                        /\ t.units = TableUnits(t.o)'
      OBVIOUS
    <2>1. t.o \in Owners
      BY <1>2 DEF TaskOk
    <2>2. Live(t.o) /\ (\E x \in tasks : x.kind = "lend" /\ x.o = t.o) /\ t.units = TableUnits(t.o)
      BY <2>1, <1>2, <1>4 DEF TaskOk
    <2> QED
      BY <2>1, <2>2, <1>4, <1>9
  <1>14. t.kind = "import" =>
           /\ t.writes = {x[2] : x \in t.places} /\ t.writes \subseteq t.units
           /\ \A x, y \in t.places : (x[1] = y[1] \/ x[2] = y[2]) => x = y
           /\ \A u \in t.writes :
                /\ alloc'[u] = t.o
                /\ \A o \in Owners, p \in Positions : table'[o][p].u # u
                /\ \A k \in Keys : cache'[k] # u
                /\ \A d \in deferred' : u \notin d.units
                /\ \A c \in copies' : c.read # u /\ c.write # u
                /\ \A x \in tasks' : x # t => u \notin x.units
           /\ \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t
           /\ ~Lending(t.o)'
           /\ Live(t.o)' => \A x \in t.places : table'[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])'
    <2> SUFFICES ASSUME t.kind = "import"
                 PROVE  /\ t.writes = {x[2] : x \in t.places} /\ t.writes \subseteq t.units
                        /\ \A x, y \in t.places : (x[1] = y[1] \/ x[2] = y[2]) => x = y
                        /\ \A u \in t.writes :
                             /\ alloc'[u] = t.o
                             /\ \A o \in Owners, p \in Positions : table'[o][p].u # u
                             /\ \A k \in Keys : cache'[k] # u
                             /\ \A d \in deferred' : u \notin d.units
                             /\ \A c \in copies' : c.read # u /\ c.write # u
                             /\ \A x \in tasks' : x # t => u \notin x.units
                        /\ \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t
                        /\ ~Lending(t.o)'
                        /\ Live(t.o)' => \A x \in t.places : table'[t.o][x[1]] = Empty
                                                             /\ ~CopyFor(t.o, x[1])'
      OBVIOUS
    <2>1. /\ t.writes = {x[2] : x \in t.places} /\ t.writes \subseteq t.units
          /\ \A x, y \in t.places : (x[1] = y[1] \/ x[2] = y[2]) => x = y
          /\ \A u \in t.writes :
               /\ alloc[u] = t.o
               /\ \A k \in Keys : cache[k] # u
               /\ \A d \in deferred : u \notin d.units
               /\ \A c \in copies : c.read # u /\ c.write # u
               /\ \A x \in tasks : x # t => u \notin x.units
          /\ \A x \in tasks : (x.kind = "import" /\ x.o = t.o) => x = t
          /\ ~Lending(t.o)
          /\ Live(t.o) => \A x \in t.places : table[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])
          /\ t.o \in Owners
      BY <1>2 DEF TaskOk
    <2>2. \A u \in t.writes : alloc'[u] = alloc[u]
      BY <1>6, <2>1
    <2>3. Live(t.o)' => \A x \in t.places : table'[t.o][x[1]] = table[t.o][x[1]]
      BY <2>1, <1>4
    <2>4. /\ \A u \in t.writes : \A d \in deferred' : u \notin d.units
          /\ \A u \in t.writes : \A c \in copies' : c.read # u /\ c.write # u
          /\ \A u \in t.writes : \A x \in tasks' : x # t => u \notin x.units
          /\ \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t
      BY <2>1, <1>3
    <2>5. \A u \in t.writes : \A k \in Keys : cache'[k] # u
      BY <2>1, <1>2
    <2>6. ~Lending(t.o)'
      BY <2>1, <1>8
    <2>7. Live(t.o)' => \A x \in t.places : table'[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])'
      <3> SUFFICES ASSUME Live(t.o)', NEW x \in t.places
                   PROVE  table'[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])'
        OBVIOUS
      <3>1. Live(t.o)
        BY <1>4, <2>1
      <3>2. table[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])
        BY <2>1, <3>1
      <3>3. x[1] \in Positions
        BY <1>2, WorkTypes DEF WorkTyped
      <3> QED
        BY <2>1, <2>3, <3>2, <3>3, <1>8
    <2> QED
      BY <2>1, <2>2, <2>4, <2>5, <2>6, <2>7, <1>10
  <1>15. \A u \in Reads(t) :
           (\/ alloc[u] = "shared"
            \/ alloc[u] \in Owners /\ (Live(alloc[u]) => \E x \in tasks : x.kind = "lend" /\ x.o = alloc[u]))'
    <2> SUFFICES ASSUME NEW u \in Reads(t)
                 PROVE  (\/ alloc[u] = "shared"
                         \/ alloc[u] \in Owners
                            /\ (Live(alloc[u]) => \E x \in tasks : x.kind = "lend" /\ x.o = alloc[u]))'
      OBVIOUS
    <2>1. alloc'[u] = alloc[u]
      BY <1>6
    <2>2. \/ alloc[u] = "shared"
          \/ alloc[u] \in Owners /\ (Live(alloc[u]) => \E x \in tasks : x.kind = "lend" /\ x.o = alloc[u])
      BY <1>2 DEF TaskOk
    <2> QED
      BY <2>1, <2>2, <1>4, <1>9
  <1>16. /\ t.kind \in {"prefetch", "put"} => t.units = {} /\ t.writes = {} /\ t.places = {}
         /\ t.kind = "prefetch" =>
              /\ t.h \in HostUnits /\ alloc'[t.h] = "task"
              /\ \A x \in tasks' : (x.kind = "prefetch" /\ x.h = t.h) => x = t
         /\ t.kind = "put" => t.h \in HostUnits /\ hcache'[t.k] = t.h /\ vis'[t.k].h = NONE
    BY <1>2, <1>3, <1>5, <1>7 DEF TaskOk
  <1> QED
    BY <1>11, <1>12, <1>13, <1>14, <1>15, <1>16 DEF TaskOk

LEMMA Lose_Plane ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  LendingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
  <1>0. PlaneTyped
    BY PlaneTypes
  <1>9. /\ \A x \in tasks' : x \in tasks
        /\ \A x : (x \in tasks' /\ x.kind = "replicate") <=> (x \in tasks /\ x.kind = "replicate")
    BY Lose_Effect, Lose_KeptTasks DEF KeptTasks
  <1>1. \A g \in Groups, o \in Owners, p \in Rows :
          dbt'[g][o][p] = IF Mine(e, g) THEN NoEntry ELSE dbt[g][o][p]
    BY Lose_Effect
  <1>2. LendingUnmapped'
    <2>1. \A q : Lending(q)' => Lending(q)
      BY <1>9, Lose_Effect DEF Lending, KeptDeferred
    <2> QED
      BY <1>1, <2>1, Lose_Effect DEF IndInv, LendingUnmapped, KeptDeferred, NoEntry
  <1>3. LostOk'
    BY <1>1, Lose_Effect DEF IndInv, LostOk, Mine, NoEntry
  <1>4. TablesMapped'
    (* An owner named in a table of a group of another epoch is of that     *)
    (* group's epoch, attached to it or to a group it reads.                *)
    <2>1. \A g \in Groups, o \in Owners :
            ~Mine(e, g) /\ (\E p \in Rows : dbt[g][o][p].u # NONE) => o \notin GoneOwners(e)
      <3> SUFFICES ASSUME NEW g \in Groups, NEW o \in Owners, ~Mine(e, g),
                          NEW p \in Rows, dbt[g][o][p].u # NONE
                   PROVE  o \notin GoneOwners(e)
        OBVIOUS
      <3>1. Live(o) /\ (ogroup[o] = g \/ g \in ReadersOf(o))
        BY DEF IndInv, TablesMapped, ReaderRow
      <3>2. ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o]
        BY <3>1 DEF IndInv, OwnersOk
      <3>3. GroupEpoch[g] = OwnerEpoch[o]
        BY <3>1, <3>2 DEF ReadersOf
      <3> QED
        BY <3>3 DEF Mine, GoneOwners
    <2>2. \A x \in deferred' : x \in deferred
      BY Lose_Effect DEF KeptDeferred
    <2>3. UNCHANGED <<ogroup, valid>>
      BY Lose_Effect
    <2> QED
      BY <1>1, <1>9, <2>1, <2>2, <2>3, Lose_Owners, Lose_Effect
         DEF IndInv, TablesMapped, HoldsAt, NoEntry, ReaderRow, ReadersOf, Mem
  <1>5. HandoffOk'
    <2>1. \A h \in handoff' : h \in handoff /\ OwnerEpoch[h.o] # e
      BY Lose_Effect DEF IndInv, HandoffOk, Mine
    <2>2. \A h \in handoff' : (~\E r \in requests' : Names(r, h)) => ~\E r \in requests : Names(r, h)
      BY <2>1, Lose_Effect DEF Names
    <2>3. \A x \in deferred' : x \in deferred
      BY Lose_Effect DEF KeptDeferred
    <2>4. UNCHANGED <<ogroup, valid>>
      BY Lose_Effect
    <2> QED
      BY <1>9, <2>1, <2>2, <2>3, <2>4, Lose_Owners, Lose_Effect
         DEF IndInv, HandoffOk, HoldsAt, ReaderRow, ReadersOf, Mem
  <1>6. RequestsOk'
    <2>1. \A c \in copies' : c \in copies
      BY Lose_Effect DEF KeptCopies
    <2> QED
      BY <1>1, <2>1, Lose_Effect, RequestTypes DEF IndInv, RequestsOk, NoEntry, RequestTyped
  <1>7. NoBad'
    BY Lose_Effect DEF IndInv, NoBad
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5, <1>6, <1>7

LEMMA Lose_Visibility ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  TrialVisibility'
  <1>1. /\ \A c \in copies' : c \in copies
        /\ \A o \in Owners, p \in Positions :
             table'[o][p] = table[o][p] \/ table'[o][p] = Empty
        /\ UNCHANGED <<cache, hcache, vis, trial>>
        /\ \A x \in tasks' : x \in tasks
    BY Lose_Effect, Lose_Owners, Lose_KeptTasks DEF KeptCopies
  <1>2. \A k \in Keys : cache[k] # NONE => Empty # Shared(cache[k])
    BY DEF Empty, Shared
  <1> QED
    BY <1>1, <1>2 DEF IndInv, TrialVisibility, MadeBy

LEMMA Lose_Batches ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  BatchesOk'
  <1>1. SUFFICES ASSUME NEW b \in inflight' PROVE BatchOk(b)'
    BY BatchesOkEach
  <1>2. b \in inflight /\ GroupEpoch[b.g] # e /\ BatchOk(b)
    BY Lose_Effect, BatchesOkEach DEF Mine, IndInv
  <1>3. ASSUME NEW o \in Owners, NEW u, Covered(b, o, u)
        PROVE  Covered(b, o, u)' /\ Held(u)'
    <2>1. CASE /\ Live(o) /\ (ogroup[o] = b.g \/ b.g \in ReadersOf(o))
               /\ \E p \in Rows :
                    /\ HoldsAt(o, p, u)
                    /\ \A r \in requests :
                         TakesRow(r, o, p) => b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
               /\ \A d \in deferred : (d.kind = "lend" /\ d.o = o) => d.after[b.g] >= b.seq
               /\ ~\E t \in tasks : t.kind = "lend" /\ t.o = o
      (* The batch's group is of the owner's epoch, which is not lost.     *)
      <3>0. OwnerEpoch[o] # e
        <4>1. ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o]
          BY <2>1 DEF IndInv, OwnersOk
        <4>2. GroupEpoch[b.g] = OwnerEpoch[o]
          BY <2>1, <4>1 DEF ReadersOf
        <4> QED
          BY <1>2, <4>2
      <3>1. o \notin GoneOwners(e) /\ Live(o)' /\ ogroup'[o] = ogroup[o]
        BY <3>0, <2>1, Lose_Owners, Lose_Effect DEF Live, GoneOwners
      <3>2. \A p \in Rows : HoldsAt(o, p, u)' = HoldsAt(o, p, u)
        BY <3>1, Lose_Owners DEF HoldsAt
      <3>3. /\ \A r \in requests' : r \in requests
            /\ \A d \in deferred' : d \in deferred
            /\ \A t \in tasks' : t \in tasks
        BY Lose_Effect, Lose_KeptTasks DEF KeptDeferred
      <3>4. Covered(b, o, u)'
        BY <2>1, <3>1, <3>2, <3>3 DEF Covered, ReadersOf
      <3>5. Held(u)'
        BY <2>1, <3>1, Lose_Owners, Lose_Effect DEF HoldsAt, Held, HeldIn, Rows
      <3> QED
        BY <3>4, <3>5
    <2>2. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units /\ d.after[b.g] >= b.seq
      <3>1. \A d \in deferred : d.kind = "release" => d \in deferred'
        BY Lose_Effect DEF KeptDeferred
      <3> QED
        BY <2>2, <3>1, Lose_Effect DEF Covered, Held, HeldIn
    <2> QED
      BY <1>3, <2>1, <2>2 DEF Covered
  <1>4. \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits
    BY <1>2, RequestTypes DEF RequestTyped
  <1>5. \A x \in b.reads : Covered(b, x[1], x[2])' /\ alloc'[x[2]] = alloc[x[2]] /\ x[2] \notin free'
    <2>1. \A x \in b.reads : Covered(b, x[1], x[2]) /\ x[2] \notin free
      BY <1>2 DEF BatchOk
    <2>2. \A x \in b.reads : Covered(b, x[1], x[2])' /\ Held(x[2])'
      BY <1>3, <1>4, <2>1
    <2>3. \A x \in b.reads : x[2] \notin Reclaimed(e)
      BY <1>4, <2>2, Lose_Alloc
    <2> QED
      BY <1>4, <2>1, <2>2, <2>3, Lose_Alloc, Lose_Effect
  <1>6. /\ \A x \in inflight' : x \in inflight
        /\ \A d \in deferred' : d \in deferred
        /\ \A r \in requests' : r \in requests
        /\ lost' = lost \cup {e}
        /\ UNCHANGED <<launched, valid>>
    BY Lose_Effect DEF KeptDeferred
  <1>7. /\ \A u \in b.cw : \E c \in copies' : c.write = u /\ c.g = b.g /\ c.seq = b.seq
        /\ \A u \in b.cr : \E c \in copies' : c.read = u /\ c.g = b.g /\ c.seq = b.seq
    BY <1>2, Lose_Effect DEF BatchOk, KeptCopies, Mine
  <1>8. \A x \in b.writes, p \in Positions : table'[x[1]][p].u = x[2] => table[x[1]][p].u = x[2]
    BY <1>2, <1>4, Lose_Owners DEF BatchOk, Empty, AllUnits
  <1> QED
    BY <1>2, <1>5, <1>6, <1>7, <1>8 DEF BatchOk, Mem

(* A crossing kept is of an epoch that is not lost and keeps what it      *)
(* holds; the copies and batches kept and the tasks read as before.       *)
LEMMA Lose_Crossings ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced' /\ ImportsPlaced'
  <1>1. /\ \A d \in deferred' : d \in deferred /\ (d.kind = "replicate" => OwnerEpoch[d.o] # e)
        /\ \A c \in copies' : c \in copies
        /\ \A x \in inflight' : x \in inflight
        /\ lost' = lost \cup {e}
        /\ UNCHANGED <<ogroup, cache, valid, vis, trial>>
        /\ \A x \in tasks' : x \in tasks
    BY Lose_Effect, Lose_KeptTasks DEF KeptDeferred, KeptCopies
  <1>2. \A o \in Owners : Lending(o)' => Lending(o)
    BY <1>1 DEF Lending
  <1>3. \A o \in Owners : ostate[o] # "new" => ostate'[o] # "new"
    BY Lose_Owners
  <1>4. \A d \in deferred' : CrossWaitOk(d)'
    <2> SUFFICES ASSUME NEW d \in deferred', d.kind = "replicate" PROVE CrossWaitOk(d)'
      BY DEF CrossWaitOk
    <2>1. d \in deferred /\ CrossWaitOk(d) /\ OwnerEpoch[d.o] # e
      BY <1>1, CrossingsOkEach DEF IndInv
    <2>2. \A u \in d.units : alloc'[u] = alloc[u]
      BY Lose_Kept
    <2> QED
      BY <1>1, <1>2, <1>3, <2>1, <2>2 DEF CrossWaitOk, ReadersOf, MadeBy
  <1>5. \A t \in tasks' : CrossTaskOk(t)'
    <2> SUFFICES ASSUME NEW t \in tasks', t.kind = "replicate" PROVE CrossTaskOk(t)'
      BY DEF CrossTaskOk
    <2>1. t \in tasks /\ CrossTaskOk(t)
      BY <1>1, CrossingsOkEach DEF IndInv
    <2>2. \A u \in t.units : u # NONE
      BY <2>1 DEF CrossTaskOk
    <2>3. \A u \in t.units : alloc'[u] = alloc[u]
      BY <2>1, <2>2, Lose_Kept
    <2> QED
      BY <1>1, <1>2, <1>3, <2>1, <2>3 DEF CrossTaskOk, ReadersOf, Mem
  <1>6. CrossingsOk'
    BY <1>4, <1>5, CrossingsOkEach
  <1>7. CopiesPlaced'
    BY <1>1 DEF IndInv, CopiesPlaced, Mem
  <1>8. CrossingsPlaced'
    BY <1>1 DEF IndInv, CrossingsPlaced, Mem
  <1>9. ImportsPlaced'
    BY <1>1 DEF IndInv, ImportsPlaced
  <1> QED
    BY <1>6, <1>7, <1>8, <1>9

LEMMA Lose_Inv ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  IndInv'
  BY Lose_Typed, Lose_Free, Lose_Shared, Lose_Tables, Lose_Deferred, Lose_Copies, Lose_Tasks,
     Lose_Plane, Lose_Visibility, Lose_Batches, Lose_Crossings DEF IndInv

THEOREM Step10 == IndInv /\ (\E e \in Epochs : Lose(e)) => IndInv'
  BY Lose_Inv
=============================================================================
