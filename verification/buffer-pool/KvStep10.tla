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
       /\ ~(d.kind \in {"commit", "transfer"} /\ OwnerEpoch[d.o] = e)}
KeptTable(e) ==
    [o \in Owners |-> IF o \in GoneOwners(e) THEN [p \in Positions |-> Empty] ELSE table[o]]
KeptSlot(e) == [o \in Owners |-> IF o \in GoneOwners(e) THEN NONE ELSE oslot[o]]
Dropped(e) ==
    (UNION {{table[o][p].u : p \in Positions} \cup {oslot[o]} : o \in GoneOwners(e)})
        \cup (UNION {{c.read, c.write} : c \in copies \ KeptCopies(e)})
        \cup (UNION {d.units : d \in deferred \ KeptDeferred(e)})
Reclaimed(e) ==
    ReclaimedWith(Dropped(e) \ {NONE}, KeptTable(e), KeptSlot(e), cache, hcache, KeptDeferred(e),
                  KeptCopies(e), tasks)

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
         /\ free' = free \cup Reclaimed(e)
         /\ alloc' = Reset(alloc, Reclaimed(e))
         /\ sharedKey' = Reset(sharedKey, Reclaimed(e))
         /\ UNCHANGED <<ogroup, cache, hcache, ncopy, tasks, trial, member, valid, vis, launched,
                        remote, bad>>
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
        /\ DropWith(Dropped(e) \ {NONE}, KeptTable(e), KeptSlot(e), cache, hcache, KeptDeferred(e),
                    KeptCopies(e), tasks)
        /\ UNCHANGED <<ogroup, cache, hcache, ncopy, tasks, trial, member, valid, vis, launched,
                       remote, bad>>
    BY DEF Lose, Mine, GoneOwners, KeptCopies, KeptDeferred, KeptTable, KeptSlot, Dropped
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
                                  KeptCopies(e), tasks)
    BY Lose_Effect DEF Held
  <1>2. \A u \in AllUnits : Held(u)' => Held(u)
    <2>1. \A u \in AllUnits, o \in Owners, p \in Positions : KeptTable(e)[o][p].u = u => table[o][p].u = u
      BY DEF KeptTable, Empty, AllUnits
    <2>2. \A u \in AllUnits, o \in Owners : KeptSlot(e)[o] = u => oslot[o] = u
      BY DEF KeptSlot, AllUnits
    <2>3. KeptDeferred(e) \subseteq deferred /\ KeptCopies(e) \subseteq copies
      BY DEF KeptDeferred, KeptCopies
    <2> QED
      BY <1>1, <2>1, <2>2, <2>3, Zenon DEF HeldIn, Held
  <1>3. \A u \in AllUnits : Held(u) => Held(u)' \/ u \in Dropped(e)
    <2>1. \A u \in AllUnits, o \in Owners, p \in Positions :
            table[o][p].u = u => KeptTable(e)[o][p].u = u \/ u \in Dropped(e)
      BY DEF KeptTable, Dropped
    <2>2. \A u \in AllUnits, o \in Owners : oslot[o] = u => KeptSlot(e)[o] = u \/ u \in Dropped(e)
      BY DEF KeptSlot, Dropped
    <2>3. \A u \in AllUnits, d \in deferred : u \in d.units => d \in KeptDeferred(e) \/ u \in Dropped(e)
      BY DEF Dropped
    <2>4. \A u \in AllUnits, c \in copies :
            (u = c.read \/ u = c.write) => c \in KeptCopies(e) \/ u \in Dropped(e)
      BY DEF Dropped
    <2> QED
      BY <1>1, <2>1, <2>2, <2>3, <2>4, Zenon DEF HeldIn, Held
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
    BY <1>1 DEF Reclaimed, ReclaimedWith, On, AllRules
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
         /\ \A t \in tasks : \A u \in t.units \cup {t.slot, t.h} : u # NONE => alloc'[u] = alloc[u]
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
  <1>5. \A t \in tasks : \A u \in t.units \cup {t.slot, t.h} : u # NONE => Held(u)' /\ u \in AllUnits
    <2>1. \A t \in tasks : t.held
      BY DEF IndInv, TasksOk
    <2>2. UNCHANGED tasks
      BY Lose_Effect
    <2> QED
      BY <1>0, <2>1, <2>2 DEF Held, HeldIn, WorkTyped, AllUnits
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
    BY <1>0, Lose_Effect DEF WorkTyped, KeptDeferred, KeptCopies
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
  <1>3. /\ d.kind \in {"commit", "transfer"} => OwnerEpoch[d.o] # e
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
  <1>7. d.kind = "transfer" => ostate'[d.o] # "new"
    BY <1>2, Lose_Owners DEF DeferOk
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
        /\ UNCHANGED <<ncopy, launched, cache, valid, tasks, ogroup>>
    BY Lose_Effect DEF KeptCopies
  <1>4. c.write \in AllUnits /\ c.read \in AllUnits
    BY <1>2, WorkTypes DEF WorkTyped
  <1>5. \A o \in Owners, p \in Positions : table'[o][p].u = c.write => table[o][p].u = c.write
    BY <1>4, Lose_Owners DEF Empty, AllUnits
  <1>6. c.kind \in {"clone", "load"} => table'[c.o][c.p] = Empty
    BY <1>2, Lose_Owners DEF CopyOk
  <1>7. \A o \in Owners : Moving(o)' => Moving(o)
    BY Lose_Effect DEF Moving, KeptDeferred
  <1>8. c.o \in Owners => ostate'[c.o] # "new" /\ (Live(c.o)' => Live(c.o))
    BY <1>2, Lose_Owners DEF CopyOk
  <1> QED
    BY <1>2, <1>3, <1>5, <1>6, <1>7, <1>8, Lose_Kept DEF CopyOk, Mem

LEMMA Lose_Tasks ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  TasksOk'
  <1>1. SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
    BY TasksOkEach
  <1>2. t \in tasks /\ TaskOk(t) /\ UNCHANGED <<tasks, valid, hcache, vis>>
    BY Lose_Effect, TasksOkEach DEF IndInv
  <1>3. /\ \A x \in deferred' : x \in deferred
        /\ \A c \in copies' : c \in copies
    BY Lose_Effect DEF KeptDeferred, KeptCopies
  <1>4. t.kind = "transfer" => \A o \in Owners : oslot'[o] # t.slot
    BY <1>2, Lose_Owners DEF TaskOk
  <1>5. (t.kind = "transfer" /\ Live(t.o)') =>
          TableUnits(t.o)' = TableUnits(t.o) /\ oslot'[t.o] = oslot[t.o] /\ Live(t.o)
    BY <1>2, Lose_Owners DEF TaskOk, TableUnits
  <1>6. \A u \in t.units \cup {t.slot, t.h} : u # NONE => alloc'[u] = alloc[u]
    BY <1>2, Lose_Kept
  <1>7. t.kind = "transfer" => t.writes \subseteq t.units /\ t.slot # NONE
    BY <1>2 DEF TaskOk
  <1>8. t.kind = "prefetch" => t.h # NONE
    BY <1>2 DEF TaskOk
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8 DEF TaskOk, Mem

LEMMA Lose_Plane ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
  <1>0. PlaneTyped
    BY PlaneTypes
  <1>1. \A g \in Groups, o \in Owners, p \in Rows :
          dbt'[g][o][p] = IF Mine(e, g) THEN NoEntry ELSE dbt[g][o][p]
    BY Lose_Effect
  <1>2. MovingUnmapped'
    BY <1>1, Lose_Effect DEF IndInv, MovingUnmapped, KeptDeferred, NoEntry
  <1>3. LostOk'
    BY <1>1, Lose_Effect DEF IndInv, LostOk, Mine, NoEntry
  <1>4. TablesMapped'
    <2>1. \A g \in Groups, o \in Owners :
            ~Mine(e, g) /\ (\E p \in Rows : dbt[g][o][p].u # NONE) => o \notin GoneOwners(e)
      BY Lose_Owners DEF IndInv, TablesMapped
    <2>2. \A x \in deferred' : x \in deferred
      BY Lose_Effect DEF KeptDeferred
    <2> QED
      BY <1>1, <2>1, <2>2, Lose_Owners, Lose_Effect DEF IndInv, TablesMapped, HoldsAt, NoEntry
  <1>5. HandoffOk'
    <2>1. \A h \in handoff' : h \in handoff /\ OwnerEpoch[h.o] # e
      BY Lose_Effect DEF IndInv, HandoffOk, Mine
    <2>2. \A h \in handoff' : (~\E r \in requests' : Names(r, h)) => ~\E r \in requests : Names(r, h)
      BY <2>1, Lose_Effect DEF Names
    <2>3. \A x \in deferred' : x \in deferred
      BY Lose_Effect DEF KeptDeferred
    <2> QED
      BY <2>1, <2>2, <2>3, Lose_Owners, Lose_Effect DEF IndInv, HandoffOk, HoldsAt
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
        /\ UNCHANGED <<cache, hcache, vis, tasks, trial>>
    BY Lose_Effect, Lose_Owners DEF KeptCopies
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
    <2>1. CASE /\ Live(o) /\ ogroup[o] = b.g
               /\ \E p \in Rows :
                    /\ HoldsAt(o, p, u)
                    /\ \A r \in requests :
                         TakesRow(r, o, p) => b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
               /\ \A d \in deferred : (d.kind = "transfer" /\ d.o = o) => d.after[b.g] >= b.seq
               /\ ~\E t \in tasks : t.kind = "transfer" /\ t.o = o
      <3>1. o \notin GoneOwners(e) /\ Live(o)' /\ ogroup'[o] = ogroup[o]
        BY <1>2, <2>1, Lose_Owners, Lose_Effect DEF Mine, Live
      <3>2. \A p \in Rows : HoldsAt(o, p, u)' = HoldsAt(o, p, u)
        BY <3>1, Lose_Owners DEF HoldsAt
      <3>3. /\ \A r \in requests' : r \in requests
            /\ \A d \in deferred' : d \in deferred
            /\ UNCHANGED tasks
        BY Lose_Effect DEF KeptDeferred
      <3>4. Covered(b, o, u)'
        BY <2>1, <3>1, <3>2, <3>3 DEF Covered
      <3>5. Held(u)'
        BY <2>1, <3>1, Lose_Owners DEF HoldsAt, Held, HeldIn, Rows
      <3> QED
        BY <3>4, <3>5
    <2>2. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units /\ d.after[b.g] >= b.seq
      <3>1. \A d \in deferred : d.kind = "release" => d \in deferred'
        BY Lose_Effect DEF KeptDeferred
      <3> QED
        BY <2>2, <3>1 DEF Covered, Held, HeldIn
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

LEMMA Lose_Inv ==
  ASSUME IndInv, NEW e \in Epochs, Lose(e)
  PROVE  IndInv'
  BY Lose_Typed, Lose_Free, Lose_Shared, Lose_Tables, Lose_Deferred, Lose_Copies, Lose_Tasks,
     Lose_Plane, Lose_Visibility, Lose_Batches DEF IndInv

THEOREM Step10 == IndInv /\ (\E e \in Epochs : Lose(e)) => IndInv'
  BY Lose_Inv
=============================================================================
