------------------------------- MODULE KvStep1 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for the handlers that attach an owner and  *)
(* fill a position of its table: Attach, Allocate and Pin.                 *)
(***************************************************************************)
EXTENDS KvFrame

USE Sizes, Design DEF On, AllRules

-----------------------------------------------------------------------------
(* Attach                                                                  *)

(* Attach(o, g) taking slot row s, with what IndInv says of o and s: o is  *)
(* new, so it holds nothing, nothing names it and it does not move; s is   *)
(* free, so nothing holds it.                                              *)
AttachAt(o, g, s) ==
  /\ ostate[o] = "new" /\ GroupEpoch[g] = OwnerEpoch[o]
  /\ trial[OwnerEpoch[o]] # "rolled" /\ OwnerEpoch[o] \notin lost
  /\ oslot[o] = NONE
  /\ ~Moving(o)
  /\ \A c \in copies : c.o # o
  /\ \A t \in tasks : t.kind = "transfer" => t.o # o
  /\ s \in free /\ SlotGroup[s] = g
  /\ alloc[s] = NONE /\ sharedKey[s] = NONE /\ ~Held(s) /\ Unheld(s)
  /\ free' = free \ {s}
  /\ oslot' = [oslot EXCEPT ![o] = s]
  /\ alloc' = [alloc EXCEPT ![s] = o]
  /\ Give([g |-> g, o |-> o, p |-> SLOT, u |-> s, w |-> TRUE])
  /\ ostate' = [ostate EXCEPT ![o] = "live"]
  /\ ogroup' = [ogroup EXCEPT ![o] = g]
  /\ DOMAIN table' = Owners
  /\ \A q \in Owners : DOMAIN table'[q] = Positions
  /\ \A q \in Owners, p \in Positions : table'[q][p] = table[q][p]
  /\ UNCHANGED <<cache, hcache, deferred, copies, ncopy, tasks, trial, member, valid, vis,
                 launched, inflight, requests, sharedKey>>

LEMMA AttachPick ==
  ASSUME IndInv, NEW o \in Owners, NEW g \in Groups, Attach(o, g)
  PROVE  \E s \in Slots : AttachAt(o, g, s)
  <1>1. /\ ostate[o] = "new" /\ GroupEpoch[g] = OwnerEpoch[o]
        /\ trial[OwnerEpoch[o]] # "rolled" /\ OwnerEpoch[o] \notin lost
    BY DEF Attach
  <1>2. PICK s \in free \cap Slots :
          /\ SlotGroup[s] = g
          /\ free' = free \ {s}
          /\ oslot' = [oslot EXCEPT ![o] = s]
          /\ alloc' = [alloc EXCEPT ![s] = o]
          /\ Give([g |-> g, o |-> o, p |-> SLOT, u |-> s, w |-> TRUE])
    BY DEF Attach
  <1>3. /\ ostate' = [ostate EXCEPT ![o] = "live"]
        /\ ogroup' = [ogroup EXCEPT ![o] = g]
        /\ table' = [table EXCEPT ![o] = [p \in Positions |-> Empty]]
        /\ UNCHANGED <<cache, hcache, deferred, copies, ncopy, tasks, trial, member, valid, vis,
                       launched, inflight, requests, sharedKey>>
    BY DEF Attach
  <1>4. alloc[s] = NONE /\ sharedKey[s] = NONE /\ ~Held(s)
    BY <1>2 DEF IndInv, FreeClean
  <1>5. Unheld(s)
    BY <1>4, TasksHeld DEF Held, HeldIn, Unheld
  <1>6. oslot[o] = NONE /\ \A p \in Positions : table[o][p] = Empty
    BY <1>1 DEF IndInv, OwnersOk, Live
  <1>7. /\ DOMAIN table' = Owners
        /\ \A q \in Owners : DOMAIN table'[q] = Positions
        /\ \A q \in Owners, p \in Positions : table'[q][p] = table[q][p]
    BY <1>3, <1>6 DEF IndInv, TypeInv
  <1>8. \A t \in tasks : t.kind = "transfer" => t.o # o
    <2> SUFFICES ASSUME NEW t \in tasks, t.kind = "transfer", t.o = o PROVE FALSE
      OBVIOUS
    <2>1. alloc[t.slot] = t.o /\ t.slot \in Slots
      BY DEF IndInv, TasksOk
    <2> QED
      BY <1>1, <2>1 DEF IndInv, AllocOwner, AllUnits
  <1>9. \A d \in deferred : d.kind = "transfer" => d.o # o
    BY <1>1 DEF IndInv, DeferredOk
  <1>10. ~Moving(o)
    BY <1>8, <1>9 DEF Moving
  <1>11. \A c \in copies : c.o # o
    BY <1>1 DEF IndInv, CopiesOk
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>10, <1>11, <1>8 DEF AttachAt

(* The units Attach leaves held: those held before, and the slot row.      *)
LEMMA AttachHeld ==
  ASSUME IndInv, NEW o \in Owners, NEW g \in Groups, NEW s \in Slots, AttachAt(o, g, s)
  PROVE  \A v \in AllUnits : Held(v)' <=> Held(v) \/ v = s
  <1>1. TypeOK
    BY TypeFacts
  <1>2. \A v \in AllUnits : (\E q \in Owners : oslot'[q] = v) <=> (\E q \in Owners : oslot[q] = v) \/ v = s
    BY <1>1 DEF AttachAt, TypeOK, AllUnits
  <1> QED
    BY <1>2 DEF AttachAt, Held, HeldIn

LEMMA AttachPool ==
  ASSUME IndInv, NEW o \in Owners, NEW g \in Groups, NEW s \in Slots, AttachAt(o, g, s),
         UNCHANGED rare
  PROVE  /\ TypeInv' /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
         /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
  <1> DEFINE h == [g |-> g, o |-> o, p |-> SLOT, u |-> s, w |-> TRUE]
  <1>1. TypeOK
    BY TypeFacts
  <1>2. \A v \in AllUnits : Held(v)' <=> Held(v) \/ v = s
    BY AttachHeld
  <1>3. s \notin Units /\ s \notin HostUnits /\ s \in AllUnits /\ s # NONE
    BY DEF AllUnits
  <1>4. TypeOK'
    BY <1>1, <1>3 DEF AttachAt, TypeOK, rare, AllUnits
  <1>5. /\ DOMAIN dbt' = Groups
        /\ \A x \in Groups : DOMAIN dbt'[x] = Owners
        /\ \A x \in Groups, q \in Owners : DOMAIN dbt'[x][q] = Rows
        /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {h}
        /\ \A x \in Groups, q \in Owners, p \in Rows :
             dbt'[x][q][p] = IF Eager /\ x = h.g /\ q = h.o /\ p = h.p /\ Unnamed(h)
                             THEN [u |-> h.u, w |-> h.w]
                             ELSE dbt[x][q][p]
    <2>1. Give(h)
      BY DEF AttachAt
    <2>2. h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows
      BY DEF Rows
    <2> QED
      BY <2>1, <2>2, GiveRow, PlaneTypes DEF PlaneTyped
  <1>6. TypeInv'
    <2>1. \A x \in Groups, q \in Owners, p \in Rows : dbt'[x][q][p].u \in AllUnits \cup {NONE}
      BY <1>3, <1>5, PlaneTypes DEF PlaneTyped
    <2>2. \A x \in handoff' : x.g \in Groups /\ x.o \in Owners /\ x.p \in Rows /\ x.u \in AllUnits
      BY <1>3, <1>5, RequestTypes DEF RequestTyped, Rows
    <2>3. \A q \in Owners, p \in Positions :
            table'[q][p] \in {Empty} \cup {Private(v) : v \in Units} \cup {Shared(v) : v \in Units}
      <3>1. \A q \in Owners, r \in Positions :
              table[q][r] \in {Empty} \cup {Private(v) : v \in Units} \cup {Shared(v) : v \in Units}
        BY DEF IndInv, TypeInv
      <3> QED
        BY <3>1 DEF AttachAt
    <2> QED
      BY <1>4, <1>5, <2>1, <2>2, <2>3, RowTypes DEF AttachAt, IndInv, rare
  <1>7. FreeClean'
    <2> SUFFICES ASSUME NEW v \in free' PROVE alloc'[v] = NONE /\ sharedKey'[v] = NONE /\ ~Held(v)'
      BY DEF FreeClean
    <2>1. v \in free /\ v # s /\ v \in AllUnits
      BY <1>1 DEF AttachAt, TypeOK
    <2>2. alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
      BY <1>1, <2>1 DEF AttachAt, TypeOK
    <2>3. alloc[v] = NONE /\ sharedKey[v] = NONE /\ ~Held(v)
      BY <2>1 DEF IndInv, FreeClean
    <2> QED
      BY <1>2, <2>1, <2>2, <2>3
  <1>8. NoLeak'
    <2> SUFFICES ASSUME NEW v \in AllUnits
                 PROVE  IF InRegion(v)' THEN v \in free' \/ Held(v)' ELSE v \notin free' /\ ~Held(v)'
      BY DEF NoLeak
    <2>1. CASE v = s
      BY <2>1, <1>2, <1>3 DEF InRegion, AttachAt
    <2>2. CASE v # s
      BY <2>2, <1>2 DEF InRegion, AttachAt, IndInv, NoLeak
    <2> QED
      BY <2>1, <2>2
  <1>9. Outside'
    <2>1. \A v \in Units \cup HostUnits : alloc'[v] = alloc[v]
      BY <1>1, <1>3 DEF AttachAt, TypeOK, AllUnits
    <2> QED
      BY <2>1 DEF AttachAt, IndInv, Outside
  <1>12. Regions'
    BY DEF AttachAt, IndInv, Regions, OneRegion
  <1>10. AllocOwner' /\ SharedIff' /\ SharedCached'
    BY <1>1 DEF AttachAt, IndInv, AllocOwner, SharedIff, SharedCached, TypeOK
  <1>11. CacheShared'
    BY CacheSharedFrame DEF AttachAt, IndInv
  <1> QED
    BY <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12

LEMMA AttachOwners ==
  ASSUME IndInv, NEW o \in Owners, NEW g \in Groups, NEW s \in Slots, AttachAt(o, g, s),
         UNCHANGED rare
  PROVE  /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk' /\ TasksOk'
         /\ TrialVisibility' /\ NoBad'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. \A q \in Owners : q # o => ostate'[q] = ostate[q] /\ ogroup'[q] = ogroup[q] /\ oslot'[q] = oslot[q]
    BY <1>1 DEF AttachAt, TypeOK
  <1>3. \A v \in Units : alloc'[v] = alloc[v]
    BY <1>1 DEF AttachAt, TypeOK, AllUnits
  <1>4. \A q \in Owners, p \in Positions : table[o][p] = Empty
    BY DEF AttachAt, IndInv, OwnersOk, Live
  <1>5. Tables'
    <2> SUFFICES ASSUME NEW q \in Owners, NEW p \in Positions
                 PROVE  /\ table'[q][p].k = "none" => table'[q][p].u = NONE
                        /\ table'[q][p].k = "private" => table'[q][p].u \in Units /\ alloc'[table'[q][p].u] = q
                        /\ table'[q][p].k = "shared" => table'[q][p].u \in Units /\ alloc'[table'[q][p].u] = "shared"
                        /\ table'[q][p].k # "none" => ostate'[q] = "live"
                        /\ \A r \in Positions :
                             (table'[q][p].k = "private" /\ table'[q][r].u = table'[q][p].u) => p = r
                        /\ table'[q][p].u # NONE => Mem(ogroup'[q]) \in valid'[table'[q][p].u]
      BY DEF Tables, Live
    <2>1. CASE q = o
      BY <2>1, <1>4 DEF AttachAt, Empty
    <2>2. CASE q # o
      BY <2>2, <1>2, <1>3 DEF AttachAt, IndInv, Tables, Live
    <2> QED
      BY <2>1, <2>2
  <1>6. OwnersOk'
    <2> SUFFICES ASSUME NEW q \in Owners
                 PROVE  /\ ostate'[q] = "live" =>
                             /\ ogroup'[q] \in Groups
                             /\ GroupEpoch[ogroup'[q]] = OwnerEpoch[q]
                             /\ OwnerEpoch[q] \notin lost'
                             /\ trial'[OwnerEpoch[q]] # "rolled"
                             /\ oslot'[q] \in Slots
                             /\ alloc'[oslot'[q]] = q
                             /\ SlotGroup[oslot'[q]] = ogroup'[q]
                        /\ ostate'[q] # "live" => oslot'[q] = NONE /\ \A p \in Positions : table'[q][p] = Empty
      BY DEF OwnersOk, Live
    <2>1. CASE q = o
      BY <2>1, <1>1 DEF AttachAt, TypeOK, rare, AllUnits
    <2>2. CASE q # o
      <3>1. ostate[q] = "live" => oslot[q] # s
        BY DEF AttachAt, Unheld
      <3>2. ostate[q] = "live" => oslot[q] \in Slots
        BY DEF IndInv, OwnersOk, Live
      <3> QED
        BY <2>2, <1>1, <1>2, <3>1, <3>2 DEF AttachAt, IndInv, OwnersOk, Live, TypeOK, rare, AllUnits
    <2> QED
      BY <2>1, <2>2
  <1>7. DeferredOk'
    <2>1. \A d \in deferred : s \notin d.units
      BY DEF AttachAt, Unheld
    <2>2. \A d \in deferred : d.kind = "commit" => d.units \subseteq Units
      BY DEF IndInv, DeferredOk
    <2>3. \A q \in Owners : ostate[q] # "new" => ostate'[q] # "new"
      BY <1>1 DEF AttachAt, TypeOK
    <2>4. \A q \in Owners : oslot'[q] = oslot[q] \/ oslot'[q] = s
      BY <1>1 DEF AttachAt, TypeOK
    <2> QED
      BY <1>1, <1>3, <2>1, <2>2, <2>3, <2>4 DEF AttachAt, IndInv, DeferredOk, rare
  <1>8. CopiesOk'
    <2>1. \A c \in copies : c.read # s /\ c.write # s /\ c.o # o
      BY DEF AttachAt, Unheld
    <2>2. \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
      BY DEF IndInv, TypeInv
    <2>3. \A v \in AllUnits : v # s => alloc'[v] = alloc[v]
      BY <1>1 DEF AttachAt, TypeOK
    <2>4. \A q \in Owners : ostate[q] # "new" => ostate'[q] # "new"
      BY <1>1 DEF AttachAt, TypeOK
    <2>5. \A q : Moving(q)' = Moving(q)
      BY DEF AttachAt, Moving
    <2> QED
      BY <1>2, <2>1, <2>2, <2>3, <2>4, <2>5 DEF AttachAt, IndInv, CopiesOk, Live, Mem, rare
  <1>9. TasksOk'
    <2>1. \A t \in tasks : t.slot # s /\ t.h # s /\ s \notin t.units
      BY DEF AttachAt, Unheld
    <2>2. \A t \in tasks : t.kind = "transfer" => t.o # o
      BY DEF AttachAt
    <2>3. \A v \in AllUnits : v # s => alloc'[v] = alloc[v]
      BY <1>1 DEF AttachAt, TypeOK
    <2>4. \A t \in tasks : /\ t.slot \in Slots \cup {NONE} /\ t.h \in HostUnits \cup {NONE}
                           /\ t.units \subseteq AllUnits /\ t.o \in Owners \cup {NONE}
      BY DEF IndInv, TypeInv
    <2>5. \A q \in Owners : q # o => TableUnits(q)' = TableUnits(q)
      BY DEF AttachAt, TableUnits
    <2>6. \A q \in Owners : oslot'[q] = oslot[q] \/ (q = o /\ oslot'[q] = s)
      BY <1>1 DEF AttachAt, TypeOK
    <2> QED
      BY <1>2, <2>1, <2>2, <2>3, <2>4, <2>5, <2>6 DEF AttachAt, IndInv, TasksOk, Live, Mem, AllUnits
  <1>10. TrialVisibility'
    BY DEF AttachAt, IndInv, TrialVisibility, MadeBy
  <1>11. NoBad'
    BY NoBadFrame DEF IndInv, rare
  <1> QED
    BY <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11

LEMMA AttachPlane ==
  ASSUME IndInv, NEW o \in Owners, NEW g \in Groups, NEW s \in Slots, AttachAt(o, g, s),
         UNCHANGED rare
  PROVE  /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
         /\ BatchesOk'
  <1> DEFINE h == [g |-> g, o |-> o, p |-> SLOT, u |-> s, w |-> TRUE]
  <1>1. TypeOK
    BY TypeFacts
  <1>2. /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {h}
        /\ \A x \in Groups, q \in Owners, p \in Rows :
             dbt'[x][q][p] = IF Eager /\ x = h.g /\ q = h.o /\ p = h.p /\ Unnamed(h)
                             THEN [u |-> h.u, w |-> h.w]
                             ELSE dbt[x][q][p]
    <2>1. Give(h)
      BY DEF AttachAt
    <2>2. h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows
      BY DEF Rows
    <2> QED
      BY <2>1, <2>2, GiveRow, PlaneTypes DEF PlaneTyped
  <1>3. \A q \in Owners : q # o =>
          /\ ostate'[q] = ostate[q] /\ ogroup'[q] = ogroup[q] /\ oslot'[q] = oslot[q]
    BY <1>1 DEF AttachAt, TypeOK
  <1>4. /\ ostate'[o] = "live" /\ ogroup'[o] = g /\ oslot'[o] = s
    BY <1>1 DEF AttachAt, TypeOK
  <1>5. \A x \in Groups, q \in Owners, p \in Rows : dbt[x][q][p].u # NONE => q # o
    BY DEF AttachAt, IndInv, TablesMapped, Live
  <1>6. \A y \in handoff : Unnamed(y) => y.o # o
    BY DEF AttachAt, IndInv, HandoffOk, Live, Unnamed
  <1>7. MovingUnmapped'
    <2> SUFFICES ASSUME NEW q \in Owners,
                        \/ \E t \in tasks : t.kind = "transfer" /\ t.o = q
                        \/ \E d \in deferred : d.kind = "transfer" /\ d.o = q
                 PROVE  /\ \A x \in Groups, p \in Rows : dbt'[x][q][p].u = NONE
                        /\ \A y \in handoff' : y.o # q
                        /\ \A d, e \in deferred :
                             (d.kind = "transfer" /\ e.kind = "transfer" /\ d.o = q /\ e.o = q) => d = e
      BY DEF MovingUnmapped, AttachAt
    <2>1. q # o
      BY DEF AttachAt, Moving
    <2> QED
      BY <2>1, <1>2 DEF IndInv, MovingUnmapped
  <1>8. LostOk'
    <2>1. GroupEpoch[g] \notin lost
      BY DEF AttachAt
    <2> QED
      BY <2>1, <1>2 DEF AttachAt, IndInv, LostOk, rare
  <1>9. TablesMapped'
    <2> SUFFICES ASSUME NEW x \in Groups, NEW q \in Owners, NEW p \in Rows, dbt'[x][q][p].u # NONE
                 PROVE  /\ ostate'[q] = "live"
                        /\ \/ ogroup'[q] = x
                           \/ /\ x \in ReadersOf(q)' /\ p \in Positions /\ ~dbt'[x][q][p].w
                              /\ Mem(x) \in valid'[dbt'[x][q][p].u]
                        /\ IF p = SLOT THEN oslot'[q] = dbt'[x][q][p].u
                                       ELSE table'[q][p].u = dbt'[x][q][p].u
                        /\ (p \in Positions /\ dbt'[x][q][p].w) =>
                             /\ table'[q][p].k = "private"
                             /\ ~\E d \in deferred' :
                                   d.kind \in {"commit", "replicate"} /\ dbt'[x][q][p].u \in d.units
                             /\ ~\E t \in tasks' : t.kind = "replicate" /\ dbt'[x][q][p].u \in t.units
      BY DEF TablesMapped, Live, HoldsAt, ReaderRow
    <2>1. CASE Eager /\ x = g /\ q = o /\ p = SLOT /\ Unnamed(h)
      BY <2>1, <1>2, <1>4
    <2>2. CASE ~(Eager /\ x = g /\ q = o /\ p = SLOT /\ Unnamed(h))
      <3>1. dbt'[x][q][p] = dbt[x][q][p]
        BY <2>2, <1>2
      <3>2. q # o
        BY <3>1, <1>5
      <3>3. /\ ostate[q] = "live"
            /\ \/ ogroup[q] = x
               \/ /\ x \in ReadersOf(q) /\ p \in Positions /\ ~dbt[x][q][p].w
                  /\ Mem(x) \in valid[dbt[x][q][p].u]
            /\ HoldsAt(q, p, dbt[x][q][p].u)
            /\ (p \in Positions /\ dbt[x][q][p].w) =>
                 /\ table[q][p].k = "private"
                 /\ ~\E d \in deferred :
                       d.kind \in {"commit", "replicate"} /\ dbt[x][q][p].u \in d.units
                 /\ ~\E t \in tasks : t.kind = "replicate" /\ dbt[x][q][p].u \in t.units
        BY <3>1 DEF IndInv, TablesMapped, Live, ReaderRow
      <3>4. /\ ostate'[q] = ostate[q] /\ ogroup'[q] = ogroup[q] /\ oslot'[q] = oslot[q]
            /\ p \in Positions => table'[q][p] = table[q][p]
            /\ UNCHANGED <<deferred, tasks, valid>>
        BY <3>2, <1>3 DEF AttachAt
      <3>5. ReadersOf(q)' = ReadersOf(q)
        BY <3>4 DEF ReadersOf
      <3> QED
        BY <3>1, <3>3, <3>4, <3>5 DEF HoldsAt, Rows
    <2> QED
      BY <2>1, <2>2
  <1>10. HandoffOk'
    <2> SUFFICES ASSUME NEW y \in handoff'
                 PROVE  /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
                        /\ Unnamed(y) =>
                             /\ ostate'[y.o] = "live"
                             /\ \/ ogroup'[y.o] = y.g
                                \/ /\ y.g \in ReadersOf(y.o)' /\ y.p \in Positions /\ ~y.w
                                   /\ Mem(y.g) \in valid'[y.u]
                             /\ IF y.p = SLOT THEN oslot'[y.o] = y.u ELSE table'[y.o][y.p].u = y.u
                             /\ (y.p \in Positions /\ y.w) =>
                                  /\ table'[y.o][y.p].k = "private"
                                  /\ ~\E d \in deferred' :
                                        d.kind \in {"commit", "replicate"} /\ y.u \in d.units
                                  /\ ~\E t \in tasks' : t.kind = "replicate" /\ y.u \in t.units
      BY DEF HandoffOk, Live, HoldsAt, Unnamed, AttachAt, ReaderRow
    <2>1. CASE y = h
      BY <2>1, <1>4 DEF AttachAt
    <2>2. CASE y # h
      <3>1. y \in handoff
        BY <2>2, <1>2
      <3>2. Unnamed(y) => y.o # o
        BY <3>1, <1>6
      <3>3. /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
            /\ Unnamed(y) =>
                 /\ ostate[y.o] = "live"
                 /\ \/ ogroup[y.o] = y.g
                    \/ /\ y.g \in ReadersOf(y.o) /\ y.p \in Positions /\ ~y.w
                       /\ Mem(y.g) \in valid[y.u]
                 /\ HoldsAt(y.o, y.p, y.u)
                 /\ (y.p \in Positions /\ y.w) =>
                      /\ table[y.o][y.p].k = "private"
                      /\ ~\E d \in deferred : d.kind \in {"commit", "replicate"} /\ y.u \in d.units
                      /\ ~\E t \in tasks : t.kind = "replicate" /\ y.u \in t.units
        BY <3>1 DEF IndInv, HandoffOk, Live, Unnamed, ReaderRow
      <3>4. Unnamed(y) =>
              /\ ostate'[y.o] = ostate[y.o] /\ ogroup'[y.o] = ogroup[y.o] /\ oslot'[y.o] = oslot[y.o]
              /\ y.p \in Positions => table'[y.o][y.p] = table[y.o][y.p]
              /\ ReadersOf(y.o)' = ReadersOf(y.o)
        BY <3>2, <3>3, <1>3 DEF AttachAt, ReadersOf
      <3>5. UNCHANGED <<deferred, requests, tasks, valid>>
        BY DEF AttachAt
      <3>6. y.p \in Rows
        BY <3>1 DEF IndInv, TypeInv
      <3> QED
        BY <3>3, <3>4, <3>5, <3>6 DEF HoldsAt, Unnamed, Rows
    <2> QED
      BY <2>1, <2>2
  <1>11. RequestsOk'
    <2>1. \A r \in requests :
            r.kind # "finish" =>
              \A x \in Groups, p \in Rows :
                (dbt'[x][r.o][p].u # NONE /\ (TakesRow(r, r.o, p) \/ FreezesRow(r, r.o, p))) =>
                  /\ x \in r.sg
                  /\ x \in r.by => FreezesRow(r, r.o, p) /\ ~dbt'[x][r.o][p].w
      <3> SUFFICES ASSUME NEW r \in requests, r.kind # "finish", NEW x \in Groups, NEW p \in Rows,
                          dbt'[x][r.o][p].u # NONE, TakesRow(r, r.o, p) \/ FreezesRow(r, r.o, p)
                   PROVE  /\ x \in r.sg
                          /\ x \in r.by => FreezesRow(r, r.o, p) /\ ~dbt'[x][r.o][p].w
        OBVIOUS
      <3>1. r.o \in Owners
        BY DEF IndInv, TypeInv
      <3>2. CASE Eager /\ x = g /\ r.o = o /\ p = SLOT /\ Unnamed(h)
        <4>1. Names(r, h)
          BY <3>2, NamesRow
        <4> QED
          BY <3>2, <4>1 DEF Unnamed
      <3>3. CASE ~(Eager /\ x = g /\ r.o = o /\ p = SLOT /\ Unnamed(h))
        <4>1. dbt'[x][r.o][p] = dbt[x][r.o][p]
          BY <3>1, <3>3, <1>2
        <4> QED
          BY <3>1, <4>1 DEF IndInv, RequestsOk
      <3> QED
        BY <3>2, <3>3
    <2> QED
      BY <2>1 DEF IndInv, RequestsOk, AttachAt
  <1>12. BatchesOk'
    <2>1. \A b \in inflight : \A x \in b.reads : x[2] # s
      BY DEF IndInv, BatchesOk, AttachAt
    <2>2. \A b \in inflight : b.writes \subseteq b.reads
      BY DEF IndInv, BatchesOk
    <2>3. \A b, v : \A q \in Owners : Covered(b, q, v) => Covered(b, q, v)'
      <3> SUFFICES ASSUME NEW b, NEW v, NEW q \in Owners, Covered(b, q, v)
                   PROVE  Covered(b, q, v)'
        OBVIOUS
      <3>1. CASE q = o
        BY <3>1, CoverStays DEF AttachAt
      <3>2. CASE q # o
        <4>1. ostate'[q] = ostate[q] /\ ogroup'[q] = ogroup[q] /\ oslot'[q] = oslot[q]
          BY <3>2, <1>3
        <4>2. \A r \in Rows : HoldsAt(q, r, v) => HoldsAt(q, r, v)'
          BY <4>1 DEF HoldsAt, AttachAt, Rows
        <4> QED
          BY <4>1, <4>2, CoverStays DEF AttachAt
      <3> QED
        BY <3>1, <3>2
    <2>4. \A v \in AllUnits : v # s => alloc'[v] = alloc[v]
      BY <1>1 DEF AttachAt, TypeOK
    <2>5. \A b \in inflight : \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits
      BY RequestTypes DEF RequestTyped
    <2>6. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY BatchesOkEach DEF AttachAt
    <2>7. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>8. /\ free' \subseteq free
          /\ UNCHANGED <<inflight, launched, lost, valid, deferred, requests, copies>>
          /\ \A q \in Owners, r \in Positions : table'[q][r] = table[q][r]
      BY DEF AttachAt, rare
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>7, <2>8 DEF BatchOk, Mem
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12

(* An owner that is new has no crossing; Attach changes nothing else a     *)
(* crossing or a copy reads.                                               *)
LEMMA AttachCross ==
  ASSUME IndInv, NEW o \in Owners, NEW g \in Groups, NEW s \in Slots, AttachAt(o, g, s),
         UNCHANGED rare
  PROVE  CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. /\ \A q \in Owners : q # o => ostate'[q] = ostate[q] /\ ogroup'[q] = ogroup[q]
        /\ \A v \in Units : alloc'[v] = alloc[v]
        /\ UNCHANGED <<cache, deferred, copies, tasks, valid, vis, inflight, lost, trial>>
    BY <1>1 DEF AttachAt, TypeOK, AllUnits, rare
  <1>3. \A q \in Owners : Moving(q)' = Moving(q)
    BY <1>2 DEF Moving
  <1>4. /\ \A d \in deferred : d.kind = "replicate" => d.o # o
        /\ \A t \in tasks : t.kind = "replicate" => t.o # o
    BY CrossingsOkEach DEF AttachAt, IndInv, CrossWaitOk, CrossTaskOk
  <1>5. CrossingsOk'
    <2>1. \A d \in deferred : CrossWaitOk(d)'
      <3> SUFFICES ASSUME NEW d \in deferred, d.kind = "replicate" PROVE CrossWaitOk(d)'
        BY DEF CrossWaitOk
      <3>1. CrossWaitOk(d) /\ d.o # o
        BY <1>4, CrossingsOkEach DEF IndInv
      <3>2. d.o \in Owners /\ d.units \subseteq Units
        BY <3>1 DEF CrossWaitOk
      <3>3. ReadersOf(d.o)' = ReadersOf(d.o)
        BY <1>2, <3>1, <3>2 DEF ReadersOf
      <3> QED
        BY <1>2, <1>3, <3>1, <3>2, <3>3 DEF CrossWaitOk, MadeBy
    <2>2. \A t \in tasks : CrossTaskOk(t)'
      <3> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate" PROVE CrossTaskOk(t)'
        BY DEF CrossTaskOk
      <3>1. CrossTaskOk(t) /\ t.o # o
        BY <1>4, CrossingsOkEach DEF IndInv
      <3>2. t.o \in Owners /\ t.units \subseteq Units /\ t.writes \subseteq Units
        BY <3>1 DEF CrossTaskOk
      <3>3. ReadersOf(t.o)' = ReadersOf(t.o)
        BY <1>2, <3>1, <3>2 DEF ReadersOf
      <3> QED
        BY <1>2, <1>3, <3>1, <3>2, <3>3 DEF CrossTaskOk, Mem
    <2> QED
      BY <1>2, <2>1, <2>2, CrossingsOkEach
  <1>6. CopiesPlaced'
    BY <1>2, CopiesPlacedFrame DEF IndInv
  <1>7. CrossingsPlaced'
    <2>1. \A t \in tasks : t.kind = "replicate" => ogroup'[t.o] = ogroup[t.o]
      BY <1>2, <1>4, CrossingsOkEach DEF IndInv, CrossTaskOk
    <2> QED
      BY <1>2, <2>1 DEF IndInv, CrossingsPlaced, Mem
  <1> QED
    BY <1>5, <1>6, <1>7

LEMMA AttachInv ==
  ASSUME IndInv, NEW o \in Owners, NEW g \in Groups, Attach(o, g), UNCHANGED rare
  PROVE  IndInv'
  <1>1. PICK s \in Slots : AttachAt(o, g, s)
    BY AttachPick
  <1> QED
    BY <1>1, AttachPool, AttachOwners, AttachPlane, AttachCross DEF IndInv

-----------------------------------------------------------------------------
(* Allocate                                                                *)

(* Allocate(o, p) taking device unit u, with what IndInv says of o and u:  *)
(* nothing but the cache held u, and no batch in flight reads it.          *)
AllocateAt(o, p, u) ==
  /\ ostate[o] = "live" /\ table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
  /\ ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
  /\ u \in Units /\ u \in member /\ Unheld(u)
  /\ \A b \in inflight : \A x \in b.reads : x[2] # u
  /\ free' = free \ {u}
  /\ alloc' = [alloc EXCEPT ![u] = o]
  /\ sharedKey' = [sharedKey EXCEPT ![u] = NONE]
  /\ valid' = [valid EXCEPT ![u] = {Mem(ogroup[o])}]
  /\ cache' = [k \in Keys |-> IF cache[k] = u THEN NONE ELSE cache[k]]
  /\ DOMAIN table' = Owners
  /\ \A q \in Owners : DOMAIN table'[q] = Positions
  /\ \A q \in Owners, r \in Positions :
       table'[q][r] = IF q = o /\ r = p THEN Private(u) ELSE table[q][r]
  /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> u, w |-> TRUE])
  /\ UNCHANGED <<ostate, ogroup, oslot, hcache, deferred, copies, ncopy, tasks, trial, member, vis,
                 launched, inflight, requests>>

LEMMA AllocatePick ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, Allocate(o, p)
  PROVE  \E u \in Units : AllocateAt(o, p, u)
  <1>1. ostate[o] = "live" /\ table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
    BY DEF Allocate
  <1>2. ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
    BY <1>1 DEF IndInv, OwnersOk, Live
  <1>3. PICK u \in RegionUnits(o) :
          /\ Obtain(u, RegionUnits(o), OwnerEpoch[o])
          /\ free' = free \ {u}
          /\ table' = [table EXCEPT ![o][p] = Private(u)]
          /\ alloc' = [alloc EXCEPT ![u] = o]
          /\ sharedKey' = [sharedKey EXCEPT ![u] = NONE]
          /\ valid' = [valid EXCEPT ![u] = {Mem(ogroup[o])}]
          /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> u, w |-> TRUE])
    BY DEF Allocate
  <1>4. UNCHANGED <<ostate, ogroup, oslot, hcache, deferred, copies, ncopy, tasks, trial, member, vis,
                    launched, inflight, requests>>
    BY DEF Allocate
  <1>5. /\ u \in Units /\ u \in member /\ Unheld(u)
        /\ cache' = [k \in Keys |-> IF cache[k] = u THEN NONE ELSE cache[k]]
        /\ \A b \in inflight : \A x \in b.reads : x[2] # u
    BY <1>1, <1>3, ObtainFacts DEF Live
  <1>6. /\ DOMAIN table' = Owners
        /\ \A q \in Owners : DOMAIN table'[q] = Positions
        /\ \A q \in Owners, r \in Positions :
             table'[q][r] = IF q = o /\ r = p THEN Private(u) ELSE table[q][r]
    BY <1>3, OwnerTypes DEF OwnerTyped
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6 DEF AllocateAt

(* The units Allocate leaves held: those held before, and the new block.   *)
LEMMA AllocateHeld ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW u \in Units, AllocateAt(o, p, u)
  PROVE  \A v \in AllUnits : Held(v)' <=> Held(v) \/ v = u
  <1>1. \A v \in AllUnits :
          (\E q \in Owners, r \in Positions : table'[q][r].u = v)
            <=> (\E q \in Owners, r \in Positions : table[q][r].u = v) \/ v = u
    BY DEF AllocateAt, Empty, Private, AllUnits
  <1>2. \A v \in AllUnits : v # u =>
          ((\E j \in Keys : cache'[j] = v \/ hcache'[j] = v)
             <=> (\E j \in Keys : cache[j] = v \/ hcache[j] = v))
    BY DEF AllocateAt, AllUnits
  <1> QED
    BY <1>1, <1>2 DEF AllocateAt, Held, HeldIn

LEMMA AllocatePool ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW u \in Units, AllocateAt(o, p, u),
         UNCHANGED rare
  PROVE  /\ TypeInv' /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
         /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
  <1> DEFINE h == [g |-> ogroup[o], o |-> o, p |-> p, u |-> u, w |-> TRUE]
  <1>1. TypeOK
    BY TypeFacts
  <1>2. \A v \in AllUnits : Held(v)' <=> Held(v) \/ v = u
    BY AllocateHeld
  <1>3. u \in AllUnits /\ u # NONE /\ u \notin Slots /\ u \notin HostUnits
    BY DEF AllUnits, AllocateAt
  <1>4. \A v \in AllUnits : v # u => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
    BY <1>1 DEF AllocateAt, TypeOK
  <1>5. /\ alloc'[u] = o /\ sharedKey'[u] = NONE
        /\ \A j \in Keys : cache'[j] = IF cache[j] = u THEN NONE ELSE cache[j]
    BY <1>1, <1>3 DEF AllocateAt, TypeOK
  <1>6. TypeOK'
    <2>1. Mem(ogroup[o]) \in Memories
      BY DEF AllocateAt, Mem, Memories
    <2>2. \A q \in Owners, r \in Positions :
            table'[q][r].k \in {"none", "private", "shared"} /\ table'[q][r].u \in Units \cup {NONE}
      BY <1>1 DEF AllocateAt, TypeOK, Private
    <2>3. /\ valid' \in [Units \cup HostUnits -> SUBSET (Memories \cup HostParts)]
          /\ \A v \in Units : valid'[v] \subseteq Memories
          /\ \A v \in HostUnits : valid'[v] \subseteq HostParts
      BY <1>1, <1>3, <2>1 DEF AllocateAt, TypeOK
    <2> QED
      BY <1>1, <1>3, <2>1, <2>2, <2>3 DEF AllocateAt, TypeOK, rare, AllUnits
  <1>7. /\ DOMAIN dbt' = Groups
        /\ \A x \in Groups : DOMAIN dbt'[x] = Owners
        /\ \A x \in Groups, q \in Owners : DOMAIN dbt'[x][q] = Rows
        /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {h}
        /\ \A x \in Groups, q \in Owners, r \in Rows :
             dbt'[x][q][r] = IF Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h)
                             THEN [u |-> h.u, w |-> h.w]
                             ELSE dbt[x][q][r]
    <2>1. Give(h)
      BY DEF AllocateAt
    <2>2. h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows
      BY DEF Rows, AllocateAt
    <2> QED
      BY <2>1, <2>2, GiveRow, PlaneTypes DEF PlaneTyped
  <1>8. TypeInv'
    <2>1. \A x \in Groups, q \in Owners, r \in Rows : dbt'[x][q][r].u \in AllUnits \cup {NONE}
      BY <1>3, <1>7, PlaneTypes DEF PlaneTyped
    <2>2. \A x \in handoff' : x.g \in Groups /\ x.o \in Owners /\ x.p \in Rows /\ x.u \in AllUnits
      BY <1>3, <1>7, RequestTypes DEF RequestTyped, Rows, AllocateAt
    <2>3. \A q \in Owners, r \in Positions :
            table'[q][r] \in {Empty} \cup {Private(v) : v \in Units} \cup {Shared(v) : v \in Units}
      <3>1. \A q \in Owners, r \in Positions :
              table[q][r] \in {Empty} \cup {Private(v) : v \in Units} \cup {Shared(v) : v \in Units}
        BY DEF IndInv, TypeInv
      <3> QED
        BY <3>1 DEF AllocateAt
    <2> QED
      BY <1>6, <1>7, <2>1, <2>2, <2>3, RowTypes DEF AllocateAt, IndInv, rare
  <1>9. FreeClean'
    <2> SUFFICES ASSUME NEW v \in free' PROVE alloc'[v] = NONE /\ sharedKey'[v] = NONE /\ ~Held(v)'
      BY DEF FreeClean
    <2>1. v \in free /\ v # u /\ v \in AllUnits
      BY <1>1 DEF AllocateAt, TypeOK
    <2>2. alloc[v] = NONE /\ sharedKey[v] = NONE /\ ~Held(v)
      BY <2>1 DEF IndInv, FreeClean
    <2> QED
      BY <1>2, <1>4, <2>1, <2>2
  <1>10. NoLeak'
    <2> SUFFICES ASSUME NEW v \in AllUnits
                 PROVE  IF InRegion(v)' THEN v \in free' \/ Held(v)' ELSE v \notin free' /\ ~Held(v)'
      BY DEF NoLeak
    <2>1. CASE v = u
      BY <2>1, <1>2 DEF InRegion, AllocateAt
    <2>2. CASE v # u
      BY <2>2, <1>2 DEF InRegion, AllocateAt, IndInv, NoLeak
    <2> QED
      BY <2>1, <2>2
  <1>11. Outside'
    <2> SUFFICES ASSUME NEW v \in (Units \cup HostUnits) \ member'
                 PROVE  alloc'[v] = NONE /\ sharedKey'[v] = NONE
      BY DEF Outside
    <2>1. v # u /\ v \in AllUnits /\ v \in (Units \cup HostUnits) \ member
      BY DEF AllocateAt, AllUnits
    <2>2. alloc[v] = NONE /\ sharedKey[v] = NONE
      BY <2>1 DEF IndInv, Outside
    <2> QED
      BY <1>4, <2>1, <2>2
  <1>12. Regions'
    BY DEF AllocateAt, IndInv, Regions, OneRegion
  <1>13. AllocOwner' /\ SharedIff'
    BY <1>4, <1>5 DEF AllocateAt, IndInv, AllocOwner, SharedIff
  <1>14. SharedCached'
    <2> SUFFICES ASSUME NEW v \in AllUnits, alloc'[v] = "shared"
                 PROVE  cache'[sharedKey'[v]] = v \/ hcache'[sharedKey'[v]] = v
      BY DEF SharedCached
    <2>1. v # u /\ alloc[v] = "shared" /\ sharedKey'[v] = sharedKey[v]
      BY <1>4, <1>5
    <2>2. cache[sharedKey[v]] = v \/ hcache[sharedKey[v]] = v
      BY <2>1 DEF IndInv, SharedCached
    <2>3. sharedKey[v] \in Keys
      <3>1. sharedKey[v] # NONE
        BY <2>1 DEF IndInv, SharedIff
      <3>2. sharedKey \in [AllUnits -> Keys \cup {NONE}]
        BY <1>1 DEF TypeOK
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <2>1, <2>2, <2>3, <1>5 DEF AllocateAt
  <1>15. CacheShared'
    <2> SUFFICES ASSUME NEW j \in Keys
                 PROVE  /\ cache'[j] # NONE => cache'[j] \in Units /\ sharedKey'[cache'[j]] = j
                        /\ hcache'[j] # NONE => hcache'[j] \in HostUnits /\ sharedKey'[hcache'[j]] = j
      BY DEF CacheShared
    <2>1. hcache[j] # u
      BY DEF AllocateAt, Unheld
    <2>2. /\ cache[j] # NONE => cache[j] \in Units /\ sharedKey[cache[j]] = j
          /\ hcache[j] # NONE => hcache[j] \in HostUnits /\ sharedKey[hcache[j]] = j
      BY DEF IndInv, CacheShared
    <2> QED
      BY <1>4, <1>5, <2>1, <2>2 DEF AllocateAt, AllUnits
  <1> QED
    BY <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15

LEMMA AllocateOwners ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW u \in Units, AllocateAt(o, p, u),
         UNCHANGED rare
  PROVE  /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk' /\ TasksOk'
         /\ TrialVisibility' /\ NoBad'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. \A v \in AllUnits : v # u => alloc'[v] = alloc[v]
    BY <1>1 DEF AllocateAt, TypeOK
  <1>3. \A v \in Units \cup HostUnits : v # u => valid'[v] = valid[v]
    BY <1>1 DEF AllocateAt, TypeOK
  <1>4. alloc'[u] = o /\ valid'[u] = {Mem(ogroup[o])}
    BY <1>1 DEF AllocateAt, TypeOK, AllUnits
  <1>5. \A j \in Keys : cache'[j] = IF cache[j] = u THEN NONE ELSE cache[j]
    BY DEF AllocateAt
  <1>6. \A q \in Owners, r \in Positions : table[q][r].u # u
    BY DEF AllocateAt, Unheld
  <1>7. Tables'
    <2> SUFFICES ASSUME NEW q \in Owners, NEW r \in Positions
                 PROVE  /\ table'[q][r].k = "none" => table'[q][r].u = NONE
                        /\ table'[q][r].k = "private" =>
                             table'[q][r].u \in Units /\ alloc'[table'[q][r].u] = q
                        /\ table'[q][r].k = "shared" =>
                             table'[q][r].u \in Units /\ alloc'[table'[q][r].u] = "shared"
                        /\ table'[q][r].k # "none" => ostate'[q] = "live"
                        /\ \A r2 \in Positions :
                             (table'[q][r].k = "private" /\ table'[q][r2].u = table'[q][r].u) => r = r2
                        /\ table'[q][r].u # NONE => Mem(ogroup'[q]) \in valid'[table'[q][r].u]
      BY DEF Tables, Live
    <2>1. CASE q = o /\ r = p
      <3>1. table'[q][r] = Private(u)
        BY <2>1 DEF AllocateAt
      <3>2. \A r2 \in Positions : r2 # p => table'[o][r2].u # u
        BY <1>6 DEF AllocateAt
      <3> QED
        BY <2>1, <3>1, <3>2, <1>4 DEF AllocateAt, Private
    <2>2. CASE ~(q = o /\ r = p)
      <3>1. table'[q][r] = table[q][r]
        BY <2>2 DEF AllocateAt
      <3>2. /\ table[q][r].k = "none" => table[q][r].u = NONE
            /\ table[q][r].k = "private" => table[q][r].u \in Units /\ alloc[table[q][r].u] = q
            /\ table[q][r].k = "shared" => table[q][r].u \in Units /\ alloc[table[q][r].u] = "shared"
            /\ table[q][r].k # "none" => ostate[q] = "live"
            /\ \A r2 \in Positions :
                 (table[q][r].k = "private" /\ table[q][r2].u = table[q][r].u) => r = r2
            /\ table[q][r].u # NONE => Mem(ogroup[q]) \in valid[table[q][r].u]
        BY DEF IndInv, Tables, Live
      <3>3. table[q][r].u \in Units \cup {NONE}
        BY <1>1 DEF TypeOK
      <3>4. \A r2 \in Positions : table'[q][r2].u = table[q][r2].u \/ table'[q][r2].u = u
        BY DEF AllocateAt, Private
      <3> QED
        BY <3>1, <3>2, <3>3, <3>4, <1>2, <1>3, <1>6 DEF AllocateAt, AllUnits
    <2> QED
      BY <2>1, <2>2
  <1>8. OwnersOk'
    <2>1. \A q \in Owners : oslot[q] \in Slots \cup {NONE}
      BY <1>1 DEF TypeOK
    <2>2. \A q \in Owners : q # o => \A r \in Positions : table'[q][r] = table[q][r]
      BY DEF AllocateAt
    <2> QED
      BY <1>2, <2>1, <2>2 DEF AllocateAt, IndInv, OwnersOk, Live, AllUnits, rare
  <1>9. DeferredOk'
    <2>1. \A d \in deferred : u \notin d.units
      BY DEF AllocateAt, Unheld
    <2>2. \A d \in deferred : d.kind = "commit" => d.units \subseteq Units
      BY DEF IndInv, DeferredOk
    <2>3. \A q \in Owners, r \in Positions : table'[q][r].u = table[q][r].u \/ table'[q][r].u = u
      BY DEF AllocateAt, Private
    <2>4. \A q \in Owners, r \in Positions :
            table'[q][r].k = "private" => table[q][r].k = "private" \/ table'[q][r].u = u
      BY DEF AllocateAt, Private
    <2>5. \A q \in Owners, r \in Positions : table'[q][r].u # u => table'[q][r] = table[q][r]
      BY DEF AllocateAt, Private
    <2> QED
      BY <1>2, <2>1, <2>2, <2>3, <2>4, <2>5 DEF AllocateAt, IndInv, DeferredOk, rare, AllUnits
  <1>10. CopiesOk'
    <2>1. \A c \in copies : c.read # u /\ c.write # u
      BY DEF AllocateAt, Unheld
    <2>2. \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
      BY DEF IndInv, TypeInv
    <2>3. \A c \in copies : c.kind \in {"clone", "load"} => c.o \in Owners /\ c.p \in Positions
      BY DEF IndInv, CopiesOk
    <2>4. \A c \in copies : c.kind \in {"clone", "load"} => (c.o # o \/ c.p # p)
      BY DEF AllocateAt, CopyFor
    <2>5. \A c \in copies : c.kind \in {"clone", "load"} => table'[c.o][c.p] = table[c.o][c.p]
      BY <2>3, <2>4 DEF AllocateAt
    <2>6. \A c \in copies : ~c.fill => \A q \in Owners, r \in Positions : table'[q][r].u # c.write
      <3> SUFFICES ASSUME NEW c \in copies, ~c.fill, NEW q \in Owners, NEW r \in Positions
                   PROVE  table'[q][r].u # c.write
        OBVIOUS
      <3>1. table[q][r].u # c.write
        BY DEF IndInv, CopiesOk
      <3> QED
        BY <3>1, <2>1 DEF AllocateAt, Private
    <2>7. \A c \in copies : c.fill /\ c.kind = "load" =>
            cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write]
      <3> SUFFICES ASSUME NEW c \in copies, c.fill, c.kind = "load"
                   PROVE  cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write]
        OBVIOUS
      <3>1. cache[c.k] = c.write /\ c.k \in Keys
        BY DEF IndInv, CopiesOk, TypeInv
      <3>2. c.write \in Units
        BY <3>1, <2>2, <1>1 DEF TypeOK, AllUnits
      <3> QED
        BY <3>1, <3>2, <2>1, <1>3, <1>5
    <2>8. \A c \in copies : alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
      BY <1>2, <2>1, <2>2
    <2>9. Transfers' = Transfers /\ TaskUnits' \subseteq TaskUnits
      BY DEF AllocateAt, Transfers, TaskUnits
    <2>10. \A c \in copies : c.fill /\ c.kind = "store" => hcache'[c.k] = hcache[c.k]
      BY DEF AllocateAt
    <2> QED
      BY <2>5, <2>6, <2>7, <2>8, <2>9, <2>10, CopiesStay DEF AllocateAt, IndInv, rare
  <1>11. TasksOk'
    <2>1. \A t \in tasks : t.slot # u /\ t.h # u /\ u \notin t.units
      BY DEF AllocateAt, Unheld
    <2>2. \A t \in tasks : t.kind = "transfer" => t.o # o
      BY DEF AllocateAt, Moving
    <2>3. \A t \in tasks : /\ t.slot \in Slots \cup {NONE} /\ t.h \in HostUnits \cup {NONE}
                           /\ t.units \subseteq AllUnits /\ t.o \in Owners \cup {NONE}
      BY DEF IndInv, TypeInv
    <2>4. \A q \in Owners : q # o => TableUnits(q)' = TableUnits(q)
      BY DEF AllocateAt, TableUnits
    <2>5. \A t \in tasks : t.kind = "transfer" => t.writes \subseteq t.units \cap Units /\ t.units \subseteq Units \cup Slots
      BY DEF IndInv, TasksOk
    <2>6. \A q \in Owners : q # o => u \notin TableUnits(q)
      BY DEF AllocateAt, Unheld, TableUnits
    <2>7. \A t \in tasks : t.kind = "transfer" =>
            /\ alloc'[t.slot] = alloc[t.slot]
            /\ \A v \in t.writes : alloc'[v] = alloc[v] /\ valid'[v] = valid[v]
            /\ ostate[t.o] = "live" =>
                 /\ TableUnits(t.o)' = TableUnits(t.o)
                 /\ \A v \in TableUnits(t.o) : valid'[v] = valid[v]
      <3> SUFFICES ASSUME NEW t \in tasks, t.kind = "transfer"
                   PROVE  /\ alloc'[t.slot] = alloc[t.slot]
                          /\ \A v \in t.writes : alloc'[v] = alloc[v] /\ valid'[v] = valid[v]
                          /\ ostate[t.o] = "live" =>
                               /\ TableUnits(t.o)' = TableUnits(t.o)
                               /\ \A v \in TableUnits(t.o) : valid'[v] = valid[v]
        OBVIOUS
      <3>1. t.slot \in Slots /\ t.o \in Owners /\ t.o # o
        BY <2>2 DEF IndInv, TasksOk
      <3>2. TableUnits(t.o) \subseteq Units
        BY <3>1, <1>1 DEF TableUnits, TypeOK
      <3> QED
        BY <3>1, <3>2, <2>1, <2>4, <2>5, <2>6, <1>2, <1>3 DEF AllUnits, TableUnits
    <2>8. \A t \in tasks : t.kind = "prefetch" => alloc'[t.h] = alloc[t.h]
      <3>1. \A t \in tasks : t.kind = "prefetch" => t.h \in HostUnits
        BY DEF IndInv, TasksOk
      <3> QED
        BY <3>1, <2>1, <1>2 DEF AllUnits
    <2> QED
      BY <2>7, <2>8, TasksKeep DEF AllocateAt, IndInv
  <1>12. TrialVisibility'
    <2>1. \A j \in Keys : cache'[j] # NONE => cache'[j] = cache[j]
      BY <1>5
    <2>2. \A q \in Owners, r \in Positions : \A v : table'[q][r] = Shared(v) => table[q][r] = Shared(v)
      BY DEF AllocateAt, Private, Shared
    (* A block a crossing holds that is cached after the step was cached    *)
    (* before it.                                                          *)
    <2>3. \A t \in tasks : NONE \notin t.units
      BY WorkTypes DEF WorkTyped, AllUnits
    <2>4. \A k \in Keys, t \in tasks : cache'[k] \in t.units => cache'[k] = cache[k]
      BY <2>1, <2>3
    <2> QED
      BY <2>1, <2>2, <2>4 DEF AllocateAt, IndInv, TrialVisibility, MadeBy
  <1>13. NoBad'
    BY NoBadFrame DEF IndInv, rare
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13

LEMMA AllocatePlane ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW u \in Units, AllocateAt(o, p, u),
         UNCHANGED rare
  PROVE  /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
         /\ BatchesOk'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
    <2>1. /\ \A q \in Owners, r \in Positions : (q # o \/ r # p) => table'[q][r] = table[q][r]
          /\ table'[o][p].u = u /\ table'[o][p].k = "private"
          /\ \A d \in deferred : u \notin d.units
      BY DEF AllocateAt, Private, Unheld
    <2>2. Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> u, w |-> TRUE])
      BY DEF AllocateAt
    <2>3. UNCHANGED <<ostate, ogroup, oslot, deferred, tasks, requests, lost, inflight, launched,
                      ncopy, copies>>
      BY DEF AllocateAt, rare
    (* The new block is held by no task; a block a table holds keeps its   *)
    (* validity, the new one being in no table.                            *)
    <2>4. /\ \A t \in tasks' : u \notin t.units
          /\ \A t \in tasks' : t.kind \in {"transfer", "replicate"} => t \in tasks
          /\ \A v \in Units : (\E q \in Owners, r \in Positions : table[q][r].u = v) =>
                                valid[v] \subseteq valid'[v]
      <3>1. \A v \in Units : v # u => valid'[v] = valid[v]
        BY <1>1 DEF AllocateAt, TypeOK
      <3> QED
        BY <2>3, <3>1 DEF AllocateAt, Unheld
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, FillPlane DEF AllocateAt
  <1>3. BatchesOk'
    <2>1. \A b \in inflight : \A x \in b.reads : x[2] # u
      BY DEF AllocateAt
    <2>2. \A b \in inflight : b.writes \subseteq b.reads
      BY DEF IndInv, BatchesOk
    <2>3. \A b \in inflight : \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits
      BY RequestTypes DEF IndInv, BatchesOk, RequestTyped
    <2>4. \A b \in inflight : \A x \in b.reads : Covered(b, x[1], x[2])'
      <3> SUFFICES ASSUME NEW b \in inflight, NEW x \in b.reads PROVE Covered(b, x[1], x[2])'
        OBVIOUS
      <3>1. Covered(b, x[1], x[2])
        BY DEF IndInv, BatchesOk
      <3>2. \A r \in Rows : HoldsAt(x[1], r, x[2]) => HoldsAt(x[1], r, x[2])'
        <4> SUFFICES ASSUME NEW r \in Rows, HoldsAt(x[1], r, x[2]) PROVE HoldsAt(x[1], r, x[2])'
          OBVIOUS
        <4>1. CASE r = SLOT
          BY <4>1 DEF HoldsAt, AllocateAt
        <4>2. CASE r \in Positions
          <5>1. ~(x[1] = o /\ r = p)
            BY <4>2, <2>3 DEF HoldsAt, AllocateAt, Empty, AllUnits
          <5> QED
            BY <4>2, <5>1, <2>3 DEF HoldsAt, AllocateAt, Rows
        <4> QED
          BY <4>1, <4>2 DEF Rows
      <3> QED
        BY <3>1, <3>2, CoverStays DEF AllocateAt
    <2>5. \A v \in AllUnits : v # u => alloc'[v] = alloc[v]
      BY <1>1 DEF AllocateAt, TypeOK
    <2>6. \A v \in Units : v # u => valid'[v] = valid[v]
      BY <1>1 DEF AllocateAt, TypeOK
    <2>7. \A b \in inflight : \A x \in b.writes : \A r \in Positions :
            table'[x[1]][r].u = x[2] => table[x[1]][r].u = x[2]
      BY <2>1, <2>2, <2>3 DEF AllocateAt, Private
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7 DEF IndInv, BatchesOk, AllocateAt, Mem, rare
  <1> QED
    BY <1>2, <1>3

(* The new block is held by nothing a crossing or a copy names, and only   *)
(* its validity changes.                                                   *)
LEMMA AllocateCross ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW u \in Units, AllocateAt(o, p, u),
         UNCHANGED rare
  PROVE  CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
  <1>1. TypeOK
    BY TypeFacts
  <1>2. /\ \A v \in AllUnits : v # u => alloc'[v] = alloc[v]
        /\ \A v \in Units \cup HostUnits : v # u => valid'[v] = valid[v]
        /\ \A j \in Keys : cache'[j] # NONE => cache'[j] = cache[j]
        /\ UNCHANGED <<ostate, ogroup, deferred, copies, tasks, vis, inflight, lost, trial>>
    BY <1>1 DEF AllocateAt, TypeOK, rare
  <1>3. /\ \A d \in deferred : u \notin d.units
        /\ \A c \in copies : c.read # u /\ c.write # u
        /\ \A t \in tasks : u \notin t.units
    BY DEF AllocateAt, Unheld
  <1>4. \A q \in Owners : Moving(q)' = Moving(q)
    BY <1>2 DEF Moving
  <1>5. CrossingsOk'
    <2>1. \A d \in deferred : CrossWaitOk(d)'
      <3> SUFFICES ASSUME NEW d \in deferred, d.kind = "replicate" PROVE CrossWaitOk(d)'
        BY DEF CrossWaitOk
      <3>1. CrossWaitOk(d)
        BY CrossingsOkEach DEF IndInv
      <3>2. /\ d.units \subseteq Units /\ u \notin d.units
            /\ \A v \in d.units : alloc'[v] = alloc[v]
        BY <1>2, <1>3, <3>1 DEF CrossWaitOk, AllUnits
      <3>3. \A k \in Keys : cache'[k] \in d.units => cache[k] \in d.units
        BY <1>2, <3>2 DEF NONE
      <3> QED
        BY <1>2, <1>4, <3>1, <3>2, <3>3 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>2. \A t \in tasks : CrossTaskOk(t)'
      <3> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate" PROVE CrossTaskOk(t)'
        BY DEF CrossTaskOk
      <3>1. CrossTaskOk(t)
        BY CrossingsOkEach DEF IndInv
      <3>2. /\ t.writes \subseteq Units /\ u \notin t.writes
            /\ \A v \in t.writes : alloc'[v] = alloc[v] /\ valid'[v] = valid[v]
        BY <1>2, <1>3, <3>1 DEF CrossTaskOk, AllUnits
      <3> QED
        BY <1>2, <1>4, <3>1, <3>2 DEF CrossTaskOk, ReadersOf, Mem
    <2> QED
      BY <1>2, <2>1, <2>2, CrossingsOkEach
  <1>6. CopiesPlaced'
    <2> SUFFICES ASSUME NEW c \in copies PROVE CopyPlaced(c)'
      BY <1>2, PlacedEach
    <2>1. CopyPlaced(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
      BY PlacedEach DEF IndInv, TypeInv
    <2>2. /\ c.read \in Units \cup HostUnits => valid'[c.read] = valid[c.read]
          /\ c.write \in Units \cup HostUnits => valid'[c.write] = valid[c.write]
      BY <1>2, <1>3
    <2> QED
      BY <1>2, <2>1, <2>2 DEF CopyPlaced, Mem
  <1>7. CrossingsPlaced'
    <2>1. \A t \in tasks : t.kind = "replicate" =>
            \A v \in t.units : valid'[v] = valid[v]
      BY <1>2, <1>3, CrossingsOkEach DEF IndInv, CrossTaskOk
    <2> QED
      BY <1>2, <2>1 DEF IndInv, CrossingsPlaced, Mem
  <1> QED
    BY <1>5, <1>6, <1>7

LEMMA AllocateInv ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, Allocate(o, p), UNCHANGED rare
  PROVE  IndInv'
  <1>1. PICK u \in Units : AllocateAt(o, p, u)
    BY AllocatePick
  <1> QED
    BY <1>1, AllocatePool, AllocateOwners, AllocatePlane, AllocateCross DEF IndInv

-----------------------------------------------------------------------------
(* Pin                                                                     *)

(* Pin(o, p, k), with what IndInv says of the cached block it pins.        *)
PinAt(o, p, k) ==
  /\ ostate[o] = "live" /\ table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
  /\ ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
  /\ cache[k] \in Units /\ cache[k] \notin free
  /\ alloc[cache[k]] = "shared" /\ sharedKey[cache[k]] = k
  /\ \A j \in Keys : cache[j] = cache[k] => j = k
  /\ Mem(ogroup[o]) \in valid[cache[k]]
  /\ vis[k].d \in {NONE, OwnerEpoch[o]}
  /\ DOMAIN table' = Owners
  /\ \A q \in Owners : DOMAIN table'[q] = Positions
  /\ \A q \in Owners, r \in Positions :
       table'[q][r] = IF q = o /\ r = p THEN Shared(cache[k]) ELSE table[q][r]
  /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> cache[k], w |-> FALSE])
  /\ UNCHANGED <<free, ostate, ogroup, oslot, cache, hcache, deferred, copies, ncopy, tasks, trial,
                 member, valid, vis, launched, inflight, requests, alloc, sharedKey>>

LEMMA PinPick ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, Pin(o, p, k)
  PROVE  PinAt(o, p, k)
  <1>1. /\ ostate[o] = "live" /\ cache[k] # NONE /\ Mem(ogroup[o]) \in valid[cache[k]]
        /\ VisibleDevice(k, o)
        /\ table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
        /\ table' = [table EXCEPT ![o][p] = Shared(cache[k])]
        /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> cache[k], w |-> FALSE])
        /\ UNCHANGED <<free, ostate, ogroup, oslot, cache, hcache, deferred, copies, ncopy, tasks,
                       trial, member, valid, vis, launched, inflight, requests, alloc, sharedKey>>
    BY DEF Pin
  <1>2. ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
    BY <1>1 DEF IndInv, OwnersOk, Live
  <1>3. /\ cache[k] \in Units /\ cache[k] \notin free
        /\ alloc[cache[k]] = "shared" /\ sharedKey[cache[k]] = k
        /\ \A j \in Keys : cache[j] = cache[k] => j = k
    BY <1>1, DeviceCached
  <1>4. vis[k].d \in {NONE, OwnerEpoch[o]}
    BY <1>1 DEF VisibleDevice
  <1>5. /\ DOMAIN table' = Owners
        /\ \A q \in Owners : DOMAIN table'[q] = Positions
        /\ \A q \in Owners, r \in Positions :
             table'[q][r] = IF q = o /\ r = p THEN Shared(cache[k]) ELSE table[q][r]
    BY <1>1, OwnerTypes DEF OwnerTyped
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5 DEF PinAt

(* Pin leaves held what was held: the block it pins is cached.             *)
LEMMA PinHeld ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, PinAt(o, p, k)
  PROVE  \A v \in AllUnits : Held(v)' <=> Held(v)
  <1>1. \A v \in AllUnits :
          (\E q \in Owners, r \in Positions : table'[q][r].u = v)
            <=> (\E q \in Owners, r \in Positions : table[q][r].u = v) \/ v = cache[k]
    BY DEF PinAt, Empty, Shared, AllUnits
  <1> QED
    BY <1>1 DEF PinAt, Held, HeldIn

LEMMA PinPool ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, PinAt(o, p, k),
         UNCHANGED rare
  PROVE  /\ TypeInv' /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
         /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
  <1> DEFINE h == [g |-> ogroup[o], o |-> o, p |-> p, u |-> cache[k], w |-> FALSE]
  <1>1. TypeOK
    BY TypeFacts
  <1>2. \A v \in AllUnits : Held(v)' <=> Held(v)
    BY PinHeld
  <1>3. cache[k] \in AllUnits
    BY DEF AllUnits, PinAt
  <1>4. TypeOK'
    <2>1. \A q \in Owners, r \in Positions :
            table'[q][r].k \in {"none", "private", "shared"} /\ table'[q][r].u \in Units \cup {NONE}
      BY <1>1 DEF PinAt, TypeOK, Shared
    <2> QED
      BY <1>1, <2>1 DEF PinAt, TypeOK, rare
  <1>5. /\ DOMAIN dbt' = Groups
        /\ \A x \in Groups : DOMAIN dbt'[x] = Owners
        /\ \A x \in Groups, q \in Owners : DOMAIN dbt'[x][q] = Rows
        /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {h}
        /\ \A x \in Groups, q \in Owners, r \in Rows :
             dbt'[x][q][r] = IF Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h)
                             THEN [u |-> h.u, w |-> h.w]
                             ELSE dbt[x][q][r]
    <2>1. Give(h)
      BY DEF PinAt
    <2>2. h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows
      BY DEF Rows, PinAt
    <2> QED
      BY <2>1, <2>2, GiveRow, PlaneTypes DEF PlaneTyped
  <1>6. TypeInv'
    <2>1. \A x \in Groups, q \in Owners, r \in Rows : dbt'[x][q][r].u \in AllUnits \cup {NONE}
      BY <1>3, <1>5, PlaneTypes DEF PlaneTyped
    <2>2. \A x \in handoff' : x.g \in Groups /\ x.o \in Owners /\ x.p \in Rows /\ x.u \in AllUnits
      BY <1>3, <1>5, RequestTypes DEF RequestTyped, Rows, PinAt
    <2>3. \A q \in Owners, r \in Positions :
            table'[q][r] \in {Empty} \cup {Private(v) : v \in Units} \cup {Shared(v) : v \in Units}
      <3>1. \A q \in Owners, r \in Positions :
              table[q][r] \in {Empty} \cup {Private(v) : v \in Units} \cup {Shared(v) : v \in Units}
        BY DEF IndInv, TypeInv
      <3> QED
        BY <3>1 DEF PinAt
    <2> QED
      BY <1>4, <1>5, <2>1, <2>2, <2>3, RowTypes DEF PinAt, IndInv, rare
  <1>7. FreeClean'
    <2> SUFFICES ASSUME NEW v \in free' PROVE alloc'[v] = NONE /\ sharedKey'[v] = NONE /\ ~Held(v)'
      BY DEF FreeClean
    <2>1. v \in free /\ v \in AllUnits
      BY <1>1 DEF PinAt, TypeOK
    <2>2. alloc[v] = NONE /\ sharedKey[v] = NONE /\ ~Held(v)
      BY <2>1 DEF IndInv, FreeClean
    <2> QED
      BY <1>2, <2>1, <2>2 DEF PinAt
  <1>8. NoLeak'
    BY <1>2 DEF PinAt, IndInv, NoLeak, InRegion
  <1>9. Outside' /\ Regions' /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. UNCHANGED <<member, alloc, sharedKey, free, ostate, cache, hcache>>
      BY DEF PinAt
    <2> QED
      BY <2>1, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame, SharedCachedFrame,
         CacheSharedFrame DEF IndInv
  <1> QED
    BY <1>6, <1>7, <1>8, <1>9

LEMMA PinOwners ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, PinAt(o, p, k),
         UNCHANGED rare
  PROVE  /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk' /\ TasksOk'
         /\ TrialVisibility' /\ NoBad'
  <1>1. TypeOK
    BY TypeFacts
  <1> DEFINE v == cache[k]
  <1>2. \A q \in Owners, r \in Positions : table[q][r].k = "private" => table[q][r].u # v
    BY DEF PinAt, IndInv, Tables
  <1>3. Tables'
    <2> SUFFICES ASSUME NEW q \in Owners, NEW r \in Positions
                 PROVE  /\ table'[q][r].k = "none" => table'[q][r].u = NONE
                        /\ table'[q][r].k = "private" =>
                             table'[q][r].u \in Units /\ alloc'[table'[q][r].u] = q
                        /\ table'[q][r].k = "shared" =>
                             table'[q][r].u \in Units /\ alloc'[table'[q][r].u] = "shared"
                        /\ table'[q][r].k # "none" => ostate'[q] = "live"
                        /\ \A r2 \in Positions :
                             (table'[q][r].k = "private" /\ table'[q][r2].u = table'[q][r].u) => r = r2
                        /\ table'[q][r].u # NONE => Mem(ogroup'[q]) \in valid'[table'[q][r].u]
      BY DEF Tables, Live
    <2>1. CASE q = o /\ r = p
      BY <2>1 DEF PinAt, Shared
    <2>2. CASE ~(q = o /\ r = p)
      <3>1. table'[q][r] = table[q][r]
        BY <2>2 DEF PinAt
      <3>2. /\ table[q][r].k = "none" => table[q][r].u = NONE
            /\ table[q][r].k = "private" => table[q][r].u \in Units /\ alloc[table[q][r].u] = q
            /\ table[q][r].k = "shared" => table[q][r].u \in Units /\ alloc[table[q][r].u] = "shared"
            /\ table[q][r].k # "none" => ostate[q] = "live"
            /\ \A r2 \in Positions :
                 (table[q][r].k = "private" /\ table[q][r2].u = table[q][r].u) => r = r2
            /\ table[q][r].u # NONE => Mem(ogroup[q]) \in valid[table[q][r].u]
        BY DEF IndInv, Tables, Live
      <3>3. \A r2 \in Positions : table'[q][r2].u = table[q][r2].u \/ (q = o /\ r2 = p /\ table'[q][r2].u = v)
        BY DEF PinAt, Shared
      <3> QED
        BY <3>1, <3>2, <3>3, <1>2 DEF PinAt
    <2> QED
      BY <2>1, <2>2
  <1>4. OwnersOk'
    <2>1. \A q \in Owners : q # o => \A r \in Positions : table'[q][r] = table[q][r]
      BY DEF PinAt
    <2> QED
      BY <2>1 DEF PinAt, IndInv, OwnersOk, Live, rare
  <1>5. DeferredOk'
    <2>1. \A d \in deferred : d.kind = "commit" => v \notin d.units
      BY DEF PinAt, IndInv, DeferredOk
    <2>2. \A q \in Owners, r \in Positions : table'[q][r] = table[q][r] \/ table'[q][r] = Shared(v)
      BY DEF PinAt
    <2>3. \A q \in Owners, r \in Positions : table'[q][r].u # table[q][r].u => table'[q][r].u = v
      BY DEF PinAt, Shared
    <2> QED
      BY <2>1, <2>2, <2>3 DEF PinAt, IndInv, DeferredOk, rare, Shared
  <1>6. CopiesOk'
    <2>1. \A c \in copies : ~c.fill => c.write # v
      <3> SUFFICES ASSUME NEW c \in copies, ~c.fill, c.write = v PROVE FALSE
        OBVIOUS
      <3>1. /\ c.kind \in {"clone", "load", "store", "save"}
            /\ c.kind \in {"clone", "load"} => alloc[c.write] = c.o /\ c.o \in Owners
            /\ c.kind \in {"store", "save"} => c.write \in HostUnits
        BY DEF IndInv, CopiesOk
      <3> QED
        BY <3>1 DEF PinAt
    <2>2. \A c \in copies : c.kind \in {"clone", "load"} => c.o \in Owners /\ c.p \in Positions
      BY DEF IndInv, CopiesOk
    <2>3. \A c \in copies : c.kind \in {"clone", "load"} => (c.o # o \/ c.p # p)
      BY DEF PinAt, CopyFor
    <2>4. \A c \in copies : c.kind \in {"clone", "load"} => table'[c.o][c.p] = table[c.o][c.p]
      BY <2>2, <2>3 DEF PinAt
    <2>5. \A c \in copies : ~c.fill => \A q \in Owners, r \in Positions : table'[q][r].u # c.write
      <3> SUFFICES ASSUME NEW c \in copies, ~c.fill, NEW q \in Owners, NEW r \in Positions
                   PROVE  table'[q][r].u # c.write
        OBVIOUS
      <3>1. table[q][r].u # c.write
        BY DEF IndInv, CopiesOk
      <3> QED
        BY <3>1, <2>1 DEF PinAt, Shared
    <2>6. Transfers' = Transfers /\ TaskUnits' \subseteq TaskUnits
      BY DEF PinAt, Transfers, TaskUnits
    <2>7. \A c \in copies : c.fill /\ c.kind = "store" => hcache'[c.k] = hcache[c.k]
      BY DEF PinAt
    <2> QED
      BY <2>4, <2>5, <2>6, <2>7, CopiesStay DEF PinAt, IndInv, rare
  <1>7. TasksOk'
    <2>1. \A t \in tasks : t.kind = "transfer" => t.o # o /\ t.o \in Owners
      BY DEF PinAt, Moving, IndInv, TasksOk
    <2>2. \A q \in Owners : q # o => TableUnits(q)' = TableUnits(q)
      BY DEF PinAt, TableUnits
    <2> QED
      BY <2>1, <2>2, TasksKeep DEF PinAt, IndInv
  <1>8. TrialVisibility'
    <2>1. \A j \in Keys : (cache[j] # NONE /\ vis[j].d # NONE) =>
            \A q \in Owners, r \in Positions : table'[q][r] = Shared(cache[j]) => OwnerEpoch[q] = vis[j].d
      <3> SUFFICES ASSUME NEW j \in Keys, cache[j] # NONE, vis[j].d # NONE,
                          NEW q \in Owners, NEW r \in Positions, table'[q][r] = Shared(cache[j])
                   PROVE  OwnerEpoch[q] = vis[j].d
        OBVIOUS
      <3>1. CASE q = o /\ r = p
        <4>1. cache[j] = v
          BY <3>1 DEF PinAt, Shared
        <4>2. j = k
          BY <4>1 DEF PinAt
        <4> QED
          BY <3>1, <4>2 DEF PinAt
      <3>2. CASE ~(q = o /\ r = p)
        <4>1. table[q][r] = Shared(cache[j])
          BY <3>2 DEF PinAt
        <4> QED
          BY <4>1 DEF IndInv, TrialVisibility
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <2>1 DEF TrialVisibility, PinAt, IndInv, MadeBy
  <1>9. NoBad'
    BY NoBadFrame DEF IndInv, rare
  <1> QED
    BY <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9

LEMMA PinPlane ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, PinAt(o, p, k),
         UNCHANGED rare
  PROVE  /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
         /\ BatchesOk'
  <1> DEFINE v == cache[k]
  <1>1. TypeOK
    BY TypeFacts
  <1>2. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
    <2>1. /\ \A q \in Owners, r \in Positions : (q # o \/ r # p) => table'[q][r] = table[q][r]
          /\ table'[o][p].u = v
          /\ v \in Units
      BY DEF PinAt, Shared
    <2>2. Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> v, w |-> FALSE])
      BY DEF PinAt
    <2>3. UNCHANGED <<ostate, ogroup, oslot, deferred, tasks, requests, lost, inflight, launched,
                      ncopy, copies, valid>>
      BY DEF PinAt, rare
    <2>4. /\ \A t \in tasks' : t.kind \in {"transfer", "replicate"} => t \in tasks
          /\ \A x \in Units : valid[x] \subseteq valid'[x]
      BY <2>3
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, FillPlane DEF PinAt
  <1>3. BatchesOk'
    <2>1. \A b \in inflight : b.writes \subseteq b.reads
      BY DEF IndInv, BatchesOk
    <2>2. \A b \in inflight : \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits
      BY RequestTypes DEF IndInv, BatchesOk, RequestTyped
    <2>3. \A b \in inflight : \A x \in b.reads : Covered(b, x[1], x[2])'
      <3> SUFFICES ASSUME NEW b \in inflight, NEW x \in b.reads PROVE Covered(b, x[1], x[2])'
        OBVIOUS
      <3>1. Covered(b, x[1], x[2])
        BY DEF IndInv, BatchesOk
      <3>2. \A r \in Rows : HoldsAt(x[1], r, x[2]) => HoldsAt(x[1], r, x[2])'
        <4> SUFFICES ASSUME NEW r \in Rows, HoldsAt(x[1], r, x[2]) PROVE HoldsAt(x[1], r, x[2])'
          OBVIOUS
        <4>1. CASE r = SLOT
          BY <4>1 DEF HoldsAt, PinAt
        <4>2. CASE r \in Positions
          <5>1. ~(x[1] = o /\ r = p)
            BY <4>2, <2>2 DEF HoldsAt, PinAt, Empty, AllUnits
          <5> QED
            BY <4>2, <5>1, <2>2 DEF HoldsAt, PinAt, Rows
        <4> QED
          BY <4>1, <4>2 DEF Rows
      <3> QED
        BY <3>1, <3>2, CoverStays DEF PinAt
    <2>4. \A b \in inflight : \A x \in b.writes : \A r \in Positions :
            table'[x[1]][r].u = x[2] => table[x[1]][r].u = x[2]
      <3> SUFFICES ASSUME NEW b \in inflight, NEW x \in b.writes, NEW r \in Positions,
                          table'[x[1]][r].u = x[2]
                   PROVE  table[x[1]][r].u = x[2]
        OBVIOUS
      <3>1. alloc[x[2]] = x[1] /\ x[1] \in Owners
        BY <2>1, <2>2 DEF IndInv, BatchesOk
      <3>2. x[2] # v
        BY <3>1 DEF PinAt
      <3> QED
        BY <3>2, <2>1, <2>2 DEF PinAt, Shared
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4 DEF IndInv, BatchesOk, PinAt, Mem, rare
  <1> QED
    BY <1>2, <1>3

(* Pin changes only an owner's table, an executor's table and the          *)
(* completions: nothing a crossing or a copy reads.                        *)
LEMMA PinCross ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, PinAt(o, p, k),
         UNCHANGED rare
  PROVE  CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
  <1>1. UNCHANGED <<deferred, tasks, copies, ostate, ogroup, lost, alloc, cache, vis, valid,
                    inflight, trial>>
    BY DEF PinAt, rare
  <1> QED
    BY <1>1, TransfersFrame, CrossingsOkFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv

LEMMA PinInv ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, Pin(o, p, k), UNCHANGED rare
  PROVE  IndInv'
  <1>1. PinAt(o, p, k)
    BY PinPick
  <1> QED
    BY <1>1, PinPool, PinOwners, PinPlane, PinCross DEF IndInv

-----------------------------------------------------------------------------
(* The group                                                               *)

Next1 ==
  /\ \/ \E o \in Owners, g \in Groups : Attach(o, g)
     \/ \E o \in Owners, p \in Positions : Allocate(o, p)
     \/ \E o \in Owners, p \in Positions, k \in Keys : Pin(o, p, k)
  /\ UNCHANGED rare

THEOREM Step1 == IndInv /\ Next1 => IndInv'
  <1> SUFFICES ASSUME IndInv, Next1 PROVE IndInv'
    OBVIOUS
  <1>1. UNCHANGED rare
    BY DEF Next1
  <1>2. CASE \E o \in Owners, g \in Groups : Attach(o, g)
    BY <1>1, <1>2, AttachInv
  <1>3. CASE \E o \in Owners, p \in Positions : Allocate(o, p)
    BY <1>1, <1>3, AllocateInv
  <1>4. CASE \E o \in Owners, p \in Positions, k \in Keys : Pin(o, p, k)
    BY <1>1, <1>4, PinInv
  <1> QED
    BY <1>2, <1>3, <1>4 DEF Next1
=============================================================================
