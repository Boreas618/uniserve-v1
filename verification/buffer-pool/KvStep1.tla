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
  <1>3. s \notin Units /\ s \in AllUnits /\ s # NONE
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
    <2>1. \A v \in Units : alloc'[v] = alloc[v]
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
                 PROVE  /\ ostate'[q] = "live" /\ ogroup'[q] = x
                        /\ IF p = SLOT THEN oslot'[q] = dbt'[x][q][p].u
                                       ELSE table'[q][p].u = dbt'[x][q][p].u
                        /\ (p \in Positions /\ dbt'[x][q][p].w) =>
                             /\ table'[q][p].k = "private"
                             /\ ~\E d \in deferred' : d.kind = "commit" /\ dbt'[x][q][p].u \in d.units
      BY DEF TablesMapped, Live, HoldsAt
    <2>1. CASE Eager /\ x = g /\ q = o /\ p = SLOT /\ Unnamed(h)
      BY <2>1, <1>2, <1>4
    <2>2. CASE ~(Eager /\ x = g /\ q = o /\ p = SLOT /\ Unnamed(h))
      <3>1. dbt'[x][q][p] = dbt[x][q][p]
        BY <2>2, <1>2
      <3>2. q # o
        BY <3>1, <1>5
      <3>3. /\ ostate[q] = "live" /\ ogroup[q] = x
            /\ HoldsAt(q, p, dbt[x][q][p].u)
            /\ (p \in Positions /\ dbt[x][q][p].w) =>
                 /\ table[q][p].k = "private"
                 /\ ~\E d \in deferred : d.kind = "commit" /\ dbt[x][q][p].u \in d.units
        BY <3>1 DEF IndInv, TablesMapped, Live
      <3>4. /\ ostate'[q] = ostate[q] /\ ogroup'[q] = ogroup[q] /\ oslot'[q] = oslot[q]
            /\ p \in Positions => table'[q][p] = table[q][p]
            /\ deferred' = deferred
        BY <3>2, <1>3 DEF AttachAt
      <3> QED
        BY <3>1, <3>3, <3>4 DEF HoldsAt, Rows
    <2> QED
      BY <2>1, <2>2
  <1>10. HandoffOk'
    <2> SUFFICES ASSUME NEW y \in handoff'
                 PROVE  /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
                        /\ Unnamed(y) =>
                             /\ ostate'[y.o] = "live" /\ ogroup'[y.o] = y.g
                             /\ IF y.p = SLOT THEN oslot'[y.o] = y.u ELSE table'[y.o][y.p].u = y.u
                             /\ (y.p \in Positions /\ y.w) =>
                                  /\ table'[y.o][y.p].k = "private"
                                  /\ ~\E d \in deferred' : d.kind = "commit" /\ y.u \in d.units
      BY DEF HandoffOk, Live, HoldsAt, Unnamed, AttachAt
    <2>1. CASE y = h
      BY <2>1, <1>4 DEF AttachAt
    <2>2. CASE y # h
      <3>1. y \in handoff
        BY <2>2, <1>2
      <3>2. Unnamed(y) => y.o # o
        BY <3>1, <1>6
      <3>3. /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
            /\ Unnamed(y) =>
                 /\ ostate[y.o] = "live" /\ ogroup[y.o] = y.g
                 /\ HoldsAt(y.o, y.p, y.u)
                 /\ (y.p \in Positions /\ y.w) =>
                      /\ table[y.o][y.p].k = "private"
                      /\ ~\E d \in deferred : d.kind = "commit" /\ y.u \in d.units
        BY <3>1 DEF IndInv, HandoffOk, Live, Unnamed
      <3>4. Unnamed(y) =>
              /\ ostate'[y.o] = ostate[y.o] /\ ogroup'[y.o] = ogroup[y.o] /\ oslot'[y.o] = oslot[y.o]
              /\ y.p \in Positions => table'[y.o][y.p] = table[y.o][y.p]
        BY <3>2, <3>3, <1>3 DEF AttachAt
      <3>5. deferred' = deferred /\ requests' = requests
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
                (dbt'[x][r.o][p].u # NONE /\ (TakesRow(r, r.o, p) \/ CommitsRow(r, r.o, p))) =>
                  /\ x \in r.sg
                  /\ x \in r.by => CommitsRow(r, r.o, p) /\ ~dbt'[x][r.o][p].w
      <3> SUFFICES ASSUME NEW r \in requests, r.kind # "finish", NEW x \in Groups, NEW p \in Rows,
                          dbt'[x][r.o][p].u # NONE, TakesRow(r, r.o, p) \/ CommitsRow(r, r.o, p)
                   PROVE  /\ x \in r.sg
                          /\ x \in r.by => CommitsRow(r, r.o, p) /\ ~dbt'[x][r.o][p].w
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
      BY RequestTypes DEF IndInv, BatchesOk, RequestTyped
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5 DEF IndInv, BatchesOk, AttachAt, Mem, rare
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12

LEMMA AttachInv ==
  ASSUME IndInv, NEW o \in Owners, NEW g \in Groups, Attach(o, g), UNCHANGED rare
  PROVE  IndInv'
  <1>1. PICK s \in Slots : AttachAt(o, g, s)
    BY AttachPick
  <1> QED
    BY <1>1, AttachPool, AttachOwners, AttachPlane DEF IndInv

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
    BY <1>3 DEF IndInv, TypeInv
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
    <2> QED
      BY <1>1, <1>3, <2>1, <2>2 DEF AllocateAt, TypeOK, rare, AllUnits
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
    BY <1>4, <1>5 DEF AllocateAt, IndInv, Outside, AllUnits
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
      BY <2>1 DEF IndInv, SharedIff, TypeInv, TypeOK
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
  <1>3. \A v \in Units : v # u => valid'[v] = valid[v]
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
    <2> QED
      BY <2>5, <2>6, <2>7, <2>8, <2>9, CopiesStay DEF AllocateAt, IndInv, rare
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
    <2> QED
      BY <2>1, <2>2 DEF AllocateAt, IndInv, TrialVisibility, MadeBy
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
    <2> QED
      BY <2>1, <2>2, <2>3, FillPlane DEF AllocateAt
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

LEMMA AllocateInv ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, Allocate(o, p), UNCHANGED rare
  PROVE  IndInv'
  <1>1. PICK u \in Units : AllocateAt(o, p, u)
    BY AllocatePick
  <1> QED
    BY <1>1, AllocatePool, AllocateOwners, AllocatePlane DEF IndInv

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
    BY <1>1 DEF IndInv, TypeInv
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
    <2> QED
      BY <2>4, <2>5, <2>6, CopiesStay DEF PinAt, IndInv, rare
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
                      ncopy, copies>>
      BY DEF PinAt, rare
    <2> QED
      BY <2>1, <2>2, <2>3, FillPlane DEF PinAt
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

LEMMA PinInv ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW k \in Keys, Pin(o, p, k), UNCHANGED rare
  PROVE  IndInv'
  <1>1. PinAt(o, p, k)
    BY PinPick
  <1> QED
    BY <1>1, PinPool, PinOwners, PinPlane DEF IndInv

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
