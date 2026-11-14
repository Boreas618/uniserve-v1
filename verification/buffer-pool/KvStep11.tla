------------------------------- MODULE KvStep11 ------------------------------
(***************************************************************************)
(* The inductive step of IndInv for crossings: TakeReplicate, ReplicateDue *)
(* and ReplicateEnd.                                                       *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, KvFrame, TLAPS

USE Sizes, Design

(* Held, written out: the six kinds of holder, before and after a step.  *)
LEMMA HeldUnfold ==
  \A x : /\ Held(x) = \/ \E o \in Owners, p \in Positions : table[o][p].u = x
                      \/ \E o \in Owners : oslot[o] = x
                      \/ \E k \in Keys : cache[k] = x \/ hcache[k] = x
                      \/ \E d \in deferred : x \in d.units
                      \/ \E c \in copies : x = c.read \/ x = c.write
                      \/ \E t \in tasks : t.held /\ (x \in t.units \/ x = t.slot \/ x = t.h)
         /\ Held(x)' = \/ \E o \in Owners, p \in Positions : table'[o][p].u = x
                       \/ \E o \in Owners : oslot'[o] = x
                       \/ \E k \in Keys : cache'[k] = x \/ hcache'[k] = x
                       \/ \E d \in deferred' : x \in d.units
                       \/ \E c \in copies' : x = c.read \/ x = c.write
                       \/ \E t \in tasks' : t.held /\ (x \in t.units \/ x = t.slot \/ x = t.h)
  BY DEF Held, HeldIn

-----------------------------------------------------------------------------
(* TakeReplicate                                                           *)

(* The block a crossing names, the crossing that waits for its stamps, and *)
(* the entry the reading group is handed when the crossing is answered at  *)
(* once.                                                                   *)
Named(r) == table[r.o][r.p].u
ReplicateWait(r) ==
    [kind |-> "replicate", units |-> {Named(r)}, after |-> r.after,
     o |-> r.o, p |-> r.p, k |-> NONE, c |-> 0, ok |-> FALSE,
     g |-> ogroup[r.o], to |-> r.to]
Answer(r) == [g |-> r.to, o |-> r.o, p |-> r.p, u |-> Named(r), w |-> FALSE]

(* What R1 and R8 require of a crossing for it to be taken.                *)
Taken(r) ==
    /\ ostate[r.o] = "live"
    /\ table[r.o][r.p] # Empty
    /\ OwnerEpoch[r.o] = r.e /\ ~Moving(r.o) /\ r.to \in ReadersOf(r.o) /\ GroupEpoch[r.to] = r.e
    /\ table[r.o][r.p].k = "shared" => vis[sharedKey[Named(r)]].d = MadeBy(OwnerEpoch[r.o])

LEMMA TakeReplicate_Effect ==
  ASSUME NEW r \in requests, TakeReplicate(r)
  PROVE  /\ r.kind = "replicate" /\ r.sg \subseteq r.by
         /\ requests' = requests \ {r}
         /\ \/ /\ ~Taken(r)
               /\ handoff' = Kept(r)
               /\ UNCHANGED <<deferred, dbt>>
            \/ /\ Taken(r) /\ Mem(r.to) \in valid[Named(r)]
               /\ UNCHANGED deferred
               /\ \/ /\ Eager
                     /\ handoff' = Kept(r)
                     /\ dbt' = IF \E x \in requests' : Names(x, Answer(r))
                               THEN dbt
                               ELSE [dbt EXCEPT ![r.to][r.o][r.p] = [u |-> Named(r), w |-> FALSE]]
                  \/ /\ ~Eager
                     /\ handoff' = Kept(r) \cup {Answer(r)}
                     /\ dbt' = dbt
            \/ /\ Taken(r) /\ Mem(r.to) \notin valid[Named(r)]
               /\ deferred' = deferred \cup {ReplicateWait(r)}
               /\ handoff' = Kept(r)
               /\ UNCHANGED dbt
         /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks,
                        trial, member, valid, vis, launched, inflight, alloc, sharedKey>>
  BY DEF TakeReplicate, Submittable, StampGroups, Named, ReplicateWait, Answer, Taken, OwnSession,
         On, AllRules

(* What IndInv says of the owner and the block a crossing that is taken    *)
(* names: its owner is attached and does not move; the block is in its     *)
(* table at that position alone if private, valid in its group's memory,  *)
(* no new block of a copy, and no group may write it.                     *)
LEMMA TakeReplicate_Facts ==
  ASSUME IndInv, NEW r \in requests, TakeReplicate(r), Taken(r)
  PROVE  /\ r.o \in Owners /\ r.p \in Positions /\ r.to \in Groups /\ r.after \in Stamps
         /\ Live(r.o) /\ ogroup[r.o] \in Groups /\ OwnerEpoch[r.o] \notin lost
         /\ GroupEpoch[r.to] = OwnerEpoch[r.o] /\ r.e = OwnerEpoch[r.o]
         /\ Named(r) \in Units /\ Named(r) \notin free /\ Held(Named(r))
         /\ alloc[Named(r)] = r.o \/ alloc[Named(r)] = "shared"
         /\ Mem(ogroup[r.o]) \in valid[Named(r)]
         /\ alloc[Named(r)] = r.o =>
              /\ table[r.o][r.p].k = "private"
              /\ \A o \in Owners, p \in Positions : table[o][p].u = Named(r) => o = r.o /\ p = r.p
         /\ alloc[Named(r)] = "shared" =>
              /\ table[r.o][r.p].k = "shared"
              /\ cache[sharedKey[Named(r)]] = Named(r)
              /\ vis[sharedKey[Named(r)]].d = MadeBy(OwnerEpoch[r.o])
         /\ \A c \in copies : ~c.fill => c.write # Named(r)
         /\ \A c \in copies : c.o = r.o => c.p # r.p \/ c.kind \in {"save", "restore", "store"}
         /\ \A g \in Groups, p \in Rows : (dbt[g][r.o][p].u # NONE /\ FreezesRow(r, r.o, p))
                                            => ~dbt[g][r.o][p].w
         /\ \A b \in inflight : b.g \in r.sg => b.g \in r.by
  <1>0. UnitTyped /\ OwnerTyped
    BY UnitTypes, OwnerTypes
  <1>1. /\ r.kind = "replicate" /\ r.sg \subseteq r.by
        /\ r.o \in Owners /\ r.p \in Positions /\ r.to \in Groups /\ r.after \in Stamps
    BY TakeReplicate_Effect, RequestTypes DEF RequestTyped
  <1>2. /\ Live(r.o) /\ ogroup[r.o] \in Groups /\ OwnerEpoch[r.o] \notin lost
        /\ GroupEpoch[r.to] = OwnerEpoch[r.o] /\ r.e = OwnerEpoch[r.o]
    BY <1>1 DEF Taken, IndInv, OwnersOk, Live
  <1>3. /\ Named(r) \in Units /\ Mem(ogroup[r.o]) \in valid[Named(r)]
        /\ table[r.o][r.p].k = "private" => alloc[Named(r)] = r.o
        /\ table[r.o][r.p].k = "shared" => alloc[Named(r)] = "shared"
        /\ table[r.o][r.p].k \in {"private", "shared"}
    BY <1>0, <1>1 DEF Taken, IndInv, Tables, OwnerTyped, Named, Empty, Private, Shared
  <1>4. Held(Named(r)) /\ Named(r) \notin free
    BY <1>1 DEF Held, HeldIn, Named, IndInv, FreeClean
  <1>5. alloc[Named(r)] = r.o =>
          /\ table[r.o][r.p].k = "private"
          /\ \A o \in Owners, p \in Positions : table[o][p].u = Named(r) => o = r.o /\ p = r.p
    <2> SUFFICES ASSUME alloc[Named(r)] = r.o
                 PROVE  /\ table[r.o][r.p].k = "private"
                        /\ \A o \in Owners, p \in Positions :
                             table[o][p].u = Named(r) => o = r.o /\ p = r.p
      OBVIOUS
    <2>1. table[r.o][r.p].k = "private"
      BY <1>1, <1>3
    <2>2. ASSUME NEW o \in Owners, NEW p \in Positions, table[o][p].u = Named(r)
          PROVE  o = r.o /\ p = r.p
      <3>1. table[o][p].k = "private"
        BY <1>0, <1>1, <1>3, <2>2 DEF IndInv, Tables, OwnerTyped, Empty, Private, Shared, NONE
      <3>2. o = r.o
        BY <1>3, <2>2, <3>1 DEF IndInv, Tables
      <3> QED
        BY <1>1, <2>1, <2>2, <3>2 DEF IndInv, Tables, Named
    <2> QED
      BY <2>1, <2>2
  <1>6. alloc[Named(r)] = "shared" =>
          /\ table[r.o][r.p].k = "shared"
          /\ cache[sharedKey[Named(r)]] = Named(r)
          /\ vis[sharedKey[Named(r)]].d = MadeBy(OwnerEpoch[r.o])
    <2> SUFFICES ASSUME alloc[Named(r)] = "shared"
                 PROVE  /\ table[r.o][r.p].k = "shared"
                        /\ cache[sharedKey[Named(r)]] = Named(r)
                        /\ vis[sharedKey[Named(r)]].d = MadeBy(OwnerEpoch[r.o])
      OBVIOUS
    <2>1. table[r.o][r.p].k = "shared"
      BY <1>1, <1>3
    <2>2. cache[sharedKey[Named(r)]] = Named(r)
      <3>1. sharedKey[Named(r)] \in Keys
        BY <1>0, <1>3 DEF IndInv, SharedIff, UnitTyped, AllUnits, NONE
      <3>2. cache[sharedKey[Named(r)]] = Named(r) \/ hcache[sharedKey[Named(r)]] = Named(r)
        BY <1>3 DEF IndInv, SharedCached, AllUnits
      <3> QED
        BY <1>0, <1>3, <3>1, <3>2 DEF UnitTyped
    <2> QED
      BY <2>1, <2>2 DEF Taken
  <1>7. \A c \in copies : ~c.fill => c.write # Named(r)
    BY <1>1, <1>3 DEF IndInv, CopiesOk, Named
  <1>8. \A c \in copies : c.o = r.o => c.p # r.p \/ c.kind \in {"save", "restore", "store"}
    <2> SUFFICES ASSUME NEW c \in copies, c.o = r.o, c.kind \in {"clone", "load"} PROVE c.p # r.p
      BY DEF IndInv, CopiesOk
    <2> QED
      BY <1>1 DEF Taken, IndInv, CopiesOk
  <1>9. /\ \A g \in Groups, p \in Rows : (dbt[g][r.o][p].u # NONE /\ FreezesRow(r, r.o, p))
                                            => ~dbt[g][r.o][p].w
        /\ \A b \in inflight : b.g \in r.sg => b.g \in r.by
    BY <1>1, Submitted
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9

(* The crossing is refused: the request leaves and nothing else changes.  *)
LEMMA TakeReplicate_Refuse ==
  ASSUME IndInv, NEW r \in requests, TakeReplicate(r), UNCHANGED rare,
         handoff' = Kept(r), UNCHANGED <<deferred, dbt>>
  PROVE  IndInv'
  <1>1. /\ requests' = requests \ {r}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks,
                       trial, member, valid, vis, launched, inflight, alloc, sharedKey>>
    BY TakeReplicate_Effect
  <1>2. handoff' \subseteq handoff /\ \A h \in handoff' : ~Names(r, h)
    BY DEF Kept
  <1> QED
    BY <1>1, <1>2, TakeIdle DEF pool, rare

(* The reading group's memory holds the block: the crossing is answered    *)
(* at once and the group is handed the entry, read only.                  *)
LEMMA TakeReplicate_Answer ==
  ASSUME IndInv, NEW r \in requests, TakeReplicate(r), UNCHANGED rare,
         Taken(r), Mem(r.to) \in valid[Named(r)], UNCHANGED deferred,
         \/ /\ Eager
            /\ handoff' = Kept(r)
            /\ dbt' = IF \E x \in requests' : Names(x, Answer(r))
                      THEN dbt
                      ELSE [dbt EXCEPT ![r.to][r.o][r.p] = [u |-> Named(r), w |-> FALSE]]
         \/ /\ ~Eager
            /\ handoff' = Kept(r) \cup {Answer(r)}
            /\ dbt' = dbt
  PROVE  IndInv'
  <1> DEFINE h == Answer(r)
  <1>0. PlaneTyped /\ RequestTyped
    BY PlaneTypes, RequestTypes
  <1>1. /\ r.kind = "replicate"
        /\ requests' = requests \ {r}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       tasks, valid, vis, trial, member, launched, inflight, alloc, sharedKey,
                       remote, lost, bad>>
    BY TakeReplicate_Effect DEF rare
  <1>2. /\ r.o \in Owners /\ r.p \in Positions /\ r.to \in Groups
        /\ Live(r.o) /\ OwnerEpoch[r.o] \notin lost /\ GroupEpoch[r.to] = OwnerEpoch[r.o]
        /\ r.e = OwnerEpoch[r.o] /\ ~Moving(r.o) /\ r.to \in ReadersOf(r.o)
        /\ Named(r) \in Units
    BY TakeReplicate_Facts DEF Taken
  <1>3. /\ h.g = r.to /\ h.o = r.o /\ h.p = r.p /\ h.u = Named(r) /\ ~h.w
        /\ h.p \in Rows /\ h.u \in AllUnits
        /\ ReaderRow(h.g, h.o, h.p, h.u, h.w)
        /\ HoldsAt(h.o, h.p, h.u)
    BY <1>2 DEF Answer, Rows, AllUnits, ReaderRow, HoldsAt, Named
  <1>4. handoff' \subseteq handoff \cup {h} /\ \A y \in handoff' : y # h => y \in handoff /\ ~Names(r, y)
    BY DEF Kept
  <1>5. /\ \A x \in Groups, q \in Owners, p \in Rows :
             dbt'[x][q][p] = IF Eager /\ x = r.to /\ q = r.o /\ p = r.p
                                /\ ~\E y \in requests' : Names(y, h)
                             THEN [u |-> Named(r), w |-> FALSE]
                             ELSE dbt[x][q][p]
        /\ DOMAIN dbt' = Groups
        /\ \A x \in Groups : DOMAIN dbt'[x] = Owners
        /\ \A x \in Groups, q \in Owners : DOMAIN dbt'[x][q] = Rows
    BY <1>0, <1>2, <1>3 DEF PlaneTyped
  <1>6. \A y \in handoff' : y # h => (\E x \in requests' : Names(x, y)) = (\E x \in requests : Names(x, y))
    BY <1>1, <1>4
  <1>7. TypeInv'
    <2>1. PlaneTyped'
      BY <1>0, <1>1, <1>3, <1>5 DEF PlaneTyped
    <2>2. RequestTyped'
      BY <1>0, <1>1, <1>3, <1>4 DEF RequestTyped
    <2> QED
      BY <1>1, <2>1, <2>2, UnitTypes, OwnerTypes, WorkTypes, UnitTypedFrame, OwnerTypedFrame,
         WorkTypedFrame, TypeInvNext
  <1>8. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
        /\ TasksOk' /\ TrialVisibility' /\ NoBad' /\ CrossingsOk' /\ CopiesPlaced'
        /\ CrossingsPlaced'
    BY <1>1, TransfersFrame, FreeCleanFrame, NoLeakFrame, OutsideFrame, RegionsFrame,
       AllocOwnerFrame, SharedIffFrame, SharedCachedFrame, CacheSharedFrame, TablesFrame,
       OwnersOkFrame, DeferredOkFrame, CopiesOkFrame, TasksOkFrame, TrialVisibilityFrame,
       NoBadFrame, CrossingsOkFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
  <1>9. MovingUnmapped'
    <2> SUFFICES ASSUME NEW q \in Owners,
                        \/ \E t \in tasks : t.kind = "transfer" /\ t.o = q
                        \/ \E d \in deferred : d.kind = "transfer" /\ d.o = q
                 PROVE  /\ \A x \in Groups, p \in Rows : dbt'[x][q][p].u = NONE
                        /\ \A y \in handoff' : y.o # q
                        /\ \A d, e \in deferred :
                             (d.kind = "transfer" /\ e.kind = "transfer" /\ d.o = q /\ e.o = q) => d = e
      BY <1>1 DEF MovingUnmapped
    <2>1. q # r.o
      BY <1>2 DEF Moving
    <2> QED
      BY <1>1, <1>3, <1>4, <1>5, <2>1 DEF IndInv, MovingUnmapped
  <1>10. LostOk'
    <2>1. GroupEpoch[r.to] \notin lost
      BY <1>2
    <2> QED
      BY <1>1, <1>3, <1>4, <1>5, <2>1 DEF IndInv, LostOk
  <1>11. TablesMapped'
    <2> SUFFICES ASSUME NEW x \in Groups, NEW q \in Owners, NEW p \in Rows, dbt'[x][q][p].u # NONE
                 PROVE  /\ ostate'[q] = "live"
                        /\ \/ ogroup'[q] = x
                           \/ ReaderRow(x, q, p, dbt[x][q][p].u, dbt[x][q][p].w)'
                        /\ HoldsAt(q, p, dbt[x][q][p].u)'
                        /\ (p \in Positions /\ dbt'[x][q][p].w) =>
                             /\ table'[q][p].k = "private"
                             /\ ~\E d \in deferred' :
                                   d.kind \in {"commit", "replicate"} /\ dbt'[x][q][p].u \in d.units
                             /\ ~\E t \in tasks' : t.kind = "replicate" /\ dbt'[x][q][p].u \in t.units
      BY DEF TablesMapped, Live
    <2>1. CASE Eager /\ x = r.to /\ q = r.o /\ p = r.p /\ ~\E y \in requests' : Names(y, h)
      <3>1. dbt'[x][q][p] = [u |-> Named(r), w |-> FALSE]
        BY <2>1, <1>5
      <3> QED
        BY <1>1, <1>2, <1>3, <2>1, <3>1 DEF Live, ReaderRow, ReadersOf, Mem, HoldsAt
    <2>2. CASE ~(Eager /\ x = r.to /\ q = r.o /\ p = r.p /\ ~\E y \in requests' : Names(y, h))
      <3>1. dbt'[x][q][p] = dbt[x][q][p]
        BY <2>2, <1>5
      <3> QED
        BY <1>1, <3>1 DEF IndInv, TablesMapped, Live, ReaderRow, ReadersOf, Mem, HoldsAt
    <2> QED
      BY <2>1, <2>2
  <1>12. HandoffOk'
    <2> SUFFICES ASSUME NEW y \in handoff'
                 PROVE  /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
                        /\ (~\E x \in requests' : Names(x, y)) =>
                             /\ ostate'[y.o] = "live"
                             /\ ogroup'[y.o] = y.g \/ ReaderRow(y.g, y.o, y.p, y.u, y.w)'
                             /\ HoldsAt(y.o, y.p, y.u)'
                             /\ (y.p \in Positions /\ y.w) =>
                                  /\ table'[y.o][y.p].k = "private"
                                  /\ ~\E d \in deferred' :
                                        d.kind \in {"commit", "replicate"} /\ y.u \in d.units
                                  /\ ~\E t \in tasks' : t.kind = "replicate" /\ y.u \in t.units
      BY DEF HandoffOk, Live
    <2>1. CASE y = h
      BY <1>1, <1>2, <1>3, <2>1 DEF Live, ReaderRow, ReadersOf, Mem, HoldsAt
    <2>2. CASE y # h
      <3>1. y \in handoff
        BY <1>4, <2>2
      <3> QED
        BY <1>1, <1>6, <2>2, <3>1 DEF IndInv, HandoffOk, Live, ReaderRow, ReadersOf, Mem, HoldsAt
    <2> QED
      BY <2>1, <2>2
  <1>13. RequestsOk'
    <2>1. \A x \in requests' :
            x.kind # "finish" =>
              \A z \in Groups, q \in Rows :
                (dbt'[z][x.o][q].u # NONE /\ (TakesRow(x, x.o, q) \/ FreezesRow(x, x.o, q))) =>
                  /\ z \in x.sg
                  /\ z \in x.by => FreezesRow(x, x.o, q) /\ ~dbt'[z][x.o][q].w
      <3> SUFFICES ASSUME NEW x \in requests', x.kind # "finish", NEW z \in Groups, NEW q \in Rows,
                          dbt'[z][x.o][q].u # NONE, TakesRow(x, x.o, q) \/ FreezesRow(x, x.o, q)
                   PROVE  /\ z \in x.sg
                          /\ z \in x.by => FreezesRow(x, x.o, q) /\ ~dbt'[z][x.o][q].w
        OBVIOUS
      <3>1. x \in requests /\ x.o \in Owners
        BY <1>0, <1>1 DEF RequestTyped
      <3>2. CASE Eager /\ z = r.to /\ x.o = r.o /\ q = r.p /\ ~\E y \in requests' : Names(y, h)
        <4>1. Names(x, h)
          BY <1>3, <3>2, NamesRow
        <4> QED
          BY <3>2, <4>1
      <3>3. CASE ~(Eager /\ z = r.to /\ x.o = r.o /\ q = r.p /\ ~\E y \in requests' : Names(y, h))
        <4>1. dbt'[z][x.o][q] = dbt[z][x.o][q]
          BY <1>5, <3>1, <3>3
        <4> QED
          BY <3>1, <4>1 DEF IndInv, RequestsOk
      <3> QED
        BY <3>2, <3>3
    <2> QED
      BY <1>1, <2>1 DEF IndInv, RequestsOk
  <1>14. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
      BY <1>1 DEF Covered, Live, HoldsAt, ReadersOf
    <2>4. \A o \in Owners, p \in Positions :
            (\A x \in requests :
               FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
            => \A x \in requests' :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
      BY <1>1
    <2> QED
      BY <1>1, <2>2, <2>3, <2>4 DEF BatchOk, Mem
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14 DEF IndInv

(* The crossing waits for its stamps holding the block, which no group may *)
(* write any more and which no batch in flight writes beyond the stamps.   *)
LEMMA TakeReplicate_Wait ==
  ASSUME IndInv, NEW r \in requests, TakeReplicate(r), UNCHANGED rare,
         Taken(r), Mem(r.to) \notin valid[Named(r)],
         deferred' = deferred \cup {ReplicateWait(r)}, handoff' = Kept(r), UNCHANGED dbt
  PROVE  IndInv'
  <1> DEFINE u == Named(r)
  <1>a. PICK nd : nd = ReplicateWait(r)
    OBVIOUS
  <1>b. deferred' = deferred \cup {nd}
    BY <1>a
  <1> USE <1>b
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, WorkTypes
  <1>1. /\ r.kind = "replicate" /\ r.sg \subseteq r.by
        /\ requests' = requests \ {r}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks,
                       trial, member, valid, vis, launched, inflight, alloc, sharedKey, dbt,
                       remote, lost, bad>>
    BY TakeReplicate_Effect DEF rare
  <1>2. /\ r.o \in Owners /\ r.p \in Positions /\ r.to \in Groups /\ r.after \in Stamps
        /\ Live(r.o) /\ ogroup[r.o] \in Groups /\ OwnerEpoch[r.o] \notin lost
        /\ r.e = OwnerEpoch[r.o] /\ ~Moving(r.o) /\ r.to \in ReadersOf(r.o)
        /\ u \in Units /\ Held(u)
        /\ alloc[u] = r.o \/ alloc[u] = "shared"
        /\ alloc[u] = r.o =>
             /\ table[r.o][r.p].k = "private"
             /\ \A o \in Owners, p \in Positions : table[o][p].u = u => o = r.o /\ p = r.p
        /\ alloc[u] = "shared" =>
             /\ table[r.o][r.p].k = "shared"
             /\ cache[sharedKey[u]] = u
             /\ vis[sharedKey[u]].d = MadeBy(OwnerEpoch[r.o])
        /\ \A c \in copies : ~c.fill => c.write # u
        /\ \A g \in Groups, p \in Rows : (dbt[g][r.o][p].u # NONE /\ FreezesRow(r, r.o, p))
                                           => ~dbt[g][r.o][p].w
    BY TakeReplicate_Facts DEF Taken
  <1>3. /\ nd.kind = "replicate" /\ nd.units = {u} /\ nd.after = r.after /\ nd.o = r.o
        /\ nd.p = r.p /\ nd.k = NONE /\ nd.c = 0 /\ nd.g = ogroup[r.o] /\ nd.to = r.to
    BY <1>a DEF ReplicateWait, Named
  <1>4. \A x : Held(x)' = Held(x)
    <2> SUFFICES ASSUME NEW x PROVE Held(x)' = Held(x)
      OBVIOUS
    <2>1. (\E d \in deferred' : x \in d.units) = ((\E d \in deferred : x \in d.units) \/ x = u)
      BY <1>b, <1>3
    <2>2. x = u => \E o \in Owners, p \in Positions : table[o][p].u = x
      BY <1>2 DEF Named
    <2> QED
      BY <1>1, <2>1, <2>2, HeldUnfold
  <1>5. \A o \in Owners : Moving(o)' = Moving(o)
    BY <1>1, <1>3 DEF Moving
  <1>6. /\ handoff' \subseteq handoff /\ \A h \in handoff' : ~Names(r, h)
        /\ \A h \in handoff' : (\E x \in requests' : Names(x, h)) = (\E x \in requests : Names(x, h))
    BY <1>1 DEF Kept
  <1>7. TypeInv'
    <2>1. WorkTyped'
      BY <1>0, <1>1, <1>2, <1>3, WorkTypes DEF WorkTyped, NONE, AllUnits
    <2>2. RequestTyped'
      BY <1>1, <1>6, RequestTypes DEF RequestTyped
    <2> QED
      BY <1>1, <2>1, <2>2, UnitTypes, OwnerTypes, PlaneTypes, UnitTypedFrame, OwnerTypedFrame,
         PlaneTypedFrame, TypeInvNext
  <1>8. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ TrialVisibility' /\ NoBad'
        /\ CopiesPlaced' /\ CrossingsPlaced'
    BY <1>1, <1>4, TransfersFrame, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
       SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame, TrialVisibilityFrame,
       NoBadFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv, FreeClean, NoLeak, InRegion
  <1>9. RequestsOk' /\ LostOk'
    BY <1>1, <1>6, TakeCommon
  <1>10. DeferredOk'
    <2>1. SUFFICES ASSUME NEW d \in deferred' PROVE DeferOk(d)'
      BY DeferredOkEach
    <2>2. CASE d = nd
      BY <1>3, <2>2 DEF DeferOk
    <2>3. CASE d # nd
      <3>1. d \in deferred /\ DeferOk(d)
        BY <2>3, DeferredOkEach DEF IndInv
      <3>2. \A e \in deferred' : e \in deferred \/ e = nd
        BY <1>3
      <3> QED
        BY <1>1, <1>3, <3>1, <3>2 DEF DeferOk
    <2> QED
      BY <2>2, <2>3
  <1>11. CopiesOk' /\ TasksOk'
    <2>1. CopiesOk'
      <3>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
        BY <1>1, CopiesOkEach
      <3> QED
        BY <1>1, <1>5, CopiesOkEach DEF CopyOk, Live, Mem, IndInv
    <2>2. TasksOk'
      <3>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
        BY <1>1, TasksOkEach
      <3>2. TaskOk(t)
        BY TasksOkEach DEF IndInv
      <3> QED
        BY <1>1, <1>2, <1>3, <3>2 DEF TaskOk, Live, TableUnits, Mem
    <2> QED
      BY <2>1, <2>2
  <1>12. CrossingsOk'
    <2>1. CrossWaitOk(nd)'
      <3>1. \A k \in Keys : cache[k] = u => vis[k].d = MadeBy(OwnerEpoch[r.o])
        <4> SUFFICES ASSUME NEW k \in Keys, cache[k] = u PROVE vis[k].d = MadeBy(OwnerEpoch[r.o])
          OBVIOUS
        <4>1. alloc[u] = "shared" /\ sharedKey[u] = k
          BY <1>2, DeviceCached
        <4> QED
          BY <1>2, <4>1
      <3>2. /\ nd.o \in Owners /\ nd.p \in Positions /\ nd.to \in Groups
            /\ ostate'[nd.o] # "new" /\ OwnerEpoch[nd.o] \notin lost'
            /\ nd.to \in ReadersOf(nd.o)'
        BY <1>1, <1>2, <1>3 DEF Live, ReadersOf
      <3>3. \E v \in Units : nd.units = {v} /\ (alloc'[v] = nd.o \/ alloc'[v] = "shared")
        BY <1>1, <1>2, <1>3
      <3>4. ~Moving(nd.o)'
        BY <1>2, <1>3, <1>5
      <3>5. \A c \in copies' : ~c.fill => c.write \notin nd.units
        BY <1>1, <1>2, <1>3
      <3>6. \A k \in Keys : cache'[k] \in nd.units => vis'[k].d = MadeBy(OwnerEpoch[nd.o])'
        BY <1>1, <1>3, <3>1 DEF MadeBy
      <3> QED
        BY <1>3, <3>2, <3>3, <3>4, <3>5, <3>6 DEF CrossWaitOk
    <2>2. \A d \in deferred : CrossWaitOk(d)'
      <3> SUFFICES ASSUME NEW d \in deferred PROVE CrossWaitOk(d)'
        OBVIOUS
      <3>1. CrossWaitOk(d)
        BY CrossingsOkEach DEF IndInv
      <3> QED
        BY <1>1, <1>5, <3>1 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>3. \A t \in tasks : CrossTaskOk(t)'
      <3> SUFFICES ASSUME NEW t \in tasks PROVE CrossTaskOk(t)'
        OBVIOUS
      <3>1. CrossTaskOk(t)
        BY CrossingsOkEach DEF IndInv
      <3> QED
        BY <1>1, <1>5, <3>1 DEF CrossTaskOk, ReadersOf, Mem
    <2>4. \A x \in deferred' : CrossWaitOk(x)'
      BY <2>1, <2>2
    <2> QED
      BY <1>1, <2>3, <2>4, CrossingsOkEach
  <1>13. MovingUnmapped'
    BY <1>1, <1>3, <1>6 DEF IndInv, MovingUnmapped
  (* No executor may write the block, and no completion kept gives the     *)
  (* right to: the request named every such row.                          *)
  <1>14. TablesMapped'
    <2>1. \A g \in Groups, o \in Owners, p \in Rows :
            (dbt[g][o][p].u = u /\ p \in Positions /\ dbt[g][o][p].w) => FALSE
      <3> SUFFICES ASSUME NEW g \in Groups, NEW o \in Owners, NEW p \in Rows,
                          dbt[g][o][p].u = u, p \in Positions, dbt[g][o][p].w
                   PROVE  FALSE
        OBVIOUS
      <3>1. table[o][p].u = u /\ table[o][p].k = "private"
        BY <1>2 DEF IndInv, TablesMapped, HoldsAt
      <3>2. alloc[u] = o
        BY <3>1 DEF IndInv, Tables
      <3>3. o = r.o /\ p = r.p
        BY <1>2, <3>1, <3>2
      <3> QED
        BY <1>1, <1>2, <3>3 DEF FreezesRow
    <2> QED
      BY <1>1, <1>3, <2>1 DEF IndInv, TablesMapped, Live, HoldsAt, ReaderRow, ReadersOf, Mem
  <1>15. HandoffOk'
    <2>1. \A h \in handoff' :
            ((~\E x \in requests : Names(x, h)) /\ h.p \in Positions /\ h.w) => h.u # u
      <3> SUFFICES ASSUME NEW h \in handoff', ~\E x \in requests : Names(x, h),
                          h.p \in Positions, h.w, h.u = u
                   PROVE  FALSE
        OBVIOUS
      <3>1. h \in handoff /\ h.o \in Owners
        BY <1>6, RequestTypes DEF RequestTyped
      <3>2. table[h.o][h.p].u = u /\ table[h.o][h.p].k = "private"
        BY <3>1 DEF IndInv, HandoffOk, HoldsAt
      <3>3. alloc[u] = h.o
        BY <3>1, <3>2 DEF IndInv, Tables
      <3>4. h.o = r.o /\ h.p = r.p
        BY <1>2, <3>1, <3>2, <3>3
      <3> QED
        BY <1>1, <1>2, <1>6, <3>4 DEF Names
    <2> QED
      BY <1>1, <1>3, <1>6, <2>1 DEF IndInv, HandoffOk, Live, HoldsAt, ReaderRow, ReadersOf, Mem
  <1>16. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A x : Covered(b, o, x) => Covered(b, o, x)'
      BY <1>1, <1>3 DEF Covered, Live, HoldsAt, ReadersOf
    <2>4. \A o \in Owners, p \in Positions :
            (\A x \in requests :
               FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
            => \A x \in requests' :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
      BY <1>1, <1>6, TakeCommon
    (* A batch that writes the block was covered by the request.           *)
    <2>5. \A y \in b.writes : y[2] = u => r.after[b.g] >= b.seq
      <3> SUFFICES ASSUME NEW y \in b.writes, y[2] = u PROVE r.after[b.g] >= b.seq
        OBVIOUS
      <3>1. alloc[u] = y[1] /\ y[1] \in Owners
        BY <2>2 DEF BatchOk
      <3>2. y[1] = r.o
        BY <1>2, <3>1, "shared" \notin Owners
      <3>3. FreezesRow(r, y[1], r.p) /\ table[y[1]][r.p].u = y[2]
        BY <1>1, <1>2, <3>2 DEF FreezesRow, Named
      <3> QED
        BY <1>1, <1>2, <2>2, <3>3 DEF BatchOk
    <2> QED
      BY <1>1, <1>3, <1>4, <2>2, <2>3, <2>4, <2>5 DEF BatchOk, Mem, IndInv, FreeClean
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16 DEF IndInv

LEMMA TakeReplicate_Inv ==
  ASSUME IndInv, NEW r \in requests, TakeReplicate(r), UNCHANGED rare
  PROVE  IndInv'
  BY TakeReplicate_Effect, TakeReplicate_Refuse, TakeReplicate_Answer, TakeReplicate_Wait


-----------------------------------------------------------------------------
(* ReplicateDue                                                            *)

(* The block a waiting crossing holds, whether its owner still holds it at *)
(* the position it named, the task that starts, and the entry the reading  *)
(* group is handed when the block is valid in its memory already.         *)
Crossed(d) == CHOOSE x \in d.units : TRUE
Holding(d) == ostate[d.o] = "live" /\ table[d.o][d.p].u = Crossed(d)
NewCrossing(d) ==
    [kind |-> "replicate", k |-> NONE, h |-> NONE, src |-> NONE, ok |-> TRUE, expired |-> FALSE,
     p |-> d.p, o |-> d.o, to |-> d.to, slot |-> NONE, held |-> TRUE,
     units |-> {Crossed(d)}, writes |-> {Crossed(d)}]
Reading(d) == [g |-> d.to, o |-> d.o, p |-> d.p, u |-> Crossed(d), w |-> FALSE]

LEMMA ReplicateDue_Effect ==
  ASSUME NEW d \in deferred, ReplicateDue(d)
  PROVE  /\ d.kind = "replicate" /\ Reached(d.after)
         /\ ~\E c \in copies : c.fill /\ c.write = Crossed(d) /\ Mem(c.g) = Mem(d.to)
         /\ ~\E t \in tasks :
              t.kind \in {"transfer", "replicate"} /\ Mem(t.to) = Mem(d.to) /\ Crossed(d) \in t.writes
         /\ deferred' = deferred \ {d}
         /\ \/ /\ Holding(d) /\ Mem(d.to) \notin valid[Crossed(d)]
               /\ tasks' = tasks \cup {NewCrossing(d)}
               /\ UNCHANGED <<free, alloc, sharedKey, dbt, handoff>>
            \/ /\ Holding(d) /\ Mem(d.to) \in valid[Crossed(d)]
               /\ Drop(d.units, table, oslot, cache, hcache, deferred', copies)
               /\ UNCHANGED tasks
               /\ Give(Reading(d))
            \/ /\ ~Holding(d)
               /\ Drop(d.units, table, oslot, cache, hcache, deferred', copies)
               /\ UNCHANGED <<tasks, dbt, handoff>>
         /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, valid, vis, trial,
                        member, launched, inflight, requests>>
  BY DEF ReplicateDue, Crossed, Holding, NewCrossing, Reading, On, AllRules

(* What IndInv says of a crossing that comes due: the block it holds, its  *)
(* owner and target, and that no batch in flight writes the block.         *)
LEMMA ReplicateDue_Facts ==
  ASSUME IndInv, NEW d \in deferred, ReplicateDue(d)
  PROVE  /\ d.o \in Owners /\ d.p \in Positions /\ d.to \in Groups
         /\ ostate[d.o] # "new" /\ OwnerEpoch[d.o] \notin lost /\ d.to \in ReadersOf(d.o)
         /\ ~Moving(d.o)
         /\ d.units = {Crossed(d)} /\ Crossed(d) \in Units
         /\ alloc[Crossed(d)] = d.o \/ alloc[Crossed(d)] = "shared"
         /\ Held(Crossed(d)) /\ Crossed(d) \notin free
         /\ \A c \in copies : ~c.fill => c.write # Crossed(d)
         /\ \A k \in Keys : cache[k] = Crossed(d) => vis[k].d = MadeBy(OwnerEpoch[d.o])
         /\ \A b \in inflight : \A w \in b.writes : w[2] # Crossed(d)
         /\ Holding(d) => Mem(ogroup[d.o]) \in valid[Crossed(d)]
         /\ Live(d.o) => GroupEpoch[d.to] = OwnerEpoch[d.o]
  <1>1. d.kind = "replicate" /\ Reached(d.after)
    BY ReplicateDue_Effect
  <1>2. CrossWaitOk(d)
    BY CrossingsOkEach DEF IndInv
  <1>3. /\ d.o \in Owners /\ d.p \in Positions /\ d.to \in Groups
        /\ ostate[d.o] # "new" /\ OwnerEpoch[d.o] \notin lost /\ d.to \in ReadersOf(d.o)
        /\ ~Moving(d.o)
        /\ \E u \in Units : d.units = {u} /\ (alloc[u] = d.o \/ alloc[u] = "shared")
        /\ \A c \in copies : ~c.fill => c.write \notin d.units
        /\ \A k \in Keys : cache[k] \in d.units => vis[k].d = MadeBy(OwnerEpoch[d.o])
    BY <1>1, <1>2 DEF CrossWaitOk
  <1>4. d.units = {Crossed(d)} /\ Crossed(d) \in Units
        /\ (alloc[Crossed(d)] = d.o \/ alloc[Crossed(d)] = "shared")
    BY <1>3 DEF Crossed
  <1>5. Held(Crossed(d)) /\ Crossed(d) \notin free
    BY <1>4 DEF Held, HeldIn, IndInv, FreeClean
  (* A batch that writes the block was stamped before the crossing's stamp, *)
  (* which is reached.                                                      *)
  <1>6. \A b \in inflight : \A w \in b.writes : w[2] # Crossed(d)
    <2> SUFFICES ASSUME NEW b \in inflight, NEW w \in b.writes, w[2] = Crossed(d) PROVE FALSE
      OBVIOUS
    <2>1. d.after[b.g] >= b.seq
      BY <1>1, <1>4 DEF IndInv, BatchesOk
    <2>2. d.after \in Stamps
      BY WorkTypes DEF WorkTyped
    <2> QED
      BY <1>1, <2>1, <2>2, NotReached
  <1>7. Holding(d) => Mem(ogroup[d.o]) \in valid[Crossed(d)]
    BY <1>3, <1>4 DEF Holding, IndInv, Tables
  <1>8. Live(d.o) => GroupEpoch[d.to] = OwnerEpoch[d.o]
    BY <1>3 DEF IndInv, OwnersOk, Live, ReadersOf
  <1> QED
    BY <1>3, <1>4, <1>5, <1>6, <1>7, <1>8


(* The crossing comes due and its task starts, holding the block the       *)
(* crossing held.                                                          *)
LEMMA ReplicateDue_Start ==
  ASSUME IndInv, NEW d \in deferred, ReplicateDue(d), UNCHANGED rare,
         Holding(d), Mem(d.to) \notin valid[Crossed(d)],
         tasks' = tasks \cup {NewCrossing(d)}, UNCHANGED <<free, alloc, sharedKey, dbt, handoff>>
  PROVE  IndInv'
  <1> DEFINE t0 == NewCrossing(d)
             u == Crossed(d)
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, WorkTypes
  <1>1. /\ d.kind = "replicate"
        /\ deferred' = deferred \ {d}
        /\ ~\E c \in copies : c.fill /\ c.write = u /\ Mem(c.g) = Mem(d.to)
        /\ ~\E t \in tasks :
             t.kind \in {"transfer", "replicate"} /\ Mem(t.to) = Mem(d.to) /\ u \in t.writes
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, valid, vis,
                       trial, member, dbt, launched, inflight, handoff, requests, alloc, sharedKey,
                       remote, lost, bad>>
    BY ReplicateDue_Effect DEF rare
  <1>2. /\ d.o \in Owners /\ d.p \in Positions /\ d.to \in Groups
        /\ ostate[d.o] # "new" /\ d.to \in ReadersOf(d.o) /\ ~Moving(d.o)
        /\ d.units = {u} /\ u \in Units
        /\ alloc[u] = d.o \/ alloc[u] = "shared"
        /\ \A c \in copies : ~c.fill => c.write # u
        /\ \A k \in Keys : cache[k] = u => vis[k].d = MadeBy(OwnerEpoch[d.o])
        /\ \A b \in inflight : \A w \in b.writes : w[2] # u
        /\ Mem(ogroup[d.o]) \in valid[u]
    BY ReplicateDue_Facts
  <1>3. /\ t0.kind = "replicate" /\ t0.k = NONE /\ t0.h = NONE /\ t0.src = NONE /\ t0.held
        /\ t0.o = d.o /\ t0.p = d.p /\ t0.to = d.to /\ t0.slot = NONE
        /\ t0.units = {u} /\ t0.writes = {u}
        /\ t0 = [kind |-> t0.kind, k |-> t0.k, h |-> t0.h, src |-> t0.src, ok |-> t0.ok,
                 expired |-> t0.expired, p |-> t0.p, o |-> t0.o, to |-> t0.to, slot |-> t0.slot,
                 held |-> t0.held, units |-> t0.units, writes |-> t0.writes]
    BY DEF NewCrossing
  <1>4. \A x \in AllUnits : Held(x)' = Held(x)
    <2>1. \A x : (\E e \in deferred : x \in e.units)
                   = ((\E e \in deferred' : x \in e.units) \/ x = u)
      BY <1>1, <1>2
    <2>2. \A x \in AllUnits :
            (\E t \in tasks' : t.held /\ (x \in t.units \/ x = t.slot \/ x = t.h))
              = ((\E t \in tasks : t.held /\ (x \in t.units \/ x = t.slot \/ x = t.h)) \/ x = u)
      <3>1. NONE \notin AllUnits
        BY DEF AllUnits
      <3> QED
        BY <1>3, <3>1
    <2>3. \A x \in AllUnits : x = u => \E t \in tasks' : t.held /\ x \in t.units
      BY <1>3
    <2> QED
      BY <1>1, <2>1, <2>2, HeldUnfold
  <1>5. /\ \A o \in Owners : Moving(o)' = Moving(o)
        /\ Transfers' = Transfers
        /\ \A x \in deferred' : x \in deferred /\ x # d
    BY <1>1, <1>3 DEF Moving, Transfers
  <1>6. TypeInv'
    <2>1. WorkTyped'
      BY <1>0, <1>1, <1>2, <1>3 DEF WorkTyped, AllUnits, NONE
    <2> QED
      BY <1>1, <2>1, UnitTypes, OwnerTypes, PlaneTypes, RequestTypes, UnitTypedFrame,
         OwnerTypedFrame, PlaneTypedFrame, RequestTypedFrame, TypeInvNext
  <1>7. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ LostOk' /\ RequestsOk'
        /\ NoBad' /\ CopiesPlaced'
    BY <1>0, <1>1, <1>4, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
       SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame, LostOkFrame,
       RequestsOkFrame, NoBadFrame, CopiesPlacedFrame DEF IndInv, FreeClean, NoLeak, InRegion,
                                                          UnitTyped
  <1>8. DeferredOk'
    <2>1. SUFFICES ASSUME NEW x \in deferred' PROVE DeferOk(x)'
      BY DeferredOkEach
    <2>2. x \in deferred /\ DeferOk(x)
      BY <1>5, DeferredOkEach DEF IndInv
    <2> QED
      BY <1>1, <1>5, <2>2 DEF DeferOk
  <1>9. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY <1>1, CopiesOkEach
    <2>2. CopyOk(c)
      BY CopiesOkEach DEF IndInv
    <2>3. (c.kind \in {"clone", "load"} /\ ~c.fill) => c.write \notin t0.units
      BY <1>2, <1>3
    <2>4. \A t \in tasks' : t.kind = "transfer" => t \in tasks
      BY <1>1, <1>3
    <2> QED
      BY <1>1, <1>5, <2>2, <2>3, <2>4 DEF CopyOk, Live, Mem
  <1>10. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. CASE t = t0
      BY <1>3, <2>2 DEF TaskOk
    <2>3. CASE t # t0
      <3>1. t \in tasks /\ TaskOk(t)
        BY <1>1, <2>3, TasksOkEach DEF IndInv
      <3>2. t.kind = "transfer" => t.slot \in Slots /\ t.slot \notin t0.units
        BY <1>2, <1>3, <3>1 DEF TaskOk
      <3> QED
        BY <1>1, <1>3, <1>5, <3>1, <3>2 DEF TaskOk, Live, TableUnits, Mem
    <2> QED
      BY <2>2, <2>3
  <1>11. CrossingsOk'
    <2>1. \A x \in deferred' : CrossWaitOk(x)'
      <3> SUFFICES ASSUME NEW x \in deferred' PROVE CrossWaitOk(x)'
        OBVIOUS
      <3>1. CrossWaitOk(x)
        BY <1>5, CrossingsOkEach DEF IndInv
      <3> QED
        BY <1>1, <1>5, <3>1 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>2. CrossTaskOk(t0)'
      <3>1. \A x \in tasks' :
              (x.kind \in {"transfer", "replicate"} /\ x # t0 /\ Mem(x.to) = Mem(t0.to))
                  => x.writes \cap t0.writes = {}
        BY <1>1, <1>3
      <3>2. \A c \in copies : (c.fill /\ c.kind = "load" /\ Mem(c.g) = Mem(t0.to))
                                => c.write \notin t0.writes
        BY <1>1, <1>3
      <3> QED
        BY <1>1, <1>2, <1>3, <1>5, <3>1, <3>2 DEF CrossTaskOk, ReadersOf
    <2>3. \A t \in tasks : CrossTaskOk(t)'
      <3> SUFFICES ASSUME NEW t \in tasks PROVE CrossTaskOk(t)'
        OBVIOUS
      <3>1. CrossTaskOk(t)
        BY CrossingsOkEach DEF IndInv
      <3>2. (t.kind = "replicate" /\ Mem(t0.to) = Mem(t.to)) => t0.writes \cap t.writes = {}
        BY <1>1, <1>3
      <3> QED
        BY <1>1, <1>3, <1>5, <3>1, <3>2 DEF CrossTaskOk, ReadersOf, Mem
    <2> QED
      BY <1>1, <2>1, <2>2, <2>3, CrossingsOkEach
  <1>12. MovingUnmapped'
    BY <1>1, <1>5 DEF IndInv, MovingUnmapped, Transfers
  (* The block was held by a waiting crossing, so no row or completion that *)
  (* may write it exists; now its crossing's task holds it.                *)
  <1>13. TablesMapped' /\ HandoffOk'
    <2>1. \A x : (\E t \in tasks' : t.kind = "replicate" /\ x \in t.units)
                   => (\E t \in tasks : t.kind = "replicate" /\ x \in t.units)
                      \/ (\E e \in deferred : e.kind = "replicate" /\ x \in e.units)
      BY <1>1, <1>2, <1>3
    <2>2. \A x : (\E e \in deferred' : e.kind \in {"commit", "replicate"} /\ x \in e.units)
                   => (\E e \in deferred : e.kind \in {"commit", "replicate"} /\ x \in e.units)
      BY <1>5
    <2>3. TablesMapped'
      BY <1>1, <2>1, <2>2 DEF IndInv, TablesMapped, Live, HoldsAt, ReaderRow, ReadersOf, Mem
    <2>4. HandoffOk'
      BY <1>1, <2>1, <2>2 DEF IndInv, HandoffOk, Live, HoldsAt, ReaderRow, ReadersOf, Mem
    <2> QED
      BY <2>3, <2>4
  <1>14. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A x : Covered(b, o, x) => Covered(b, o, x)'
      <3> SUFFICES ASSUME NEW o \in Owners, NEW x, Covered(b, o, x) PROVE Covered(b, o, x)'
        OBVIOUS
      <3>1. \A e \in deferred : e.kind \in {"release", "transfer"} => e \in deferred'
        BY <1>1
      <3>2. (\E t \in tasks' : t.kind = "transfer" /\ t.o = o)
              => (\E t \in tasks : t.kind = "transfer" /\ t.o = o)
        BY <1>1, <1>3
      <3> QED
        BY <1>1, <1>5, <3>1, <3>2 DEF Covered, Live, HoldsAt, ReadersOf
    <2> QED
      BY <1>1, <1>5, <2>2, <2>3 DEF BatchOk, Mem
  <1>15. TrialVisibility'
    <2>1. \A k \in Keys : cache[k] \in t0.units => vis[k].d = MadeBy(OwnerEpoch[t0.o])
      BY <1>2, <1>3
    <2> QED
      BY <1>1, <1>3, <2>1 DEF IndInv, TrialVisibility, MadeBy
  <1>16. CrossingsPlaced'
    <2>1. CrossingPlaced(t0)'
      BY <1>1, <1>2, <1>3 DEF CrossingPlaced, Mem
    <2>2. \A t \in tasks : CrossingPlaced(t)'
      BY <1>1, PlacedEach DEF IndInv, CrossingPlaced, Mem
    <2> QED
      BY <1>1, <2>1, <2>2, PlacedEach
  <1> QED
    BY <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16 DEF IndInv

(* The crossing comes due and its owner no longer holds the block where it *)
(* named it: it lets go of the block.                                      *)
LEMMA ReplicateDue_Drop ==
  ASSUME IndInv, NEW d \in deferred, ReplicateDue(d), UNCHANGED rare, ~Holding(d),
         Drop(d.units, table, oslot, cache, hcache, deferred', copies),
         UNCHANGED <<tasks, dbt, handoff>>
  PROVE  IndInv'
  <1>1. /\ d.kind = "replicate"
        /\ deferred' = deferred \ {d}
        /\ DropWith(d.units, table, oslot, cache, hcache, deferred', copies, tasks)
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks, trial,
                       member, valid, vis, dbt, launched, inflight, handoff, requests,
                       remote, lost, bad>>
    BY ReplicateDue_Effect DEF Drop, rare
  <1>2. /\ d.units = d.units \cup UNION {{c.read, c.write} : c \in {}}
        /\ copies' = copies \ {}
    BY <1>1
  <1>3. ValidTyped'
    BY <1>1, UnitTypes DEF UnitTyped, ValidTyped
  <1> QED
    BY <1>1, <1>2, <1>3, Retire


(* The block became valid in the reading group's memory while the crossing *)
(* waited: it lets go of the block, which its owner still holds, and the   *)
(* reading group is handed the entry.                                      *)
LEMMA ReplicateDue_Answer ==
  ASSUME IndInv, NEW d \in deferred, ReplicateDue(d), UNCHANGED rare,
         Holding(d), Mem(d.to) \in valid[Crossed(d)],
         Drop(d.units, table, oslot, cache, hcache, deferred', copies), UNCHANGED tasks,
         Give(Reading(d))
  PROVE  IndInv'
  <1> DEFINE h == Reading(d)
             u == Crossed(d)
  <1>0. PlaneTyped /\ RequestTyped /\ WorkTyped
    BY PlaneTypes, RequestTypes, WorkTypes
  <1>1. /\ d.kind = "replicate"
        /\ deferred' = deferred \ {d}
        /\ DropWith(d.units, table, oslot, cache, hcache, deferred', copies, tasks)
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks, valid, vis,
                       trial, member, launched, inflight, requests, remote, lost, bad>>
    BY ReplicateDue_Effect DEF Drop, rare
  <1>2. /\ d.o \in Owners /\ d.p \in Positions /\ d.to \in Groups
        /\ d.to \in ReadersOf(d.o) /\ ~Moving(d.o)
        /\ d.units = {u} /\ u \in Units
        /\ Live(d.o) /\ table[d.o][d.p].u = u
        /\ GroupEpoch[d.to] = OwnerEpoch[d.o] /\ OwnerEpoch[d.o] \notin lost
    BY ReplicateDue_Facts DEF Holding, Live
  (* The owner's table holds the block: nothing is reclaimed.             *)
  <1>3. UNCHANGED <<free, alloc, sharedKey>>
    <2>1. HeldIn(u, table, oslot, cache, hcache, deferred', copies, tasks)
      BY <1>2 DEF HeldIn
    <2>2. ReclaimedWith(d.units, table, oslot, cache, hcache, deferred', copies, tasks) = {}
      BY <1>2, <2>1 DEF ReclaimedWith, On, AllRules
    <2>3. Reset(alloc, {}) = alloc /\ Reset(sharedKey, {}) = sharedKey
      BY UnitTypes DEF UnitTyped, Reset
    <2> QED
      BY <1>1, <2>2, <2>3 DEF DropWith
  <1>4. /\ h.g = d.to /\ h.o = d.o /\ h.p = d.p /\ h.u = u /\ h.w = FALSE
        /\ h.p \in Rows /\ h.u \in AllUnits
        /\ ReaderRow(h.g, h.o, h.p, h.u, h.w)
        /\ HoldsAt(h.o, h.p, h.u)
    BY <1>2 DEF Reading, Rows, AllUnits, ReaderRow, HoldsAt, Crossed
  <1>5. /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {h}
        /\ \A x \in Groups, q \in Owners, r \in Rows :
             dbt'[x][q][r] = IF Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h)
                             THEN [u |-> h.u, w |-> h.w]
                             ELSE dbt[x][q][r]
        /\ DOMAIN dbt' = Groups
        /\ \A x \in Groups : DOMAIN dbt'[x] = Owners
        /\ \A x \in Groups, q \in Owners : DOMAIN dbt'[x][q] = Rows
    BY <1>0, <1>2, <1>4, GiveRow DEF PlaneTyped
  <1>6. /\ \A x : Held(x)' = Held(x)
        /\ \A x \in deferred' : x \in deferred /\ x # d
        /\ \A o \in Owners : Moving(o)' = Moving(o)
    <2>1. \A x : (\E e \in deferred : x \in e.units) => (\E e \in deferred' : x \in e.units) \/ x = u
      BY <1>1, <1>2
    <2>2. \A x : (\E e \in deferred' : x \in e.units) => (\E e \in deferred : x \in e.units)
      BY <1>1
    <2>3. Held(u)'
      BY <1>1, <1>2 DEF Held, HeldIn
    <2>4. \A x : Held(x)' = Held(x)
      <3> SUFFICES ASSUME NEW x PROVE Held(x)' = Held(x)
        OBVIOUS
      <3>1. x = u => \E o \in Owners, p \in Positions : table[o][p].u = x
        BY <1>2
      <3> QED
        BY <1>1, <1>3, <2>1, <2>2, <3>1, HeldUnfold
    <2>5. \A x \in deferred' : x \in deferred /\ x # d
      BY <1>1
    <2> QED
      BY <1>1, <1>2, <2>4, <2>5 DEF Moving
  <1>7. TypeInv'
    <2>1. PlaneTyped'
      BY <1>0, <1>1, <1>4, <1>5 DEF PlaneTyped
    <2>2. RequestTyped'
      BY <1>0, <1>1, <1>2, <1>4, <1>5 DEF RequestTyped
    <2>3. WorkTyped'
      BY <1>0, <1>1, <1>6 DEF WorkTyped
    <2> QED
      BY <1>1, <1>3, <2>1, <2>2, <2>3, UnitTypes, OwnerTypes, UnitTypedFrame, OwnerTypedFrame,
         TypeInvNext
  <1>8. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ TrialVisibility' /\ NoBad'
        /\ CopiesPlaced' /\ CrossingsPlaced'
    BY <1>1, <1>3, <1>6, TransfersFrame, OutsideFrame, RegionsFrame, AllocOwnerFrame,
       SharedIffFrame, SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame,
       TrialVisibilityFrame, NoBadFrame, CopiesPlacedFrame, CrossingsPlacedFrame
       DEF IndInv, FreeClean, NoLeak, InRegion
  <1>9. DeferredOk' /\ CopiesOk' /\ TasksOk' /\ CrossingsOk'
    <2>1. DeferredOk'
      <3>1. SUFFICES ASSUME NEW x \in deferred' PROVE DeferOk(x)'
        BY DeferredOkEach
      <3>2. x \in deferred /\ DeferOk(x)
        BY <1>6, DeferredOkEach DEF IndInv
      <3> QED
        BY <1>1, <1>3, <1>6, <3>2 DEF DeferOk
    <2>2. CopiesOk'
      <3>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
        BY <1>1, CopiesOkEach
      <3> QED
        BY <1>1, <1>3, <1>6, CopiesOkEach DEF CopyOk, Live, Mem, IndInv
    <2>3. TasksOk'
      <3>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
        BY <1>1, TasksOkEach
      <3>2. TaskOk(t)
        BY TasksOkEach DEF IndInv
      <3> QED
        BY <1>1, <1>3, <1>6, <3>2 DEF TaskOk, Live, TableUnits, Mem
    <2>4. CrossingsOk'
      <3>1. \A x \in deferred' : CrossWaitOk(x)'
        BY <1>1, <1>3, <1>6, CrossingsOkEach DEF IndInv, CrossWaitOk, ReadersOf, MadeBy
      <3>2. \A t \in tasks' : CrossTaskOk(t)'
        BY <1>1, <1>3, <1>6, CrossingsOkEach DEF IndInv, CrossTaskOk, ReadersOf, Mem
      <3> QED
        BY <3>1, <3>2, CrossingsOkEach
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4
  <1>10. MovingUnmapped'
    <2> SUFFICES ASSUME NEW q \in Owners,
                        \/ \E t \in tasks : t.kind = "transfer" /\ t.o = q
                        \/ \E e \in deferred' : e.kind = "transfer" /\ e.o = q
                 PROVE  /\ \A x \in Groups, r \in Rows : dbt'[x][q][r].u = NONE
                        /\ \A y \in handoff' : y.o # q
                        /\ \A e1, e2 \in deferred' :
                             (e1.kind = "transfer" /\ e2.kind = "transfer" /\ e1.o = q /\ e2.o = q)
                                 => e1 = e2
      BY <1>1 DEF MovingUnmapped
    <2>1. q # d.o
      BY <1>2, <1>6 DEF Moving
    <2> QED
      BY <1>4, <1>5, <1>6, <2>1 DEF IndInv, MovingUnmapped
  <1>11. LostOk'
    BY <1>1, <1>2, <1>4, <1>5 DEF IndInv, LostOk
  <1>12. TablesMapped'
    <2> SUFFICES ASSUME NEW x \in Groups, NEW q \in Owners, NEW r \in Rows, dbt'[x][q][r].u # NONE
                 PROVE  /\ ostate'[q] = "live"
                        /\ \/ ogroup'[q] = x
                           \/ ReaderRow(x, q, r, dbt[x][q][r].u, dbt[x][q][r].w)'
                        /\ HoldsAt(q, r, dbt[x][q][r].u)'
                        /\ (r \in Positions /\ dbt'[x][q][r].w) =>
                             /\ table'[q][r].k = "private"
                             /\ ~\E e \in deferred' :
                                   e.kind \in {"commit", "replicate"} /\ dbt'[x][q][r].u \in e.units
                             /\ ~\E t \in tasks' : t.kind = "replicate" /\ dbt'[x][q][r].u \in t.units
      BY DEF TablesMapped, Live
    <2>1. CASE Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h)
      <3>1. dbt'[x][q][r] = [u |-> u, w |-> FALSE]
        BY <2>1, <1>4, <1>5
      <3> QED
        BY <1>1, <1>2, <1>4, <2>1, <3>1 DEF Live, ReaderRow, ReadersOf, Mem, HoldsAt
    <2>2. CASE ~(Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h))
      <3>1. dbt'[x][q][r] = dbt[x][q][r]
        BY <2>2, <1>5
      <3> QED
        BY <1>1, <1>6, <3>1 DEF IndInv, TablesMapped, Live, ReaderRow, ReadersOf, Mem, HoldsAt
    <2> QED
      BY <2>1, <2>2
  <1>13. HandoffOk'
    <2> SUFFICES ASSUME NEW y \in handoff'
                 PROVE  /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
                        /\ (~\E x \in requests' : Names(x, y)) =>
                             /\ ostate'[y.o] = "live"
                             /\ ogroup'[y.o] = y.g \/ ReaderRow(y.g, y.o, y.p, y.u, y.w)'
                             /\ HoldsAt(y.o, y.p, y.u)'
                             /\ (y.p \in Positions /\ y.w) =>
                                  /\ table'[y.o][y.p].k = "private"
                                  /\ ~\E e \in deferred' :
                                        e.kind \in {"commit", "replicate"} /\ y.u \in e.units
                                  /\ ~\E t \in tasks' : t.kind = "replicate" /\ y.u \in t.units
      BY DEF HandoffOk, Live
    <2>1. CASE y = h
      BY <1>1, <1>2, <1>4, <2>1 DEF Live, ReaderRow, ReadersOf, Mem, HoldsAt
    <2>2. CASE y # h
      <3>1. y \in handoff
        BY <1>5, <2>2
      <3> QED
        BY <1>1, <1>6, <3>1 DEF IndInv, HandoffOk, Live, ReaderRow, ReadersOf, Mem, HoldsAt
    <2> QED
      BY <2>1, <2>2
  <1>14. RequestsOk'
    <2>1. \A x \in requests :
            x.kind # "finish" =>
              \A z \in Groups, q \in Rows :
                (dbt'[z][x.o][q].u # NONE /\ (TakesRow(x, x.o, q) \/ FreezesRow(x, x.o, q))) =>
                  /\ z \in x.sg
                  /\ z \in x.by => FreezesRow(x, x.o, q) /\ ~dbt'[z][x.o][q].w
      <3> SUFFICES ASSUME NEW x \in requests, x.kind # "finish", NEW z \in Groups, NEW q \in Rows,
                          dbt'[z][x.o][q].u # NONE, TakesRow(x, x.o, q) \/ FreezesRow(x, x.o, q)
                   PROVE  /\ z \in x.sg
                          /\ z \in x.by => FreezesRow(x, x.o, q) /\ ~dbt'[z][x.o][q].w
        OBVIOUS
      <3>1. x.o \in Owners
        BY <1>0 DEF RequestTyped
      <3>2. CASE Eager /\ z = h.g /\ x.o = h.o /\ q = h.p /\ Unnamed(h)
        <4>1. Names(x, h)
          BY <3>2, NamesRow
        <4> QED
          BY <3>2, <4>1 DEF Unnamed
      <3>3. CASE ~(Eager /\ z = h.g /\ x.o = h.o /\ q = h.p /\ Unnamed(h))
        <4>1. dbt'[z][x.o][q] = dbt[z][x.o][q]
          BY <1>5, <3>1, <3>3
        <4> QED
          BY <4>1 DEF IndInv, RequestsOk
      <3> QED
        BY <3>2, <3>3
    <2> QED
      BY <1>1, <2>1 DEF IndInv, RequestsOk
  <1>15. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A x : Covered(b, o, x) => Covered(b, o, x)'
      <3> SUFFICES ASSUME NEW o \in Owners, NEW x, Covered(b, o, x) PROVE Covered(b, o, x)'
        OBVIOUS
      <3>1. \A e \in deferred : e.kind \in {"release", "transfer"} => e \in deferred'
        BY <1>1, <1>6
      <3> QED
        BY <1>1, <3>1 DEF Covered, Live, HoldsAt, ReadersOf
    <2> QED
      BY <1>1, <1>3, <1>6, <2>2, <2>3 DEF BatchOk, Mem
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15 DEF IndInv

LEMMA ReplicateDue_Inv ==
  ASSUME IndInv, NEW d \in deferred, ReplicateDue(d), UNCHANGED rare
  PROVE  IndInv'
  BY ReplicateDue_Effect, ReplicateDue_Start, ReplicateDue_Answer, ReplicateDue_Drop

-----------------------------------------------------------------------------
(* ReplicateEnd                                                            *)

(* The block a crossing's task holds, whether its owner still holds it at  *)
(* the position the crossing named, the validity after a task that         *)
(* succeeded, and the entry the reading group is handed.                   *)
Ended(t) == CHOOSE x \in t.units : TRUE
Back(t) == ostate[t.o] = "live" /\ table[t.o][t.p].u = Ended(t)
Grown(t) == [valid EXCEPT ![Ended(t)] = @ \cup {Mem(t.to)}]
Handed(t) == [g |-> t.to, o |-> t.o, p |-> t.p, u |-> Ended(t), w |-> FALSE]

LEMMA ReplicateEnd_Effect ==
  ASSUME NEW t \in tasks, NEW ok \in BOOLEAN, ReplicateEnd(t, ok)
  PROVE  /\ t.kind = "replicate" /\ tasks' = tasks \ {t}
         /\ valid' = IF ok THEN Grown(t) ELSE valid
         /\ \/ /\ ok /\ Back(t) /\ Give(Handed(t))
            \/ /\ ~(ok /\ Back(t)) /\ UNCHANGED <<dbt, handoff>>
         /\ DropWith(t.units, table, oslot, cache, hcache, deferred, copies, tasks')
         /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, vis,
                        trial, member, launched, inflight, requests>>
  BY DEF ReplicateEnd, Ended, Back, Grown, Handed

(* What CrossingsOk and CrossingsPlaced say of the task that ends.         *)
LEMMA ReplicateEnd_Facts ==
  ASSUME IndInv, NEW t \in tasks, t.kind = "replicate"
  PROVE  /\ t.held /\ t.h = NONE /\ t.slot = NONE
         /\ t.o \in Owners /\ t.p \in Positions /\ t.to \in Groups
         /\ ostate[t.o] # "new" /\ t.to \in ReadersOf(t.o) /\ ~Moving(t.o)
         /\ t.units = {Ended(t)} /\ t.writes = {Ended(t)} /\ Ended(t) \in Units
         /\ Mem(t.to) \notin valid[Ended(t)]
         /\ alloc[Ended(t)] = t.o \/ alloc[Ended(t)] = "shared"
         /\ \A x \in tasks :
              (x.kind \in {"transfer", "replicate"} /\ x # t /\ Mem(x.to) = Mem(t.to))
                  => Ended(t) \notin x.writes
         /\ \A c \in copies : (c.fill /\ c.kind = "load" /\ Mem(c.g) = Mem(t.to))
                                => c.write # Ended(t)
         /\ Live(t.o) => GroupEpoch[t.to] = OwnerEpoch[t.o] /\ OwnerEpoch[t.o] \notin lost
  <1>1. CrossTaskOk(t) /\ t.held
    BY CrossingsOkEach DEF IndInv, TasksOk
  <1>2. \E u \in Units : t.units = {u} /\ t.writes = {u}
    BY <1>1 DEF CrossTaskOk
  <1>3. t.units = {Ended(t)} /\ t.writes = {Ended(t)} /\ Ended(t) \in Units
    BY <1>2 DEF Ended
  <1>4. Live(t.o) => GroupEpoch[t.to] = OwnerEpoch[t.o] /\ OwnerEpoch[t.o] \notin lost
    BY <1>1 DEF CrossTaskOk, IndInv, OwnersOk, Live, ReadersOf
  <1> QED
    BY <1>1, <1>3, <1>4 DEF CrossTaskOk

(* The task succeeded and the owner still holds the block where the        *)
(* crossing named it: the block is valid in the reading group's memory,    *)
(* which is handed the entry; nothing is reclaimed.                       *)
LEMMA ReplicateEnd_Back ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, ReplicateEnd(t, ok), UNCHANGED rare,
         ok, Back(t), Give(Handed(t))
  PROVE  IndInv'
  <1> DEFINE h == Handed(t)
             u == Ended(t)
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ RequestTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, RequestTypes, WorkTypes
  <1>1. /\ t.kind = "replicate" /\ tasks' = tasks \ {t}
        /\ valid' = Grown(t)
        /\ DropWith(t.units, table, oslot, cache, hcache, deferred, copies, tasks')
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, vis,
                       trial, member, launched, inflight, requests, remote, lost, bad>>
    BY ReplicateEnd_Effect DEF rare
  <1>2. /\ t.o \in Owners /\ t.p \in Positions /\ t.to \in Groups
        /\ t.to \in ReadersOf(t.o) /\ ~Moving(t.o) /\ t.h = NONE /\ t.slot = NONE
        /\ t.units = {u} /\ t.writes = {u} /\ u \in Units
        /\ Mem(t.to) \notin valid[u]
        /\ Live(t.o) /\ table[t.o][t.p].u = u
        /\ GroupEpoch[t.to] = OwnerEpoch[t.o] /\ OwnerEpoch[t.o] \notin lost
        /\ \A x \in tasks :
             (x.kind \in {"transfer", "replicate"} /\ x # t /\ Mem(x.to) = Mem(t.to)) => u \notin x.writes
        /\ \A c \in copies : (c.fill /\ c.kind = "load" /\ Mem(c.g) = Mem(t.to)) => c.write # u
    BY <1>1, ReplicateEnd_Facts DEF Back, Live
  (* The owner's table holds the block: nothing is reclaimed.             *)
  <1>3. UNCHANGED <<free, alloc, sharedKey>>
    <2>1. HeldIn(u, table, oslot, cache, hcache, deferred, copies, tasks')
      BY <1>2 DEF HeldIn
    <2>2. ReclaimedWith(t.units, table, oslot, cache, hcache, deferred, copies, tasks') = {}
      BY <1>2, <2>1 DEF ReclaimedWith, On, AllRules
    <2>3. Reset(alloc, {}) = alloc /\ Reset(sharedKey, {}) = sharedKey
      BY <1>0 DEF UnitTyped, Reset
    <2> QED
      BY <1>1, <2>2, <2>3 DEF DropWith
  (* Validity grows by the reading group's memory, for the block alone.   *)
  <1>4. /\ \A v \in Units \cup HostUnits : v # u => valid'[v] = valid[v]
        /\ valid'[u] = valid[u] \cup {Mem(t.to)}
        /\ \A v \in Units \cup HostUnits : valid[v] \subseteq valid'[v]
        /\ ValidTyped'
    <2>1. /\ valid \in [Units \cup HostUnits -> SUBSET (Memories \cup HostParts)]
          /\ \A v \in Units : valid[v] \subseteq Memories
          /\ \A v \in HostUnits : valid[v] \subseteq HostParts
      BY <1>0 DEF UnitTyped
    <2>2. Mem(t.to) \in Memories
      BY <1>2 DEF Mem, Memories
    <2>3. valid' = [valid EXCEPT ![u] = valid[u] \cup {Mem(t.to)}]
      BY <1>1 DEF Grown, Ended
    <2>4. \A v \in Units \cup HostUnits : v # u => valid'[v] = valid[v]
      BY <2>1, <2>3
    <2>5. valid'[u] = valid[u] \cup {Mem(t.to)}
      BY <1>2, <2>1, <2>3
    <2>6. u \notin HostUnits
      BY <1>2
    <2>7. ValidTyped'
      BY <1>2, <2>1, <2>2, <2>3, <2>4, <2>5, <2>6 DEF ValidTyped
    <2> QED
      BY <2>4, <2>5, <2>7
  <1>5. /\ h.g = t.to /\ h.o = t.o /\ h.p = t.p /\ h.u = u /\ h.w = FALSE
        /\ h.p \in Rows /\ h.u \in AllUnits
        /\ ReaderRow(h.g, h.o, h.p, h.u, h.w)'
        /\ HoldsAt(h.o, h.p, h.u)'
    BY <1>1, <1>2, <1>4 DEF Handed, Rows, AllUnits, ReaderRow, ReadersOf, Mem, HoldsAt, Ended
  <1>6. /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {h}
        /\ \A x \in Groups, q \in Owners, r \in Rows :
             dbt'[x][q][r] = IF Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h)
                             THEN [u |-> h.u, w |-> h.w]
                             ELSE dbt[x][q][r]
        /\ DOMAIN dbt' = Groups
        /\ \A x \in Groups : DOMAIN dbt'[x] = Owners
        /\ \A x \in Groups, q \in Owners : DOMAIN dbt'[x][q] = Rows
    BY <1>0, <1>2, <1>5, GiveRow DEF PlaneTyped
  <1>7. /\ \A x \in AllUnits : Held(x)' = Held(x)
        /\ \A o \in Owners : Moving(o)' = Moving(o)
        /\ Transfers' = Transfers
    <2>1. Held(u)'
      BY <1>1, <1>2 DEF Held, HeldIn
    <2>2. \A x \in AllUnits :
            HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
                => HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks') \/ x = u
      <3>1. NONE \notin AllUnits
        BY DEF AllUnits
      <3> QED
        BY <1>1, <1>2, <3>1 DEF HeldIn
    <2>3. \A x : HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks')
                   => HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
      BY <1>1 DEF HeldIn
    <2>4. \A x \in AllUnits : Held(x)' = Held(x)
      <3> SUFFICES ASSUME NEW x \in AllUnits PROVE Held(x)' = Held(x)
        OBVIOUS
      <3>1. (\E y \in tasks' : y.held /\ (x \in y.units \/ x = y.slot \/ x = y.h))
              => (\E y \in tasks : y.held /\ (x \in y.units \/ x = y.slot \/ x = y.h))
        BY <1>1
      <3>2. (\E y \in tasks : y.held /\ (x \in y.units \/ x = y.slot \/ x = y.h))
              => (\E y \in tasks' : y.held /\ (x \in y.units \/ x = y.slot \/ x = y.h)) \/ x = u
        <4>1. x # NONE
          BY DEF AllUnits
        <4> QED
          BY <1>1, <1>2, <4>1
      <3>3. x = u => \E o \in Owners, p \in Positions : table[o][p].u = x
        BY <1>2
      <3> QED
        BY <1>1, <1>3, <3>1, <3>2, <3>3, HeldUnfold
    <2> QED
      BY <1>1, <2>4 DEF Moving, Transfers
  <1>8. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>3, <1>4 DEF UnitTyped, ValidTyped
    <2>2. PlaneTyped'
      BY <1>0, <1>1, <1>5, <1>6 DEF PlaneTyped
    <2>3. RequestTyped'
      BY <1>0, <1>1, <1>2, <1>5, <1>6 DEF RequestTyped
    <2>4. WorkTyped'
      BY <1>0, <1>1 DEF WorkTyped
    <2> QED
      BY <1>1, <2>1, <2>2, <2>3, <2>4, OwnerTypes, OwnerTypedFrame, TypeInvNext
  <1>9. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ OwnersOk' /\ NoBad'
    BY <1>0, <1>1, <1>3, <1>7, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
       SharedCachedFrame, CacheSharedFrame, OwnersOkFrame, NoBadFrame
       DEF IndInv, FreeClean, NoLeak, InRegion, UnitTyped
  <1>10. Tables'
    BY <1>1, <1>3, <1>4, OwnerTypes DEF IndInv, Tables, Live, Mem, OwnerTyped, Empty, Private,
                                        Shared
  <1>11. DeferredOk'
    BY <1>1, <1>3 DEF IndInv, DeferredOk
  <1>12. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY <1>1, CopiesOkEach
    <2>2. CopyOk(c) /\ c.write \in AllUnits
      BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. (c.fill /\ c.kind = "load") => Mem(c.g) \notin valid'[c.write]
      <3>1. (c.fill /\ c.kind = "load") => Mem(c.g) \notin valid[c.write] /\ c.write \in Units
        BY <1>0, <2>2 DEF CopyOk, UnitTyped, WorkTyped, AllUnits
      <3>2. (c.fill /\ c.kind = "load" /\ c.write = u) => Mem(c.g) # Mem(t.to)
        BY <1>2
      <3> QED
        BY <1>4, <3>1, <3>2
    <2>4. \A x \in tasks' : x \in tasks
      BY <1>1
    <2> QED
      BY <1>1, <1>3, <1>7, <2>2, <2>3, <2>4 DEF CopyOk, Live, Mem
  <1>13. TasksOk'
    <2>1. SUFFICES ASSUME NEW x \in tasks' PROVE TaskOk(x)'
      BY TasksOkEach
    <2>2. x \in tasks /\ x # t /\ TaskOk(x)
      BY <1>1, TasksOkEach DEF IndInv
    <2>3. x.kind = "transfer" =>
            /\ \A v \in x.writes : Mem(x.to) \notin valid'[v]
            /\ Live(x.o) => \A v \in TableUnits(x.o) \ x.writes : Mem(x.to) \in valid'[v]
      <3>1. x.kind = "transfer" =>
              /\ x.writes \subseteq Units /\ \A v \in x.writes : Mem(x.to) \notin valid[v]
              /\ Live(x.o) => \A v \in TableUnits(x.o) \ x.writes : Mem(x.to) \in valid[v]
        BY <2>2 DEF TaskOk
      <3>2. (x.kind = "transfer" /\ Mem(x.to) = Mem(t.to)) => u \notin x.writes
        BY <1>2, <2>2
      <3>3. x.kind = "transfer" => \A v \in TableUnits(x.o) : v \in Units
        BY <1>0, <2>2 DEF TaskOk, TableUnits, OwnerTyped, Empty, Private, Shared
      <3>4. ASSUME x.kind = "transfer", NEW v \in x.writes PROVE Mem(x.to) \notin valid'[v]
        <4>1. v \in Units /\ Mem(x.to) \notin valid[v]
          BY <3>1, <3>4
        <4>2. CASE v = u
          <5>1. Mem(x.to) # Mem(t.to)
            BY <3>2, <3>4, <4>2
          <5> QED
            BY <1>4, <4>1, <4>2, <5>1
        <4>3. CASE v # u
          BY <1>4, <4>1, <4>3
        <4> QED
          BY <4>2, <4>3
      <3>5. ASSUME x.kind = "transfer", Live(x.o), NEW v \in TableUnits(x.o) \ x.writes
            PROVE  Mem(x.to) \in valid'[v]
        BY <1>4, <3>1, <3>3, <3>5
      <3> QED
        BY <3>4, <3>5
    <2> QED
      BY <1>1, <1>3, <2>2, <2>3 DEF TaskOk, Live, TableUnits, Mem
  <1>14. CrossingsOk'
    <2>1. \A x \in deferred' : CrossWaitOk(x)'
      BY <1>1, <1>3, <1>7, CrossingsOkEach DEF IndInv, CrossWaitOk, ReadersOf, MadeBy
    <2>2. \A x \in tasks' : CrossTaskOk(x)'
      <3> SUFFICES ASSUME NEW x \in tasks' PROVE CrossTaskOk(x)'
        OBVIOUS
      <3>1. x \in tasks /\ x # t /\ CrossTaskOk(x)
        BY <1>1, CrossingsOkEach DEF IndInv
      <3>2. x.kind = "replicate" =>
              /\ x.writes \subseteq Units
              /\ \A v \in x.writes : Mem(x.to) \notin valid'[v]
        <4>1. x.kind = "replicate" => x.writes \subseteq Units /\ \A v \in x.writes : Mem(x.to) \notin valid[v]
          BY <3>1 DEF CrossTaskOk
        <4>2. (x.kind = "replicate" /\ Mem(x.to) = Mem(t.to)) => u \notin x.writes
          BY <1>2, <3>1
        <4> QED
          BY <1>4, <4>1, <4>2
      <3> QED
        BY <1>1, <1>3, <1>7, <3>1, <3>2 DEF CrossTaskOk, ReadersOf, Mem
    <2> QED
      BY <2>1, <2>2, CrossingsOkEach
  <1>15. MovingUnmapped'
    <2> SUFFICES ASSUME NEW q \in Owners,
                        \/ \E x \in tasks' : x.kind = "transfer" /\ x.o = q
                        \/ \E e \in deferred : e.kind = "transfer" /\ e.o = q
                 PROVE  /\ \A x \in Groups, r \in Rows : dbt'[x][q][r].u = NONE
                        /\ \A y \in handoff' : y.o # q
                        /\ \A e1, e2 \in deferred :
                             (e1.kind = "transfer" /\ e2.kind = "transfer" /\ e1.o = q /\ e2.o = q)
                                 => e1 = e2
      BY <1>1 DEF MovingUnmapped
    <2>1. q # t.o
      BY <1>1, <1>2 DEF Moving
    <2> QED
      BY <1>1, <1>5, <1>6, <2>1 DEF IndInv, MovingUnmapped
  <1>16. LostOk'
    BY <1>1, <1>2, <1>5, <1>6 DEF IndInv, LostOk
  <1>17. TablesMapped'
    <2> SUFFICES ASSUME NEW x \in Groups, NEW q \in Owners, NEW r \in Rows, dbt'[x][q][r].u # NONE
                 PROVE  /\ ostate'[q] = "live"
                        /\ \/ ogroup'[q] = x
                           \/ /\ x \in ReadersOf(q)' /\ r \in Positions /\ ~dbt'[x][q][r].w
                              /\ Mem(x) \in valid'[dbt'[x][q][r].u]
                        /\ IF r = SLOT THEN oslot'[q] = dbt'[x][q][r].u
                                       ELSE table'[q][r].u = dbt'[x][q][r].u
                        /\ (r \in Positions /\ dbt'[x][q][r].w) =>
                             /\ table'[q][r].k = "private"
                             /\ ~\E e \in deferred' :
                                   e.kind \in {"commit", "replicate"} /\ dbt'[x][q][r].u \in e.units
                             /\ ~\E y \in tasks' : y.kind = "replicate" /\ dbt'[x][q][r].u \in y.units
      BY DEF TablesMapped, Live, ReaderRow, HoldsAt
    <2>0. ReadersOf(q)' = ReadersOf(q)
      BY <1>1 DEF ReadersOf
    <2>1. CASE Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h)
      <3>1. dbt'[x][q][r] = [u |-> u, w |-> FALSE] /\ x = t.to /\ q = t.o /\ r = t.p
        BY <2>1, <1>5, <1>6
      <3>2. Mem(t.to) \in valid'[u]
        BY <1>4
      <3> QED
        BY <1>1, <1>2, <2>0, <3>1, <3>2, SLOT \notin Positions DEF Live
    <2>2. CASE ~(Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h))
      <3>1. dbt'[x][q][r] = dbt[x][q][r]
        BY <2>2, <1>6
      <3>2. /\ ostate[q] = "live"
            /\ \/ ogroup[q] = x
               \/ /\ x \in ReadersOf(q) /\ r \in Positions /\ ~dbt[x][q][r].w
                  /\ Mem(x) \in valid[dbt[x][q][r].u]
            /\ IF r = SLOT THEN oslot[q] = dbt[x][q][r].u ELSE table[q][r].u = dbt[x][q][r].u
            /\ (r \in Positions /\ dbt[x][q][r].w) =>
                 /\ table[q][r].k = "private"
                 /\ ~\E e \in deferred :
                       e.kind \in {"commit", "replicate"} /\ dbt[x][q][r].u \in e.units
                 /\ ~\E y \in tasks : y.kind = "replicate" /\ dbt[x][q][r].u \in y.units
        BY <3>1 DEF IndInv, TablesMapped, Live, ReaderRow, HoldsAt
      (* A reader's row names a block of a table, which stays valid where it  *)
      (* was.                                                                *)
      <3>3. r \in Positions => dbt[x][q][r].u \in Units
        BY <3>1, <3>2, OwnerTypes DEF OwnerTyped, Empty, Private, Shared
      <3>4. r \in Positions => valid[dbt[x][q][r].u] \subseteq valid'[dbt[x][q][r].u]
        BY <1>4, <3>3
      <3> QED
        BY <1>1, <2>0, <3>1, <3>2, <3>4
    <2> QED
      BY <2>1, <2>2
  <1>18. HandoffOk'
    <2> SUFFICES ASSUME NEW y \in handoff'
                 PROVE  /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
                        /\ (~\E x \in requests' : Names(x, y)) =>
                             /\ ostate'[y.o] = "live"
                             /\ ogroup'[y.o] = y.g \/ ReaderRow(y.g, y.o, y.p, y.u, y.w)'
                             /\ HoldsAt(y.o, y.p, y.u)'
                             /\ (y.p \in Positions /\ y.w) =>
                                  /\ table'[y.o][y.p].k = "private"
                                  /\ ~\E e \in deferred' :
                                        e.kind \in {"commit", "replicate"} /\ y.u \in e.units
                                  /\ ~\E z \in tasks' : z.kind = "replicate" /\ y.u \in z.units
      BY DEF HandoffOk, Live
    <2>1. CASE y = h
      BY <1>1, <1>2, <1>5, <2>1 DEF Live
    <2>2. CASE y # h
      <3>1. y \in handoff /\ y.u \in AllUnits /\ y.o \in Owners /\ y.p \in Rows
        BY <1>0, <1>6, <2>2 DEF RequestTyped
      <3>2. /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
            /\ (~\E x \in requests : Names(x, y)) =>
                 /\ ostate[y.o] = "live"
                 /\ ogroup[y.o] = y.g \/ ReaderRow(y.g, y.o, y.p, y.u, y.w)
                 /\ HoldsAt(y.o, y.p, y.u)
                 /\ (y.p \in Positions /\ y.w) =>
                      /\ table[y.o][y.p].k = "private"
                      /\ ~\E e \in deferred : e.kind \in {"commit", "replicate"} /\ y.u \in e.units
                      /\ ~\E z \in tasks : z.kind = "replicate" /\ y.u \in z.units
        BY <3>1 DEF IndInv, HandoffOk, Live
      <3>3. ReaderRow(y.g, y.o, y.p, y.u, y.w) /\ HoldsAt(y.o, y.p, y.u)
              => ReaderRow(y.g, y.o, y.p, y.u, y.w)'
        <4> SUFFICES ASSUME ReaderRow(y.g, y.o, y.p, y.u, y.w), HoldsAt(y.o, y.p, y.u)
                     PROVE  ReaderRow(y.g, y.o, y.p, y.u, y.w)'
          OBVIOUS
        <4>1. y.p \in Positions /\ table[y.o][y.p].u = y.u
          BY DEF ReaderRow, HoldsAt
        <4>2. y.u \in Units
          BY <3>1, <4>1, OwnerTypes DEF OwnerTyped, Empty, Private, Shared, AllUnits
        <4>3. valid[y.u] \subseteq valid'[y.u] /\ ReadersOf(y.o)' = ReadersOf(y.o)
          BY <1>1, <1>4, <4>2 DEF ReadersOf
        <4> QED
          BY <4>3 DEF ReaderRow
      <3> QED
        BY <1>1, <3>2, <3>3 DEF Live, HoldsAt
    <2> QED
      BY <2>1, <2>2
  <1>19. RequestsOk'
    <2>1. \A x \in requests :
            x.kind # "finish" =>
              \A z \in Groups, q \in Rows :
                (dbt'[z][x.o][q].u # NONE /\ (TakesRow(x, x.o, q) \/ FreezesRow(x, x.o, q))) =>
                  /\ z \in x.sg
                  /\ z \in x.by => FreezesRow(x, x.o, q) /\ ~dbt'[z][x.o][q].w
      <3> SUFFICES ASSUME NEW x \in requests, x.kind # "finish", NEW z \in Groups, NEW q \in Rows,
                          dbt'[z][x.o][q].u # NONE, TakesRow(x, x.o, q) \/ FreezesRow(x, x.o, q)
                   PROVE  /\ z \in x.sg
                          /\ z \in x.by => FreezesRow(x, x.o, q) /\ ~dbt'[z][x.o][q].w
        OBVIOUS
      <3>1. x.o \in Owners
        BY <1>0 DEF RequestTyped
      <3>2. CASE Eager /\ z = h.g /\ x.o = h.o /\ q = h.p /\ Unnamed(h)
        <4>1. Names(x, h)
          BY <3>2, NamesRow
        <4> QED
          BY <3>2, <4>1 DEF Unnamed
      <3>3. CASE ~(Eager /\ z = h.g /\ x.o = h.o /\ q = h.p /\ Unnamed(h))
        <4>1. dbt'[z][x.o][q] = dbt[z][x.o][q]
          BY <1>6, <3>1, <3>3
        <4> QED
          BY <4>1 DEF IndInv, RequestsOk
      <3> QED
        BY <3>2, <3>3
    <2> QED
      BY <1>1, <2>1 DEF IndInv, RequestsOk
  <1>20. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A x : Covered(b, o, x) => Covered(b, o, x)'
      BY <1>1, <1>7 DEF Covered, Live, HoldsAt, ReadersOf, Transfers
    <2>4. \A x \in b.reads : x[2] \in AllUnits
      BY <1>0, <2>2 DEF RequestTyped
    <2> QED
      BY <1>1, <1>3, <1>4, <2>2, <2>3, <2>4 DEF BatchOk, Mem, AllUnits
  <1>21. TrialVisibility'
    BY <1>1 DEF IndInv, TrialVisibility, MadeBy
  <1>22. CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A c \in copies : CopyPlaced(c)'
      <3> SUFFICES ASSUME NEW c \in copies PROVE CopyPlaced(c)'
        OBVIOUS
      <3>1. CopyPlaced(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
        BY <1>0, PlacedEach DEF IndInv, WorkTyped
      <3>2. c.write \in HostUnits => valid'[c.write] = valid[c.write]
        BY <1>2, <1>4
      <3> QED
        BY <1>1, <1>4, <3>1, <3>2 DEF CopyPlaced, AllUnits
    <2>2. \A x \in tasks' : CrossingPlaced(x)'
      <3> SUFFICES ASSUME NEW x \in tasks' PROVE CrossingPlaced(x)'
        OBVIOUS
      <3>1. x \in tasks /\ CrossingPlaced(x)
        BY <1>1, PlacedEach DEF IndInv
      <3>2. x.kind = "replicate" => x.units \subseteq Units
        BY <3>1, CrossingsOkEach DEF IndInv, CrossTaskOk
      <3> QED
        BY <1>1, <1>4, <3>1, <3>2 DEF CrossingPlaced
    <2> QED
      BY <1>1, <2>1, <2>2, PlacedEach
  <1> QED
    BY <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18, <1>19, <1>20,
       <1>21, <1>22 DEF IndInv

(* The task failed, or the owner no longer holds the block where the       *)
(* crossing named it: the block is let go when nothing else holds it, and *)
(* a task that succeeded leaves it valid in the reading group's memory.   *)
LEMMA ReplicateEnd_Away ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, ReplicateEnd(t, ok), UNCHANGED rare,
         ~(ok /\ Back(t)), UNCHANGED <<dbt, handoff>>
  PROVE  IndInv'
  <1> DEFINE u == Ended(t)
             G == ReclaimedWith(t.units, table, oslot, cache, hcache, deferred, copies, tasks \ {t})
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ RequestTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, RequestTypes, WorkTypes
  <1>1. /\ t.kind = "replicate" /\ tasks' = tasks \ {t}
        /\ valid' = IF ok THEN Grown(t) ELSE valid
        /\ free' = free \cup G /\ alloc' = Reset(alloc, G) /\ sharedKey' = Reset(sharedKey, G)
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, vis,
                       trial, member, dbt, launched, inflight, handoff, requests, remote, lost, bad>>
    BY ReplicateEnd_Effect DEF DropWith, rare
  <1>2. /\ t.held /\ t.o \in Owners /\ t.to \in Groups
        /\ t.units = {u} /\ t.writes = {u} /\ u \in Units /\ t.h = NONE /\ t.slot = NONE
        /\ ~Moving(t.o)
        /\ \A x \in tasks :
             (x.kind \in {"transfer", "replicate"} /\ x # t /\ Mem(x.to) = Mem(t.to)) => u \notin x.writes
        /\ \A c \in copies : (c.fill /\ c.kind = "load" /\ Mem(c.g) = Mem(t.to)) => c.write # u
    BY <1>1, ReplicateEnd_Facts
  <1>3. /\ \A x : Held(x)' = HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks \ {t})
        /\ G = {x \in t.units : ~Held(x)'}
    BY <1>1 DEF Held, ReclaimedWith, On, AllRules
  <1>4. /\ \A x \in AllUnits : Held(x)' => Held(x)
        /\ \A x \in AllUnits : Held(x) => Held(x)' \/ x \in G
        /\ \A x \in G : Held(x) /\ ~Held(x)'
        /\ G \subseteq AllUnits /\ G \subseteq {u}
    <2>1. \A x : Held(x)' => Held(x)
      BY <1>3, Zenon DEF Held, HeldIn
    <2>2. \A x \in AllUnits : Held(x) => Held(x)' \/ x = u
      <3> SUFFICES ASSUME NEW x \in AllUnits, Held(x) PROVE Held(x)' \/ x = u
        OBVIOUS
      <3>1. (\E y \in tasks : y.held /\ (x \in y.units \/ x = y.slot \/ x = y.h))
              => (\E y \in tasks \ {t} : y.held /\ (x \in y.units \/ x = y.slot \/ x = y.h)) \/ x = u
        <4>1. x # NONE
          BY DEF AllUnits
        <4> QED
          BY <1>2, <4>1
      <3> QED
        BY <1>3, <3>1 DEF Held, HeldIn
    <2>3. Held(u)
      BY <1>2 DEF Held, HeldIn
    <2> QED
      BY <1>2, <1>3, <2>1, <2>2, <2>3 DEF AllUnits
  <1>5. /\ \A v \in AllUnits : alloc'[v] = IF v \in G THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v \in G THEN NONE ELSE sharedKey[v]
        /\ \A v \in AllUnits : Held(v)' => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
    BY <1>1, <1>4 DEF Reset
  (* Validity grows by the reading group's memory at most, for the block   *)
  (* alone.                                                                *)
  <1>6. /\ \A v \in Units \cup HostUnits : v # u => valid'[v] = valid[v]
        /\ valid'[u] \subseteq valid[u] \cup {Mem(t.to)}
        /\ \A v \in Units \cup HostUnits : valid[v] \subseteq valid'[v]
        /\ ValidTyped'
    <2>1. /\ valid \in [Units \cup HostUnits -> SUBSET (Memories \cup HostParts)]
          /\ \A v \in Units : valid[v] \subseteq Memories
          /\ \A v \in HostUnits : valid[v] \subseteq HostParts
      BY <1>0 DEF UnitTyped
    <2>2. Mem(t.to) \in Memories /\ u \notin HostUnits
      BY <1>2 DEF Mem, Memories
    <2>3. CASE ok
      <3>1. valid' = [valid EXCEPT ![u] = valid[u] \cup {Mem(t.to)}]
        BY <1>1, <2>3 DEF Grown, Ended
      <3>2. \A v \in Units \cup HostUnits : v # u => valid'[v] = valid[v]
        BY <2>1, <3>1
      <3>3. valid'[u] = valid[u] \cup {Mem(t.to)}
        BY <1>2, <2>1, <3>1
      <3>4. ValidTyped'
        BY <1>2, <2>1, <2>2, <3>1, <3>2, <3>3 DEF ValidTyped
      <3> QED
        BY <3>2, <3>3, <3>4
    <2>4. CASE ~ok
      BY <1>1, <2>1, <2>4 DEF ValidTyped
    <2> QED
      BY <2>3, <2>4
  (* What stays held keeps its allocation.                                 *)
  <1>7. /\ \A o \in Owners, p \in Positions :
             table[o][p].u # NONE => alloc'[table[o][p].u] = alloc[table[o][p].u]
        /\ \A o \in Owners : oslot[o] # NONE => alloc'[oslot[o]] = alloc[oslot[o]]
        /\ \A d \in deferred : \A v \in d.units : alloc'[v] = alloc[v]
        /\ \A c \in copies : alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
        /\ \A x \in tasks' : \A v \in x.units \cup {x.slot, x.h} : v # NONE => alloc'[v] = alloc[v]
        /\ \A k \in Keys : /\ cache[k] # NONE => sharedKey'[cache[k]] = sharedKey[cache[k]]
                           /\ hcache[k] # NONE => sharedKey'[hcache[k]] = sharedKey[hcache[k]]
    <2>1. \A x \in tasks' : x.held
      BY <1>1 DEF IndInv, TasksOk
    <2>2. \A v \in AllUnits : (\/ \E o \in Owners, p \in Positions : table[o][p].u = v
                               \/ \E o \in Owners : oslot[o] = v
                               \/ \E k \in Keys : cache[k] = v \/ hcache[k] = v
                               \/ \E d \in deferred : v \in d.units
                               \/ \E c \in copies : v = c.read \/ v = c.write
                               \/ \E x \in tasks' : x.held /\ (v \in x.units \/ v = x.slot \/ v = x.h))
                              => Held(v)'
      BY <1>1, <1>3 DEF HeldIn
    <2>3. /\ \A o \in Owners, p \in Positions : table[o][p].u # NONE => table[o][p].u \in AllUnits
          /\ \A o \in Owners : oslot[o] # NONE => oslot[o] \in AllUnits
          /\ \A d \in deferred : d.units \subseteq AllUnits
          /\ \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
          /\ \A x \in tasks' : \A v \in x.units \cup {x.slot, x.h} : v # NONE => v \in AllUnits
          /\ \A k \in Keys : (cache[k] # NONE => cache[k] \in AllUnits)
                             /\ (hcache[k] # NONE => hcache[k] \in AllUnits)
      BY <1>0, <1>1 DEF OwnerTyped, WorkTyped, UnitTyped, AllUnits, Empty, Private, Shared
    <2> QED
      BY <1>5, <2>1, <2>2, <2>3
  <1>8. /\ \A o \in Owners : Moving(o)' = Moving(o)
        /\ Transfers' = Transfers
    BY <1>1 DEF Moving, Transfers
  <1>9. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>4, <1>6 DEF UnitTyped, ValidTyped, Reset
    <2>2. WorkTyped'
      BY <1>0, <1>1 DEF WorkTyped
    <2> QED
      BY <1>1, <2>1, <2>2, OwnerTypes, PlaneTypes, RequestTypes, OwnerTypedFrame, PlaneTypedFrame,
         RequestTypedFrame, TypeInvNext
  <1>10. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
    BY <1>1, <1>4, <1>5, FreeReclaim, UnitTypes DEF IndInv
  <1>11. AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. AllocOwner' /\ SharedIff'
      BY <1>1, <1>5 DEF IndInv, AllocOwner, SharedIff, NONE
    <2>2. \A v \in AllUnits : alloc[v] = "shared" /\ v \notin G =>
            cache[sharedKey[v]] = v \/ hcache[sharedKey[v]] = v
      BY DEF IndInv, SharedCached
    <2>3. SharedCached'
      BY <1>1, <1>5, <2>2 DEF SharedCached, NONE
    <2>4. CacheShared'
      BY <1>1, <1>7 DEF IndInv, CacheShared
    <2> QED
      BY <2>1, <2>3, <2>4
  <1>12. Tables' /\ OwnersOk' /\ DeferredOk'
    BY <1>1, <1>6, <1>7, OwnerTypes DEF IndInv, Tables, OwnersOk, DeferredOk, Live, Mem, OwnerTyped,
                                        Empty, Private, Shared
  <1>13. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY <1>1, CopiesOkEach
    <2>2. CopyOk(c) /\ c.write \in AllUnits
      BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. (c.fill /\ c.kind = "load") => Mem(c.g) \notin valid'[c.write]
      <3>1. (c.fill /\ c.kind = "load") => Mem(c.g) \notin valid[c.write] /\ c.write \in Units
        BY <1>0, <2>2 DEF CopyOk, UnitTyped, WorkTyped, AllUnits
      <3>2. (c.fill /\ c.kind = "load" /\ c.write = u) => Mem(c.g) # Mem(t.to)
        BY <1>2
      <3> QED
        BY <1>6, <3>1, <3>2
    <2> QED
      BY <1>1, <1>7, <1>8, <2>2, <2>3 DEF CopyOk, Live, Mem
  <1>14. TasksOk'
    <2>1. SUFFICES ASSUME NEW x \in tasks' PROVE TaskOk(x)'
      BY TasksOkEach
    <2>2. x \in tasks /\ x # t /\ TaskOk(x)
      BY <1>1, TasksOkEach DEF IndInv
    <2>3. \A v \in x.units \cup {x.slot, x.h} : v # NONE => alloc'[v] = alloc[v]
      BY <1>7
    <2>4. x.kind = "transfer" =>
            /\ x.writes \subseteq x.units /\ x.slot # NONE
            /\ \A v \in x.writes : Mem(x.to) \notin valid'[v]
            /\ Live(x.o) => \A v \in TableUnits(x.o) \ x.writes : Mem(x.to) \in valid'[v]
      <3>1. x.kind = "transfer" =>
              /\ x.writes \subseteq x.units /\ x.slot \in Slots /\ x.writes \subseteq Units
              /\ \A v \in x.writes : Mem(x.to) \notin valid[v]
              /\ Live(x.o) => \A v \in TableUnits(x.o) \ x.writes : Mem(x.to) \in valid[v]
        BY <2>2 DEF TaskOk
      <3>2. (x.kind = "transfer" /\ Mem(x.to) = Mem(t.to)) => u \notin x.writes
        BY <1>2, <2>2
      <3>3. x.kind = "transfer" => \A v \in TableUnits(x.o) : v \in Units
        BY <1>0, <2>2 DEF TaskOk, TableUnits, OwnerTyped, Empty, Private, Shared
      <3>4. ASSUME x.kind = "transfer", NEW v \in x.writes PROVE Mem(x.to) \notin valid'[v]
        <4>1. v \in Units /\ Mem(x.to) \notin valid[v]
          BY <3>1, <3>4
        <4>2. CASE v = u
          <5>1. Mem(x.to) # Mem(t.to)
            BY <3>2, <3>4, <4>2
          <5> QED
            BY <1>6, <4>1, <4>2, <5>1
        <4>3. CASE v # u
          BY <1>6, <4>1, <4>3
        <4> QED
          BY <4>2, <4>3
      <3>5. ASSUME x.kind = "transfer", Live(x.o), NEW v \in TableUnits(x.o) \ x.writes
            PROVE  Mem(x.to) \in valid'[v]
        BY <1>6, <3>1, <3>3, <3>5
      <3> QED
        BY <3>1, <3>4, <3>5
    <2>5. x.kind = "prefetch" => x.h # NONE
      BY <2>2 DEF TaskOk
    <2> QED
      BY <1>1, <2>2, <2>3, <2>4, <2>5 DEF TaskOk, Live, TableUnits, Mem
  <1>15. CrossingsOk'
    <2>1. \A x \in deferred' : CrossWaitOk(x)'
      <3> SUFFICES ASSUME NEW x \in deferred PROVE CrossWaitOk(x)'
        BY <1>1
      <3>1. CrossWaitOk(x)
        BY CrossingsOkEach DEF IndInv
      <3>2. \A v \in x.units : alloc'[v] = alloc[v]
        BY <1>7
      <3> QED
        BY <1>1, <1>8, <3>1, <3>2 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>2. \A x \in tasks' : CrossTaskOk(x)'
      <3> SUFFICES ASSUME NEW x \in tasks' PROVE CrossTaskOk(x)'
        OBVIOUS
      <3>1. x \in tasks /\ x # t /\ CrossTaskOk(x)
        BY <1>1, CrossingsOkEach DEF IndInv
      <3>2. x.kind = "replicate" =>
              /\ \A v \in x.writes : alloc'[v] = alloc[v]
              /\ \A v \in x.writes : Mem(x.to) \notin valid'[v]
        <4>1. x.kind = "replicate" =>
                /\ x.writes \subseteq x.units /\ x.writes \subseteq Units
                /\ \A v \in x.writes : Mem(x.to) \notin valid[v]
          BY <3>1 DEF CrossTaskOk
        <4>2. (x.kind = "replicate" /\ Mem(x.to) = Mem(t.to)) => u \notin x.writes
          BY <1>2, <3>1
        <4> QED
          BY <1>6, <1>7, <4>1, <4>2
      <3> QED
        BY <1>1, <1>8, <3>1, <3>2 DEF CrossTaskOk, ReadersOf, Mem
    <2> QED
      BY <2>1, <2>2, CrossingsOkEach
  <1>16. /\ MovingUnmapped' /\ LostOk' /\ RequestsOk' /\ NoBad' /\ TrialVisibility'
    BY <1>1, <1>8, MovingUnmappedFrame, LostOkFrame, RequestsOkFrame, NoBadFrame
       DEF IndInv, TrialVisibility, MadeBy
  <1>17. TablesMapped' /\ HandoffOk'
    <2>1. \A v \in Units : valid[v] \subseteq valid'[v]
      BY <1>6
    <2>2. \A v : (\E x \in tasks' : x.kind = "replicate" /\ v \in x.units)
                   => (\E x \in tasks : x.kind = "replicate" /\ v \in x.units)
      BY <1>1
    <2>3. \A q \in Owners : ReadersOf(q)' = ReadersOf(q)
      BY <1>1 DEF ReadersOf
    (* A reader's row or completion names a block of a table.              *)
    <2>4. \A q \in Owners, r \in Positions : table[q][r].u # NONE => table[q][r].u \in Units
      BY <1>0 DEF OwnerTyped, Empty, Private, Shared
    <2>5. TablesMapped'
      <3> SUFFICES ASSUME NEW x \in Groups, NEW q \in Owners, NEW r \in Rows, dbt[x][q][r].u # NONE
                   PROVE  /\ ostate'[q] = "live"
                          /\ \/ ogroup'[q] = x
                             \/ /\ x \in ReadersOf(q)' /\ r \in Positions /\ ~dbt[x][q][r].w
                                /\ Mem(x) \in valid'[dbt[x][q][r].u]
                          /\ IF r = SLOT THEN oslot'[q] = dbt[x][q][r].u
                                         ELSE table'[q][r].u = dbt[x][q][r].u
                          /\ (r \in Positions /\ dbt[x][q][r].w) =>
                               /\ table'[q][r].k = "private"
                               /\ ~\E e \in deferred' :
                                     e.kind \in {"commit", "replicate"} /\ dbt[x][q][r].u \in e.units
                               /\ ~\E y \in tasks' : y.kind = "replicate" /\ dbt[x][q][r].u \in y.units
        BY <1>1 DEF TablesMapped, Live, ReaderRow, HoldsAt
      <3>1. /\ ostate[q] = "live"
            /\ \/ ogroup[q] = x
               \/ /\ x \in ReadersOf(q) /\ r \in Positions /\ ~dbt[x][q][r].w
                  /\ Mem(x) \in valid[dbt[x][q][r].u]
            /\ IF r = SLOT THEN oslot[q] = dbt[x][q][r].u ELSE table[q][r].u = dbt[x][q][r].u
            /\ (r \in Positions /\ dbt[x][q][r].w) =>
                 /\ table[q][r].k = "private"
                 /\ ~\E e \in deferred :
                       e.kind \in {"commit", "replicate"} /\ dbt[x][q][r].u \in e.units
                 /\ ~\E y \in tasks : y.kind = "replicate" /\ dbt[x][q][r].u \in y.units
        BY DEF IndInv, TablesMapped, Live, ReaderRow, HoldsAt
      <3>2. r \in Positions => valid[dbt[x][q][r].u] \subseteq valid'[dbt[x][q][r].u]
        BY <2>1, <2>4, <3>1
      <3> QED
        BY <1>1, <2>2, <2>3, <3>1, <3>2
    <2>6. HandoffOk'
      <3> SUFFICES ASSUME NEW y \in handoff
                   PROVE  /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
                          /\ (~\E x \in requests : Names(x, y)) =>
                               /\ ostate'[y.o] = "live"
                               /\ \/ ogroup'[y.o] = y.g
                                  \/ /\ y.g \in ReadersOf(y.o)' /\ y.p \in Positions /\ ~y.w
                                     /\ Mem(y.g) \in valid'[y.u]
                               /\ IF y.p = SLOT THEN oslot'[y.o] = y.u ELSE table'[y.o][y.p].u = y.u
                               /\ (y.p \in Positions /\ y.w) =>
                                    /\ table'[y.o][y.p].k = "private"
                                    /\ ~\E e \in deferred' :
                                          e.kind \in {"commit", "replicate"} /\ y.u \in e.units
                                    /\ ~\E z \in tasks' : z.kind = "replicate" /\ y.u \in z.units
        BY <1>1 DEF HandoffOk, Live, ReaderRow, HoldsAt
      <3>1. /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
            /\ (~\E x \in requests : Names(x, y)) =>
                 /\ ostate[y.o] = "live"
                 /\ \/ ogroup[y.o] = y.g
                    \/ /\ y.g \in ReadersOf(y.o) /\ y.p \in Positions /\ ~y.w
                       /\ Mem(y.g) \in valid[y.u]
                 /\ IF y.p = SLOT THEN oslot[y.o] = y.u ELSE table[y.o][y.p].u = y.u
                 /\ (y.p \in Positions /\ y.w) =>
                      /\ table[y.o][y.p].k = "private"
                      /\ ~\E e \in deferred : e.kind \in {"commit", "replicate"} /\ y.u \in e.units
                      /\ ~\E z \in tasks : z.kind = "replicate" /\ y.u \in z.units
        BY DEF IndInv, HandoffOk, Live, ReaderRow, HoldsAt
      <3>2. y.u \in AllUnits
        BY <1>0 DEF RequestTyped
      <3>3. (y.p \in Positions /\ table[y.o][y.p].u = y.u /\ y.o \in Owners)
              => valid[y.u] \subseteq valid'[y.u]
        BY <2>1, <2>4, <3>2 DEF AllUnits
      <3> QED
        BY <1>1, <2>2, <2>3, <3>1, <3>3
    <2> QED
      BY <2>5, <2>6
  <1>18. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight' PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b) /\ b \in inflight
      BY <1>1, BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A x : Covered(b, o, x) => Covered(b, o, x)' /\ Held(x)'
      <3>1. \A o \in Owners : \A x : Covered(b, o, x) => Covered(b, o, x)'
        BY <1>1, <1>8 DEF Covered, Live, HoldsAt, ReadersOf, Transfers
      <3>2. \A o \in Owners : \A x : Covered(b, o, x) => Held(x)'
        BY <1>1, <1>3 DEF Covered, HoldsAt, Rows, HeldIn
      <3> QED
        BY <3>1, <3>2
    <2>4. \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits
      BY <1>0, <2>2 DEF RequestTyped
    <2> QED
      BY <1>1, <1>4, <1>5, <1>6, <2>2, <2>3, <2>4 DEF BatchOk, Mem, AllUnits
  <1>19. CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A c \in copies : CopyPlaced(c)'
      <3> SUFFICES ASSUME NEW c \in copies PROVE CopyPlaced(c)'
        OBVIOUS
      <3>1. CopyPlaced(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
        BY <1>0, PlacedEach DEF IndInv, WorkTyped
      <3>2. c.write \in HostUnits => valid'[c.write] = valid[c.write]
        BY <1>2, <1>6
      <3> QED
        BY <1>1, <1>6, <3>1, <3>2 DEF CopyPlaced, AllUnits
    <2>2. \A x \in tasks' : CrossingPlaced(x)'
      <3> SUFFICES ASSUME NEW x \in tasks' PROVE CrossingPlaced(x)'
        OBVIOUS
      <3>1. x \in tasks /\ CrossingPlaced(x)
        BY <1>1, PlacedEach DEF IndInv
      <3>2. x.kind = "replicate" => x.units \subseteq Units
        BY <3>1, CrossingsOkEach DEF IndInv, CrossTaskOk
      <3> QED
        BY <1>1, <1>6, <3>1, <3>2 DEF CrossingPlaced
    <2> QED
      BY <1>1, <2>1, <2>2, PlacedEach
  <1> QED
    BY <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18, <1>19 DEF IndInv

LEMMA ReplicateEnd_Inv ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, ReplicateEnd(t, ok), UNCHANGED rare
  PROVE  IndInv'
  BY ReplicateEnd_Effect, ReplicateEnd_Back, ReplicateEnd_Away

-----------------------------------------------------------------------------

THEOREM Step11 ==
    IndInv /\ (/\ \/ \E r \in requests : TakeReplicate(r)
                  \/ \E d \in deferred : ReplicateDue(d)
                  \/ \E t \in tasks, ok \in BOOLEAN : ReplicateEnd(t, ok)
               /\ UNCHANGED rare)
    => IndInv'
  BY TakeReplicate_Inv, ReplicateDue_Inv, ReplicateEnd_Inv

=============================================================================
