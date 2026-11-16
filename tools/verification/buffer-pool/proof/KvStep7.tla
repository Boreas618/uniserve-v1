------------------------------- MODULE KvStep7 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for the deferred work that comes due:      *)
(* ReleaseDue, CommitDue and FinishDue.                                    *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, KvFrame, TLAPS

USE Sizes, Design

-----------------------------------------------------------------------------
(* ReleaseDue                                                              *)

LEMMA ReleaseDue_Inv ==
  ASSUME IndInv, NEW d \in deferred, ReleaseDue(d), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ d.kind = "release" /\ Reached(d.after)
        /\ deferred' = deferred \ {d}
        /\ DropWith(d.units, table, oslot, cache, hcache, deferred', copies, tasks)
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks, trial,
                       member, valid, vis, dbt, launched, inflight, handoff, requests,
                       remote, lost, bad>>
    BY DEF ReleaseDue, Drop, plane, rare
  <1>2. \A b \in inflight : d.after[b.g] < b.seq
    BY <1>1 DEF Reached, IndInv, BatchesOk
  <1>3. /\ d.units = d.units \cup UNION {{c.read, c.write} : c \in {}}
        /\ copies' = copies \ {}
    BY <1>1
  <1>4. ValidTyped'
    BY <1>1, UnitTypes DEF UnitTyped, ValidTyped
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, Retire

-----------------------------------------------------------------------------
(* CommitDue                                                               *)

LEMMA CommitDue_Effect ==
  ASSUME IndInv, NEW d \in deferred, CommitDue(d), UNCHANGED rare
  PROVE  /\ d.kind = "commit" /\ Reached(d.after) /\ deferred' = deferred \ {d}
         /\ d.o \in Owners /\ d.p \in Positions /\ d.k \in Keys
         /\ UNCHANGED <<ostate, ogroup, oslot, hcache, copies, ncopy, tasks, trial, member, dbt,
                        launched, inflight, handoff, requests, remote, lost, bad>>
         /\ \E u \in Units :
              /\ d.units = {u} /\ alloc[u] = d.o
              /\ \/ /\ cache[d.k] # NONE
                    /\ DropWith(d.units, table, oslot, cache, hcache, deferred', copies, tasks)
                    /\ UNCHANGED <<table, cache, valid, vis>>
                 \/ /\ cache[d.k] = NONE
                    /\ cache' = [cache EXCEPT ![d.k] = u]
                    /\ valid' = valid
                    /\ vis' = [vis EXCEPT ![d.k].d = MadeBy(OwnerEpoch[d.o])]
                    /\ table' = [table EXCEPT ![d.o][d.p] = IF @ = Private(u) THEN Shared(u) ELSE @]
                    /\ free' = free
                    /\ alloc' = [alloc EXCEPT ![u] = "shared"]
                    /\ sharedKey' = [sharedKey EXCEPT ![u] = IF @ = NONE THEN d.k ELSE @]
  <1>1. d.kind = "commit"
    BY DEF CommitDue
  <1>2. PICK u \in Units : d.units = {u} /\ alloc[u] = d.o
    BY <1>1 DEF IndInv, DeferredOk
  <1>3. /\ d.o \in Owners /\ d.p \in Positions /\ d.k \in Keys
    BY <1>1 DEF IndInv, DeferredOk
  <1>4. (CHOOSE x \in d.units : TRUE) = u /\ u # NONE
    BY <1>2
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4 DEF CommitDue, Drop, plane, rare, On, AllRules

(* The commit shares its block: the key had no device block.               *)
LEMMA CommitDue_Share ==
  ASSUME IndInv, NEW d \in deferred, d.kind = "commit", Reached(d.after),
         deferred' = deferred \ {d}, d.o \in Owners, d.p \in Positions, d.k \in Keys,
         NEW u \in Units, d.units = {u}, alloc[u] = d.o, cache[d.k] = NONE,
         cache' = [cache EXCEPT ![d.k] = u],
         valid' = valid,
         vis' = [vis EXCEPT ![d.k].d = MadeBy(OwnerEpoch[d.o])],
         table' = [table EXCEPT ![d.o][d.p] = IF @ = Private(u) THEN Shared(u) ELSE @],
         free' = free,
         alloc' = [alloc EXCEPT ![u] = "shared"],
         sharedKey' = [sharedKey EXCEPT ![u] = IF @ = NONE THEN d.k ELSE @],
         UNCHANGED <<ostate, ogroup, oslot, hcache, copies, ncopy, tasks, trial, member, dbt,
                     launched, inflight, handoff, requests, remote, lost, bad>>
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  (* What IndInv says of the block the commit holds.                       *)
  <1>1. /\ \A o \in Owners, p \in Positions :
             table[o][p].u = u => o = d.o /\ p = d.p /\ table[o][p].k = "private"
        /\ \A c \in copies : c.write # u /\ c.read # u
        /\ \A k \in Keys : cache[k] # u /\ hcache[k] # u
        /\ sharedKey[u] = NONE
        /\ Held(u) /\ u \in member /\ u \notin free
        /\ \A e \in deferred : (e.kind = "commit" /\ e # d) => u \notin e.units
        /\ \A o \in Owners : oslot[o] # u
        /\ \A b \in inflight : \A x \in b.writes : x[2] # u
        /\ OwnerEpoch[d.o] \in Epochs
    <2>1. \A o \in Owners, p \in Positions :
            table[o][p].u = u => o = d.o /\ p = d.p /\ table[o][p].k = "private"
      BY DEF IndInv, DeferredOk
    <2>2. \A c \in copies : c.write # u /\ c.read # u
      <3>1. \A c \in copies : c.write # u
        BY DEF IndInv, DeferredOk
      <3>2. \A c \in copies : c.read # u
        BY <1>0 DEF IndInv, CopiesOk, OwnerTyped, NONE, AllUnits
      <3> QED
        BY <3>1, <3>2
    <2>3. /\ \A k \in Keys : cache[k] # u /\ hcache[k] # u
          /\ sharedKey[u] = NONE
      BY <1>0 DEF IndInv, CacheShared, SharedIff, UnitTyped, AllUnits
    <2>4. Held(u) /\ u \in member /\ u \notin free
      BY DEF Held, HeldIn, IndInv, NoLeak, FreeClean, InRegion, AllUnits
    <2>5. \A e \in deferred : (e.kind = "commit" /\ e # d) => u \notin e.units
      <3> SUFFICES ASSUME NEW e \in deferred, e.kind = "commit", e # d, u \in e.units PROVE FALSE
        OBVIOUS
      <3>1. \E v \in Units : e.units = {v}
        BY DEF IndInv, DeferredOk
      <3> QED
        BY <3>1 DEF IndInv, DeferredOk
    <2>6. \A o \in Owners : oslot[o] # u
      BY <1>0 DEF OwnerTyped
    <2>7. \A b \in inflight : \A x \in b.writes : x[2] # u
      <3> SUFFICES ASSUME NEW b \in inflight, NEW x \in b.writes, x[2] = u PROVE FALSE
        OBVIOUS
      <3>1. d.after[b.g] >= b.seq /\ d.after[b.g] < b.seq
        BY DEF IndInv, BatchesOk, Reached
      <3>2. b.seq \in Nat /\ d.after[b.g] \in Nat
        BY <1>0 DEF IndInv, BatchesOk, WorkTyped, Stamps
      <3> QED
        BY <3>1, <3>2
    <2>8. OwnerEpoch[d.o] \in Epochs
      OBVIOUS
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8
  (* The values after the step, pointwise.                                 *)
  <1>2. /\ \A k \in Keys : cache'[k] = IF k = d.k THEN u ELSE cache[k]
        /\ \A v \in AllUnits : alloc'[v] = IF v = u THEN "shared" ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v = u THEN d.k ELSE sharedKey[v]
        /\ \A k \in Keys : vis'[k] = IF k = d.k THEN [d |-> MadeBy(OwnerEpoch[d.o]), h |-> vis[k].h]
                                                ELSE vis[k]
        /\ \A o \in Owners, p \in Positions :
             table'[o][p] = IF o = d.o /\ p = d.p /\ table[o][p] = Private(u) THEN Shared(u)
                            ELSE table[o][p]
        /\ \A o \in Owners, p \in Positions : table'[o][p].u = table[o][p].u
        /\ \A o \in Owners, p \in Positions :
             table'[o][p].k # table[o][p].k => o = d.o /\ p = d.p /\ table'[o][p] = Shared(u)
    BY <1>0, <1>1 DEF UnitTyped, OwnerTyped, PlaneTyped, AllUnits, Private, Shared
  <1>3. \A x \in AllUnits : Held(x)' = Held(x)
    <2>1. \A x \in AllUnits : x # u =>
            /\ (\E e \in deferred' : x \in e.units) = (\E e \in deferred : x \in e.units)
            /\ (\E k \in Keys : cache'[k] = x \/ hcache[k] = x)
                 = (\E k \in Keys : cache[k] = x \/ hcache[k] = x)
            /\ (\E o \in Owners, p \in Positions : table'[o][p].u = x)
                 = (\E o \in Owners, p \in Positions : table[o][p].u = x)
      BY <1>2 DEF AllUnits
    <2>2. Held(u)' /\ Held(u)
      BY <1>1, <1>2 DEF Held, HeldIn
    <2>3. \A x \in AllUnits : x # u => Held(x)' = Held(x)
      BY <2>1 DEF Held, HeldIn
    <2> QED
      BY <2>2, <2>3
  <1>4. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>2 DEF UnitTyped, AllUnits
    <2>2. OwnerTyped'
      BY <1>0, <1>2 DEF OwnerTyped, Empty, Private, Shared
    <2>3. PlaneTyped'
      <3>1. MadeBy(OwnerEpoch[d.o]) \in Epochs \cup {NONE}
        BY <1>1 DEF MadeBy
      <3> QED
        BY <1>0, <1>2, <3>1 DEF PlaneTyped
    <2>4. WorkTyped' /\ RequestTyped'
      BY <1>0 DEF WorkTyped, RequestTyped
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, TypeInvNext
  <1>5. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
    BY <1>0, <1>1, <1>2, <1>3 DEF IndInv, FreeClean, NoLeak, InRegion, Outside, Regions,
                                  OneRegion, UnitTyped, AllUnits
  <1>6. AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. AllocOwner' /\ SharedIff'
      BY <1>1, <1>2 DEF IndInv, AllocOwner, SharedIff, NONE
    <2>2. SharedCached'
      <3>1. \A v \in AllUnits : alloc[v] = "shared" => sharedKey[v] \in Keys
        BY <1>0 DEF IndInv, SharedIff, UnitTyped
      <3> QED
        BY <1>1, <1>2, <3>1 DEF IndInv, SharedCached, AllUnits
    <2>3. CacheShared'
      BY <1>0, <1>1, <1>2 DEF IndInv, CacheShared, UnitTyped, AllUnits
    <2> QED
      BY <2>1, <2>2, <2>3
  <1>7. Tables' /\ OwnersOk'
    <2>1. \A o \in Owners, p \in Positions :
            /\ table[o][p].u \in Units \cup {NONE}
            /\ table[o][p].k = "none" <=> table[o][p].u = NONE
      BY <1>0 DEF OwnerTyped, Empty, Private, Shared
    <2>2. \A o \in Owners, p \in Positions :
            table[o][p].u # NONE /\ table[o][p].u # u =>
                alloc'[table[o][p].u] = alloc[table[o][p].u]
      BY <1>2, <2>1 DEF AllUnits
    <2>3. Tables'
      <3> SUFFICES ASSUME NEW o \in Owners, NEW p \in Positions
                   PROVE  /\ table'[o][p].k = "none" => table'[o][p].u = NONE
                          /\ table'[o][p].k = "private" =>
                               table'[o][p].u \in Units /\ alloc'[table'[o][p].u] = o
                          /\ table'[o][p].k = "shared" =>
                               table'[o][p].u \in Units /\ alloc'[table'[o][p].u] = "shared"
                          /\ table'[o][p].k # "none" => Live(o)'
                          /\ \A q \in Positions :
                               (table'[o][p].k = "private" /\ table'[o][q].u = table'[o][p].u) => p = q
                          /\ table'[o][p].u # NONE => Mem(ogroup'[o]) \in valid'[table'[o][p].u]
        BY DEF Tables
      <3>1. CASE o = d.o /\ p = d.p /\ table[o][p] = Private(u)
        <4>1. table'[o][p] = Shared(u) /\ Live(o) /\ Mem(ogroup[o]) \in valid[u]
          BY <1>2, <3>1 DEF IndInv, Tables, Private, Live
        <4> QED
          BY <1>2, <4>1 DEF Shared, Live, AllUnits
      <3>2. CASE ~(o = d.o /\ p = d.p /\ table[o][p] = Private(u))
        <4>1. table'[o][p] = table[o][p]
          BY <1>2, <3>2
        <4>2. table[o][p].u = u => o = d.o /\ p = d.p /\ table[o][p].k = "private"
          BY <1>1
        <4>3. table[o][p].u = u => FALSE
          <5>1. table[o][p].u = u => table[o][p] = Private(u)
            BY <1>0, <4>2 DEF OwnerTyped, Private, Shared, Empty
          <5> QED
            BY <3>2, <4>2, <5>1
        <4>4. \A q \in Positions : table'[o][q].u = table[o][q].u /\ table'[o][q].k = table[o][q].k
                                    \/ table'[o][q] = Shared(u)
          BY <1>2
        <4> QED
          BY <1>2, <2>1, <2>2, <4>1, <4>3, <4>4 DEF IndInv, Tables, Live, Mem, Shared
      <3> QED
        BY <3>1, <3>2
    <2>4. OwnersOk'
      BY <1>0, <1>1, <1>2 DEF IndInv, OwnersOk, Live, OwnerTyped, AllUnits, Empty, Private
    <2> QED
      BY <2>3, <2>4
  <1>8. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred' PROVE DeferOk(e)'
      BY DeferredOkEach
    <2>2. e \in deferred /\ e # d /\ DeferOk(e)
      BY DeferredOkEach DEF IndInv
    <2>3. \A o \in Owners, p \in Positions :
            /\ table'[o][p].u = table[o][p].u
            /\ table'[o][p].k = "private" => table[o][p].k = "private"
      BY <1>2 DEF Shared
    <2>4. e.kind = "commit" => \A v \in e.units : v # u /\ v \in AllUnits
      BY <1>0, <1>1, <2>2 DEF WorkTyped
    <2>5. e.kind = "commit" => \A v \in e.units : alloc'[v] = alloc[v]
      BY <1>2, <2>4
    <2>6. \A x \in deferred' : x \in deferred
      OBVIOUS
    <2>7. \A o \in Owners, p \in Positions : table[o][p].u # u => table'[o][p] = table[o][p]
      BY <1>2 DEF Private
    <2> QED
      BY <2>2, <2>3, <2>4, <2>5, <2>6, <2>7 DEF DeferOk
  <1>9. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY CopiesOkEach
    <2>2. CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits /\ c.k \in Keys
      BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. (c.fill /\ c.kind = "load") => c.k # d.k
      BY <1>1, <2>2 DEF CopyOk, AllUnits
    <2>4. c.kind \in {"clone", "load"} => table'[c.o][c.p] = table[c.o][c.p]
      BY <1>2, <2>2 DEF CopyOk, Empty, Private
    <2>5. \A o \in Owners : Moving(o)' = Moving(o)
      BY DEF Moving
    <2> QED
      BY <1>1, <1>2, <2>2, <2>3, <2>4, <2>5 DEF CopyOk, Live, Mem
  <1>10. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. TaskOk(t)
      BY TasksOkEach DEF IndInv
    <2>3. \A o \in Owners : TableUnits(o)' = TableUnits(o)
      BY <1>2 DEF TableUnits
    <2>4. /\ t.kind = "transfer" => t.slot \in Slots /\ t.writes \subseteq Units
          /\ t.kind = "prefetch" => t.h \in HostUnits
          /\ t.kind = "put" => t.k \in Keys /\ t.k # NONE
      BY <1>0, <2>2 DEF TaskOk, WorkTyped
    <2> QED
      BY <1>1, <1>2, <2>2, <2>3, <2>4 DEF TaskOk, Live, Mem, AllUnits
  <1>11. /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
    <2>1. MovingUnmapped' /\ LostOk' /\ RequestsOk' /\ NoBad'
      BY DEF IndInv, MovingUnmapped, LostOk, RequestsOk, NoBad
    <2>2. \A o \in Owners, p \in Rows : \A x : HoldsAt(o, p, x)' = HoldsAt(o, p, x)
      BY <1>2 DEF HoldsAt, Rows
    <2>3. \A o \in Owners, p \in Positions :
            table'[o][p].k = "private" => table[o][p].k = "private"
      BY <1>2 DEF Shared
    <2>4. \A x \in deferred' : x \in deferred /\ (x.kind = "commit" => u \notin x.units)
      BY <1>1
    (* A reader's row reads validity and the owner's group, which stay.    *)
    <2>7. \A g \in Groups, o \in Owners, p \in Rows : \A x, wr :
            ReaderRow(g, o, p, x, wr)' = ReaderRow(g, o, p, x, wr)
      BY DEF ReaderRow, ReadersOf, Mem
    <2>5. TablesMapped'
      <3> SUFFICES ASSUME NEW g \in Groups, NEW o \in Owners, NEW p \in Rows, dbt[g][o][p].u # NONE
                   PROVE  /\ Live(o)'
                          /\ \/ ogroup'[o] = g
                             \/ ReaderRow(g, o, p, dbt[g][o][p].u, dbt[g][o][p].w)'
                          /\ HoldsAt(o, p, dbt[g][o][p].u)'
                          /\ (p \in Positions /\ dbt[g][o][p].w) =>
                               /\ table'[o][p].k = "private"
                               /\ ~\E e \in deferred' :
                                     e.kind \in {"commit", "replicate"} /\ dbt[g][o][p].u \in e.units
                               /\ ~\E t \in tasks' : t.kind = "replicate" /\ dbt[g][o][p].u \in t.units
        BY DEF TablesMapped
      <3>1. /\ Live(o) /\ (ogroup[o] = g \/ ReaderRow(g, o, p, dbt[g][o][p].u, dbt[g][o][p].w))
            /\ HoldsAt(o, p, dbt[g][o][p].u)
            /\ (p \in Positions /\ dbt[g][o][p].w) =>
                 /\ table[o][p].k = "private"
                 /\ ~\E e \in deferred :
                       e.kind \in {"commit", "replicate"} /\ dbt[g][o][p].u \in e.units
                 /\ ~\E t \in tasks : t.kind = "replicate" /\ dbt[g][o][p].u \in t.units
        BY DEF IndInv, TablesMapped
      <3>2. (p \in Positions /\ dbt[g][o][p].w) => dbt[g][o][p].u # u
        BY <3>1
      <3>3. (p \in Positions /\ dbt[g][o][p].w) => table'[o][p].k = "private"
        <4>1. (p \in Positions /\ dbt[g][o][p].w) => table[o][p].u # u
          BY <3>1, <3>2 DEF HoldsAt, Rows
        <4> QED
          BY <1>2, <3>1, <4>1 DEF Private
      <3> QED
        BY <2>2, <2>4, <2>7, <3>1, <3>3 DEF Live
    <2>6. HandoffOk'
      <3> SUFFICES ASSUME NEW h \in handoff
                   PROVE  /\ h.o \in Owners /\ h.g \in Groups /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
                          /\ (~\E r \in requests' : Names(r, h)') =>
                               /\ Live(h.o)'
                               /\ ogroup'[h.o] = h.g \/ ReaderRow(h.g, h.o, h.p, h.u, h.w)'
                               /\ HoldsAt(h.o, h.p, h.u)'
                               /\ (h.p \in Positions /\ h.w) =>
                                    /\ table'[h.o][h.p].k = "private"
                                    /\ ~\E e \in deferred' :
                                          e.kind \in {"commit", "replicate"} /\ h.u \in e.units
                                    /\ ~\E t \in tasks' : t.kind = "replicate" /\ h.u \in t.units
        BY DEF HandoffOk
      <3>1. /\ h.o \in Owners /\ h.g \in Groups /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
            /\ (~\E r \in requests : Names(r, h)) =>
                 /\ Live(h.o) /\ (ogroup[h.o] = h.g \/ ReaderRow(h.g, h.o, h.p, h.u, h.w))
                 /\ HoldsAt(h.o, h.p, h.u)
                 /\ (h.p \in Positions /\ h.w) =>
                      /\ table[h.o][h.p].k = "private"
                      /\ ~\E e \in deferred : e.kind \in {"commit", "replicate"} /\ h.u \in e.units
                      /\ ~\E t \in tasks : t.kind = "replicate" /\ h.u \in t.units
        BY DEF IndInv, HandoffOk
      <3>2. ((~\E r \in requests : Names(r, h)) /\ h.p \in Positions /\ h.w) =>
              table'[h.o][h.p].k = "private"
        <4>1. ((~\E r \in requests : Names(r, h)) /\ h.p \in Positions /\ h.w) => h.u # u
          BY <3>1
        <4> QED
          BY <1>2, <3>1, <4>1 DEF HoldsAt, Private
      <3>3. h.p \in Rows
        BY <1>0 DEF RequestTyped
      <3>4. (\E r \in requests' : Names(r, h)') <=> (\E r \in requests : Names(r, h))
        BY DEF Names
      <3> QED
        BY <2>2, <2>4, <2>7, <3>1, <3>2, <3>3, <3>4 DEF Live
    <2> QED
      BY <2>1, <2>5, <2>6
  <1>12. TrialVisibility'
    <2>0. TrialVisibility
      BY DEF IndInv
    <2>1. \A k \in Keys : k # d.k => cache'[k] = cache[k] /\ cache[k] # u /\ vis'[k] = vis[k]
      BY <1>1, <1>2
    <2>2. \A o \in Owners, p \in Positions : table'[o][p] = Shared(u) => o = d.o
      BY <1>1, <1>2 DEF Shared, Private
    <2>3. \A c \in copies : (c.kind = "load" /\ c.fill) => c.k # d.k
      <3> SUFFICES ASSUME NEW c \in copies, c.kind = "load", c.fill PROVE c.k # d.k
        OBVIOUS
      <3>1. CopyOk(c) /\ c.write \in AllUnits
        BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
      <3> QED
        BY <3>1 DEF CopyOk, AllUnits
    <2>4. MadeBy(OwnerEpoch[d.o]) # NONE => MadeBy(OwnerEpoch[d.o]) = OwnerEpoch[d.o]
      BY DEF MadeBy
    <2>5. \A o \in Owners, p \in Positions, x \in Units :
            (table'[o][p] = Shared(x) /\ x # u) => table[o][p] = Shared(x)
      BY <1>2 DEF Shared, Private
    <2>6. /\ cache'[d.k] = u
          /\ vis'[d.k].d = MadeBy(OwnerEpoch[d.o]) /\ vis'[d.k].h = vis[d.k].h
      BY <1>1, <1>2
    (* A crossing of the block is its owner's: the block is private to it. *)
    <2>9. \A t \in tasks : (t.kind = "replicate" /\ u \in t.units) => t.o = d.o
      <3> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate", u \in t.units PROVE t.o = d.o
        OBVIOUS
      <3>1. CrossTaskOk(t)
        BY CrossingsOkEach DEF IndInv
      <3> QED
        BY <3>1 DEF CrossTaskOk
    <2> SUFFICES ASSUME NEW k \in Keys
                 PROVE  /\ (cache'[k] # NONE /\ vis'[k].d # NONE) =>
                             /\ \A o \in Owners, p \in Positions :
                                  table'[o][p] = Shared(cache'[k]) => OwnerEpoch[o] = vis'[k].d
                             /\ \A c \in copies : c.read = cache'[k] => GroupEpoch[c.g] = vis'[k].d
                        /\ (hcache[k] # NONE /\ vis'[k].h # NONE) =>
                             /\ \A c \in copies : c.read = hcache[k] => GroupEpoch[c.g] = vis'[k].h
                             /\ \A t \in tasks : t.kind = "put" => t.h # hcache[k]
                        /\ \A c \in copies :
                             (c.kind = "load" /\ c.fill /\ c.k = k) => vis'[k].d = MadeBy(GroupEpoch[c.g])
                        /\ \A c \in copies :
                             (c.kind = "store" /\ c.fill /\ c.k = k) => vis'[k].h = MadeBy(GroupEpoch[c.g])
                        /\ \A t \in tasks :
                             (t.kind = "replicate" /\ cache'[k] \in t.units) =>
                                 vis'[k].d = MadeBy(OwnerEpoch[t.o])
      BY DEF TrialVisibility, MadeBy
    <2>7. CASE k = d.k
      BY <1>1, <2>0, <2>2, <2>3, <2>4, <2>6, <2>7, <2>9 DEF TrialVisibility
    <2>8. CASE k # d.k
      <3>1. cache[k] # NONE => cache[k] \in Units
        BY <1>0 DEF UnitTyped
      <3> QED
        BY <2>0, <2>1, <2>5, <2>8, <3>1 DEF TrialVisibility
    <2> QED
      BY <2>7, <2>8
  <1>13. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A x : Covered(b, o, x) => Covered(b, o, x)'
      <3>1. \A o \in Owners, p \in Rows : \A x : HoldsAt(o, p, x)' = HoldsAt(o, p, x)
        BY <1>2 DEF HoldsAt, Rows
      <3>2. \A e \in deferred : e.kind \in {"release", "transfer"} => e \in deferred'
        OBVIOUS
      <3> QED
        BY <3>1, <3>2 DEF Covered, Live, ReadersOf
    <2>4. \A x \in b.reads : x[2] \in AllUnits
      BY <1>0, <2>2 DEF RequestTyped
    <2> QED
      BY <1>1, <1>2, <2>2, <2>3, <2>4 DEF BatchOk, Mem
  (* A crossing of the block is its owner's, and the commit makes the block *)
  (* visible as that owner's epoch makes it.                               *)
  <1>14. CrossingsOk'
    <2>1. \A o \in Owners : Moving(o)' = Moving(o)
      BY DEF Moving
    <2>2. \A e \in deferred' : CrossWaitOk(e)'
      <3> SUFFICES ASSUME NEW e \in deferred', e.kind = "replicate" PROVE CrossWaitOk(e)'
        BY DEF CrossWaitOk
      <3>1. e \in deferred /\ CrossWaitOk(e)
        BY CrossingsOkEach DEF IndInv
      <3>2. PICK v \in Units : e.units = {v} /\ (alloc[v] = e.o \/ alloc[v] = "shared")
        BY <3>1 DEF CrossWaitOk
      <3>3. v = u => e.o = d.o
        BY <3>2
      <3>4. alloc'[v] = e.o \/ alloc'[v] = "shared"
        BY <1>2, <3>2 DEF AllUnits
      <3>5. \A k \in Keys : cache'[k] \in e.units => vis'[k].d = MadeBy(OwnerEpoch[e.o])
        <4> SUFFICES ASSUME NEW k \in Keys, cache'[k] \in e.units
                     PROVE  vis'[k].d = MadeBy(OwnerEpoch[e.o])
          OBVIOUS
        <4>1. CASE k = d.k
          BY <1>2, <3>2, <3>3, <4>1
        <4>2. CASE k # d.k
          BY <1>2, <3>1, <4>2 DEF CrossWaitOk
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <2>1, <3>1, <3>2, <3>4, <3>5 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>3. \A t \in tasks' : CrossTaskOk(t)'
      <3> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate" PROVE CrossTaskOk(t)'
        BY DEF CrossTaskOk
      <3>1. CrossTaskOk(t)
        BY CrossingsOkEach DEF IndInv
      <3>2. \A x \in t.writes : x \in AllUnits /\ (alloc'[x] = t.o \/ alloc'[x] = "shared")
        <4> SUFFICES ASSUME NEW x \in t.writes PROVE x \in AllUnits /\ (alloc'[x] = t.o \/ alloc'[x] = "shared")
          OBVIOUS
        <4>1. x \in Units /\ (alloc[x] = t.o \/ alloc[x] = "shared")
          BY <3>1 DEF CrossTaskOk
        <4> QED
          BY <1>2, <4>1 DEF AllUnits
      <3> QED
        BY <2>1, <3>1, <3>2 DEF CrossTaskOk, ReadersOf, Mem
    <2> QED
      BY <2>2, <2>3, CrossingsOkEach
  <1>15. CopiesPlaced' /\ CrossingsPlaced'
    BY TransfersFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
  <1> QED
    BY <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15 DEF IndInv

LEMMA CommitDue_Inv ==
  ASSUME IndInv, NEW d \in deferred, CommitDue(d), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ d.kind = "commit" /\ Reached(d.after) /\ deferred' = deferred \ {d}
        /\ d.o \in Owners /\ d.p \in Positions /\ d.k \in Keys
        /\ UNCHANGED <<ostate, ogroup, oslot, hcache, copies, ncopy, tasks, trial, member, dbt,
                       launched, inflight, handoff, requests, remote, lost, bad>>
    BY CommitDue_Effect
  <1>2. PICK u \in Units :
          /\ d.units = {u} /\ alloc[u] = d.o
          /\ \/ /\ cache[d.k] # NONE
                /\ DropWith(d.units, table, oslot, cache, hcache, deferred', copies, tasks)
                /\ UNCHANGED <<table, cache, valid, vis>>
             \/ /\ cache[d.k] = NONE
                /\ cache' = [cache EXCEPT ![d.k] = u]
                /\ valid' = valid
                /\ vis' = [vis EXCEPT ![d.k].d = MadeBy(OwnerEpoch[d.o])]
                /\ table' = [table EXCEPT ![d.o][d.p] = IF @ = Private(u) THEN Shared(u) ELSE @]
                /\ free' = free
                /\ alloc' = [alloc EXCEPT ![u] = "shared"]
                /\ sharedKey' = [sharedKey EXCEPT ![u] = IF @ = NONE THEN d.k ELSE @]
    BY CommitDue_Effect
  <1>3. CASE cache[d.k] # NONE
    <2>1. /\ d.units = d.units \cup UNION {{c.read, c.write} : c \in {}}
          /\ copies' = copies \ {}
          /\ ValidTyped'
          /\ DropWith(d.units, table, oslot, cache, hcache, deferred', copies', tasks)
          /\ UNCHANGED <<table, cache, valid, vis>>
      BY <1>1, <1>2, <1>3, UnitTypes DEF UnitTyped, ValidTyped
    <2> QED
      BY <1>1, <2>1, Retire
  <1>4. CASE cache[d.k] = NONE
    BY <1>1, <1>2, <1>4, CommitDue_Share
  <1> QED
    BY <1>3, <1>4

-----------------------------------------------------------------------------
(* FinishDue                                                               *)

(* The end of copy c as FinishDue states it, and whether its owner takes   *)
(* its new block.                                                          *)
EndStep(d, c) ==
    /\ c.id = d.c
    /\ copies' = copies \ {c}
    /\ vis' = IF d.ok /\ c.kind = "load" /\ ~c.fill /\ ~(On("R4") /\ cache[c.k] # NONE)
              THEN [vis EXCEPT ![c.k].d = MadeBy(GroupEpoch[c.g])]
              ELSE IF /\ d.ok /\ c.kind \in {"store", "save"}
                      /\ ~(On("R4") /\ hcache[c.k] # NONE)
              THEN [vis EXCEPT ![c.k].h = MadeBy(GroupEpoch[c.g])]
              ELSE vis
    /\ LET target == c.o # NONE /\ ostate[c.o] = "live" /\ table[c.o][c.p] = Empty
       IN CASE ~d.ok ->
                 /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                 /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff>>
            [] d.ok /\ c.kind = "clone" ->
                 IF target
                 THEN /\ table' = [table EXCEPT ![c.o][c.p] = Private(c.write)]
                      /\ valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}]
                      /\ Give([g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write, w |-> TRUE])
                      /\ Drop({c.read}, table', oslot, cache, hcache, deferred', copies')
                      /\ UNCHANGED <<cache, hcache>>
                 ELSE /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                      /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff>>
            [] d.ok /\ c.kind = "load" /\ c.fill ->
                 /\ valid' = [valid EXCEPT ![c.write] = @ \cup {Mem(c.g)}]
                 /\ IF target
                    THEN /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
                         /\ Give([g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write, w |-> FALSE])
                    ELSE UNCHANGED <<table, dbt, handoff>>
                 /\ UNCHANGED <<free, cache, hcache, alloc, sharedKey>>
            [] d.ok /\ c.kind = "load" /\ ~c.fill ->
                 IF On("R4") /\ cache[c.k] # NONE
                 THEN /\ IF target
                         THEN /\ table' = [table EXCEPT ![c.o][c.p] = Private(c.write)]
                              /\ Give([g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write,
                                       w |-> FALSE])
                              /\ Drop({c.read}, table', oslot, cache, hcache, deferred',
                                      copies')
                         ELSE /\ Drop({c.read, c.write}, table, oslot, cache, hcache,
                                      deferred', copies')
                              /\ UNCHANGED <<table, dbt, handoff>>
                      /\ valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}]
                      /\ UNCHANGED <<cache, hcache>>
                 ELSE /\ cache' = [cache EXCEPT ![c.k] = c.write]
                      /\ valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}]
                      /\ IF target
                         THEN /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
                              /\ Give([g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write,
                                       w |-> FALSE])
                         ELSE UNCHANGED <<table, dbt, handoff>>
                      /\ free' = free
                      /\ alloc' = [alloc EXCEPT ![c.write] = "shared"]
                      /\ sharedKey' = [sharedKey EXCEPT ![c.write] = c.k]
                      /\ UNCHANGED hcache
            [] d.ok /\ c.kind = "restore" ->
                 /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                 /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff>>
            [] d.ok /\ c.kind = "store" /\ c.fill ->
                 /\ valid' = [valid EXCEPT ![c.write] = @ \cup c.parts]
                 /\ Drop({c.read}, table, oslot, cache, hcache, deferred', copies')
                 /\ UNCHANGED <<table, cache, hcache, dbt, handoff>>
            [] d.ok /\ c.kind \in {"store", "save"} /\ ~c.fill ->
                 IF On("R4") /\ hcache[c.k] # NONE
                 THEN /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                      /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff>>
                 ELSE /\ hcache' = [hcache EXCEPT ![c.k] = c.write]
                      /\ valid' = [valid EXCEPT ![c.write] = GroupParts[c.g]]
                      /\ LET gone == ReclaimedWith({c.read}, table, oslot, cache, hcache',
                                                   deferred', copies', tasks)
                         IN /\ free' = free \cup gone
                            /\ alloc' = [Reset(alloc, gone) EXCEPT ![c.write] = "shared"]
                            /\ sharedKey' = [Reset(sharedKey, gone) EXCEPT ![c.write] = c.k]
                      /\ UNCHANGED <<table, cache, dbt, handoff>>

Target(c) == c.o # NONE /\ ostate[c.o] = "live" /\ table[c.o][c.p] = Empty
Entry(c, wr) == [g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write, w |-> wr]

LEMMA FinishDue_Effect ==
  ASSUME NEW d \in deferred, FinishDue(d)
  PROVE  /\ d.kind = "finish" /\ Reached(d.after) /\ deferred' = deferred \ {d}
         /\ \E c \in copies : EndStep(d, c)
         /\ UNCHANGED <<ostate, ogroup, oslot, ncopy, tasks, trial, member, launched, inflight,
                        requests>>
  BY DEF FinishDue, EndStep, On, AllRules

(* What IndInv says of the copy that ends and of its Finish.               *)
LEMMA FinishDue_Facts ==
  ASSUME IndInv, NEW d \in deferred, d.kind = "finish", Reached(d.after),
         NEW c \in copies, c.id = d.c
  PROVE  /\ d.units = {}
         /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
         /\ CopyOk(c)
         /\ c.read \in AllUnits /\ c.write \in AllUnits /\ c.k \in Keys
         /\ c.g \in Groups /\ GroupEpoch[c.g] \notin lost
         /\ (c.o \in Owners /\ Live(c.o)) => ogroup[c.o] = c.g /\ ~Moving(c.o)
         /\ c.o \in Owners => OwnerEpoch[c.o] = GroupEpoch[c.g]
         /\ \A x \in copies : x.id = c.id => x = c
         (* The new block a clone or a load writes is the copy's alone.    *)
         /\ (~c.fill /\ c.kind \in {"clone", "load"}) =>
              /\ c.write \in Units /\ alloc[c.write] = c.o
              /\ \A o \in Owners, p \in Positions : table[o][p].u # c.write
              /\ \A o \in Owners : oslot[o] # c.write
              /\ \A k \in Keys : cache[k] # c.write /\ hcache[k] # c.write
              /\ \A e \in deferred : c.write \notin e.units
              /\ \A x \in copies : x # c => x.write # c.write /\ x.read # c.write
              /\ \A t \in tasks : c.write \notin t.units /\ t.slot # c.write /\ t.h # c.write
              /\ \A b \in inflight : \A x \in b.reads : x[2] # c.write
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, WorkTypes, RequestTypes
  <1>1. CopyOk(c)
    BY CopiesOkEach DEF IndInv
  <1>2. d.units = {}
    BY DEF IndInv, DeferredOk
  <1>3. \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
    <2> SUFFICES ASSUME NEW b \in inflight, c.g = b.g, c.seq = b.seq PROVE FALSE
      OBVIOUS
    <2>1. PICK x \in copies : x.id = d.c /\ (x.seq > 0 => d.after[x.g] >= x.seq) /\ (x.seq = 0 => x.stopped)
      BY DEF IndInv, DeferredOk
    <2>2. x = c
      BY <1>1, <2>1 DEF CopyOk
    <2>3. b.g \in Groups /\ b.seq \in 1 .. launched[b.g] /\ d.after[b.g] < b.seq
      BY DEF IndInv, BatchesOk, Reached
    <2>4. d.after[b.g] \in Nat
      BY <1>0, <2>3 DEF WorkTyped, Stamps
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4
  <1>4. (~c.fill /\ c.kind \in {"clone", "load"}) =>
          /\ c.write \in Units /\ alloc[c.write] = c.o
          /\ \A o \in Owners, p \in Positions : table[o][p].u # c.write
          /\ \A o \in Owners : oslot[o] # c.write
          /\ \A k \in Keys : cache[k] # c.write /\ hcache[k] # c.write
          /\ \A e \in deferred : c.write \notin e.units
          /\ \A x \in copies : x # c => x.write # c.write /\ x.read # c.write
          /\ \A t \in tasks : c.write \notin t.units /\ t.slot # c.write /\ t.h # c.write
          /\ \A b \in inflight : \A x \in b.reads : x[2] # c.write
    <2> SUFFICES ASSUME ~c.fill, c.kind \in {"clone", "load"}
                 PROVE  /\ c.write \in Units /\ alloc[c.write] = c.o
                        /\ \A o \in Owners, p \in Positions : table[o][p].u # c.write
                        /\ \A o \in Owners : oslot[o] # c.write
                        /\ \A k \in Keys : cache[k] # c.write /\ hcache[k] # c.write
                        /\ \A e \in deferred : c.write \notin e.units
                        /\ \A x \in copies : x # c => x.write # c.write /\ x.read # c.write
                        /\ \A t \in tasks : c.write \notin t.units /\ t.slot # c.write /\ t.h # c.write
                        /\ \A b \in inflight : \A x \in b.reads : x[2] # c.write
      OBVIOUS
    <2>1. /\ c.write \in Units /\ alloc[c.write] = c.o /\ c.o \in Owners
          /\ \A o \in Owners, p \in Positions : table[o][p].u # c.write
          /\ \A t \in tasks : c.write \notin t.units
          /\ \A x \in copies : x.write = c.write => x = c
      BY <1>1 DEF CopyOk
    <2>2. /\ \A o \in Owners : oslot[o] # c.write
          /\ \A t \in tasks : t.slot # c.write /\ t.h # c.write
      BY <1>0, <2>1 DEF OwnerTyped, WorkTyped
    <2>3. \A k \in Keys : cache[k] # c.write /\ hcache[k] # c.write
      BY <1>0, <2>1 DEF IndInv, CacheShared, SharedIff, UnitTyped, AllUnits
    <2>4. \A e \in deferred : c.write \notin e.units
      BY <2>1 DEF IndInv, DeferredOk, CrossingsOk
    <2>5. \A x \in copies : x # c => x.read # c.write
      BY <2>1 DEF IndInv, CopiesOk
    <2>6. \A b \in inflight : \A x \in b.reads : x[2] # c.write
      BY <2>1, <2>2, <2>4, ReadHeld, RequestTypes DEF RequestTyped
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>6
  <1> QED
    BY <1>0, <1>1, <1>2, <1>3, <1>4 DEF CopyOk, WorkTyped

(* What IndInv says of a copy that stays.                                  *)
LEMMA CopyFacts ==
  ASSUME IndInv, NEW x \in copies
  PROVE  /\ CopyOk(x)
         /\ x.read \in AllUnits /\ x.write \in AllUnits /\ x.k \in Keys
         /\ (x.fill /\ x.kind = "load") => x.write \in Units /\ cache[x.k] = x.write
  <1>0. UnitTyped /\ WorkTyped
    BY UnitTypes, WorkTypes
  <1>1. CopyOk(x)
    BY CopiesOkEach DEF IndInv
  <1>2. x.read \in AllUnits /\ x.write \in AllUnits /\ x.k \in Keys
    BY <1>0 DEF WorkTyped
  <1>3. (x.fill /\ x.kind = "load") => cache[x.k] = x.write
    BY <1>1 DEF CopyOk
  <1>4. (x.fill /\ x.kind = "load") => x.write \in Units
    BY <1>0, <1>2, <1>3 DEF UnitTyped, AllUnits
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4

(* The blocks the tables hold are device blocks.                           *)
LEMMA TableTypes ==
  ASSUME OwnerTyped
  PROVE  /\ \A q \in Owners, r \in Positions :
              /\ table[q][r].u # NONE => table[q][r].u \in Units
              /\ table[q][r].k = "none" <=> table[q][r].u = NONE
              /\ table[q][r].k \in {"none", "private", "shared"}
         /\ \A q \in Owners : TableUnits(q) \subseteq Units
  <1>1. \A q \in Owners, r \in Positions :
          /\ table[q][r].u # NONE => table[q][r].u \in Units
          /\ table[q][r].k = "none" <=> table[q][r].u = NONE
          /\ table[q][r].k \in {"none", "private", "shared"}
    BY DEF OwnerTyped, Empty, Private, Shared
  <1> QED
    BY <1>1 DEF TableUnits

(* The conjuncts on units after a step that keeps every unit's allocation, *)
(* the caches, and whether anything holds each unit.                       *)
LEMMA EndUnits ==
  ASSUME FreeClean, NoLeak, Outside, Regions, AllocOwner, SharedIff, SharedCached, CacheShared,
         UnitTyped, UNCHANGED <<free, alloc, sharedKey, cache, hcache, member, ostate>>,
         \A x \in AllUnits : Held(x)' = Held(x)
  PROVE  /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
         /\ SharedCached' /\ CacheShared'
  <1>1. Outside' /\ Regions' /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    BY OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame, SharedCachedFrame,
       CacheSharedFrame
  <1>2. FreeClean'
    BY DEF FreeClean, UnitTyped
  <1>3. NoLeak'
    BY DEF NoLeak, InRegion
  <1> QED
    BY <1>1, <1>2, <1>3

(* The control plane after deferred work and copies end and nothing else  *)
(* it reads changes.                                                       *)
LEMMA EndPlane ==
  ASSUME TypeInv, MovingUnmapped, LostOk, TablesMapped, HandoffOk, RequestsOk, NoBad,
         deferred' \subseteq deferred, copies' \subseteq copies,
         UNCHANGED <<ostate, ogroup, oslot, table, tasks, dbt, launched, ncopy, inflight, handoff,
                     requests, lost, bad>>,
         \A u \in Units : (\E q \in Owners, r \in Positions : table[q][r].u = u) =>
                           valid[u] \subseteq valid'[u]
  PROVE  MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
  <1>1. MovingUnmapped'
    BY DEF MovingUnmapped
  <1>2. LostOk' /\ NoBad'
    BY DEF LostOk, NoBad
  (* A reader's row names a block a table holds, which stays valid there.  *)
  <1>5. \A o \in Owners, p \in Positions, g \in Groups : \A x, wr :
          (x = table[o][p].u /\ x # NONE /\ ReaderRow(g, o, p, x, wr)) => ReaderRow(g, o, p, x, wr)'
    <2> SUFFICES ASSUME NEW o \in Owners, NEW p \in Positions, NEW g \in Groups, NEW x, NEW wr,
                        x = table[o][p].u, x # NONE, ReaderRow(g, o, p, x, wr)
                 PROVE  ReaderRow(g, o, p, x, wr)'
      OBVIOUS
    <2>1. x \in Units
      BY DEF TypeInv, TypeOK
    <2>2. valid[x] \subseteq valid'[x]
      BY <2>1
    <2>3. ReadersOf(o)' = ReadersOf(o)
      BY DEF ReadersOf
    <2> QED
      BY <2>2, <2>3 DEF ReaderRow, Mem
  <1>3. TablesMapped' /\ HandoffOk'
    <2>1. TablesMapped'
      <3> SUFFICES ASSUME NEW g \in Groups, NEW o \in Owners, NEW p \in Rows, dbt[g][o][p].u # NONE
                   PROVE  ReaderRow(g, o, p, dbt[g][o][p].u, dbt[g][o][p].w)
                            => ReaderRow(g, o, p, dbt[g][o][p].u, dbt[g][o][p].w)'
        BY DEF TablesMapped, Live, HoldsAt
      <3> SUFFICES ASSUME ReaderRow(g, o, p, dbt[g][o][p].u, dbt[g][o][p].w)
                   PROVE  ReaderRow(g, o, p, dbt[g][o][p].u, dbt[g][o][p].w)'
        OBVIOUS
      <3>1. p \in Positions /\ dbt[g][o][p].u = table[o][p].u
        BY DEF TablesMapped, ReaderRow, HoldsAt
      <3> QED
        BY <1>5, <3>1 DEF ReaderRow, Mem
    <2>2. HandoffOk'
      <3> SUFFICES ASSUME NEW h \in handoff, ~\E r \in requests : Names(r, h)
                   PROVE  ReaderRow(h.g, h.o, h.p, h.u, h.w) => ReaderRow(h.g, h.o, h.p, h.u, h.w)'
        BY DEF HandoffOk, Live, HoldsAt
      <3> SUFFICES ASSUME ReaderRow(h.g, h.o, h.p, h.u, h.w)
                   PROVE  ReaderRow(h.g, h.o, h.p, h.u, h.w)'
        OBVIOUS
      <3>1. h.o \in Owners /\ h.g \in Groups /\ h.u \in AllUnits
        BY DEF HandoffOk, TypeInv
      <3>2. h.p \in Positions /\ h.u = table[h.o][h.p].u /\ h.u # NONE
        BY <3>1 DEF HandoffOk, ReaderRow, HoldsAt, AllUnits
      <3> QED
        BY <1>5, <3>1, <3>2 DEF ReaderRow, Mem
    <2> QED
      BY <2>1, <2>2
  <1>4. RequestsOk'
    BY DEF RequestsOk
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4

(* Trial visibility after copies end, with the caches and tables kept.     *)
LEMMA EndVisible ==
  ASSUME TrialVisibility, copies' \subseteq copies,
         UNCHANGED <<cache, hcache, vis, table, tasks, trial>>
  PROVE  TrialVisibility'
  BY DEF TrialVisibility, MadeBy

(* The crossings after deferred work and copies end: the tasks and the     *)
(* owners stay, no block a crossing holds changes its allocation or        *)
(* becomes valid in the memory a crossing writes, and a key whose device   *)
(* block or its visibility changes is cached under no block a waiting      *)
(* crossing holds.                                                         *)
LEMMA EndCrossings ==
  ASSUME CrossingsOk, deferred' \subseteq deferred, copies' \subseteq copies,
         UNCHANGED <<tasks, ostate, ogroup, lost, trial>>,
         \A d \in deferred' : d.kind = "replicate" => \A u \in d.units : alloc'[u] = alloc[u],
         \A t \in tasks : t.kind = "replicate" =>
           \A u \in t.writes :
             alloc'[u] = alloc[u] /\ (Mem(t.to) \notin valid[u] => Mem(t.to) \notin valid'[u]),
         \A k \in Keys : ~(cache'[k] = cache[k] /\ vis'[k].d = vis[k].d) =>
           \A d \in deferred' : d.kind = "replicate" => cache'[k] \notin d.units
  PROVE  CrossingsOk'
  <1>1. \A o \in Owners : Moving(o)' => Moving(o)
    BY DEF Moving
  <1>2. \A d \in deferred' : CrossWaitOk(d)'
    <2> SUFFICES ASSUME NEW d \in deferred', d.kind = "replicate" PROVE CrossWaitOk(d)'
      BY DEF CrossWaitOk
    <2>1. d \in deferred /\ CrossWaitOk(d)
      BY CrossingsOkEach
    <2>2. \A k \in Keys : cache'[k] \in d.units => vis'[k].d = MadeBy(OwnerEpoch[d.o])'
      <3> SUFFICES ASSUME NEW k \in Keys, cache'[k] \in d.units
                   PROVE  vis'[k].d = MadeBy(OwnerEpoch[d.o])'
        OBVIOUS
      <3>1. cache'[k] = cache[k] /\ vis'[k].d = vis[k].d
        OBVIOUS
      <3> QED
        BY <2>1, <3>1 DEF CrossWaitOk, MadeBy
    <2> QED
      BY <1>1, <2>1, <2>2 DEF CrossWaitOk, ReadersOf
  <1>3. \A t \in tasks' : CrossTaskOk(t)'
    <2> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate" PROVE CrossTaskOk(t)'
      BY DEF CrossTaskOk
    <2>1. CrossTaskOk(t)
      BY CrossingsOkEach
    <2> QED
      BY <1>1, <2>1 DEF CrossTaskOk, ReadersOf, Mem
  <1> QED
    BY <1>2, <1>3, CrossingsOkEach

(* Placement after copies end: a copy that stays reads a block that stays  *)
(* valid where it was, and the host parts a fill writes keep their         *)
(* validity; a crossing's block stays valid where it was.                  *)
LEMMA EndPlaced ==
  ASSUME CopiesPlaced, CrossingsPlaced, copies' \subseteq copies,
         UNCHANGED <<tasks, ogroup, inflight>>,
         \A c \in copies' :
           /\ c.read \in Units \cup HostUnits => valid[c.read] \subseteq valid'[c.read]
           /\ (c.fill /\ c.write \in HostUnits) => valid'[c.write] = valid[c.write],
         \A t \in tasks : t.kind = "replicate" => \A u \in t.units : valid[u] \subseteq valid'[u]
  PROVE  CopiesPlaced' /\ CrossingsPlaced'
  <1>1. CopiesPlaced'
    BY DEF CopiesPlaced, Mem
  <1>2. CrossingsPlaced'
    BY DEF CrossingsPlaced, Mem
  <1> QED
    BY <1>1, <1>2

(* The tables after a step that fills at most one empty position, of a     *)
(* live owner, with a block valid in the owner's memory, and keeps the     *)
(* allocation and the validity of every other block a table holds.         *)
LEMMA TablesEntry ==
  ASSUME Tables, OwnerTyped, NEW o \in Owners, NEW p \in Positions, NEW v \in Units, NEW e,
         e = Private(v) \/ e = Shared(v), table[o][p] = Empty, Live(o),
         \A q \in Owners, r \in Positions :
             table'[q][r] = table[q][r] \/ (q = o /\ r = p /\ table'[q][r] = e),
         e = Private(v) => alloc'[v] = o /\ \A q \in Owners, r \in Positions : table[q][r].u # v,
         e = Shared(v) => alloc'[v] = "shared",
         Mem(ogroup[o]) \in valid'[v],
         \A q \in Owners, r \in Positions :
             (table[q][r].u # NONE /\ table[q][r].u # v) =>
                 /\ alloc'[table[q][r].u] = alloc[table[q][r].u]
                 /\ valid[table[q][r].u] \subseteq valid'[table[q][r].u],
         \A q \in Owners, r \in Positions :
             table[q][r].u = v => table[q][r].k = "shared" /\ valid[v] \subseteq valid'[v],
         UNCHANGED <<ostate, ogroup>>
  PROVE  Tables'
  <1>1. /\ e.u = v /\ e.k \in {"private", "shared"}
        /\ e.k = "private" => e = Private(v)
    BY DEF Private, Shared
  <1> SUFFICES ASSUME NEW q \in Owners, NEW r \in Positions
               PROVE  /\ table'[q][r].k = "none" => table'[q][r].u = NONE
                      /\ table'[q][r].k = "private" =>
                           table'[q][r].u \in Units /\ alloc'[table'[q][r].u] = q
                      /\ table'[q][r].k = "shared" =>
                           table'[q][r].u \in Units /\ alloc'[table'[q][r].u] = "shared"
                      /\ table'[q][r].k # "none" => Live(q)'
                      /\ \A s \in Positions :
                           (table'[q][r].k = "private" /\ table'[q][s].u = table'[q][r].u) => r = s
                      /\ table'[q][r].u # NONE => Mem(ogroup'[q]) \in valid'[table'[q][r].u]
    BY DEF Tables
  <1>2. Live(q)' = Live(q)
    BY DEF Live
  <1>3. CASE q = o /\ r = p /\ table'[q][r] = e
    <2>1. e.k = "private" => \A s \in Positions : s # p => table'[o][s].u # v
      <3> SUFFICES ASSUME e.k = "private", NEW s \in Positions, s # p PROVE table'[o][s].u # v
        OBVIOUS
      <3>1. table'[o][s] = table[o][s]
        OBVIOUS
      <3> QED
        BY <1>1, <3>1
    <2> QED
      BY <1>1, <1>2, <1>3, <2>1 DEF Private, Shared
  <1>4. CASE table'[q][r] = table[q][r]
    <2>1. /\ table[q][r].k = "none" => table[q][r].u = NONE
          /\ table[q][r].k = "private" => table[q][r].u \in Units /\ alloc[table[q][r].u] = q
          /\ table[q][r].k = "shared" => table[q][r].u \in Units /\ alloc[table[q][r].u] = "shared"
          /\ table[q][r].k # "none" => Live(q)
          /\ \A s \in Positions :
               (table[q][r].k = "private" /\ table[q][s].u = table[q][r].u) => r = s
          /\ table[q][r].u # NONE => Mem(ogroup[q]) \in valid[table[q][r].u]
      BY DEF Tables
    <2>2. table[q][r].k = "private" => table[q][r].u # v
      OBVIOUS
    <2>3. table[q][r].u = v => e = Shared(v)
      OBVIOUS
    <2>4. \A s \in Positions :
            table'[q][s].u = table[q][s].u \/ (q = o /\ s = p /\ table'[q][s].u = v)
      BY <1>1
    <2> QED
      BY <1>2, <1>4, <2>1, <2>2, <2>3, <2>4
  <1> QED
    BY <1>3, <1>4

(* The tables after a step that keeps them and the owners, keeps the       *)
(* allocation of what they hold, and shrinks no validity.                  *)
LEMMA TablesKeep ==
  ASSUME Tables, UNCHANGED <<table, ostate, ogroup>>,
         \A q \in Owners, r \in Positions :
             table[q][r].u # NONE =>
                 /\ alloc'[table[q][r].u] = alloc[table[q][r].u]
                 /\ valid[table[q][r].u] \subseteq valid'[table[q][r].u]
  PROVE  Tables'
  BY DEF Tables, Live, Mem

(* The owners after a step that changes at most the table of live owner o *)
(* and keeps the allocation of every live owner's slot row.               *)
LEMMA OwnersEntry ==
  ASSUME OwnersOk, NEW o \in Owners, Live(o), UNCHANGED <<ostate, ogroup, lost, trial, oslot>>,
         \A q \in Owners : Live(q) => alloc'[oslot[q]] = alloc[oslot[q]],
         \A q \in Owners, r \in Positions : q # o => table'[q][r] = table[q][r]
  PROVE  OwnersOk'
  BY DEF OwnersOk, Live

(* A copy that ends and whose new block, if any, its owner does not take.  *)
LEMMA FinishDue_Drop ==
  ASSUME IndInv, NEW d \in deferred, d.kind = "finish", Reached(d.after),
         deferred' = deferred \ {d}, NEW c \in copies, c.id = d.c, copies' = copies \ {c},
         DropWith({c.read, c.write}, table, oslot, cache, hcache, deferred', copies', tasks),
         \/ valid' = valid
         \/ /\ ~c.fill /\ c.kind \in {"clone", "load"}
            /\ valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}],
         UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, ncopy, tasks, trial, member, vis,
                     dbt, launched, inflight, handoff, requests, remote, lost, bad>>
  PROVE  IndInv'
  <1>0. UnitTyped
    BY UnitTypes
  <1>1. /\ d.units = {}
        /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
        /\ c.g \in Groups
    BY FinishDue_Facts
  <1>2. {c.read, c.write} = d.units \cup UNION {{x.read, x.write} : x \in {c}}
    BY <1>1
  <1>3. ValidTyped'
    <2>1. (~c.fill /\ c.kind \in {"clone", "load"}) => c.write \in Units
      BY FinishDue_Facts
    <2> QED
      BY <1>0, <1>1, <2>1 DEF UnitTyped, ValidTyped, Memories, Mem
  <1>4. \A u \in Units \cup HostUnits : valid'[u] = valid[u] \/ (u \in {c.read, c.write} /\ ~Held(u)')
    <2>1. CASE valid' = valid
      BY <2>1
    <2>2. CASE /\ ~c.fill /\ c.kind \in {"clone", "load"}
               /\ valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}]
      <3>1. ~Held(c.write)'
        BY <2>2, FinishDue_Facts DEF Held, HeldIn
      <3> QED
        BY <1>0, <2>2, <3>1 DEF UnitTyped
    <2> QED
      BY <2>1, <2>2
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, Retire

(* The step after which nothing but a new table entry, the validity of the *)
(* copy's new block and the executors' tables changed: the owner takes the *)
(* block a clone or a load wrote as its private block.                     *)
LEMMA FinishDue_Private ==
  ASSUME IndInv, NEW d \in deferred, d.kind = "finish", Reached(d.after),
         deferred' = deferred \ {d}, NEW c \in copies, c.id = d.c, copies' = copies \ {c},
         c.kind \in {"clone", "load"}, ~c.fill, Target(c), NEW wr \in BOOLEAN,
         table' = [table EXCEPT ![c.o][c.p] = Private(c.write)],
         valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}],
         Give(Entry(c, wr)),
         DropWith({c.read}, table', oslot, cache, hcache, deferred', copies', tasks),
         UNCHANGED <<ostate, ogroup, oslot, cache, hcache, ncopy, tasks, trial, member, vis,
                     launched, inflight, requests, remote, lost, bad>>
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1> DEFINE o == c.o  p == c.p  w == c.write
  <1>1. /\ o \in Owners /\ p \in Positions /\ Live(o) /\ table[o][p] = Empty
        /\ ogroup[o] = c.g /\ ~Moving(o) /\ c.g \in Groups
        /\ OwnerEpoch[o] = GroupEpoch[c.g]
        /\ w \in Units /\ alloc[w] = o
        /\ \A q \in Owners, r \in Positions : table[q][r].u # w
        /\ \A q \in Owners : oslot[q] # w
        /\ \A k \in Keys : cache[k] # w /\ hcache[k] # w
        /\ \A e \in deferred : w \notin e.units
        /\ \A x \in copies : x # c => x.write # w /\ x.read # w
        /\ \A t \in tasks : w \notin t.units /\ t.slot # w /\ t.h # w
        /\ \A b \in inflight : \A x \in b.reads : x[2] # w
        /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
        /\ \A x \in copies : x.id = c.id => x = c
        /\ CopyOk(c)
        /\ d.units = {} /\ c.read \in AllUnits /\ alloc[c.read] = "shared"
    BY FinishDue_Facts DEF Target, Live, CopyOk
  <1>2. \A x \in copies : x # c => ~(x.o = o /\ x.p = p /\ x.kind \in {"clone", "load"})
    BY <1>1 DEF IndInv, CopiesOk
  (* What the copy read stays cached, so nothing is reclaimed.             *)
  <1>3. \E k \in Keys : cache[k] = c.read \/ hcache[k] = c.read
    <2>1. SharedIff /\ SharedCached
      BY DEF IndInv
    <2>2. sharedKey[c.read] \in Keys
      BY <1>0, <1>1, <2>1 DEF SharedIff, UnitTyped
    <2> QED
      BY <1>1, <2>1, <2>2 DEF SharedCached
  <1>4. /\ \A q \in Owners, r \in Positions :
             table'[q][r] = IF q = o /\ r = p THEN Private(w) ELSE table[q][r]
        /\ \A u \in Units : valid'[u] = IF u = w THEN {Mem(c.g)} ELSE valid[u]
        /\ \A u \in HostUnits : valid'[u] = valid[u]
    BY <1>0, <1>1 DEF OwnerTyped, UnitTyped
  <1>5. \A x \in AllUnits : Held(x)' = Held(x)
    <2>1. \A x \in AllUnits : (x # w /\ x # c.read) =>
            /\ (\E q \in Owners, r \in Positions : table'[q][r].u = x)
                 <=> (\E q \in Owners, r \in Positions : table[q][r].u = x)
            /\ (\E e \in deferred' : x \in e.units) <=> (\E e \in deferred : x \in e.units)
            /\ (\E y \in copies' : x = y.read \/ x = y.write)
                 <=> (\E y \in copies : x = y.read \/ x = y.write)
      BY <1>1, <1>4 DEF Private, Empty, AllUnits
    <2>2. \A x \in AllUnits : (x # w /\ x # c.read) =>
            (HeldIn(x, table', oslot, cache, hcache, deferred', copies', tasks)
               <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks))
      BY <2>1 DEF HeldIn
    <2>3. /\ HeldIn(w, table', oslot, cache, hcache, deferred', copies', tasks)
          /\ HeldIn(w, table, oslot, cache, hcache, deferred, copies, tasks)
          /\ HeldIn(c.read, table', oslot, cache, hcache, deferred', copies', tasks)
          /\ HeldIn(c.read, table, oslot, cache, hcache, deferred, copies, tasks)
      BY <1>1, <1>3, <1>4 DEF HeldIn, Private
    <2>4. \A x \in AllUnits :
            HeldIn(x, table', oslot, cache, hcache, deferred', copies', tasks)
              <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
      <3> SUFFICES ASSUME NEW x \in AllUnits
                   PROVE  HeldIn(x, table', oslot, cache, hcache, deferred', copies', tasks)
                            <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
        OBVIOUS
      <3>1. CASE x = w
        BY <2>3, <3>1
      <3>2. CASE x = c.read
        BY <2>3, <3>2
      <3>3. CASE x # w /\ x # c.read
        BY <2>2, <3>3
      <3> QED
        BY <3>1, <3>2, <3>3
    <2>5. \A x : Held(x)' = HeldIn(x, table', oslot, cache, hcache, deferred', copies', tasks)
      BY DEF Held
    <2> QED
      BY <2>4, <2>5 DEF Held, HeldIn
  <1>6. UNCHANGED <<free, alloc, sharedKey>>
    <2>1. ReclaimedWith({c.read}, table', oslot, cache, hcache, deferred', copies', tasks) = {}
      BY <1>3, ReclaimedHeld DEF HeldIn, On, AllRules
    <2> QED
      BY <1>0, <2>1 DEF DropWith, Reset, UnitTyped
  <1>7. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>6 DEF UnitTyped, Memories, Mem
    <2>2. OwnerTyped'
      BY <1>0, <1>1, <1>4 DEF OwnerTyped, Private
    <2>3. PlaneTyped' /\ RequestTyped'
      <3>1. /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {Entry(c, wr)}
            /\ DOMAIN dbt' = Groups
            /\ \A g \in Groups : DOMAIN dbt'[g] = Owners
            /\ \A g \in Groups, q \in Owners : DOMAIN dbt'[g][q] = Rows
            /\ \A g \in Groups, q \in Owners, r \in Rows :
                 dbt'[g][q][r] = IF Eager /\ g = c.g /\ q = o /\ r = p /\ Unnamed(Entry(c, wr))
                                 THEN [u |-> w, w |-> wr] ELSE dbt[g][q][r]
        BY <1>0, <1>1, GiveRow DEF PlaneTyped, Entry, Rows
      <3> QED
        BY <1>0, <1>1, <3>1 DEF PlaneTyped, RequestTyped, Entry, Rows, AllUnits
    <2>4. WorkTyped'
      BY <1>0 DEF WorkTyped
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, TypeInvNext
  <1>8. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared'
    BY <1>0, <1>5, <1>6, EndUnits DEF IndInv
  <1>9. Tables' /\ OwnersOk'
    <2>1. \A q \in Owners, r \in Positions :
            table'[q][r] = table[q][r] \/ (q = o /\ r = p /\ table'[q][r] = Private(w))
      BY <1>4
    <2>2. \A q \in Owners, r \in Positions :
            (table[q][r].u # NONE /\ table[q][r].u # w) =>
                /\ alloc'[table[q][r].u] = alloc[table[q][r].u]
                /\ valid[table[q][r].u] \subseteq valid'[table[q][r].u]
      <3>1. \A q \in Owners, r \in Positions : table[q][r].u # NONE => table[q][r].u \in Units
        BY <1>0, TableTypes
      <3> QED
        BY <1>4, <1>6, <3>1
    <2>3. Mem(ogroup[o]) \in valid'[w]
      BY <1>1, <1>4
    <2>4. Tables'
      BY <1>0, <1>1, <1>6, <2>1, <2>2, <2>3, TablesEntry DEF IndInv, Private, Shared
    <2>5. OwnersOk'
      <3>1. \A q \in Owners, r \in Positions : q # o => table'[q][r] = table[q][r]
        BY <1>4
      <3> QED
        BY <1>1, <1>6, <3>1, OwnersEntry DEF IndInv
    <2> QED
      BY <2>4, <2>5
  <1>10. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred' PROVE DeferOk(e)'
      BY DeferredOkEach
    <2>2. e \in deferred /\ e # d /\ DeferOk(e) /\ DeferOk(d)
      BY DeferredOkEach DEF IndInv
    <2>3. e.kind = "finish" => \A x \in copies : x.id = e.c => x \in copies'
      BY <2>2 DEF DeferOk
    <2> QED
      BY <1>1, <1>4, <1>6, <2>2, <2>3 DEF DeferOk, Private
  <1>11. CopiesOk'
    <2>1. SUFFICES ASSUME NEW x \in copies' PROVE CopyOk(x)'
      BY CopiesOkEach
    <2>2. /\ x \in copies /\ x # c /\ CopyOk(x)
          /\ (x.fill /\ x.kind = "load") => x.write \in Units
      BY CopyFacts
    <2>3. /\ x.write # w /\ x.read # w
          /\ x.kind \in {"clone", "load"} => (x.o # o \/ x.p # p)
      BY <1>1, <1>2, <2>2 DEF CopyOk
    <2>4. \A q \in Owners : Moving(q)' => Moving(q)
      BY DEF Moving
    <2> QED
      BY <1>1, <1>4, <1>6, <2>2, <2>3, <2>4 DEF CopyOk, Live, Mem, Private
  <1>12. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. TaskOk(t)
      BY TasksOkEach DEF IndInv
    <2>3. t.kind = "transfer" => t.o # o
      BY <1>1 DEF Moving
    <2>4. \A q \in Owners : q # o => TableUnits(q)' = TableUnits(q)
      BY <1>4 DEF TableUnits
    <2>5. \A q \in Owners : w \notin TableUnits(q)
      BY <1>1 DEF TableUnits
    <2>6. \A q \in Owners : TableUnits(q) \subseteq Units
      BY <1>0, TableTypes
    <2> QED
      BY <1>1, <1>4, <1>6, <2>2, <2>3, <2>4, <2>5, <2>6 DEF TaskOk, Live, Mem
  <1>13. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
    <2>1. Entry(c, wr) = [g |-> ogroup[o], o |-> o, p |-> p, u |-> w, w |-> wr]
      BY <1>1 DEF Entry
    <2>2. wr => table'[o][p].k = "private" /\ \A e \in deferred : w \notin e.units
      BY <1>1, <1>4 DEF Private
    <2>3. \A q \in Owners, r \in Positions : (q # o \/ r # p) => table'[q][r] = table[q][r]
      BY <1>4
    <2>4. table'[o][p].u = w
      BY <1>1, <1>4 DEF Private
    <2>5. /\ wr => \A t \in tasks' : w \notin t.units
          /\ \A t \in tasks' : t.kind \in {"transfer", "replicate"} => t \in tasks
          /\ \A u \in Units : (\E q \in Owners, r \in Positions : table[q][r].u = u) =>
                              valid[u] \subseteq valid'[u]
      BY <1>1, <1>4
    <2> QED
      BY <1>1, <2>1, <2>2, <2>3, <2>4, <2>5, FillPlane DEF Live
  <1>14. TrialVisibility' /\ NoBad'
    BY <1>1, <1>4 DEF IndInv, TrialVisibility, MadeBy, NoBad, Private, Shared
  <1>15. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A q \in Owners : \A x \in AllUnits : Covered(b, q, x) => Covered(b, q, x)'
      <3>1. \A q \in Owners, r \in Rows : \A x \in AllUnits : HoldsAt(q, r, x) => HoldsAt(q, r, x)'
        BY <1>1, <1>4 DEF HoldsAt, Rows, Empty, AllUnits
      <3> QED
        BY <3>1 DEF Covered, Live, ReadersOf
    <2>4. \A y \in b.reads : y[1] \in Owners /\ y[2] \in AllUnits /\ y[2] # w
      BY <1>0, <1>1, <2>2 DEF BatchOk, RequestTyped
    <2>5. \A y \in b.writes, r \in Positions : table'[y[1]][r].u = y[2] => table[y[1]][r].u = y[2]
      BY <1>4, <2>2, <2>4 DEF BatchOk, Private
    <2>6. /\ \A x \in b.cw : \E y \in copies' : y.write = x /\ y.g = b.g /\ y.seq = b.seq
          /\ \A x \in b.cr : \E y \in copies' : y.read = x /\ y.g = b.g /\ y.seq = b.seq
      BY <1>1, <2>2 DEF BatchOk
    <2> QED
      BY <1>4, <1>6, <2>2, <2>3, <2>4, <2>5, <2>6 DEF BatchOk, Mem
  (* The new block is no crossing's, no copy's but this one's, and in no   *)
  (* table: what the crossings and the placement read stays.               *)
  <1>16. /\ \A t \in tasks : t.kind = "replicate" => w \notin t.writes
         /\ \A x \in copies' : x.read # w /\ x.write # w
    <2>1. \A t \in tasks : t.kind = "replicate" => t.writes = t.units
      BY CrossingsOkEach DEF IndInv, CrossTaskOk
    <2> QED
      BY <1>1, <2>1
  <1>17. CrossingsOk'
    <2>1. \A d0 \in deferred' : d0.kind = "replicate" => \A u \in d0.units : alloc'[u] = alloc[u]
      BY <1>6
    <2>2. \A t \in tasks : t.kind = "replicate" =>
            \A u \in t.writes :
              alloc'[u] = alloc[u] /\ (Mem(t.to) \notin valid[u] => Mem(t.to) \notin valid'[u])
      <3>1. \A t \in tasks : t.kind = "replicate" => t.writes \subseteq Units
        BY DEF IndInv, CrossingsOk
      <3> QED
        BY <1>4, <1>6, <1>16, <3>1
    <2>3. \A k \in Keys : cache'[k] = cache[k] /\ vis'[k].d = vis[k].d
      OBVIOUS
    <2> QED
      BY <2>1, <2>2, <2>3, EndCrossings DEF IndInv
  <1>18. CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A t \in tasks : t.kind = "replicate" => \A u \in t.units : u \in Units /\ u # w
      BY <1>1 DEF IndInv, CrossingsOk
    <2>2. \A x \in copies' :
            /\ x.read \in Units \cup HostUnits => valid[x.read] \subseteq valid'[x.read]
            /\ (x.fill /\ x.write \in HostUnits) => valid'[x.write] = valid[x.write]
      BY <1>4, <1>16
    <2>3. \A t \in tasks : t.kind = "replicate" => \A u \in t.units : valid[u] \subseteq valid'[u]
      BY <1>4, <2>1
    <2> QED
      BY <2>2, <2>3, EndPlaced DEF IndInv
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>17, <1>18 DEF IndInv

(* A load that filled its group's memory of a shared block: the block is   *)
(* valid there now, and pinned for the owner when it takes it.             *)
LEMMA FinishDue_Fill ==
  ASSUME IndInv, NEW d \in deferred, d.kind = "finish", Reached(d.after),
         deferred' = deferred \ {d}, NEW c \in copies, c.id = d.c, copies' = copies \ {c},
         c.kind = "load", c.fill,
         valid' = [valid EXCEPT ![c.write] = @ \cup {Mem(c.g)}],
         \/ /\ Target(c) /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
            /\ Give(Entry(c, FALSE))
         \/ /\ ~Target(c) /\ UNCHANGED <<table, dbt, handoff>>,
         UNCHANGED <<free, cache, hcache, alloc, sharedKey, ostate, ogroup, oslot, ncopy, tasks,
                     trial, member, vis, launched, inflight, requests, remote, lost, bad>>
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1> DEFINE o == c.o  p == c.p  w == c.write
  <1>1. /\ CopyOk(c) /\ c.g \in Groups /\ c.k \in Keys
        /\ cache[c.k] = w /\ Mem(c.g) \notin valid[w] /\ w \in Units
        /\ alloc[w] = "shared" /\ sharedKey[w] = c.k
        /\ \A x \in copies : (x.write = w /\ Mem(x.g) = Mem(c.g)) => x = c
        /\ \A t \in tasks : (t.kind = "transfer" /\ Mem(t.to) = Mem(c.g)) => w \notin t.writes
        /\ alloc[c.read] = "shared" /\ c.read \in AllUnits /\ d.units = {}
        /\ vis[c.k].d = MadeBy(GroupEpoch[c.g])
        /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
        /\ \A x \in copies : x.id = c.id => x = c
        /\ c.o \in Owners => OwnerEpoch[c.o] = GroupEpoch[c.g]
        /\ o \in Owners /\ p \in Positions
    <2>1. /\ CopyOk(c) /\ c.g \in Groups /\ c.k \in Keys
          /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
          /\ \A x \in copies : x.id = c.id => x = c
          /\ c.o \in Owners => OwnerEpoch[c.o] = GroupEpoch[c.g]
          /\ c.read \in AllUnits /\ c.write \in AllUnits /\ d.units = {}
      BY FinishDue_Facts
    <2>2. /\ cache[c.k] = w /\ Mem(c.g) \notin valid[w]
          /\ \A x \in copies : (x.write = w /\ Mem(x.g) = Mem(c.g)) => x = c
          /\ \A t \in tasks : (t.kind = "transfer" /\ Mem(t.to) = Mem(c.g)) => w \notin t.writes
          /\ alloc[c.read] = "shared"
          /\ o \in Owners /\ p \in Positions
      BY <2>1 DEF CopyOk
    <2>3. w \in Units /\ alloc[w] = "shared" /\ sharedKey[w] = c.k
      <3>1. CacheShared /\ SharedIff
        BY DEF IndInv
      <3>2. cache[c.k] # NONE
        BY <2>1, <2>2 DEF AllUnits
      <3>3. w \in Units /\ sharedKey[w] = c.k
        BY <2>1, <2>2, <3>1, <3>2 DEF CacheShared
      <3> QED
        BY <1>0, <2>1, <3>1, <3>3 DEF SharedIff, UnitTyped, AllUnits
    <2>4. vis[c.k].d = MadeBy(GroupEpoch[c.g])
      BY <2>1 DEF IndInv, TrialVisibility
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4
  <1>2. /\ \A u \in Units : valid'[u] = IF u = w THEN valid[w] \cup {Mem(c.g)} ELSE valid[u]
        /\ \A u \in HostUnits : valid'[u] = valid[u]
        /\ \A u \in Units \cup HostUnits : valid[u] \subseteq valid'[u]
    BY <1>0, <1>1 DEF UnitTyped
  <1>3. \A x \in AllUnits : Held(x)' = Held(x)
    <2>1. \E k \in Keys : cache[k] = c.read \/ hcache[k] = c.read
      <3>1. SharedIff /\ SharedCached
        BY DEF IndInv
      <3>2. sharedKey[c.read] \in Keys
        BY <1>0, <1>1, <3>1 DEF SharedIff, UnitTyped
      <3> QED
        BY <1>1, <3>1, <3>2 DEF SharedCached
    <2>2. \A q \in Owners, r \in Positions :
            \/ table'[q][r] = table[q][r]
            \/ q = o /\ r = p /\ table'[q][r] = Shared(w) /\ table[q][r] = Empty
      <3>1. CASE Target(c) /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
        BY <1>0, <1>1, <3>1 DEF OwnerTyped, Target
      <3>2. CASE ~Target(c) /\ UNCHANGED <<table, dbt, handoff>>
        BY <3>2
      <3> QED
        BY <3>1, <3>2
    <2>3. \A x \in AllUnits : (x # w /\ x # c.read) =>
            (HeldIn(x, table', oslot, cache, hcache, deferred', copies', tasks)
               <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks))
      <3>1. \A x \in AllUnits : x # w =>
              ((\E q \in Owners, r \in Positions : table'[q][r].u = x)
                 <=> (\E q \in Owners, r \in Positions : table[q][r].u = x))
        BY <2>2 DEF Shared, Empty, AllUnits
      <3>2. \A x \in AllUnits : (x # w /\ x # c.read) =>
              /\ (\E e \in deferred' : x \in e.units) <=> (\E e \in deferred : x \in e.units)
              /\ (\E y \in copies' : x = y.read \/ x = y.write)
                   <=> (\E y \in copies : x = y.read \/ x = y.write)
        BY <1>1
      <3> QED
        BY <3>1, <3>2 DEF HeldIn
    <2>4. /\ HeldIn(w, table', oslot, cache, hcache, deferred', copies', tasks)
          /\ HeldIn(w, table, oslot, cache, hcache, deferred, copies, tasks)
          /\ HeldIn(c.read, table', oslot, cache, hcache, deferred', copies', tasks)
          /\ HeldIn(c.read, table, oslot, cache, hcache, deferred, copies, tasks)
      BY <1>1, <2>1 DEF HeldIn
    <2>5. \A x \in AllUnits :
            HeldIn(x, table', oslot, cache, hcache, deferred', copies', tasks)
              <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
      <3> SUFFICES ASSUME NEW x \in AllUnits
                   PROVE  HeldIn(x, table', oslot, cache, hcache, deferred', copies', tasks)
                            <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
        OBVIOUS
      <3>1. CASE x = w
        BY <2>4, <3>1
      <3>2. CASE x = c.read
        BY <2>4, <3>2
      <3>3. CASE x # w /\ x # c.read
        BY <2>3, <3>3
      <3> QED
        BY <3>1, <3>2, <3>3
    <2>6. \A x : Held(x)' = HeldIn(x, table', oslot, cache, hcache, deferred', copies', tasks)
      BY DEF Held
    <2> QED
      BY <2>5, <2>6 DEF Held, HeldIn
  <1>4. \A q \in Owners : Moving(q)' => Moving(q)
    BY DEF Moving
  (* No crossing writes the block into the memory the load filled; the     *)
  (* blocks crossings and copies read only gain validity.                  *)
  <1>20. CrossingsOk'
    <2>1. \A t \in tasks : t.kind = "replicate" =>
            \A u \in t.writes :
              alloc'[u] = alloc[u] /\ (Mem(t.to) \notin valid[u] => Mem(t.to) \notin valid'[u])
      <3> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate", NEW u \in t.writes
                   PROVE  Mem(t.to) \notin valid[u] => Mem(t.to) \notin valid'[u]
        OBVIOUS
      <3>1. CrossTaskOk(t)
        BY CrossingsOkEach DEF IndInv
      <3>2. u \in Units /\ (Mem(c.g) = Mem(t.to) => w \notin t.writes)
        BY <1>1, <3>1 DEF CrossTaskOk
      <3> QED
        BY <1>2, <3>2
    <2>2. \A k \in Keys : cache'[k] = cache[k] /\ vis'[k].d = vis[k].d
      OBVIOUS
    <2> QED
      BY <2>1, <2>2, EndCrossings DEF IndInv
  <1>21. CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A x \in copies' :
            /\ x.read \in Units \cup HostUnits => valid[x.read] \subseteq valid'[x.read]
            /\ (x.fill /\ x.write \in HostUnits) => valid'[x.write] = valid[x.write]
      BY <1>2
    <2>2. \A t \in tasks : t.kind = "replicate" => \A u \in t.units : valid[u] \subseteq valid'[u]
      <3>1. \A t \in tasks : t.kind = "replicate" => t.units \subseteq Units
        BY DEF IndInv, CrossingsOk
      <3> QED
        BY <1>2, <3>1
    <2> QED
      BY <2>1, <2>2, EndPlaced DEF IndInv
  <1>5. CASE /\ Target(c) /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
             /\ Give(Entry(c, FALSE))
    <2>1. /\ o \in Owners /\ p \in Positions /\ Live(o) /\ table[o][p] = Empty
          /\ ogroup[o] = c.g /\ ~Moving(o)
      BY <1>1, <1>5, FinishDue_Facts DEF Target, Live, CopyOk
    <2>2. \A q \in Owners, r \in Positions :
            table'[q][r] = IF q = o /\ r = p THEN Shared(w) ELSE table[q][r]
      BY <1>0, <1>5, <2>1 DEF OwnerTyped
    <2>3. TypeInv'
      <3>1. UnitTyped'
        BY <1>0, <1>1, <1>2 DEF UnitTyped, Memories, Mem
      <3>2. OwnerTyped'
        BY <1>0, <1>1, <2>2 DEF OwnerTyped, Shared
      <3>3. PlaneTyped' /\ RequestTyped'
        <4>1. /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {Entry(c, FALSE)}
              /\ DOMAIN dbt' = Groups
              /\ \A g \in Groups : DOMAIN dbt'[g] = Owners
              /\ \A g \in Groups, q \in Owners : DOMAIN dbt'[g][q] = Rows
              /\ \A g \in Groups, q \in Owners, r \in Rows :
                   dbt'[g][q][r] = IF Eager /\ g = c.g /\ q = o /\ r = p /\ Unnamed(Entry(c, FALSE))
                                   THEN [u |-> w, w |-> FALSE] ELSE dbt[g][q][r]
          BY <1>0, <1>1, <1>5, <2>1, GiveRow DEF PlaneTyped, Entry, Rows
        <4> QED
          BY <1>0, <1>1, <2>1, <4>1 DEF PlaneTyped, RequestTyped, Entry, Rows, AllUnits
      <3>4. WorkTyped'
        BY <1>0 DEF WorkTyped
      <3> QED
        BY <3>1, <3>2, <3>3, <3>4, TypeInvNext
    <2>4. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
          /\ SharedCached' /\ CacheShared'
      BY <1>0, <1>3, EndUnits DEF IndInv
    <2>5. Tables' /\ OwnersOk'
      <3>1. \A q \in Owners, r \in Positions :
              table'[q][r] = table[q][r] \/ (q = o /\ r = p /\ table'[q][r] = Shared(w))
        BY <2>2
      <3>2. \A q \in Owners, r \in Positions :
              (table[q][r].u # NONE /\ table[q][r].u # w) =>
                  /\ alloc'[table[q][r].u] = alloc[table[q][r].u]
                  /\ valid[table[q][r].u] \subseteq valid'[table[q][r].u]
        <4>1. \A q \in Owners, r \in Positions : table[q][r].u # NONE => table[q][r].u \in Units
          BY <1>0, TableTypes
        <4> QED
          BY <1>2, <4>1
      <3>3. \A q \in Owners, r \in Positions :
              table[q][r].u = w => table[q][r].k = "shared" /\ valid[w] \subseteq valid'[w]
        <4>1. \A q \in Owners, r \in Positions :
                table[q][r].k = "private" => alloc[table[q][r].u] = q
          BY DEF IndInv, Tables
        <4>2. \A q \in Owners, r \in Positions :
                /\ table[q][r].k \in {"none", "private", "shared"}
                /\ table[q][r].k = "none" <=> table[q][r].u = NONE
          BY <1>0, TableTypes
        <4> QED
          BY <1>1, <1>2, <4>1, <4>2
      <3>4. Mem(ogroup[o]) \in valid'[w]
        BY <1>1, <1>2, <2>1
      <3>5. Tables'
        BY <1>0, <1>1, <2>1, <3>1, <3>2, <3>3, <3>4, TablesEntry DEF IndInv, Private, Shared
      <3>6. OwnersOk'
        <4>1. \A q \in Owners, r \in Positions : q # o => table'[q][r] = table[q][r]
          BY <2>2
        <4> QED
          BY <2>1, <4>1, OwnersEntry DEF IndInv
      <3> QED
        BY <3>5, <3>6
    <2>6. DeferredOk'
      <3>1. SUFFICES ASSUME NEW e \in deferred' PROVE DeferOk(e)'
        BY DeferredOkEach
      <3>2. e \in deferred /\ e # d /\ DeferOk(e) /\ DeferOk(d)
        BY DeferredOkEach DEF IndInv
      <3>3. e.kind = "commit" => w \notin e.units
        BY <1>1, <3>2 DEF DeferOk
      <3>4. e.kind = "finish" => \A x \in copies : x.id = e.c => x \in copies'
        BY <3>2 DEF DeferOk
      <3> QED
        BY <2>2, <3>2, <3>3, <3>4 DEF DeferOk, Shared
    <2>7. CopiesOk'
      <3>1. SUFFICES ASSUME NEW x \in copies' PROVE CopyOk(x)'
        BY CopiesOkEach
      <3>2. /\ x \in copies /\ x # c /\ CopyOk(x)
            /\ (x.fill /\ x.kind = "load") => x.write \in Units
        BY CopyFacts
      <3>3. (x.fill /\ x.kind = "load" /\ x.write = w) => Mem(x.g) # Mem(c.g)
        BY <1>1, <3>2
      <3>4. ~x.fill => x.write # w
        <4>1. ~x.fill =>
                /\ x.kind \in {"clone", "load", "store", "save"}
                /\ x.kind \in {"clone", "load"} => alloc[x.write] = x.o /\ x.o \in Owners
                /\ x.kind \in {"store", "save"} => x.write \in HostUnits
          BY <3>2 DEF CopyOk
        <4> QED
          BY <1>1, <4>1
      <3>5. x.kind \in {"clone", "load"} => (x.o # o \/ x.p # p)
        BY <1>1, <3>2 DEF CopyOk
      <3> QED
        BY <1>1, <1>2, <1>4, <2>2, <3>2, <3>3, <3>4, <3>5 DEF CopyOk, Live, Mem, Shared
    <2>8. TasksOk'
      <3>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
        BY TasksOkEach
      <3>2. TaskOk(t)
        BY TasksOkEach DEF IndInv
      <3>3. t.kind = "transfer" => t.o # o /\ (w \in t.writes => Mem(t.to) # Mem(c.g))
        BY <1>1, <2>1, <3>2 DEF Moving, TaskOk
      <3>4. \A q \in Owners : q # o => TableUnits(q)' = TableUnits(q)
        BY <2>2 DEF TableUnits
      <3>5. \A q \in Owners : TableUnits(q) \subseteq Units
        BY <1>0, TableTypes
      <3> QED
        BY <1>2, <2>2, <3>2, <3>3, <3>4, <3>5 DEF TaskOk, Live, Mem
    <2>9. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
      <3>1. Entry(c, FALSE) = [g |-> ogroup[o], o |-> o, p |-> p, u |-> w, w |-> FALSE]
        BY <2>1 DEF Entry
      <3>2. \A q \in Owners, r \in Positions : (q # o \/ r # p) => table'[q][r] = table[q][r]
        BY <2>2
      <3>3. table'[o][p].u = w
        BY <2>1, <2>2 DEF Shared
      <3>4. /\ \A t \in tasks' : t.kind \in {"transfer", "replicate"} => t \in tasks
            /\ \A u \in Units : (\E q \in Owners, r \in Positions : table[q][r].u = u) =>
                                valid[u] \subseteq valid'[u]
        BY <1>2
      <3> QED
        BY <1>1, <1>5, <2>1, <3>1, <3>2, <3>3, <3>4, FillPlane DEF Live
    <2>10. TrialVisibility' /\ NoBad'
      <3>1. vis[c.k].d # NONE => vis[c.k].d = OwnerEpoch[o]
        BY <1>1, <2>1 DEF MadeBy
      <3>2. \A k \in Keys : cache[k] = w => k = c.k
        BY <1>1 DEF IndInv, CacheShared
      <3> QED
        BY <1>1, <2>2, <3>1, <3>2 DEF IndInv, TrialVisibility, MadeBy, NoBad, Shared
    <2>11. BatchesOk'
      <3>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
        BY BatchesOkEach
      <3>2. BatchOk(b)
        BY BatchesOkEach DEF IndInv
      <3>3. \A q \in Owners : \A x \in AllUnits : Covered(b, q, x) => Covered(b, q, x)'
        <4>1. \A q \in Owners, r \in Rows : \A x \in AllUnits : HoldsAt(q, r, x) => HoldsAt(q, r, x)'
          BY <2>1, <2>2 DEF HoldsAt, Rows, Empty, AllUnits
        <4> QED
          BY <4>1 DEF Covered, Live, ReadersOf
      <3>4. \A y \in b.reads : y[1] \in Owners /\ y[2] \in AllUnits
        BY <1>0, <3>2 DEF BatchOk, RequestTyped
      <3>5. \A y \in b.writes, r \in Positions : table'[y[1]][r].u = y[2] => table[y[1]][r].u = y[2]
        BY <1>1, <2>2, <3>2, <3>4 DEF BatchOk, Shared
      <3> QED
        BY <1>1, <1>2, <3>2, <3>3, <3>4, <3>5 DEF BatchOk, Mem
    <2> QED
      BY <1>20, <1>21, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9, <2>10, <2>11 DEF IndInv
  <1>6. CASE ~Target(c) /\ UNCHANGED <<table, dbt, handoff>>
    <2>1. TypeInv'
      <3>1. UnitTyped'
        BY <1>0, <1>1, <1>2 DEF UnitTyped, Memories, Mem
      <3>2. WorkTyped'
        BY <1>0 DEF WorkTyped
      <3>3. OwnerTyped' /\ PlaneTyped' /\ RequestTyped'
        BY <1>0, <1>6, OwnerTypedFrame, PlaneTypedFrame, RequestTypedFrame
      <3> QED
        BY <3>1, <3>2, <3>3, TypeInvNext
    <2>2. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
          /\ SharedCached' /\ CacheShared' /\ OwnersOk'
      <3>1. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
            /\ SharedCached' /\ CacheShared'
        BY <1>0, <1>3, EndUnits DEF IndInv
      <3>2. OwnersOk'
        BY <1>6, OwnersOkFrame DEF IndInv
      <3> QED
        BY <3>1, <3>2
    <2>3. Tables'
      <3>1. \A q \in Owners, r \in Positions :
              table[q][r].u # NONE =>
                  /\ alloc'[table[q][r].u] = alloc[table[q][r].u]
                  /\ valid[table[q][r].u] \subseteq valid'[table[q][r].u]
        <4>1. \A q \in Owners, r \in Positions : table[q][r].u # NONE => table[q][r].u \in Units
          BY <1>0, TableTypes
        <4> QED
          BY <1>2, <4>1
      <3> QED
        BY <1>6, <3>1, TablesKeep DEF IndInv
    <2>4. DeferredOk'
      <3>1. SUFFICES ASSUME NEW e \in deferred' PROVE DeferOk(e)'
        BY DeferredOkEach
      <3>2. e \in deferred /\ e # d /\ DeferOk(e) /\ DeferOk(d)
        BY DeferredOkEach DEF IndInv
      <3>3. e.kind = "finish" => \A x \in copies : x.id = e.c => x \in copies'
        BY <3>2 DEF DeferOk
      <3> QED
        BY <1>6, <3>2, <3>3 DEF DeferOk
    <2>5. CopiesOk'
      <3>1. SUFFICES ASSUME NEW x \in copies' PROVE CopyOk(x)'
        BY CopiesOkEach
      <3>2. /\ x \in copies /\ x # c /\ CopyOk(x)
            /\ (x.fill /\ x.kind = "load") => x.write \in Units
        BY CopyFacts
      <3>3. x.fill /\ x.kind = "load" /\ x.write = w => Mem(x.g) # Mem(c.g)
        BY <1>1, <3>2 DEF CopyOk
      <3> QED
        BY <1>1, <1>2, <1>4, <1>6, <3>2, <3>3 DEF CopyOk, Live, Mem
    <2>6. TasksOk'
      <3>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
        BY TasksOkEach
      <3>2. TaskOk(t)
        BY TasksOkEach DEF IndInv
      <3>3. t.kind = "transfer" => (w \in t.writes => Mem(t.to) # Mem(c.g))
        BY <1>1, <3>2 DEF TaskOk
      <3>4. \A q \in Owners : TableUnits(q) \subseteq Units
        BY <1>0, TableTypes
      <3> QED
        BY <1>2, <1>6, <3>2, <3>3, <3>4 DEF TaskOk, Live, Mem, TableUnits
    <2>7. /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
          /\ TrialVisibility' /\ NoBad'
      BY <1>2, <1>6, EndPlane, EndVisible DEF IndInv
    <2>8. BatchesOk'
      <3>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
        BY BatchesOkEach
      <3>2. BatchOk(b)
        BY BatchesOkEach DEF IndInv
      <3>3. \A q \in Owners : \A x : Covered(b, q, x) => Covered(b, q, x)'
        BY <1>6 DEF Covered, Live, HoldsAt, ReadersOf
      <3> QED
        BY <1>1, <1>2, <1>6, <3>2, <3>3 DEF BatchOk, Mem
    <2> QED
      BY <1>20, <1>21, <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8 DEF IndInv
  <1> QED
    BY <1>5, <1>6

(* A load whose key has no device block: its new block becomes the key's   *)
(* shared block, pinned for the owner when it takes it.                    *)
LEMMA FinishDue_Cache ==
  ASSUME IndInv, NEW d \in deferred, d.kind = "finish", Reached(d.after),
         deferred' = deferred \ {d}, NEW c \in copies, c.id = d.c, copies' = copies \ {c},
         c.kind = "load", ~c.fill, cache[c.k] = NONE,
         cache' = [cache EXCEPT ![c.k] = c.write],
         valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}],
         vis' = [vis EXCEPT ![c.k].d = MadeBy(GroupEpoch[c.g])],
         \/ /\ Target(c) /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
            /\ Give(Entry(c, FALSE))
         \/ /\ ~Target(c) /\ UNCHANGED <<table, dbt, handoff>>,
         free' = free,
         alloc' = [alloc EXCEPT ![c.write] = "shared"],
         sharedKey' = [sharedKey EXCEPT ![c.write] = c.k],
         UNCHANGED <<hcache, ostate, ogroup, oslot, ncopy, tasks, trial, member, launched,
                     inflight, requests, remote, lost, bad>>
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1> DEFINE o == c.o  p == c.p  w == c.write
  <1>1. /\ CopyOk(c) /\ c.g \in Groups /\ c.k \in Keys /\ o \in Owners /\ p \in Positions
        /\ OwnerEpoch[o] = GroupEpoch[c.g] /\ GroupEpoch[c.g] \in Epochs
        /\ w \in Units /\ alloc[w] = o /\ sharedKey[w] = NONE /\ w \in member /\ w \notin free
        /\ \A q \in Owners, r \in Positions : table[q][r].u # w
        /\ \A q \in Owners : oslot[q] # w
        /\ \A k \in Keys : cache[k] # w /\ hcache[k] # w
        /\ \A e \in deferred : w \notin e.units
        /\ \A x \in copies : x # c => x.write # w /\ x.read # w
        /\ \A t \in tasks : w \notin t.units /\ t.slot # w /\ t.h # w
        /\ \A b \in inflight : \A x \in b.reads : x[2] # w
        /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
        /\ \A x \in copies : x.id = c.id => x = c
        /\ alloc[c.read] = "shared" /\ c.read \in AllUnits /\ d.units = {}
        /\ \E k \in Keys : cache[k] = c.read \/ hcache[k] = c.read
        /\ \A x \in copies : (x.kind = "load" /\ x.fill) => x.k # c.k
    <2>1. /\ CopyOk(c) /\ c.g \in Groups /\ c.k \in Keys
          /\ c.o \in Owners => OwnerEpoch[c.o] = GroupEpoch[c.g]
          /\ w \in Units /\ alloc[w] = o
          /\ \A q \in Owners, r \in Positions : table[q][r].u # w
          /\ \A q \in Owners : oslot[q] # w
          /\ \A k \in Keys : cache[k] # w /\ hcache[k] # w
          /\ \A e \in deferred : w \notin e.units
          /\ \A x \in copies : x # c => x.write # w /\ x.read # w
          /\ \A t \in tasks : w \notin t.units /\ t.slot # w /\ t.h # w
          /\ \A b \in inflight : \A x \in b.reads : x[2] # w
          /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
          /\ \A x \in copies : x.id = c.id => x = c
          /\ c.read \in AllUnits /\ d.units = {}
      BY FinishDue_Facts
    <2>2. o \in Owners /\ p \in Positions /\ alloc[c.read] = "shared"
      BY <2>1 DEF CopyOk
    <2>3. sharedKey[w] = NONE /\ w \in member /\ w \notin free
      <3>1. SharedIff /\ FreeClean /\ NoLeak
        BY DEF IndInv
      <3>2. Held(w)
        BY DEF Held, HeldIn
      <3> QED
        BY <2>1, <2>2, <3>1, <3>2 DEF SharedIff, FreeClean, NoLeak, InRegion, AllUnits
    <2>4. \E k \in Keys : cache[k] = c.read \/ hcache[k] = c.read
      <3>1. SharedIff /\ SharedCached
        BY DEF IndInv
      <3>2. sharedKey[c.read] \in Keys
        BY <1>0, <2>1, <2>2, <3>1 DEF SharedIff, UnitTyped
      <3> QED
        BY <2>1, <2>2, <3>1, <3>2 DEF SharedCached
    <2>5. \A x \in copies : (x.kind = "load" /\ x.fill) => x.k # c.k
      <3> SUFFICES ASSUME NEW x \in copies, x.kind = "load", x.fill PROVE x.k # c.k
        OBVIOUS
      <3>1. cache[x.k] = x.write /\ x.write \in AllUnits
        BY CopyFacts
      <3> QED
        BY <3>1 DEF AllUnits
    <2>6. GroupEpoch[c.g] \in Epochs
      BY <2>1
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>6
  (* The values after the step, pointwise.                                 *)
  <1>2. /\ \A k \in Keys : cache'[k] = IF k = c.k THEN w ELSE cache[k]
        /\ \A v \in AllUnits : alloc'[v] = IF v = w THEN "shared" ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v = w THEN c.k ELSE sharedKey[v]
        /\ \A v \in Units : valid'[v] = IF v = w THEN {Mem(c.g)} ELSE valid[v]
        /\ \A v \in HostUnits : valid'[v] = valid[v]
        /\ \A k \in Keys : vis'[k] = IF k = c.k THEN [d |-> MadeBy(GroupEpoch[c.g]), h |-> vis[k].h]
                                                ELSE vis[k]
        /\ \A q \in Owners, r \in Positions :
             table'[q][r] = IF Target(c) /\ q = o /\ r = p THEN Shared(w) ELSE table[q][r]
    <2>1. \A q \in Owners, r \in Positions :
            table'[q][r] = IF Target(c) /\ q = o /\ r = p THEN Shared(w) ELSE table[q][r]
      BY <1>0, <1>1 DEF OwnerTyped
    <2> QED
      BY <1>0, <1>1, <2>1 DEF UnitTyped, PlaneTyped, AllUnits
  <1>3. \A x \in AllUnits : Held(x)' = Held(x)
    <2>1. \A x \in AllUnits : (x # w /\ x # c.read) =>
            /\ (\E k \in Keys : cache'[k] = x \/ hcache[k] = x)
                 <=> (\E k \in Keys : cache[k] = x \/ hcache[k] = x)
            /\ (\E q \in Owners, r \in Positions : table'[q][r].u = x)
                 <=> (\E q \in Owners, r \in Positions : table[q][r].u = x)
            /\ (\E e \in deferred' : x \in e.units) <=> (\E e \in deferred : x \in e.units)
            /\ (\E y \in copies' : x = y.read \/ x = y.write)
                 <=> (\E y \in copies : x = y.read \/ x = y.write)
      BY <1>1, <1>2 DEF Shared, Target, Empty, AllUnits
    <2>2. \A x \in AllUnits : (x # w /\ x # c.read) =>
            (HeldIn(x, table', oslot, cache', hcache, deferred', copies', tasks)
               <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks))
      BY <2>1 DEF HeldIn
    <2>3. /\ HeldIn(w, table', oslot, cache', hcache, deferred', copies', tasks)
          /\ HeldIn(w, table, oslot, cache, hcache, deferred, copies, tasks)
          /\ HeldIn(c.read, table', oslot, cache', hcache, deferred', copies', tasks)
          /\ HeldIn(c.read, table, oslot, cache, hcache, deferred, copies, tasks)
      <3>1. cache'[c.k] = w /\ \E k \in Keys : cache'[k] = c.read \/ hcache[k] = c.read
        BY <1>1, <1>2 DEF AllUnits
      <3> QED
        BY <1>1, <3>1 DEF HeldIn
    <2>4. \A x \in AllUnits :
            HeldIn(x, table', oslot, cache', hcache, deferred', copies', tasks)
              <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
      <3> SUFFICES ASSUME NEW x \in AllUnits
                   PROVE  HeldIn(x, table', oslot, cache', hcache, deferred', copies', tasks)
                            <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
        OBVIOUS
      <3>1. CASE x = w
        BY <2>3, <3>1
      <3>2. CASE x = c.read
        BY <2>3, <3>2
      <3>3. CASE x # w /\ x # c.read
        BY <2>2, <3>3
      <3> QED
        BY <3>1, <3>2, <3>3
    <2>5. \A x : Held(x)' = HeldIn(x, table', oslot, cache', hcache, deferred', copies', tasks)
      BY DEF Held
    <2> QED
      BY <2>4, <2>5 DEF Held, HeldIn
  <1>4. \A q \in Owners : Moving(q)' => Moving(q)
    BY DEF Moving
  <1>5. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>2 DEF UnitTyped, Memories, Mem, AllUnits
    <2>2. OwnerTyped'
      BY <1>0, <1>1, <1>2 DEF OwnerTyped, Shared
    <2>3. PlaneTyped' /\ RequestTyped'
      <3>1. MadeBy(GroupEpoch[c.g]) \in Epochs \cup {NONE}
        BY <1>1 DEF MadeBy
      <3>2. CASE Target(c) /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
                 /\ Give(Entry(c, FALSE))
        <4>1. /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {Entry(c, FALSE)}
              /\ DOMAIN dbt' = Groups
              /\ \A g \in Groups : DOMAIN dbt'[g] = Owners
              /\ \A g \in Groups, q \in Owners : DOMAIN dbt'[g][q] = Rows
              /\ \A g \in Groups, q \in Owners, r \in Rows :
                   dbt'[g][q][r] = IF Eager /\ g = c.g /\ q = o /\ r = p /\ Unnamed(Entry(c, FALSE))
                                   THEN [u |-> w, w |-> FALSE] ELSE dbt[g][q][r]
          BY <1>0, <1>1, <3>2, GiveRow DEF PlaneTyped, Entry, Rows
        <4> QED
          BY <1>0, <1>1, <1>2, <3>1, <4>1 DEF PlaneTyped, RequestTyped, Entry, Rows, AllUnits
      <3>3. CASE ~Target(c) /\ UNCHANGED <<table, dbt, handoff>>
        BY <1>0, <1>2, <3>1, <3>3 DEF PlaneTyped, RequestTyped
      <3> QED
        BY <3>2, <3>3
    <2>4. WorkTyped'
      BY <1>0 DEF WorkTyped
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, TypeInvNext
  <1>6. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared'
    <2>0. /\ FreeClean /\ NoLeak /\ Outside /\ Regions /\ AllocOwner /\ SharedIff
          /\ SharedCached /\ CacheShared
      BY DEF IndInv
    <2>1. FreeClean'
      BY <1>0, <1>1, <1>2, <1>3, <2>0 DEF FreeClean, UnitTyped, AllUnits
    <2>2. NoLeak' /\ Regions'
      BY <1>3, <2>0, RegionsFrame DEF NoLeak, InRegion
    <2>3. Outside' /\ AllocOwner' /\ SharedIff'
      BY <1>1, <1>2, <2>0 DEF Outside, AllocOwner, SharedIff, AllUnits
    <2>4. SharedCached'
      <3>1. \A v \in AllUnits : alloc[v] = "shared" => sharedKey[v] \in Keys
        BY <1>0, <2>0 DEF SharedIff, UnitTyped
      <3> QED
        BY <1>1, <1>2, <2>0, <3>1 DEF SharedCached, AllUnits
    <2>5. CacheShared'
      BY <1>0, <1>1, <1>2, <2>0 DEF CacheShared, UnitTyped, AllUnits
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5
  <1>7. Tables' /\ OwnersOk'
    <2>1. Target(c) => Live(o) /\ ogroup[o] = c.g /\ table[o][p] = Empty
      BY <1>1, FinishDue_Facts DEF Target, Live, CopyOk
    <2>2. \A q \in Owners, r \in Positions :
            (table[q][r].u # NONE /\ table[q][r].u # w) =>
                /\ alloc'[table[q][r].u] = alloc[table[q][r].u]
                /\ valid[table[q][r].u] \subseteq valid'[table[q][r].u]
      <3>1. \A q \in Owners, r \in Positions : table[q][r].u # NONE => table[q][r].u \in Units
        BY <1>0, TableTypes
      <3> QED
        BY <1>2, <3>1 DEF AllUnits
    <2>3. Tables /\ OwnersOk
      BY DEF IndInv
    <2>4. \A q \in Owners : Live(q) => alloc'[oslot[q]] = alloc[oslot[q]]
      BY <1>1, <1>2, <2>3 DEF OwnersOk, AllUnits
    <2>5. CASE Target(c)
      <3>1. \A q \in Owners, r \in Positions :
              table'[q][r] = table[q][r] \/ (q = o /\ r = p /\ table'[q][r] = Shared(w))
        BY <1>2, <2>5
      <3>2. Mem(ogroup[o]) \in valid'[w]
        BY <1>1, <1>2, <2>1, <2>5
      <3>3. Tables'
        <4>1. /\ Live(o) /\ table[o][p] = Empty /\ o \in Owners /\ p \in Positions /\ w \in Units
              /\ alloc'[w] = "shared"
              /\ \A q \in Owners, r \in Positions : table[q][r].u # w
          BY <1>1, <1>2, <2>1, <2>5 DEF AllUnits
        <4>2. Tables /\ OwnerTyped
          BY <1>0, <2>3
        <4> QED
          BY <2>2, <3>1, <3>2, <4>1, <4>2, TablesEntry, SMTT(20) DEF Private, Shared
      <3>4. OwnersOk'
        <4>1. \A q \in Owners, r \in Positions : q # o => table'[q][r] = table[q][r]
          BY <1>2
        <4> QED
          BY <1>1, <2>1, <2>3, <2>4, <2>5, <4>1, OwnersEntry
      <3> QED
        BY <3>3, <3>4
    <2>6. CASE ~Target(c)
      <3>1. table' = table
        BY <2>6
      <3>2. Tables'
        BY <1>1, <2>2, <2>3, <3>1, TablesKeep
      <3>3. OwnersOk'
        BY <2>3, <2>4, <3>1 DEF OwnersOk, Live
      <3> QED
        BY <3>2, <3>3
    <2> QED
      BY <2>5, <2>6
  <1>8. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred' PROVE DeferOk(e)'
      BY DeferredOkEach
    <2>2. e \in deferred /\ e # d /\ DeferOk(e) /\ DeferOk(d)
      BY DeferredOkEach DEF IndInv
    <2>3. e.kind = "commit" => \A v \in e.units : alloc'[v] = alloc[v] /\ v # w
      BY <1>0, <1>1, <1>2, <2>2 DEF WorkTyped
    <2>4. e.kind = "finish" => \A x \in copies : x.id = e.c => x \in copies'
      BY <2>2 DEF DeferOk
    <2>5. \A q \in Owners, r \in Positions :
            /\ table'[q][r].k = "private" => table'[q][r] = table[q][r]
            /\ table'[q][r].u # w => table'[q][r] = table[q][r]
      BY <1>2 DEF Shared
    <2> QED
      BY <1>1, <2>2, <2>3, <2>4, <2>5 DEF DeferOk
  <1>9. CopiesOk'
    <2>1. SUFFICES ASSUME NEW x \in copies' PROVE CopyOk(x)'
      BY CopiesOkEach
    <2>2. /\ x \in copies /\ x # c /\ CopyOk(x) /\ x.read \in AllUnits /\ x.write \in AllUnits
          /\ x.k \in Keys
          /\ (x.fill /\ x.kind = "load") => x.write \in Units
      BY CopyFacts
    <2>3. /\ x.write # w /\ x.read # w
          /\ (x.kind = "load" /\ x.fill) => x.k # c.k
          /\ x.kind \in {"clone", "load"} => (x.o # o \/ x.p # p)
      BY <1>1, <2>2 DEF CopyOk
    <2> QED
      BY <1>1, <1>2, <1>4, <2>2, <2>3 DEF CopyOk, Live, Mem, Shared
  <1>10. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. TaskOk(t)
      BY TasksOkEach DEF IndInv
    <2>3. \A q \in Owners : ~(Target(c) /\ q = o) => TableUnits(q)' = TableUnits(q)
      BY <1>2 DEF TableUnits
    <2>4. t.kind = "transfer" => (Target(c) => t.o # o)
      <3>1. Target(c) => ~Moving(o)
        BY <1>1, FinishDue_Facts DEF Target, Live
      <3> QED
        BY <3>1 DEF Moving
    <2>5. /\ t.kind = "transfer" =>
               t.slot \in Slots /\ t.writes \subseteq t.units /\ t.o \in Owners
          /\ t.kind = "prefetch" => t.h \in HostUnits
          /\ t.kind = "put" => t.k \in Keys
      BY <1>0, <2>2 DEF TaskOk, WorkTyped
    <2>6. \A q \in Owners : TableUnits(q) \subseteq Units
      BY <1>0, TableTypes
    <2> QED
      BY <1>1, <1>2, <2>2, <2>3, <2>4, <2>5, <2>6 DEF TaskOk, Live, Mem, AllUnits
  <1>11. MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
    <2>1. CASE Target(c) /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
               /\ Give(Entry(c, FALSE))
      <3>1. /\ Live(o) /\ table[o][p] = Empty /\ ogroup[o] = c.g /\ ~Moving(o)
        BY <1>1, <2>1, FinishDue_Facts DEF Target, Live
      <3>2. Entry(c, FALSE) = [g |-> ogroup[o], o |-> o, p |-> p, u |-> w, w |-> FALSE]
        BY <3>1 DEF Entry
      <3>3. \A q \in Owners, r \in Positions : (q # o \/ r # p) => table'[q][r] = table[q][r]
        BY <1>2
      <3>4. table'[o][p].u = w
        BY <1>1, <1>2, <2>1 DEF Shared
      <3>5. /\ \A t \in tasks' : t.kind \in {"transfer", "replicate"} => t \in tasks
            /\ \A u \in Units : (\E q \in Owners, r \in Positions : table[q][r].u = u) =>
                                valid[u] \subseteq valid'[u]
        BY <1>1, <1>2
      <3> QED
        BY <1>1, <2>1, <3>1, <3>2, <3>3, <3>4, <3>5, FillPlane DEF Live
    <2>2. CASE ~Target(c) /\ UNCHANGED <<table, dbt, handoff>>
      <3>1. \A u \in Units : (\E q \in Owners, r \in Positions : table[q][r].u = u) =>
                             valid[u] \subseteq valid'[u]
        BY <1>1, <1>2
      <3> QED
        BY <2>2, <3>1, EndPlane DEF IndInv
    <2> QED
      BY <2>1, <2>2
  <1>12. TrialVisibility' /\ NoBad'
    <2>0. TrialVisibility
      BY DEF IndInv
    <2>1. \A q \in Owners, r \in Positions : table'[q][r] = Shared(w) => q = o
      BY <1>1, <1>2 DEF Shared
    <2>2. MadeBy(GroupEpoch[c.g]) # NONE => MadeBy(GroupEpoch[c.g]) = GroupEpoch[c.g]
      BY DEF MadeBy
    <2>3. \A q \in Owners, r \in Positions, x \in Units :
            (table'[q][r] = Shared(x) /\ x # w) => table[q][r] = Shared(x)
      BY <1>2 DEF Shared
    <2>4. /\ cache'[c.k] = w
          /\ vis'[c.k].d = MadeBy(GroupEpoch[c.g]) /\ vis'[c.k].h = vis[c.k].h
          /\ \A k \in Keys : k # c.k => cache'[k] = cache[k] /\ vis'[k] = vis[k] /\ cache[k] # w
      BY <1>1, <1>2
    <2>5. NoBad'
      BY NoBadFrame DEF IndInv
    <2>6. TrialVisibility'
      <3> SUFFICES ASSUME NEW k \in Keys
                   PROVE  /\ (cache'[k] # NONE /\ vis'[k].d # NONE) =>
                               /\ \A q \in Owners, r \in Positions :
                                    table'[q][r] = Shared(cache'[k]) => OwnerEpoch[q] = vis'[k].d
                               /\ \A y \in copies' : y.read = cache'[k] => GroupEpoch[y.g] = vis'[k].d
                          /\ (hcache[k] # NONE /\ vis'[k].h # NONE) =>
                               /\ \A y \in copies' : y.read = hcache[k] => GroupEpoch[y.g] = vis'[k].h
                               /\ \A t \in tasks : t.kind = "put" => t.h # hcache[k]
                          /\ \A y \in copies' :
                               (y.kind = "load" /\ y.fill /\ y.k = k) =>
                                   vis'[k].d = MadeBy(GroupEpoch[y.g])
                          /\ \A y \in copies' :
                               (y.kind = "store" /\ y.fill /\ y.k = k) =>
                                   vis'[k].h = MadeBy(GroupEpoch[y.g])
                          /\ \A t \in tasks :
                               (t.kind = "replicate" /\ cache'[k] \in t.units) =>
                                   vis'[k].d = MadeBy(OwnerEpoch[t.o])
        BY DEF TrialVisibility, MadeBy
      <3>1. CASE k = c.k
        BY <1>1, <2>0, <2>1, <2>2, <2>4, <3>1 DEF TrialVisibility
      <3>2. CASE k # c.k
        <4>1. cache[k] # NONE => cache[k] \in Units
          BY <1>0 DEF UnitTyped
        <4> QED
          BY <2>0, <2>3, <2>4, <3>2, <4>1 DEF TrialVisibility
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <2>5, <2>6
  <1>13. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A q \in Owners : \A x \in AllUnits : Covered(b, q, x) => Covered(b, q, x)'
      <3>1. \A q \in Owners, r \in Rows : \A x \in AllUnits : HoldsAt(q, r, x) => HoldsAt(q, r, x)'
        <4>1. Target(c) => table[o][p] = Empty
          BY DEF Target
        <4> QED
          BY <1>2, <4>1 DEF HoldsAt, Rows, Empty, AllUnits
      <3> QED
        BY <3>1 DEF Covered, Live, ReadersOf
    <2>4. \A y \in b.reads : y[1] \in Owners /\ y[2] \in AllUnits /\ y[2] # w
      BY <1>0, <1>1, <2>2 DEF BatchOk, RequestTyped
    <2>5. \A y \in b.writes, r \in Positions : table'[y[1]][r].u = y[2] => table[y[1]][r].u = y[2]
      BY <1>2, <2>2, <2>4 DEF BatchOk, Shared
    <2>6. /\ \A x \in b.cw : \E y \in copies' : y.write = x /\ y.g = b.g /\ y.seq = b.seq
          /\ \A x \in b.cr : \E y \in copies' : y.read = x /\ y.g = b.g /\ y.seq = b.seq
      BY <1>1, <2>2 DEF BatchOk
    <2> QED
      BY <1>1, <1>2, <2>2, <2>3, <2>4, <2>5, <2>6 DEF BatchOk, Mem
  (* The new block is no crossing's and no other copy's.                   *)
  <1>14. CrossingsOk'
    <2>1. \A d0 \in deferred' : d0.kind = "replicate" => \A u \in d0.units : alloc'[u] = alloc[u]
      <3>1. \A d0 \in deferred' : \A u \in d0.units : u \in AllUnits /\ u # w
        BY <1>0, <1>1 DEF WorkTyped
      <3> QED
        BY <1>2, <3>1
    <2>2. \A t \in tasks : t.kind = "replicate" =>
            \A u \in t.writes :
              alloc'[u] = alloc[u] /\ (Mem(t.to) \notin valid[u] => Mem(t.to) \notin valid'[u])
      <3>1. \A t \in tasks : t.kind = "replicate" => t.writes = t.units /\ t.units \subseteq Units
        BY DEF IndInv, CrossingsOk
      <3> QED
        BY <1>1, <1>2, <3>1 DEF AllUnits
    <2>3. \A k \in Keys : ~(cache'[k] = cache[k] /\ vis'[k].d = vis[k].d) =>
            \A d0 \in deferred' : d0.kind = "replicate" => cache'[k] \notin d0.units
      BY <1>1, <1>2
    <2> QED
      BY <2>1, <2>2, <2>3, EndCrossings DEF IndInv
  <1>15. CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A x \in copies' :
            /\ x.read \in Units \cup HostUnits => valid[x.read] \subseteq valid'[x.read]
            /\ (x.fill /\ x.write \in HostUnits) => valid'[x.write] = valid[x.write]
      BY <1>1, <1>2
    <2>2. \A t \in tasks : t.kind = "replicate" => \A u \in t.units : valid[u] \subseteq valid'[u]
      <3>1. \A t \in tasks : t.kind = "replicate" => t.units \subseteq Units
        BY DEF IndInv, CrossingsOk
      <3> QED
        BY <1>1, <1>2, <3>1
    <2> QED
      BY <2>1, <2>2, EndPlaced DEF IndInv
  <1> QED
    BY <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15 DEF IndInv

(* A store or a save whose key has no host block: its new host block      *)
(* becomes the key's host copy, and what it read is let go when nothing   *)
(* else holds it.                                                          *)
LEMMA FinishDue_Host ==
  ASSUME IndInv, NEW d \in deferred, d.kind = "finish", Reached(d.after),
         deferred' = deferred \ {d}, NEW c \in copies, c.id = d.c, copies' = copies \ {c},
         c.kind \in {"store", "save"}, ~c.fill, hcache[c.k] = NONE,
         hcache' = [hcache EXCEPT ![c.k] = c.write],
         valid' = [valid EXCEPT ![c.write] = GroupParts[c.g]],
         vis' = [vis EXCEPT ![c.k].h = MadeBy(GroupEpoch[c.g])],
         free' = free \cup ReclaimedWith({c.read}, table, oslot, cache, hcache', deferred', copies',
                                         tasks),
         alloc' = [Reset(alloc, ReclaimedWith({c.read}, table, oslot, cache, hcache', deferred',
                                              copies', tasks))
                     EXCEPT ![c.write] = "shared"],
         sharedKey' = [Reset(sharedKey, ReclaimedWith({c.read}, table, oslot, cache, hcache',
                                                      deferred', copies', tasks))
                         EXCEPT ![c.write] = c.k],
         UNCHANGED <<table, cache, dbt, handoff, ostate, ogroup, oslot, ncopy, tasks, trial,
                     member, launched, inflight, requests, remote, lost, bad>>
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1> DEFINE w == c.write
             G == ReclaimedWith({c.read}, table, oslot, cache, hcache', deferred', copies', tasks)
  (* The host block a store or a save writes is the copy's alone.          *)
  <1>1. /\ CopyOk(c) /\ c.g \in Groups /\ c.k \in Keys /\ GroupEpoch[c.g] \in Epochs
        /\ ~c.fill /\ w \in HostUnits /\ alloc[w] = "copy"
        /\ c.read \in AllUnits
        /\ \A q \in Owners, r \in Positions : table[q][r].u # w
        /\ \A q \in Owners : oslot[q] # w
        /\ \A k \in Keys : cache[k] # w /\ hcache[k] # w
        /\ \A e \in deferred : w \notin e.units
        /\ \A x \in copies : x # c => x.write # w /\ x.read # w
        /\ \A t \in tasks : w \notin t.units /\ t.slot # w /\ t.h # w
        /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
        /\ \A b \in inflight : \A x \in b.reads : x[2] # w
        /\ sharedKey[w] = NONE /\ w \notin free /\ c.read # w
    <2>1. /\ CopyOk(c) /\ c.g \in Groups /\ c.k \in Keys
          /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
          /\ \A x \in copies : x.id = c.id => x = c
          /\ c.read \in AllUnits /\ c.write \in AllUnits
      BY FinishDue_Facts
    <2>2. /\ ~c.fill /\ w \in HostUnits /\ alloc[w] = "copy"
          /\ \A q \in Owners, r \in Positions : table[q][r].u # w
          /\ \A x \in copies : x.write = w => x = c
      BY <2>1 DEF CopyOk
    <2>3. /\ \A q \in Owners : oslot[q] # w
          /\ \A k \in Keys : cache[k] # w /\ hcache[k] # w
          /\ sharedKey[w] = NONE
      <3>1. CacheShared /\ SharedIff
        BY DEF IndInv
      <3>2. sharedKey[w] = NONE
        BY <2>2, <3>1 DEF SharedIff, AllUnits
      <3> QED
        BY <1>0, <2>2, <3>1, <3>2 DEF OwnerTyped, UnitTyped, CacheShared, AllUnits
    <2>4. \A e \in deferred : w \notin e.units
      <3> SUFFICES ASSUME NEW e \in deferred, w \in e.units PROVE FALSE
        OBVIOUS
      <3>1. DeferOk(e)
        BY DeferredOkEach DEF IndInv
      <3>2. CrossWaitOk(e)
        BY CrossingsOkEach DEF IndInv
      <3> QED
        BY <2>1, <2>2, <3>1, <3>2 DEF DeferOk, CrossWaitOk
    <2>5. \A x \in copies : x # c => x.read # w
      <3> SUFFICES ASSUME NEW x \in copies, x # c, x.read = w PROVE FALSE
        OBVIOUS
      <3>1. CopyOk(x)
        BY CopiesOkEach DEF IndInv
      <3> QED
        BY <2>2, <3>1 DEF CopyOk
    <2>6. \A t \in tasks : w \notin t.units /\ t.slot # w /\ t.h # w
      <3> SUFFICES ASSUME NEW t \in tasks PROVE w \notin t.units /\ t.slot # w /\ t.h # w
        OBVIOUS
      <3>1. TaskOk(t)
        BY TasksOkEach DEF IndInv
      <3>2. t.kind = "put" => t.h = hcache[t.k] /\ t.k \in Keys
        BY <1>0, <3>1 DEF TaskOk, WorkTyped
      <3>3. CrossTaskOk(t)
        BY CrossingsOkEach DEF IndInv
      <3> QED
        BY <1>0, <2>2, <2>3, <3>1, <3>2, <3>3 DEF TaskOk, WorkTyped, CrossTaskOk
    <2>7. \A b \in inflight : \A x \in b.reads : x[2] # w
      <3> SUFFICES ASSUME NEW b \in inflight, NEW x \in b.reads, x[2] = w PROVE FALSE
        OBVIOUS
      <3>1. \/ \E q \in Positions : table[x[1]][q].u = x[2]
            \/ oslot[x[1]] = x[2]
            \/ \E e \in deferred : e.kind = "release" /\ x[2] \in e.units
        BY ReadHeld
      <3>2. x[1] \in Owners
        BY <1>0 DEF RequestTyped
      <3> QED
        BY <2>2, <2>3, <2>4, <3>1, <3>2
    <2>8. w \notin free
      <3>1. FreeClean
        BY DEF IndInv
      <3>2. Held(w)
        BY DEF Held, HeldIn
      <3> QED
        BY <3>1, <3>2 DEF FreeClean
    <2>9. c.read # w
      BY <2>1, <2>2 DEF CopyOk
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9
  <1>2. /\ \A x : Held(x)' = HeldIn(x, table, oslot, cache, hcache', deferred', copies', tasks)
        /\ G = {x \in {c.read} : ~Held(x)'}
        /\ G \subseteq AllUnits
    BY <1>1, ReclaimedHeld DEF Held, On, AllRules
  <1>3. /\ \A k \in Keys : hcache'[k] = IF k = c.k THEN w ELSE hcache[k]
        /\ \A v \in AllUnits : alloc'[v] = IF v = w THEN "shared" ELSE IF v \in G THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v = w THEN c.k ELSE IF v \in G THEN NONE ELSE sharedKey[v]
        /\ \A k \in Keys : vis'[k] = IF k = c.k THEN [d |-> vis[k].d, h |-> MadeBy(GroupEpoch[c.g])]
                                                ELSE vis[k]
        /\ \A u \in Units : valid'[u] = valid[u]
        /\ \A u \in HostUnits : valid'[u] = IF u = w THEN GroupParts[c.g] ELSE valid[u]
    BY <1>0, <1>1 DEF UnitTyped, PlaneTyped, Reset, AllUnits
  <1>4. /\ \A x \in AllUnits : Held(x)' => Held(x)
        /\ \A x \in AllUnits : Held(x) => Held(x)' \/ x \in G
        /\ \A x \in G : Held(x) /\ ~Held(x)' /\ x # w
        /\ Held(w)'
    <2>1. \A x \in AllUnits : x # w =>
            /\ (\E k \in Keys : cache[k] = x \/ hcache'[k] = x)
                 = (\E k \in Keys : cache[k] = x \/ hcache[k] = x)
      BY <1>1, <1>3 DEF AllUnits
    <2>2. \A x \in AllUnits : x # w =>
            HeldIn(x, table, oslot, cache, hcache', deferred', copies', tasks)
              => HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
      BY <2>1 DEF HeldIn
    <2>3. \A x \in AllUnits : x # w =>
            HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
              => HeldIn(x, table, oslot, cache, hcache', deferred', copies', tasks) \/ x = c.read
      <3>1. \A x \in AllUnits : (\E e \in deferred : x \in e.units) => \E e \in deferred' : x \in e.units
        <4>1. d.units = {}
          BY FinishDue_Facts
        <4> QED
          BY <4>1
      <3> QED
        BY <2>1, <3>1 DEF HeldIn
    <2>4. Held(w)' /\ Held(w) /\ Held(c.read)
      BY <1>1, <1>2, <1>3 DEF Held, HeldIn
    <2> QED
      BY <1>1, <1>2, <2>2, <2>3, <2>4 DEF Held
  <1>5. TypeInv'
    <2>1. UnitTyped'
      <3>1. GroupParts[c.g] \subseteq HostParts
        BY <1>1
      <3> QED
        BY <1>0, <1>1, <1>2, <1>3, <3>1 DEF UnitTyped, AllUnits, Reset
    <2>2. PlaneTyped'
      <3>1. MadeBy(GroupEpoch[c.g]) \in Epochs \cup {NONE}
        BY <1>1 DEF MadeBy
      <3> QED
        BY <1>0, <1>3, <3>1 DEF PlaneTyped
    <2>3. WorkTyped'
      BY <1>0 DEF WorkTyped
    <2> QED
      BY <2>1, <2>2, <2>3, OwnerTypes, RequestTypes, OwnerTypedFrame, RequestTypedFrame,
         TypeInvNext
  <1>6. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared'
    <2>0. /\ FreeClean /\ NoLeak /\ Outside /\ Regions /\ AllocOwner /\ SharedIff
          /\ SharedCached /\ CacheShared
      BY DEF IndInv
    <2>1. FreeClean'
      BY <1>0, <1>1, <1>2, <1>3, <1>4, <2>0 DEF FreeClean, UnitTyped, AllUnits
    <2>2. NoLeak'
      BY <1>2, <1>4, <2>0 DEF NoLeak, InRegion, AllUnits
    <2>3. Outside' /\ Regions'
      <3>1. \A x \in G : x \in Units \cup HostUnits => x \in member
        BY <1>4, <2>0 DEF NoLeak, InRegion, AllUnits
      (* The host block the copy writes is allocated, so it is in its region. *)
      <3>2. w \in member
        BY <1>1, <2>0 DEF Outside, NONE
      <3>3. Outside'
        <4> SUFFICES ASSUME NEW v \in (Units \cup HostUnits) \ member'
                     PROVE  alloc'[v] = NONE /\ sharedKey'[v] = NONE
          BY DEF Outside
        <4>1. v # w /\ v \in AllUnits /\ v \in (Units \cup HostUnits) \ member
          BY <3>2 DEF AllUnits
        <4> QED
          BY <1>3, <2>0, <4>1 DEF Outside
      <3>4. Regions'
        BY <1>2, <2>0, <3>1 DEF Regions, OneRegion, AllUnits
      <3> QED
        BY <3>3, <3>4
    <2>4. AllocOwner' /\ SharedIff'
      BY <1>1, <1>3, <2>0 DEF AllocOwner, SharedIff, NONE, AllUnits
    <2>5. SharedCached'
      <3>1. \A v \in AllUnits : alloc[v] = "shared" => sharedKey[v] \in Keys
        BY <1>0, <2>0 DEF SharedIff, UnitTyped
      <3> QED
        BY <1>1, <1>3, <2>0, <3>1 DEF SharedCached, NONE, AllUnits
    <2>6. CacheShared'
      <3>1. \A k \in Keys : \A x \in AllUnits : (cache[k] = x \/ hcache'[k] = x) => Held(x)'
        BY <1>2 DEF HeldIn
      <3> QED
        BY <1>0, <1>1, <1>3, <1>4, <2>0, <3>1 DEF CacheShared, UnitTyped, AllUnits
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>6
  (* What stays held keeps its allocation.                                 *)
  <1>7. \A x \in AllUnits : Held(x)' /\ x # w => alloc'[x] = alloc[x] /\ sharedKey'[x] = sharedKey[x]
    BY <1>2, <1>3
  <1>8. /\ \A q \in Owners, r \in Positions : \A x \in AllUnits : table[q][r].u = x => Held(x)'
        /\ \A q \in Owners : \A x \in AllUnits : oslot[q] = x => Held(x)'
        /\ \A e \in deferred' : \A x \in e.units : Held(x)'
        /\ \A y \in copies' : Held(y.read)' /\ Held(y.write)'
        /\ \A t \in tasks : t.held => \A x \in t.units \cup {t.slot, t.h} : Held(x)'
    BY <1>2 DEF HeldIn
  <1>9. Tables' /\ OwnersOk'
    <2>1. \A q \in Owners, r \in Positions :
            table[q][r].u # NONE => alloc'[table[q][r].u] = alloc[table[q][r].u]
      BY <1>0, <1>1, <1>7, <1>8 DEF OwnerTyped, Empty, Private, Shared, AllUnits
    <2>2. \A q \in Owners : oslot[q] # NONE => alloc'[oslot[q]] = alloc[oslot[q]]
      BY <1>0, <1>1, <1>7, <1>8 DEF OwnerTyped, AllUnits
    <2>3. Tables'
      <3>1. \A q \in Owners, r \in Positions :
              table[q][r].u # NONE =>
                /\ alloc'[table[q][r].u] = alloc[table[q][r].u]
                /\ valid[table[q][r].u] \subseteq valid'[table[q][r].u]
        BY <1>0, <1>3, <2>1, TableTypes
      <3> QED
        BY <3>1, TablesKeep DEF IndInv
    <2>4. OwnersOk'
      <3>1. OwnersOk
        BY DEF IndInv
      <3> QED
        BY <2>2, <3>1 DEF OwnersOk, Live
    <2> QED
      BY <2>3, <2>4
  <1>10. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred' PROVE DeferOk(e)'
      BY DeferredOkEach
    <2>2. e \in deferred /\ e # d /\ DeferOk(e) /\ DeferOk(d)
      BY DeferredOkEach DEF IndInv
    <2>3. e.kind = "commit" => \A v \in e.units : alloc'[v] = alloc[v]
      BY <1>0, <1>1, <1>7, <1>8, <2>2 DEF WorkTyped
    <2>4. e.kind = "finish" => \A x \in copies : x.id = e.c => x \in copies'
      BY <2>2 DEF DeferOk
    <2> QED
      BY <2>2, <2>3, <2>4 DEF DeferOk
  <1>11. CopiesOk'
    <2>1. SUFFICES ASSUME NEW x \in copies' PROVE CopyOk(x)'
      BY CopiesOkEach
    <2>2. x \in copies /\ x # c /\ CopyOk(x) /\ x.read \in AllUnits /\ x.write \in AllUnits
      BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. alloc'[x.read] = alloc[x.read] /\ alloc'[x.write] = alloc[x.write]
      BY <1>1, <1>7, <1>8, <2>2
    <2>4. \A q \in Owners : Moving(q)' => Moving(q)
      BY DEF Moving
    (* A fill writes a key's host block, so its key is not the one stored.  *)
    <2>5. (x.fill /\ x.kind = "store") => x.k # c.k
      BY <2>2 DEF CopyOk
    <2>6. (x.fill /\ x.kind = "load") => x.write \in Units /\ valid'[x.write] = valid[x.write]
      <3>1. (x.fill /\ x.kind = "load") => x.write \in Units
        BY <2>2, CopyFacts
      <3> QED
        BY <1>3, <3>1
    <2>7. (x.fill /\ x.kind = "store") => hcache'[x.k] = hcache[x.k]
      <3>1. x.k \in Keys
        BY <2>2, CopyFacts
      <3> QED
        BY <1>3, <2>5, <3>1
    <2> QED
      BY <1>3, <2>2, <2>3, <2>4, <2>6, <2>7 DEF CopyOk, Live, Mem
  <1>12. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. TaskOk(t)
      BY TasksOkEach DEF IndInv
    <2>3. \A u \in t.units \cup {t.slot, t.h} : u # NONE => alloc'[u] = alloc[u]
      BY <1>0, <1>1, <1>7, <1>8, <2>2 DEF TaskOk, WorkTyped, AllUnits
    <2>4. t.kind = "put" => t.k # c.k /\ t.k \in Keys
      BY <1>0, <1>1, <2>2 DEF TaskOk, WorkTyped
    <2>5. /\ t.kind = "transfer" => t.writes \subseteq t.units /\ t.slot # NONE
                                    /\ t.writes \subseteq Units
          /\ t.kind = "prefetch" => t.h # NONE
      BY <2>2 DEF TaskOk
    <2>6. \A q \in Owners : TableUnits(q) \subseteq Units
      BY <1>0, TableTypes
    <2> QED
      BY <1>3, <2>2, <2>3, <2>4, <2>5, <2>6 DEF TaskOk, Live, Mem, TableUnits
  <1>13. /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk' /\ NoBad'
    BY <1>3, EndPlane DEF IndInv
  <1>14. TrialVisibility'
    <2>0. TrialVisibility
      BY DEF IndInv
    <2>1. \A k \in Keys : k # c.k => hcache'[k] = hcache[k] /\ vis'[k] = vis[k]
      BY <1>3
    <2>2. /\ hcache'[c.k] = w /\ vis'[c.k].d = vis[c.k].d
          /\ \A y \in copies' : y.read # w
          /\ \A t \in tasks : t.h # w
      BY <1>1, <1>3
    <2> SUFFICES ASSUME NEW k \in Keys
                 PROVE  /\ (cache[k] # NONE /\ vis'[k].d # NONE) =>
                             /\ \A q \in Owners, r \in Positions :
                                  table[q][r] = Shared(cache[k]) => OwnerEpoch[q] = vis'[k].d
                             /\ \A y \in copies' : y.read = cache[k] => GroupEpoch[y.g] = vis'[k].d
                        /\ (hcache'[k] # NONE /\ vis'[k].h # NONE) =>
                             /\ \A y \in copies' : y.read = hcache'[k] => GroupEpoch[y.g] = vis'[k].h
                             /\ \A t \in tasks : t.kind = "put" => t.h # hcache'[k]
                        /\ \A y \in copies' :
                             (y.kind = "load" /\ y.fill /\ y.k = k) =>
                                 vis'[k].d = MadeBy(GroupEpoch[y.g])
                        /\ \A y \in copies' :
                             (y.kind = "store" /\ y.fill /\ y.k = k) =>
                                 vis'[k].h = MadeBy(GroupEpoch[y.g])
                        /\ \A t \in tasks :
                             (t.kind = "replicate" /\ cache[k] \in t.units) =>
                                 vis'[k].d = MadeBy(OwnerEpoch[t.o])
      BY DEF TrialVisibility, MadeBy
    (* No fill writes a host block of the key that had none.               *)
    <2>5. \A y \in copies : (y.kind = "store" /\ y.fill) => y.k # c.k
      <3> SUFFICES ASSUME NEW y \in copies, y.kind = "store", y.fill PROVE y.k # c.k
        OBVIOUS
      <3>1. CopyOk(y) /\ y.write \in AllUnits
        BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
      <3> QED
        BY <3>1 DEF CopyOk, AllUnits
    <2>3. CASE k = c.k
      BY <2>0, <2>2, <2>3, <2>5 DEF TrialVisibility
    <2>4. CASE k # c.k
      BY <2>0, <2>1, <2>4 DEF TrialVisibility
    <2> QED
      BY <2>3, <2>4
  <1>15. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A q \in Owners : \A x \in AllUnits : Covered(b, q, x) => Covered(b, q, x)' /\ Held(x)'
      <3>1. \A q \in Owners : \A x \in AllUnits : Covered(b, q, x) => Covered(b, q, x)'
        BY DEF Covered, Live, HoldsAt, ReadersOf
      <3>2. \A q \in Owners : \A x \in AllUnits : Covered(b, q, x) => Held(x)'
        BY <1>8 DEF Covered, HoldsAt, Rows
      <3> QED
        BY <3>1, <3>2
    <2>4. \A y \in b.reads : y[1] \in Owners /\ y[2] \in AllUnits /\ y[2] # w /\ y[2] \notin free
      BY <1>0, <1>1, <2>2 DEF BatchOk, RequestTyped
    <2>5. \A y \in b.reads : Covered(b, y[1], y[2])' /\ y[2] \notin free' /\ alloc'[y[2]] = alloc[y[2]]
      BY <1>2, <1>7, <2>2, <2>3, <2>4 DEF BatchOk
    <2>6. /\ \A x \in b.cw : \E y \in copies' : y.write = x /\ y.g = b.g /\ y.seq = b.seq
          /\ \A x \in b.cr : \E y \in copies' : y.read = x /\ y.g = b.g /\ y.seq = b.seq
      BY <1>1, <2>2 DEF BatchOk
    <2>7. \A y \in b.reads : y[2] \in Units => valid'[y[2]] = valid[y[2]]
      BY <1>3
    <2> QED
      BY <2>2, <2>5, <2>6, <2>7 DEF BatchOk, Mem
  (* What a crossing holds stays held with its allocation; device validity  *)
  (* and the device cache stay.                                           *)
  <1>16. CrossingsOk'
    <2>1. \A d0 \in deferred' : d0.kind = "replicate" => \A u \in d0.units : alloc'[u] = alloc[u]
      <3>1. \A d0 \in deferred' : \A u \in d0.units : u \in AllUnits /\ u # w
        BY <1>0, <1>1 DEF WorkTyped
      <3> QED
        BY <1>7, <1>8, <3>1
    <2>2. \A t \in tasks : t.kind = "replicate" =>
            \A u \in t.writes :
              alloc'[u] = alloc[u] /\ (Mem(t.to) \notin valid[u] => Mem(t.to) \notin valid'[u])
      <3>1. \A t \in tasks : t.kind = "replicate" =>
              t.held /\ t.writes = t.units /\ t.units \subseteq Units
        BY DEF IndInv, CrossingsOk, TasksOk
      <3> QED
        BY <1>1, <1>3, <1>7, <1>8, <3>1 DEF AllUnits
    <2>3. \A k \in Keys : cache'[k] = cache[k] /\ vis'[k].d = vis[k].d
      BY <1>3
    <2> QED
      BY <2>1, <2>2, <2>3, EndCrossings DEF IndInv
  <1>17. CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A x \in copies' :
            /\ x.read \in Units \cup HostUnits => valid[x.read] \subseteq valid'[x.read]
            /\ (x.fill /\ x.write \in HostUnits) => valid'[x.write] = valid[x.write]
      BY <1>1, <1>3
    <2>2. \A t \in tasks : t.kind = "replicate" => \A u \in t.units : valid[u] \subseteq valid'[u]
      <3>1. \A t \in tasks : t.kind = "replicate" => t.units \subseteq Units
        BY DEF IndInv, CrossingsOk
      <3> QED
        BY <1>3, <3>1
    <2> QED
      BY <2>1, <2>2, EndPlaced DEF IndInv
  <1> QED
    BY <1>5, <1>6, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17 DEF IndInv

(* A store that filled host parts of its key's host block: the block is   *)
(* valid on those parts now. Nothing is let go: the device cache holds     *)
(* what the copy read and the host cache what it wrote.                    *)
LEMMA FinishDue_HostFill ==
  ASSUME IndInv, NEW d \in deferred, d.kind = "finish", Reached(d.after),
         deferred' = deferred \ {d}, NEW c \in copies, c.id = d.c, copies' = copies \ {c},
         c.kind = "store", c.fill,
         valid' = [valid EXCEPT ![c.write] = @ \cup c.parts],
         DropWith({c.read}, table, oslot, cache, hcache, deferred', copies', tasks),
         UNCHANGED <<table, cache, hcache, dbt, handoff, vis, ostate, ogroup, oslot, ncopy, tasks,
                     trial, member, launched, inflight, requests, remote, lost, bad>>
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1> DEFINE w == c.write
  <1>1. /\ CopyOk(c) /\ c.g \in Groups /\ c.k \in Keys /\ c.parts \subseteq HostParts
        /\ w \in HostUnits /\ hcache[c.k] = w
        /\ \A x \in copies : x.write = w => x = c
        /\ alloc[c.read] = "shared" /\ c.read \in AllUnits /\ d.units = {}
        /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
        /\ \A x \in copies : x.id = c.id => x = c
    <2>1. /\ CopyOk(c) /\ c.read \in AllUnits /\ c.k \in Keys /\ c.g \in Groups /\ d.units = {}
          /\ \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq)
          /\ \A x \in copies : x.id = c.id => x = c
      BY FinishDue_Facts
    <2>2. c.parts \subseteq HostParts
      BY <1>0 DEF WorkTyped
    <2> QED
      BY <2>1, <2>2 DEF CopyOk
  (* What the copy read is cached.                                         *)
  <1>2. \E k \in Keys : cache[k] = c.read \/ hcache[k] = c.read
    <2>1. SharedIff /\ SharedCached
      BY DEF IndInv
    <2>2. sharedKey[c.read] \in Keys
      BY <1>0, <1>1, <2>1 DEF SharedIff, UnitTyped
    <2> QED
      BY <1>1, <2>1, <2>2 DEF SharedCached
  <1>3. /\ \A u \in Units \cup HostUnits : valid'[u] = IF u = w THEN valid[w] \cup c.parts ELSE valid[u]
        /\ \A u \in Units \cup HostUnits : valid[u] \subseteq valid'[u]
        /\ \A u \in Units : valid'[u] = valid[u]
    BY <1>0, <1>1 DEF UnitTyped
  <1>4. \A x \in AllUnits : Held(x)' = Held(x)
    <2>1. \A x \in AllUnits : (x # w /\ x # c.read) =>
            /\ (\E e \in deferred' : x \in e.units) <=> (\E e \in deferred : x \in e.units)
            /\ (\E y \in copies' : x = y.read \/ x = y.write)
                 <=> (\E y \in copies : x = y.read \/ x = y.write)
      BY <1>1
    <2>2. /\ HeldIn(w, table, oslot, cache, hcache, deferred', copies', tasks)
          /\ HeldIn(w, table, oslot, cache, hcache, deferred, copies, tasks)
          /\ HeldIn(c.read, table, oslot, cache, hcache, deferred', copies', tasks)
          /\ HeldIn(c.read, table, oslot, cache, hcache, deferred, copies, tasks)
      BY <1>1, <1>2 DEF HeldIn
    <2>3. \A x \in AllUnits :
            HeldIn(x, table, oslot, cache, hcache, deferred', copies', tasks)
              <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
      <3> SUFFICES ASSUME NEW x \in AllUnits
                   PROVE  HeldIn(x, table, oslot, cache, hcache, deferred', copies', tasks)
                            <=> HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
        OBVIOUS
      <3>1. CASE x = w \/ x = c.read
        BY <2>2, <3>1
      <3>2. CASE x # w /\ x # c.read
        BY <2>1, <3>2 DEF HeldIn
      <3> QED
        BY <3>1, <3>2
    <2>4. \A x : Held(x)' = HeldIn(x, table, oslot, cache, hcache, deferred', copies', tasks)
      BY DEF Held
    <2> QED
      BY <2>3, <2>4 DEF Held, HeldIn
  <1>5. UNCHANGED <<free, alloc, sharedKey>>
    <2>1. ReclaimedWith({c.read}, table, oslot, cache, hcache, deferred', copies', tasks) = {}
      BY <1>2, ReclaimedHeld DEF HeldIn, On, AllRules
    <2> QED
      BY <1>0, <2>1 DEF DropWith, Reset, UnitTyped
  <1>6. \A q \in Owners : Moving(q)' => Moving(q)
    BY DEF Moving
  <1>7. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>3, <1>5 DEF UnitTyped
    <2>2. WorkTyped'
      BY <1>0 DEF WorkTyped
    <2>3. OwnerTyped' /\ PlaneTyped' /\ RequestTyped'
      BY <1>0, OwnerTypedFrame, PlaneTypedFrame, RequestTypedFrame
    <2> QED
      BY <2>1, <2>2, <2>3, TypeInvNext
  <1>8. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ OwnersOk'
    <2>1. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
          /\ SharedCached' /\ CacheShared'
      BY <1>0, <1>4, <1>5, EndUnits DEF IndInv
    <2>2. OwnersOk'
      BY <1>5, OwnersOkFrame DEF IndInv
    <2> QED
      BY <2>1, <2>2
  <1>9. Tables'
    <2>1. \A q \in Owners, r \in Positions :
            table[q][r].u # NONE =>
                /\ alloc'[table[q][r].u] = alloc[table[q][r].u]
                /\ valid[table[q][r].u] \subseteq valid'[table[q][r].u]
      <3>1. \A q \in Owners, r \in Positions : table[q][r].u # NONE => table[q][r].u \in Units
        BY <1>0, TableTypes
      <3> QED
        BY <1>3, <1>5, <3>1
    <2> QED
      BY <2>1, TablesKeep DEF IndInv
  <1>10. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred' PROVE DeferOk(e)'
      BY DeferredOkEach
    <2>2. e \in deferred /\ e # d /\ DeferOk(e) /\ DeferOk(d)
      BY DeferredOkEach DEF IndInv
    <2>3. e.kind = "finish" => \A x \in copies : x.id = e.c => x \in copies'
      BY <2>2 DEF DeferOk
    <2> QED
      BY <1>5, <2>2, <2>3 DEF DeferOk
  <1>11. CopiesOk'
    <2>1. SUFFICES ASSUME NEW x \in copies' PROVE CopyOk(x)'
      BY CopiesOkEach
    <2>2. /\ x \in copies /\ x # c /\ CopyOk(x)
          /\ (x.fill /\ x.kind = "load") => x.write \in Units
      BY CopyFacts
    <2>3. (x.fill /\ x.kind = "load") => valid'[x.write] = valid[x.write]
      BY <1>3, <2>2
    <2> QED
      BY <1>5, <1>6, <2>2, <2>3 DEF CopyOk, Live, Mem
  <1>12. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. TaskOk(t)
      BY TasksOkEach DEF IndInv
    <2>3. t.kind = "transfer" => t.writes \subseteq Units
      BY <2>2 DEF TaskOk
    <2>4. \A q \in Owners : TableUnits(q) \subseteq Units
      BY <1>0, TableTypes
    <2> QED
      BY <1>3, <1>5, <2>2, <2>3, <2>4 DEF TaskOk, Live, Mem, TableUnits
  <1>13. /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
         /\ TrialVisibility' /\ NoBad'
    BY <1>3, EndPlane, EndVisible DEF IndInv
  <1>14. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A q \in Owners : \A x : Covered(b, q, x) => Covered(b, q, x)'
      BY DEF Covered, Live, HoldsAt, ReadersOf
    <2>4. \A y \in b.reads : y[2] \in Units => valid'[y[2]] = valid[y[2]]
      BY <1>3
    <2>5. /\ \A x \in b.cw : \E y \in copies' : y.write = x /\ y.g = b.g /\ y.seq = b.seq
          /\ \A x \in b.cr : \E y \in copies' : y.read = x /\ y.g = b.g /\ y.seq = b.seq
      BY <1>1, <2>2 DEF BatchOk
    <2> QED
      BY <1>5, <2>2, <2>3, <2>4, <2>5 DEF BatchOk, Mem
  <1>15. CrossingsOk'
    <2>1. \A t \in tasks : t.kind = "replicate" =>
            \A u \in t.writes :
              alloc'[u] = alloc[u] /\ (Mem(t.to) \notin valid[u] => Mem(t.to) \notin valid'[u])
      <3>1. \A t \in tasks : t.kind = "replicate" => t.writes \subseteq Units
        BY DEF IndInv, CrossingsOk
      <3> QED
        BY <1>3, <1>5, <3>1
    <2> QED
      BY <1>5, <2>1, EndCrossings DEF IndInv
  <1>16. CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A x \in copies' :
            /\ x.read \in Units \cup HostUnits => valid[x.read] \subseteq valid'[x.read]
            /\ (x.fill /\ x.write \in HostUnits) => valid'[x.write] = valid[x.write]
      BY <1>1, <1>3
    <2>2. \A t \in tasks : t.kind = "replicate" => \A u \in t.units : valid[u] \subseteq valid'[u]
      <3>1. \A t \in tasks : t.kind = "replicate" => t.units \subseteq Units
        BY DEF IndInv, CrossingsOk
      <3> QED
        BY <1>3, <3>1
    <2> QED
      BY <2>1, <2>2, EndPlaced DEF IndInv
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16 DEF IndInv

LEMMA FinishDue_Inv ==
  ASSUME IndInv, NEW d \in deferred, FinishDue(d), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ d.kind = "finish" /\ Reached(d.after) /\ deferred' = deferred \ {d}
        /\ UNCHANGED <<ostate, ogroup, oslot, ncopy, tasks, trial, member, launched, inflight,
                       requests, remote, lost, bad>>
    BY FinishDue_Effect DEF rare
  <1>2. PICK c \in copies : EndStep(d, c)
    BY FinishDue_Effect
  <1>3. c.id = d.c /\ copies' = copies \ {c}
    BY <1>2 DEF EndStep
  <1>4. /\ c.kind \in {"clone", "load", "store", "save", "restore"}
        /\ c.kind \in {"clone", "save"} => ~c.fill
    BY <1>1, <1>3, FinishDue_Facts DEF CopyOk
  <1>5. CASE ~d.ok
    <2>1. /\ DropWith({c.read, c.write}, table, oslot, cache, hcache, deferred', copies', tasks)
          /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff, vis>>
      BY <1>2, <1>5 DEF EndStep, Drop
    <2> QED
      BY <1>1, <1>3, <2>1, FinishDue_Drop
  <1>6. CASE d.ok /\ c.kind = "clone"
    <2>1. vis' = vis
      BY <1>2, <1>6 DEF EndStep
    <2>2. CASE Target(c)
      <3>1. /\ table' = [table EXCEPT ![c.o][c.p] = Private(c.write)]
            /\ valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}]
            /\ Give(Entry(c, TRUE))
            /\ DropWith({c.read}, table', oslot, cache, hcache, deferred', copies', tasks)
            /\ UNCHANGED <<cache, hcache>>
        BY <1>2, <1>6, <2>2 DEF EndStep, Target, Entry, Drop
      <3> QED
        BY <1>1, <1>3, <1>4, <1>6, <2>1, <2>2, <3>1, FinishDue_Private
    <2>3. CASE ~Target(c)
      <3>1. /\ DropWith({c.read, c.write}, table, oslot, cache, hcache, deferred', copies', tasks)
            /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff>>
        BY <1>2, <1>6, <2>3 DEF EndStep, Target, Drop
      <3> QED
        BY <1>1, <1>3, <2>1, <3>1, FinishDue_Drop
    <2> QED
      BY <2>2, <2>3
  <1>7. CASE d.ok /\ c.kind = "load" /\ c.fill
    <2>1. /\ vis' = vis
          /\ valid' = [valid EXCEPT ![c.write] = @ \cup {Mem(c.g)}]
          /\ \/ /\ Target(c) /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
                /\ Give(Entry(c, FALSE))
             \/ /\ ~Target(c) /\ UNCHANGED <<table, dbt, handoff>>
          /\ UNCHANGED <<free, cache, hcache, alloc, sharedKey>>
      BY <1>2, <1>7 DEF EndStep, Target, Entry
    <2> QED
      BY <1>1, <1>3, <1>7, <2>1, FinishDue_Fill
  <1>8. CASE d.ok /\ c.kind = "load" /\ ~c.fill
    <2>1. CASE cache[c.k] # NONE
      <3>1. /\ vis' = vis
            /\ valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}]
            /\ UNCHANGED <<cache, hcache>>
        BY <1>2, <1>8, <2>1 DEF EndStep, On, AllRules
      <3>2. CASE Target(c)
        <4>1. /\ table' = [table EXCEPT ![c.o][c.p] = Private(c.write)]
              /\ Give(Entry(c, FALSE))
              /\ DropWith({c.read}, table', oslot, cache, hcache, deferred', copies', tasks)
          BY <1>2, <1>8, <2>1, <3>2 DEF EndStep, Target, Entry, Drop, On, AllRules
        <4> QED
          BY <1>1, <1>3, <1>8, <3>1, <3>2, <4>1, FinishDue_Private
      <3>3. CASE ~Target(c)
        <4>1. /\ DropWith({c.read, c.write}, table, oslot, cache, hcache, deferred', copies', tasks)
              /\ UNCHANGED <<table, dbt, handoff>>
          BY <1>2, <1>8, <2>1, <3>3 DEF EndStep, Target, Drop, On, AllRules
        <4> QED
          BY <1>1, <1>3, <1>8, <3>1, <4>1, FinishDue_Drop
      <3> QED
        BY <3>2, <3>3
    <2>2. CASE cache[c.k] = NONE
      <3>1. /\ cache' = [cache EXCEPT ![c.k] = c.write]
            /\ valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}]
            /\ vis' = [vis EXCEPT ![c.k].d = MadeBy(GroupEpoch[c.g])]
            /\ \/ /\ Target(c) /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
                  /\ Give(Entry(c, FALSE))
               \/ /\ ~Target(c) /\ UNCHANGED <<table, dbt, handoff>>
            /\ free' = free
            /\ alloc' = [alloc EXCEPT ![c.write] = "shared"]
            /\ sharedKey' = [sharedKey EXCEPT ![c.write] = c.k]
            /\ UNCHANGED hcache
        BY <1>2, <1>8, <2>2 DEF EndStep, Target, Entry, On, AllRules
      <3> QED
        BY <1>1, <1>3, <1>8, <2>2, <3>1, FinishDue_Cache
    <2> QED
      BY <2>1, <2>2
  <1>9. CASE d.ok /\ c.kind = "restore"
    <2>1. /\ DropWith({c.read, c.write}, table, oslot, cache, hcache, deferred', copies', tasks)
          /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff, vis>>
      BY <1>2, <1>9 DEF EndStep, Drop
    <2> QED
      BY <1>1, <1>3, <2>1, FinishDue_Drop
  <1>11. CASE d.ok /\ c.kind = "store" /\ c.fill
    <2>0. hcache[c.k] # NONE
      BY <1>1, <1>3, <1>11, FinishDue_Facts DEF CopyOk
    <2>1. /\ valid' = [valid EXCEPT ![c.write] = @ \cup c.parts]
          /\ DropWith({c.read}, table, oslot, cache, hcache, deferred', copies', tasks)
          /\ UNCHANGED <<table, cache, hcache, dbt, handoff, vis>>
      BY <1>2, <1>11, <2>0 DEF EndStep, Drop, On, AllRules
    <2> QED
      BY <1>1, <1>3, <1>11, <2>1, FinishDue_HostFill
  <1>10. CASE d.ok /\ c.kind \in {"store", "save"} /\ ~c.fill
    <2>1. CASE hcache[c.k] # NONE
      <3>1. /\ DropWith({c.read, c.write}, table, oslot, cache, hcache, deferred', copies', tasks)
            /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff, vis>>
        BY <1>2, <1>10, <2>1 DEF EndStep, Drop, On, AllRules
      <3> QED
        BY <1>1, <1>3, <3>1, FinishDue_Drop
    <2>2. CASE hcache[c.k] = NONE
      <3>1. /\ hcache' = [hcache EXCEPT ![c.k] = c.write]
            /\ valid' = [valid EXCEPT ![c.write] = GroupParts[c.g]]
            /\ vis' = [vis EXCEPT ![c.k].h = MadeBy(GroupEpoch[c.g])]
            /\ free' = free \cup ReclaimedWith({c.read}, table, oslot, cache, hcache', deferred',
                                               copies', tasks)
            /\ alloc' = [Reset(alloc, ReclaimedWith({c.read}, table, oslot, cache, hcache', deferred',
                                                    copies', tasks))
                           EXCEPT ![c.write] = "shared"]
            /\ sharedKey' = [Reset(sharedKey, ReclaimedWith({c.read}, table, oslot, cache, hcache',
                                                            deferred', copies', tasks))
                               EXCEPT ![c.write] = c.k]
            /\ UNCHANGED <<table, cache, dbt, handoff>>
        BY <1>2, <1>10, <2>2 DEF EndStep, On, AllRules
      <3> QED
        BY <1>1, <1>3, <1>10, <2>2, <3>1, FinishDue_Host
    <2> QED
      BY <2>1, <2>2
  <1> QED
    BY <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11

-----------------------------------------------------------------------------

THEOREM Step7 ==
    IndInv /\ (/\ \E d \in deferred : ReleaseDue(d) \/ CommitDue(d) \/ FinishDue(d)
               /\ UNCHANGED rare)
    => IndInv'
  BY ReleaseDue_Inv, CommitDue_Inv, FinishDue_Inv
=============================================================================
