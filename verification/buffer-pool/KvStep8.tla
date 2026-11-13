------------------------------- MODULE KvStep8 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for transfers: TransferDue and             *)
(* TransferEnd.                                                            *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, KvFrame, TLAPS

USE Sizes, Design

-----------------------------------------------------------------------------
(* TransferDue                                                             *)

(* The free slot rows of the target group, and the task that starts.       *)
FreeRows(d) == {s \in free \cap Slots : SlotGroup[s] = d.to}
NewTask(d, s) ==
    [kind |-> "transfer", k |-> NONE, h |-> NONE, src |-> NONE, ok |-> TRUE, expired |-> FALSE,
     o |-> d.o, to |-> d.to, slot |-> s, held |-> TRUE,
     units |-> TableUnits(d.o) \cup {oslot[d.o]},
     writes |-> {u \in TableUnits(d.o) : Mem(d.to) \notin valid[u]}]

LEMMA TransferDue_Effect ==
  ASSUME NEW d \in deferred, TransferDue(d)
  PROVE  /\ d.kind = "transfer" /\ Reached(d.after)
         /\ ~\E c \in copies : c.fill /\ Mem(c.g) = Mem(d.to) /\ c.write \in TableUnits(d.o)
         /\ ~\E t \in tasks :
              t.kind = "transfer" /\ Mem(t.to) = Mem(d.to) /\ t.writes \cap TableUnits(d.o) # {}
         /\ deferred' = deferred \ {d}
         /\ \/ /\ ostate[d.o] # "live" \/ FreeRows(d) = {}
               /\ UNCHANGED <<free, tasks, alloc>>
            \/ /\ ostate[d.o] = "live"
               /\ \E s \in FreeRows(d) :
                    /\ free' = free \ {s}
                    /\ alloc' = [alloc EXCEPT ![s] = d.o]
                    /\ tasks' = tasks \cup {NewTask(d, s)}
         /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, valid, vis,
                        trial, member, dbt, launched, inflight, handoff, requests, sharedKey>>
  BY DEF TransferDue, TableUnits, FreeRows, NewTask, plane, On, AllRules

(* What IndInv says of a transfer that comes due.                          *)
LEMMA TransferDue_Facts ==
  ASSUME IndInv, NEW d \in deferred, TransferDue(d)
  PROVE  /\ d.o \in Owners /\ d.to \in Groups /\ GroupEpoch[d.to] = OwnerEpoch[d.o]
         /\ ostate[d.o] # "new" /\ d.units = {}
         /\ \A g \in Groups, p \in Rows : dbt[g][d.o][p].u = NONE
         /\ \A h \in handoff : h.o # d.o
         /\ \A x \in deferred : (x.kind = "transfer" /\ x.o = d.o) => x = d
         /\ \A t \in tasks : t.kind = "transfer" => t.o # d.o
         /\ Live(d.o) => \A c \in copies : c.o # d.o
         /\ \A b \in inflight : d.after[b.g] < b.seq
  <1>1. d.kind = "transfer" /\ Reached(d.after)
    BY TransferDue_Effect
  <1>2. /\ d.o \in Owners /\ d.to \in Groups /\ GroupEpoch[d.to] = OwnerEpoch[d.o]
        /\ ostate[d.o] # "new" /\ d.units = {}
    BY <1>1 DEF IndInv, DeferredOk
  <1>3. /\ \A g \in Groups, p \in Rows : dbt[g][d.o][p].u = NONE
        /\ \A h \in handoff : h.o # d.o
        /\ \A x \in deferred : (x.kind = "transfer" /\ x.o = d.o) => x = d
    BY <1>1, <1>2 DEF IndInv, MovingUnmapped
  <1>4. \A t \in tasks : t.kind = "transfer" => t.o # d.o
    BY <1>1 DEF IndInv, TasksOk
  <1>5. Live(d.o) => \A c \in copies : c.o # d.o
    BY <1>1, <1>2 DEF IndInv, CopiesOk, Moving
  <1>6. \A b \in inflight : d.after[b.g] < b.seq
    BY <1>1 DEF Reached, IndInv, BatchesOk
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5, <1>6

(* Removing the transfer that comes due changes no unit's holders, and no  *)
(* owner but the transfer's own starts to move.                            *)
LEMMA TransferDue_Deferred ==
  ASSUME IndInv, NEW d \in deferred, TransferDue(d)
  PROVE  /\ \A u : HeldIn(u, table, oslot, cache, hcache, deferred', copies, tasks) = Held(u)
         /\ \A x \in deferred' : x \in deferred /\ x # d
         /\ \A x \in deferred : x # d => x \in deferred'
         /\ \A o \in Owners : Moving(o)' => Moving(o) \/ o = d.o
  <1>1. d.units = {} /\ deferred' = deferred \ {d}
    BY TransferDue_Facts, TransferDue_Effect
  <1>2. \A u : HeldIn(u, table, oslot, cache, hcache, deferred', copies, tasks) = Held(u)
    BY <1>1, Zenon DEF HeldIn, Held
  <1>3. \A o \in Owners : Moving(o)' => Moving(o) \/ o = d.o
    BY <1>1, TransferDue_Effect DEF Moving, NewTask
  <1> QED
    BY <1>1, <1>2, <1>3

(* A cover that held before the step holds after it, but for the cover of  *)
(* the transfer's owner by its table.                                      *)
LEMMA TransferDue_Covered ==
  ASSUME IndInv, NEW d \in deferred, TransferDue(d),
         \A t \in tasks' : t \in tasks \/ (t.kind = "transfer" /\ t.o = d.o),
         UNCHANGED <<ostate, ogroup, oslot, table, requests>>
  PROVE  \A b \in inflight, o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
  <1> SUFFICES ASSUME NEW b \in inflight, NEW o \in Owners, NEW u, Covered(b, o, u)
               PROVE  Covered(b, o, u)'
    OBVIOUS
  <1>1. /\ d.kind = "transfer" /\ d.after[b.g] < b.seq /\ d.o \in Owners
        /\ d.after[b.g] \in Nat /\ b.seq \in Nat
    <2>1. b.g \in Groups /\ b.seq \in Nat
      BY DEF IndInv, BatchesOk
    <2> QED
      BY <2>1, TransferDue_Effect, TransferDue_Facts, WorkTypes DEF WorkTyped, Stamps
  <1>2. /\ \A x \in deferred' : x \in deferred /\ x # d
        /\ \A x \in deferred : x # d => x \in deferred'
    BY TransferDue_Deferred
  <1>3. \A p \in Rows : HoldsAt(o, p, u)' = HoldsAt(o, p, u)
    BY DEF HoldsAt
  <1>4. CASE /\ Live(o) /\ ogroup[o] = b.g
             /\ \E p \in Rows :
                  /\ HoldsAt(o, p, u)
                  /\ \A r \in requests :
                       TakesRow(r, o, p) => b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
             /\ \A x \in deferred : (x.kind = "transfer" /\ x.o = o) => x.after[b.g] >= b.seq
             /\ ~\E t \in tasks : t.kind = "transfer" /\ t.o = o
    <2>1. o # d.o
      BY <1>1, <1>4
    <2> QED
      BY <1>2, <1>3, <1>4, <2>1 DEF Covered, Live
  <1>5. CASE \E x \in deferred : x.kind = "release" /\ u \in x.units /\ x.after[b.g] >= b.seq
    BY <1>1, <1>2, <1>5 DEF Covered
  <1> QED
    BY <1>4, <1>5 DEF Covered

(* The transfer comes due and no task starts: its owner is gone, or the    *)
(* target group has no free slot row.                                      *)
LEMMA TransferDue_Skip ==
  ASSUME IndInv, NEW d \in deferred, TransferDue(d), UNCHANGED rare,
         UNCHANGED <<free, tasks, alloc>>
  PROVE  IndInv'
  <1>1. /\ deferred' = deferred \ {d}
        /\ UNCHANGED <<free, tasks, alloc, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy,
                       valid, vis, trial, member, dbt, launched, inflight, handoff, requests,
                       sharedKey, remote, lost, bad>>
    BY TransferDue_Effect DEF rare
  <1>2. /\ \A u : Held(u)' = Held(u)
        /\ \A x \in deferred' : x \in deferred /\ x # d
        /\ \A o \in Owners : Moving(o)' => Moving(o)
    BY <1>1, TransferDue_Deferred DEF Held, Moving
  <1>3. TypeInv'
    <2>1. WorkTyped'
      BY <1>1, <1>2, WorkTypes DEF WorkTyped
    <2> QED
      BY <1>1, <2>1, UnitTypes, OwnerTypes, PlaneTypes, RequestTypes, UnitTypedFrame,
         OwnerTypedFrame, PlaneTypedFrame, RequestTypedFrame, TypeInvNext
  <1>4. FreeClean' /\ NoLeak'
    BY <1>1, <1>2 DEF IndInv, FreeClean, NoLeak, InRegion
  <1>5. /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
        /\ Tables' /\ OwnersOk' /\ LostOk' /\ RequestsOk' /\ TrialVisibility' /\ NoBad'
    BY <1>1, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame, SharedCachedFrame,
       CacheSharedFrame, TablesFrame, OwnersOkFrame, LostOkFrame, RequestsOkFrame,
       TrialVisibilityFrame, NoBadFrame DEF IndInv
  <1>6. DeferredOk'
    <2>1. SUFFICES ASSUME NEW x \in deferred' PROVE DeferOk(x)'
      BY DeferredOkEach
    <2>2. DeferOk(x)
      BY <1>2, DeferredOkEach DEF IndInv
    <2> QED
      BY <1>1, <1>2, <2>2 DEF DeferOk
  <1>7. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies' PROVE CopyOk(c)'
      BY CopiesOkEach
    <2>2. CopyOk(c)
      BY <1>1, CopiesOkEach DEF IndInv
    <2> QED
      BY <1>1, <1>2, <2>2 DEF CopyOk, Live, Mem
  <1>8. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. TaskOk(t)
      BY <1>1, TasksOkEach DEF IndInv
    <2> QED
      BY <1>1, <1>2, <2>2 DEF TaskOk, Live, TableUnits, Mem
  <1>9. MovingUnmapped' /\ TablesMapped' /\ HandoffOk'
    BY <1>1, <1>2 DEF IndInv, MovingUnmapped, TablesMapped, HandoffOk, Live, HoldsAt
  <1>10. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight' PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b)
      BY <1>1, BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
      BY <1>1, TransferDue_Covered
    <2>4. \A x \in b.reads : x[1] \in Owners
      BY <2>2 DEF BatchOk
    <2> QED
      BY <1>1, <1>2, <2>2, <2>3, <2>4 DEF BatchOk, Mem
  <1> QED
    BY <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10 DEF IndInv

(* The transfer comes due and its task starts with slot row s.             *)
LEMMA TransferDue_Start ==
  ASSUME IndInv, NEW d \in deferred, TransferDue(d), UNCHANGED rare,
         ostate[d.o] = "live", NEW s \in FreeRows(d),
         free' = free \ {s}, alloc' = [alloc EXCEPT ![s] = d.o],
         tasks' = tasks \cup {NewTask(d, s)}
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, WorkTypes
  <1>1. PICK t0 : t0 = NewTask(d, s)
    OBVIOUS
  <1>2. /\ deferred' = deferred \ {d}
        /\ tasks' = tasks \cup {t0}
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, valid, vis,
                       trial, member, dbt, launched, inflight, handoff, requests, sharedKey,
                       remote, lost, bad>>
    BY <1>1, TransferDue_Effect DEF rare
  <1>3. /\ d.o \in Owners /\ d.to \in Groups /\ GroupEpoch[d.to] = OwnerEpoch[d.o]
        /\ d.units = {} /\ Live(d.o)
        /\ \A x \in deferred : (x.kind = "transfer" /\ x.o = d.o) => x = d
        /\ \A t \in tasks : t.kind = "transfer" => t.o # d.o
        /\ \A c \in copies : c.o # d.o
        /\ \A b \in inflight : d.after[b.g] < b.seq
        /\ d.kind = "transfer"
        /\ ~\E c \in copies : c.fill /\ Mem(c.g) = Mem(d.to) /\ c.write \in TableUnits(d.o)
        /\ ~\E t \in tasks :
             t.kind = "transfer" /\ Mem(t.to) = Mem(d.to) /\ t.writes \cap TableUnits(d.o) # {}
    BY TransferDue_Facts, TransferDue_Effect DEF Live
  (* The new slot row: free, so nothing holds it.                          *)
  <1>4. /\ s \in Slots /\ s \in free /\ SlotGroup[s] = d.to
        /\ alloc[s] = NONE /\ sharedKey[s] = NONE /\ ~Held(s)
        /\ \A o \in Owners : oslot[o] # s
        /\ \A x \in deferred : s \notin x.units
        /\ \A c \in copies : c.read # s /\ c.write # s
        /\ \A t \in tasks : s \notin t.units /\ t.slot # s /\ t.h # s
    <2>1. s \in Slots /\ s \in free /\ SlotGroup[s] = d.to
      BY DEF FreeRows
    <2>2. alloc[s] = NONE /\ sharedKey[s] = NONE /\ ~Held(s)
      BY <2>1 DEF IndInv, FreeClean
    <2>3. \A t \in tasks : t.held
      BY DEF IndInv, TasksOk
    <2> QED
      BY <2>1, <2>2, <2>3 DEF Held, HeldIn
  (* The owner and the units its task holds.                               *)
  <1>5. /\ oslot[d.o] \in Slots /\ alloc[oslot[d.o]] = d.o /\ oslot[d.o] # s
        /\ TableUnits(d.o) \subseteq Units
        /\ \A u \in TableUnits(d.o) : (alloc[u] = d.o \/ alloc[u] = "shared") /\ Held(u)
        /\ \A u \in TableUnits(d.o) : u # s
    <2>1. oslot[d.o] \in Slots /\ alloc[oslot[d.o]] = d.o
      BY <1>3 DEF IndInv, OwnersOk
    <2>2. \A u \in TableUnits(d.o) : \E p \in Positions : table[d.o][p].u = u /\ u # NONE
      BY DEF TableUnits
    <2>3. \A p \in Positions : table[d.o][p].u # NONE =>
            /\ table[d.o][p].u \in Units
            /\ alloc[table[d.o][p].u] = d.o \/ alloc[table[d.o][p].u] = "shared"
      BY <1>0, <1>3 DEF IndInv, Tables, OwnerTyped, Empty, Private, Shared
    <2> QED
      BY <1>3, <1>4, <2>1, <2>2, <2>3 DEF Held, HeldIn
  <1>6. /\ t0.kind = "transfer" /\ t0.k = NONE /\ t0.h = NONE /\ t0.src = NONE /\ t0.held
        /\ t0.o = d.o /\ t0.to = d.to /\ t0.slot = s
        /\ t0.units = TableUnits(d.o) \cup {oslot[d.o]}
        /\ t0.writes = {u \in TableUnits(d.o) : Mem(d.to) \notin valid[u]}
        /\ t0 = [kind |-> t0.kind, k |-> t0.k, h |-> t0.h, src |-> t0.src, ok |-> t0.ok,
                 expired |-> t0.expired, o |-> t0.o, to |-> t0.to, slot |-> t0.slot,
                 held |-> t0.held, units |-> t0.units, writes |-> t0.writes]
    BY <1>1 DEF NewTask
  <1>7. /\ \A v \in AllUnits : alloc'[v] = IF v = s THEN d.o ELSE alloc[v]
        /\ \A v \in AllUnits : Held(v)' = (Held(v) \/ v = s)
    <2>1. \A v \in AllUnits : alloc'[v] = IF v = s THEN d.o ELSE alloc[v]
      BY <1>0 DEF UnitTyped
    <2>2. \A v : HeldIn(v, table, oslot, cache, hcache, deferred', copies, tasks) = Held(v)
      BY TransferDue_Deferred
    <2>3. \A v : Held(v)' = (HeldIn(v, table, oslot, cache, hcache, deferred', copies, tasks)
                             \/ v \in t0.units \/ v = t0.slot \/ v = t0.h)
      BY <1>2, <1>6, Zenon DEF Held, HeldIn
    <2>4. \A v \in AllUnits : v \in t0.units => Held(v)
      BY <1>3, <1>5, <1>6 DEF Held, HeldIn
    <2> QED
      BY <1>6, <2>1, <2>2, <2>3, <2>4 DEF AllUnits
  <1>8. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>2, <1>3, <1>4 DEF UnitTyped
    <2>2. WorkTyped'
      BY <1>0, <1>2, <1>3, <1>4, <1>5, <1>6 DEF WorkTyped, AllUnits
    <2> QED
      BY <1>2, <2>1, <2>2, OwnerTypes, PlaneTypes, RequestTypes, OwnerTypedFrame, PlaneTypedFrame,
         RequestTypedFrame, TypeInvNext
  <1>9. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
    <2>1. FreeClean'
      BY <1>2, <1>4, <1>7, <1>0 DEF IndInv, FreeClean, UnitTyped
    <2>2. NoLeak'
      BY <1>2, <1>4, <1>7 DEF IndInv, NoLeak, InRegion, AllUnits
    <2>3. Outside' /\ Regions'
      BY <1>2, <1>4, <1>7, <1>0 DEF IndInv, Outside, Regions, OneRegion, UnitTyped, AllUnits
    <2> QED
      BY <2>1, <2>2, <2>3
  <1>10. AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    BY <1>0, <1>2, <1>3, <1>4, <1>7 DEF IndInv, AllocOwner, SharedIff, SharedCached, CacheShared,
                                        Live, UnitTyped, AllUnits
  <1>11. Tables' /\ OwnersOk'
    <2>1. \A o \in Owners, p \in Positions : table[o][p].u # NONE => table[o][p].u \in Units
      BY <1>0 DEF OwnerTyped, Empty, Private, Shared
    <2> QED
      BY <1>2, <1>4, <1>7, <2>1 DEF IndInv, Tables, OwnersOk, Live, Mem, AllUnits
  <1>12. DeferredOk'
    <2>1. SUFFICES ASSUME NEW x \in deferred' PROVE DeferOk(x)'
      BY DeferredOkEach
    <2>2. x \in deferred /\ x # d /\ DeferOk(x)
      BY <1>2, DeferredOkEach DEF IndInv
    <2>3. x.kind = "commit" => \A u \in x.units : u \in Units
      BY <2>2 DEF DeferOk
    <2> QED
      BY <1>2, <1>4, <1>7, <2>2, <2>3 DEF DeferOk, AllUnits
  <1>13. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies' PROVE CopyOk(c)'
      BY CopiesOkEach
    <2>2. c \in copies /\ CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
      BY <1>0, <1>2, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. \A o \in Owners : Moving(o)' => Moving(o) \/ o = d.o
      BY TransferDue_Deferred
    <2>4. (c.fill /\ Mem(c.g) = Mem(d.to)) => c.write \notin t0.writes
      BY <1>3, <1>6, <2>2
    <2>5. c.o \in Owners /\ Live(c.o) => c.o # d.o
      BY <1>3, <2>2
    (* A private block a copy writes is in no table, and its owner is not  *)
    (* d.o, which owns d.o's slot.                                         *)
    <2>6. (c.kind \in {"clone", "load"} /\ ~c.fill) => c.write \notin t0.units
      BY <1>3, <1>5, <1>6, <2>2 DEF CopyOk, TableUnits
    <2> QED
      BY <1>2, <1>4, <1>6, <1>7, <2>2, <2>3, <2>4, <2>5, <2>6 DEF CopyOk, Live, Mem
  <1>14. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. \A x \in deferred' : x \in deferred /\ x # d
      BY TransferDue_Deferred
    <2>3. CASE t = t0
      <3>1. /\ \A x \in tasks' : (x.kind = "transfer" /\ (x.o = t0.o \/ x.slot = t0.slot)) => x = t0
            /\ \A x \in tasks' :
                 (x.kind = "transfer" /\ x # t0 /\ Mem(x.to) = Mem(t0.to)) =>
                     x.writes \cap t0.writes = {}
            /\ \A x \in tasks' : t0.slot \notin x.units
        BY <1>2, <1>3, <1>4, <1>5, <1>6
      <3>2. ~\E x \in deferred' : x.kind = "transfer" /\ x.o = t0.o
        BY <1>3, <1>6, <2>2
      <3>3. /\ t0.units \subseteq Units \cup Slots /\ t0.writes \subseteq t0.units \cap Units
            /\ \A u \in t0.writes : u \in TableUnits(d.o) /\ Mem(t0.to) \notin valid[u]
        BY <1>5, <1>6
      <3>4. \A u \in t0.writes : alloc'[u] = t0.o \/ alloc'[u] = "shared"
        BY <1>5, <1>6, <1>7, <3>3 DEF AllUnits
      <3>5. alloc'[t0.slot] = t0.o
        BY <1>4, <1>6, <1>7 DEF AllUnits
      <3> QED
        BY <1>2, <1>3, <1>4, <1>5, <1>6, <2>3, <3>1, <3>2, <3>3, <3>4, <3>5 DEF TaskOk, Live,
                                                                               TableUnits
    <2>4. CASE t # t0
      <3>1. t \in tasks /\ TaskOk(t)
        BY <1>2, <2>4, TasksOkEach DEF IndInv
      <3>2. t.kind = "transfer" =>
              /\ t.slot \notin t0.units /\ t.o # t0.o /\ t.slot # t0.slot
              /\ (Mem(t.to) = Mem(t0.to) => t.writes \cap t0.writes = {})
              /\ alloc'[t.slot] = alloc[t.slot]
              /\ \A u \in t.writes : alloc'[u] = alloc[u]
        BY <1>3, <1>4, <1>5, <1>6, <1>7, <3>1 DEF TaskOk, AllUnits
      <3>3. t.kind = "prefetch" => alloc'[t.h] = alloc[t.h] /\ t0.kind # "prefetch"
        BY <1>4, <1>6, <1>7, <3>1 DEF TaskOk, AllUnits
      <3> QED
        BY <1>2, <1>6, <2>2, <3>1, <3>2, <3>3 DEF TaskOk, Live, TableUnits, Mem
    <2> QED
      BY <2>3, <2>4
  <1>15. /\ MovingUnmapped' /\ TablesMapped' /\ HandoffOk' /\ LostOk' /\ RequestsOk'
         /\ TrialVisibility' /\ NoBad'
    <2>1. \A x \in deferred' : x \in deferred /\ x # d
      BY TransferDue_Deferred
    <2>2. \A o \in Owners : Moving(o)' => Moving(o)
      BY <1>2, <1>3, <1>6 DEF Moving
    <2>3. MovingUnmapped'
      BY <1>2, <2>1, <2>2 DEF IndInv, MovingUnmapped, Moving
    <2>4. TablesMapped' /\ HandoffOk'
      BY <1>2, <2>1 DEF IndInv, TablesMapped, HandoffOk, Live, HoldsAt
    <2>5. LostOk' /\ RequestsOk' /\ NoBad'
      BY <1>2, LostOkFrame, RequestsOkFrame, NoBadFrame DEF IndInv
    <2>6. TrialVisibility'
      BY <1>2, <1>6 DEF IndInv, TrialVisibility, MadeBy
    <2> QED
      BY <2>3, <2>4, <2>5, <2>6
  <1>16. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight' PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b) /\ b \in inflight
      BY <1>2, BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
      <3>1. \A t \in tasks' : t \in tasks \/ (t.kind = "transfer" /\ t.o = d.o)
        BY <1>2, <1>6
      <3> QED
        BY <1>2, <2>2, <3>1, TransferDue_Covered
    <2>4. \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits /\ x[2] # s
      BY <1>4, <2>2, RequestTypes DEF BatchOk, RequestTyped
    <2>5. \A x \in deferred' : x \in deferred
      BY TransferDue_Deferred
    <2> QED
      BY <1>2, <1>7, <2>2, <2>3, <2>4, <2>5 DEF BatchOk, Mem
  <1> QED
    BY <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16 DEF IndInv

LEMMA TransferDue_Inv ==
  ASSUME IndInv, NEW d \in deferred, TransferDue(d), UNCHANGED rare
  PROVE  IndInv'
  <1>1. CASE UNCHANGED <<free, tasks, alloc>>
    BY <1>1, TransferDue_Skip
  <1>2. CASE /\ ostate[d.o] = "live"
             /\ \E s \in FreeRows(d) :
                  /\ free' = free \ {s}
                  /\ alloc' = [alloc EXCEPT ![s] = d.o]
                  /\ tasks' = tasks \cup {NewTask(d, s)}
    BY <1>2, TransferDue_Start
  <1> QED
    BY <1>1, <1>2, TransferDue_Effect

-----------------------------------------------------------------------------
(* TransferEnd                                                             *)

(* The values TransferEnd(t, ok) computes: the entries the target group's  *)
(* executor is handed, those no request under way names, the rows of its   *)
(* table when they are handed at once, and where each block is valid.      *)
Entries(t) ==
    {[g |-> t.to, o |-> t.o, p |-> p, u |-> table[t.o][p].u,
      w |-> /\ table[t.o][p].k = "private"
            /\ ~\E d \in deferred : d.kind = "commit" /\ d.o = t.o /\ d.p = p] :
       p \in {q \in Positions : table[t.o][q] # Empty}}
    \cup {[g |-> t.to, o |-> t.o, p |-> SLOT, u |-> t.slot, w |-> TRUE]}
Given(t) == {h \in Entries(t) : ~\E r \in requests : Names(r, h)}
GivenRows(t) ==
    [p \in Rows |->
       IF \E h \in Given(t) : h.p = p
       THEN LET h == CHOOSE x \in Given(t) : x.p = p IN [u |-> h.u, w |-> h.w]
       ELSE NoEntry]
NewValid(t) ==
    [u \in Units |->
       IF u \notin t.writes THEN valid[u]
       ELSE IF alloc[u] = "shared" \/ \E d \in deferred : d.kind = "commit" /\ u \in d.units
            THEN valid[u] \cup {Mem(t.to)}
            ELSE {Mem(t.to)}]

LEMMA TransferEnd_Effect ==
  ASSUME NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok)
  PROVE  /\ t.kind = "transfer"
         /\ tasks' = tasks \ {t}
         /\ \/ /\ ok /\ ostate[t.o] = "live"
               /\ ogroup' = [ogroup EXCEPT ![t.o] = t.to]
               /\ oslot' = [oslot EXCEPT ![t.o] = t.slot]
               /\ valid' = NewValid(t)
               /\ DropWith(t.units, table, oslot', cache, hcache, deferred, copies, tasks')
               /\ \/ /\ Eager
                     /\ handoff' = handoff
                     /\ dbt' = [dbt EXCEPT ![t.to][t.o] = GivenRows(t)]
                  \/ /\ ~Eager
                     /\ handoff' = handoff \cup Given(t)
                     /\ dbt' = dbt
            \/ /\ ~(ok /\ ostate[t.o] = "live")
               /\ DropWith(t.units \cup {t.slot}, table, oslot, cache, hcache, deferred, copies,
                           tasks')
               /\ UNCHANGED <<ogroup, oslot, valid, dbt, handoff>>
         /\ UNCHANGED <<ostate, table, cache, hcache, deferred, copies, ncopy, vis, trial, member,
                        launched, inflight, requests>>
  BY DEF TransferEnd, Entries, Given, GivenRows, NewValid

(* What TasksOk says of the task that ends.                                *)
LEMMA TransferEnd_Facts ==
  ASSUME IndInv, NEW t \in tasks, t.kind = "transfer"
  PROVE  /\ t.held /\ t.h = NONE
         /\ t.o \in Owners /\ t.to \in Groups /\ GroupEpoch[t.to] = OwnerEpoch[t.o]
         /\ t.slot \in Slots /\ alloc[t.slot] = t.o /\ SlotGroup[t.slot] = t.to
         /\ \A o \in Owners : oslot[o] # t.slot
         /\ \A d \in deferred : t.slot \notin d.units
         /\ \A c \in copies : c.read # t.slot /\ c.write # t.slot
         /\ \A x \in tasks : t.slot \notin x.units
         /\ t.units \subseteq Units \cup Slots
         /\ t.writes \subseteq t.units \cap Units
         /\ \A u \in t.writes : Mem(t.to) \notin valid[u] /\ (alloc[u] = t.o \/ alloc[u] = "shared")
         /\ \A x \in tasks : (x.kind = "transfer" /\ (x.o = t.o \/ x.slot = t.slot)) => x = t
         /\ \A x \in tasks :
              (x.kind = "transfer" /\ x # t /\ Mem(x.to) = Mem(t.to)) => x.writes \cap t.writes = {}
         /\ ~\E d \in deferred : d.kind = "transfer" /\ d.o = t.o
         /\ Live(t.o) =>
              /\ t.units = TableUnits(t.o) \cup {oslot[t.o]}
              /\ ~\E c \in copies : c.o = t.o
              /\ \A u \in TableUnits(t.o) \ t.writes : Mem(t.to) \in valid[u]
         /\ \A g \in Groups, p \in Rows : dbt[g][t.o][p].u = NONE
         /\ \A h \in handoff : h.o # t.o
  <1>1. TaskOk(t)
    BY TasksOkEach DEF IndInv
  <1>2. /\ \A g \in Groups, p \in Rows : dbt[g][t.o][p].u = NONE
        /\ \A h \in handoff : h.o # t.o
    BY <1>1 DEF IndInv, MovingUnmapped, TaskOk
  <1> QED
    BY <1>1, <1>2 DEF TaskOk

(* The task fails, or its owner was detached: what it held retires.        *)
LEMMA TransferEnd_Fail ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ~(ok /\ ostate[t.o] = "live")
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, WorkTypes
  <1> DEFINE G == ReclaimedWith(t.units \cup {t.slot}, table, oslot, cache, hcache, deferred,
                                copies, tasks \ {t})
  <1>1. /\ t.kind = "transfer" /\ tasks' = tasks \ {t}
        /\ free' = free \cup G /\ alloc' = Reset(alloc, G) /\ sharedKey' = Reset(sharedKey, G)
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, valid,
                       vis, trial, member, dbt, launched, inflight, handoff, requests,
                       remote, lost, bad>>
    BY TransferEnd_Effect DEF DropWith, rare
  <1>2. /\ t.held /\ t.units \cup {t.slot} \subseteq AllUnits
    BY <1>1, TransferEnd_Facts DEF AllUnits
  <1>3. /\ \A u : Held(u)' = HeldIn(u, table, oslot, cache, hcache, deferred, copies, tasks \ {t})
        /\ G = {u \in t.units \cup {t.slot} : ~Held(u)'}
    BY <1>1 DEF Held, ReclaimedWith, On, AllRules
  <1>4. /\ \A u \in AllUnits : Held(u)' => Held(u)
        /\ \A u \in AllUnits : Held(u) => Held(u)' \/ u \in G
        /\ \A u \in G : Held(u) /\ ~Held(u)'
        /\ G \subseteq AllUnits
    <2>1. \A u : Held(u)' => Held(u)
      BY <1>3, Zenon DEF Held, HeldIn
    <2>2. \A u : Held(u) => Held(u)' \/ u \in t.units \/ u = t.slot \/ u = t.h
      BY <1>3, Zenon DEF Held, HeldIn
    <2>3. \A u \in t.units \cup {t.slot} : Held(u)
      BY <1>2 DEF Held, HeldIn
    <2>4. t.h = NONE
      BY <1>1, TransferEnd_Facts
    <2> QED
      BY <1>2, <1>3, <2>1, <2>2, <2>3, <2>4 DEF AllUnits
  <1>5. /\ \A v \in AllUnits : alloc'[v] = IF v \in G THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v \in G THEN NONE ELSE sharedKey[v]
        /\ \A v \in AllUnits : Held(v)' => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
    BY <1>1, <1>4 DEF Reset
  <1>6. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>4 DEF UnitTyped, Reset
    <2>2. WorkTyped'
      BY <1>0, <1>1 DEF WorkTyped
    <2> QED
      BY <1>1, <2>1, <2>2, OwnerTypes, PlaneTypes, RequestTypes, OwnerTypedFrame, PlaneTypedFrame,
         RequestTypedFrame, TypeInvNext
  <1>7. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
    BY <1>1, <1>4, <1>5, FreeReclaim, UnitTypes DEF IndInv
  (* What stays held keeps its allocation.                                 *)
  <1>8. /\ \A o \in Owners, p \in Positions :
             table[o][p].u # NONE => alloc'[table[o][p].u] = alloc[table[o][p].u]
        /\ \A o \in Owners : oslot[o] # NONE => alloc'[oslot[o]] = alloc[oslot[o]]
        /\ \A d \in deferred : \A u \in d.units : alloc'[u] = alloc[u]
        /\ \A c \in copies : alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
        /\ \A x \in tasks' : \A u \in x.units \cup {x.slot, x.h} : u # NONE => alloc'[u] = alloc[u]
        /\ \A k \in Keys : /\ cache[k] # NONE => sharedKey'[cache[k]] = sharedKey[cache[k]]
                           /\ hcache[k] # NONE => sharedKey'[hcache[k]] = sharedKey[hcache[k]]
    <2>1. \A x \in tasks' : x.held
      BY <1>1 DEF IndInv, TasksOk
    <2>2. \A u \in AllUnits : (\/ \E o \in Owners, p \in Positions : table[o][p].u = u
                               \/ \E o \in Owners : oslot[o] = u
                               \/ \E k \in Keys : cache[k] = u \/ hcache[k] = u
                               \/ \E d \in deferred : u \in d.units
                               \/ \E c \in copies : u = c.read \/ u = c.write
                               \/ \E x \in tasks' : x.held /\ (u \in x.units \/ u = x.slot \/ u = x.h))
                              => Held(u)'
      BY <1>1, <1>3 DEF HeldIn
    <2>3. /\ \A o \in Owners, p \in Positions : table[o][p].u # NONE => table[o][p].u \in AllUnits
          /\ \A o \in Owners : oslot[o] # NONE => oslot[o] \in AllUnits
          /\ \A d \in deferred : d.units \subseteq AllUnits
          /\ \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
          /\ \A x \in tasks' : \A u \in x.units \cup {x.slot, x.h} : u # NONE => u \in AllUnits
          /\ \A k \in Keys : (cache[k] # NONE => cache[k] \in AllUnits)
                             /\ (hcache[k] # NONE => hcache[k] \in AllUnits)
      BY <1>0, <1>1 DEF OwnerTyped, WorkTyped, UnitTyped, AllUnits, Empty, Private, Shared
    <2> QED
      BY <1>5, <2>1, <2>2, <2>3
  <1>9. AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. AllocOwner' /\ SharedIff'
      BY <1>1, <1>5 DEF IndInv, AllocOwner, SharedIff, NONE
    <2>2. \A v \in AllUnits : alloc[v] = "shared" /\ v \notin G =>
            cache[sharedKey[v]] = v \/ hcache[sharedKey[v]] = v
      BY DEF IndInv, SharedCached
    <2>3. SharedCached'
      BY <1>1, <1>5, <2>2 DEF SharedCached, NONE
    <2>4. CacheShared'
      BY <1>1, <1>8 DEF IndInv, CacheShared
    <2> QED
      BY <2>1, <2>3, <2>4
  <1>10. Tables' /\ OwnersOk'
    BY <1>1, <1>8 DEF IndInv, Tables, OwnersOk, Live, Mem
  <1>11. DeferredOk'
    BY <1>1, <1>8 DEF IndInv, DeferredOk
  <1>12. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies' PROVE CopyOk(c)'
      BY CopiesOkEach
    <2>2. CopyOk(c)
      BY <1>1, CopiesOkEach DEF IndInv
    <2>3. \A o \in Owners : Moving(o)' => Moving(o)
      BY <1>1 DEF Moving
    <2> QED
      BY <1>1, <1>8, <2>2, <2>3 DEF CopyOk, Live, Mem
  <1>13. TasksOk'
    <2>1. SUFFICES ASSUME NEW x \in tasks' PROVE TaskOk(x)'
      BY TasksOkEach
    <2>2. x \in tasks /\ TaskOk(x)
      BY <1>1, TasksOkEach DEF IndInv
    <2>3. \A u \in x.units \cup {x.slot, x.h} : u # NONE => alloc'[u] = alloc[u]
      BY <1>8
    <2>4. /\ x.kind = "transfer" => x.writes \subseteq x.units /\ x.slot # NONE
          /\ x.kind = "prefetch" => x.h # NONE
      BY <2>2 DEF TaskOk
    <2> QED
      BY <1>1, <2>2, <2>3, <2>4 DEF TaskOk, Live, TableUnits, Mem
  <1>14. /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
         /\ TrialVisibility' /\ NoBad'
    BY <1>1, LostOkFrame, TablesMappedFrame, HandoffOkFrame, RequestsOkFrame, NoBadFrame
       DEF IndInv, MovingUnmapped, TrialVisibility, MadeBy
  <1>15. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight' PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b) /\ b \in inflight
      BY <1>1, BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)' /\ Held(u)'
      <3>1. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
        BY <1>1 DEF Covered, Live, HoldsAt
      <3>2. \A o \in Owners : \A u : Covered(b, o, u) => Held(u)'
        BY <1>1, <1>3 DEF Covered, HoldsAt, Rows, HeldIn
      <3> QED
        BY <3>1, <3>2
    <2>4. \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits
      BY <2>2, RequestTypes DEF RequestTyped
    <2> QED
      BY <1>1, <1>4, <1>5, <2>2, <2>3, <2>4 DEF BatchOk, Mem
  <1> QED
    BY <1>6, <1>7, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15 DEF IndInv

(* What the end of a task that moves its owner reclaims: the units it held *)
(* that nothing holds once the owner has its new slot row.                 *)
EndFreed(t) == ReclaimedWith(t.units, table, oslot', cache, hcache, deferred, copies, tasks')

(* The entries handed to the target group's executor.                      *)
LEMMA TransferEnd_Entries ==
  ASSUME IndInv, NEW t \in tasks, t.kind = "transfer"
  PROVE  \A h \in Entries(t) :
           /\ h.g = t.to /\ h.o = t.o /\ h.p \in Rows /\ h.u \in AllUnits
           /\ h.p = SLOT => h.u = t.slot
           /\ h.p \in Positions => table[t.o][h.p].u = h.u
           /\ (h.p \in Positions /\ h.w) =>
                /\ table[t.o][h.p].k = "private"
                /\ ~\E d \in deferred : d.kind = "commit" /\ h.u \in d.units
  <1>0. OwnerTyped /\ t.o \in Owners /\ t.slot \in Slots
    BY OwnerTypes, TransferEnd_Facts
  <1>1. \A p \in Positions : table[t.o][p] # Empty => table[t.o][p].u \in Units
    BY <1>0 DEF OwnerTyped, Empty, Private, Shared
  <1>2. \A p \in Positions, d \in deferred :
          (d.kind = "commit" /\ table[t.o][p].u \in d.units) => d.o = t.o /\ d.p = p
    BY <1>0 DEF IndInv, DeferredOk
  <1> QED
    BY <1>0, <1>1, <1>2 DEF Entries, Rows, AllUnits

(* A row of the target group's table handed at once is a given entry.      *)
LEMMA TransferEnd_Rows ==
  ASSUME NEW t
  PROVE  \A p \in Rows :
           \/ GivenRows(t)[p] = NoEntry
           \/ \E h \in Given(t) : h.p = p /\ GivenRows(t)[p] = [u |-> h.u, w |-> h.w]
  BY DEF GivenRows

(* A request under way that takes or commits a row names its entry.        *)
LEMMA TransferEnd_Names ==
  ASSUME NEW t, NEW h \in Given(t), NEW r \in requests
  PROVE  /\ ~Names(r, h)
         /\ (TakesRow(r, h.o, h.p) \/ CommitsRow(r, h.o, h.p)) => Names(r, h)
  BY DEF Given, Names, TakesRow, CommitsRow

LEMMA TransferEnd_Values ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  /\ t.kind = "transfer" /\ tasks' = tasks \ {t}
         /\ UNCHANGED <<ostate, table, cache, hcache, deferred, copies, ncopy, vis, trial, member,
                        launched, inflight, requests, remote, lost, bad>>
         /\ \A o \in Owners : ogroup'[o] = IF o = t.o THEN t.to ELSE ogroup[o]
         /\ \A o \in Owners : oslot'[o] = IF o = t.o THEN t.slot ELSE oslot[o]
         /\ ogroup' \in [Owners -> Groups \cup {NONE}] /\ oslot' \in [Owners -> Slots \cup {NONE}]
         /\ valid' = NewValid(t)
         /\ free' = free \cup EndFreed(t)
         /\ \A v \in AllUnits : alloc'[v] = IF v \in EndFreed(t) THEN NONE ELSE alloc[v]
         /\ \A v \in AllUnits : sharedKey'[v] = IF v \in EndFreed(t) THEN NONE ELSE sharedKey[v]
         /\ alloc' = Reset(alloc, EndFreed(t)) /\ sharedKey' = Reset(sharedKey, EndFreed(t))
         /\ \A u : Held(u)' = HeldIn(u, table, oslot', cache, hcache, deferred, copies, tasks \ {t})
         /\ EndFreed(t) = {u \in t.units : ~Held(u)'}
         /\ EndFreed(t) \subseteq AllUnits
         /\ \A u \in AllUnits : Held(u)' => Held(u)
         /\ \A u \in AllUnits : Held(u) => Held(u)' \/ u \in EndFreed(t)
         /\ \A u \in EndFreed(t) : Held(u) /\ ~Held(u)'
         /\ \A u \in AllUnits : Held(u)' => alloc'[u] = alloc[u] /\ sharedKey'[u] = sharedKey[u]
         /\ \/ /\ Eager
               /\ handoff' = handoff
               /\ dbt' = [dbt EXCEPT ![t.to][t.o] = GivenRows(t)]
            \/ /\ ~Eager
               /\ handoff' = handoff \cup Given(t)
               /\ dbt' = dbt
  <1>0. UnitTyped /\ OwnerTyped
    BY UnitTypes, OwnerTypes
  <1>1. /\ t.kind = "transfer" /\ tasks' = tasks \ {t}
        /\ ogroup' = [ogroup EXCEPT ![t.o] = t.to]
        /\ oslot' = [oslot EXCEPT ![t.o] = t.slot]
        /\ valid' = NewValid(t)
        /\ free' = free \cup EndFreed(t)
        /\ alloc' = Reset(alloc, EndFreed(t)) /\ sharedKey' = Reset(sharedKey, EndFreed(t))
        /\ \/ /\ Eager
              /\ handoff' = handoff
              /\ dbt' = [dbt EXCEPT ![t.to][t.o] = GivenRows(t)]
           \/ /\ ~Eager
              /\ handoff' = handoff \cup Given(t)
              /\ dbt' = dbt
        /\ UNCHANGED <<ostate, table, cache, hcache, deferred, copies, ncopy, vis, trial, member,
                       launched, inflight, requests, remote, lost, bad>>
    BY TransferEnd_Effect DEF DropWith, EndFreed, rare
  <1>2. /\ t.held /\ t.h = NONE /\ t.o \in Owners /\ t.to \in Groups /\ t.slot \in Slots
        /\ t.units \subseteq Units \cup Slots
        /\ \A x \in tasks : t.slot \notin x.units
        /\ Live(t.o) => t.units = TableUnits(t.o) \cup {oslot[t.o]}
    BY <1>1, TransferEnd_Facts
  <1>3. /\ \A o \in Owners : ogroup'[o] = IF o = t.o THEN t.to ELSE ogroup[o]
        /\ \A o \in Owners : oslot'[o] = IF o = t.o THEN t.slot ELSE oslot[o]
        /\ ogroup' \in [Owners -> Groups \cup {NONE}] /\ oslot' \in [Owners -> Slots \cup {NONE}]
    BY <1>0, <1>1, <1>2 DEF OwnerTyped
  <1>4. /\ \A u : Held(u)' = HeldIn(u, table, oslot', cache, hcache, deferred, copies, tasks \ {t})
        /\ EndFreed(t) = {u \in t.units : ~Held(u)'}
    BY <1>1 DEF Held, EndFreed, ReclaimedWith, On, AllRules
  <1>5. EndFreed(t) \subseteq AllUnits
    BY <1>2, <1>4 DEF AllUnits
  <1>6. \A u \in AllUnits : Held(u)' => Held(u)
    <2>1. \A u : HeldIn(u, table, oslot', cache, hcache, deferred, copies, tasks \ {t}) =>
                 HeldIn(u, table, oslot, cache, hcache, deferred, copies, tasks) \/ u = t.slot
      BY <1>3, Zenon DEF HeldIn
    <2>2. Held(t.slot)
      BY <1>2 DEF Held, HeldIn
    <2> QED
      BY <1>4, <2>1, <2>2 DEF Held
  <1>7. \A u \in AllUnits : Held(u) => Held(u)' \/ u \in EndFreed(t)
    <2>1. \A u : HeldIn(u, table, oslot, cache, hcache, deferred, copies, tasks) =>
                 \/ HeldIn(u, table, oslot', cache, hcache, deferred, copies, tasks \ {t})
                 \/ u = oslot[t.o] \/ u \in t.units \/ u = t.slot \/ u = t.h
      BY <1>3, Zenon DEF HeldIn
    <2>2. HeldIn(t.slot, table, oslot', cache, hcache, deferred, copies, tasks \ {t})
      BY <1>2, <1>3 DEF HeldIn
    <2>3. oslot[t.o] \in t.units
      BY <1>2 DEF Live
    <2> QED
      BY <1>2, <1>4, <2>1, <2>2, <2>3 DEF Held, AllUnits
  <1>8. \A u \in EndFreed(t) : Held(u) /\ ~Held(u)'
    BY <1>2, <1>4 DEF Held, HeldIn
  <1>9. /\ \A v \in AllUnits : alloc'[v] = IF v \in EndFreed(t) THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v \in EndFreed(t) THEN NONE ELSE sharedKey[v]
    BY <1>1 DEF Reset
  <1>10. \A u \in AllUnits : Held(u)' => alloc'[u] = alloc[u] /\ sharedKey'[u] = sharedKey[u]
    BY <1>4, <1>9
  <1> QED
    BY <1>1, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10

(* What stays held after the step keeps its allocation; where each block   *)
(* is valid after it.                                                      *)
LEMMA TransferEnd_Kept ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  /\ \A o \in Owners, p \in Positions :
              table[o][p].u # NONE => alloc'[table[o][p].u] = alloc[table[o][p].u]
         /\ \A o \in Owners : oslot'[o] # NONE => alloc'[oslot'[o]] = alloc[oslot'[o]]
         /\ \A d \in deferred : \A u \in d.units : alloc'[u] = alloc[u]
         /\ \A c \in copies : alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
         /\ \A x \in tasks' : \A u \in x.units \cup {x.slot, x.h} : u # NONE => alloc'[u] = alloc[u]
         /\ \A k \in Keys : /\ cache[k] # NONE => sharedKey'[cache[k]] = sharedKey[cache[k]]
                            /\ hcache[k] # NONE => sharedKey'[hcache[k]] = sharedKey[hcache[k]]
  <1>0. UnitTyped /\ OwnerTyped /\ WorkTyped
    BY UnitTypes, OwnerTypes, WorkTypes
  <1>1. \A x \in tasks' : x.held
    BY TransferEnd_Values DEF IndInv, TasksOk
  <1>2. \A u \in AllUnits : (\/ \E o \in Owners, p \in Positions : table[o][p].u = u
                             \/ \E o \in Owners : oslot'[o] = u
                             \/ \E k \in Keys : cache[k] = u \/ hcache[k] = u
                             \/ \E d \in deferred : u \in d.units
                             \/ \E c \in copies : u = c.read \/ u = c.write
                             \/ \E x \in tasks' : x.held /\ (u \in x.units \/ u = x.slot \/ u = x.h))
                            => Held(u)'
    BY TransferEnd_Values DEF HeldIn
  <1>3. /\ \A o \in Owners, p \in Positions : table[o][p].u # NONE => table[o][p].u \in AllUnits
        /\ \A o \in Owners : oslot'[o] # NONE => oslot'[o] \in AllUnits
        /\ \A d \in deferred : d.units \subseteq AllUnits
        /\ \A c \in copies : c.read \in AllUnits /\ c.write \in AllUnits
        /\ \A x \in tasks' : \A u \in x.units \cup {x.slot, x.h} : u # NONE => u \in AllUnits
        /\ \A k \in Keys : (cache[k] # NONE => cache[k] \in AllUnits)
                           /\ (hcache[k] # NONE => hcache[k] \in AllUnits)
    BY <1>0, TransferEnd_Values DEF OwnerTyped, WorkTyped, UnitTyped, AllUnits, Empty, Private,
                                    Shared
  <1> QED
    BY <1>1, <1>2, <1>3, TransferEnd_Values

LEMMA TransferEnd_Valid ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  /\ \A u \in Units : valid'[u] \subseteq valid[u] \cup {Mem(t.to)}
         /\ \A u \in Units \ t.writes : valid'[u] = valid[u]
         /\ \A u \in t.writes : Mem(t.to) \in valid'[u]
         /\ \A u \in Units :
              \/ valid[u] \subseteq valid'[u]
              \/ /\ u \in t.writes /\ alloc[u] = t.o
                 /\ ~\E d \in deferred : d.kind = "commit" /\ u \in d.units
         /\ \A o \in Owners, p \in Positions :
              Mem(ogroup'[o]) \in valid'[table[o][p].u] \/ table[o][p].u = NONE
  <1>1. valid' = NewValid(t) /\ t.writes \subseteq Units
    BY TransferEnd_Values, TransferEnd_Facts
  <1>2. /\ \A u \in Units : valid'[u] \subseteq valid[u] \cup {Mem(t.to)}
        /\ \A u \in Units \ t.writes : valid'[u] = valid[u]
        /\ \A u \in t.writes : Mem(t.to) \in valid'[u]
        /\ \A u \in Units :
             \/ valid[u] \subseteq valid'[u]
             \/ /\ u \in t.writes /\ alloc[u] # "shared"
                /\ ~\E d \in deferred : d.kind = "commit" /\ u \in d.units
    BY <1>1 DEF NewValid
  <1>3. \A u \in t.writes : alloc[u] = t.o \/ alloc[u] = "shared"
    BY TransferEnd_Facts, TransferEnd_Values
  (* Each block of a table stays valid in its holder's memory.             *)
  <1>4. ASSUME NEW o \in Owners, NEW p \in Positions, table[o][p].u # NONE
        PROVE  Mem(ogroup'[o]) \in valid'[table[o][p].u]
    <2> DEFINE u == table[o][p].u
    <2>0. OwnerTyped
      BY OwnerTypes
    <2>1. /\ u \in Units /\ Mem(ogroup[o]) \in valid[u] /\ Live(o)
          /\ table[o][p].k = "private" => alloc[u] = o
          /\ table[o][p].k # "private" => alloc[u] = "shared"
      BY <1>4, <2>0 DEF IndInv, Tables, OwnerTyped, Empty, Private, Shared
    <2>2. CASE o = t.o
      <3>1. ogroup'[o] = t.to
        BY <2>2, TransferEnd_Values
      <3>2. u \in TableUnits(t.o)
        BY <1>4, <2>2 DEF TableUnits
      <3>3. u \notin t.writes => Mem(t.to) \in valid[u]
        BY <2>1, <2>2, <3>2, TransferEnd_Facts, TransferEnd_Values
      <3> QED
        BY <1>2, <2>1, <3>1, <3>3
    <2>3. CASE o # t.o
      <3>1. ogroup'[o] = ogroup[o]
        BY <2>3, TransferEnd_Values
      <3>2. u \in t.writes => alloc[u] # t.o
        BY <2>1, <2>3, TransferEnd_Facts, TransferEnd_Values DEF NONE
      <3> QED
        BY <1>2, <1>3, <2>1, <3>1, <3>2
    <2> QED
      BY <2>2, <2>3
  <1> QED
    BY <1>2, <1>3, <1>4

LEMMA TransferEnd_MoveTyped ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  TypeInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. t.to \in Groups /\ t.o \in Owners
    BY TransferEnd_Facts, TransferEnd_Values
  <1>2. UnitTyped'
    <2>1. valid' \in [Units -> SUBSET Memories]
      BY <1>0, <1>1, TransferEnd_Values DEF UnitTyped, NewValid, Memories, Mem
    <2> QED
      BY <1>0, <2>1, TransferEnd_Values DEF UnitTyped, Reset
  <1>3. OwnerTyped'
    BY <1>0, TransferEnd_Values DEF OwnerTyped
  <1>4. PlaneTyped'
    <2>1. \A p \in Rows : GivenRows(t)[p].u \in AllUnits \cup {NONE}
      BY TransferEnd_Rows, TransferEnd_Entries, TransferEnd_Values DEF Given, NoEntry
    <2>2. DOMAIN GivenRows(t) = Rows
      BY DEF GivenRows
    <2> QED
      BY <1>0, <1>1, <2>1, <2>2, TransferEnd_Values DEF PlaneTyped
  <1>5. WorkTyped'
    BY <1>0, TransferEnd_Values DEF WorkTyped
  <1>6. RequestTyped'
    <2>1. \A h \in Given(t) : h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows /\ h.u \in AllUnits
      BY <1>1, TransferEnd_Entries, TransferEnd_Values DEF Given
    <2> QED
      BY <1>0, <2>1, TransferEnd_Values DEF RequestTyped
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5, <1>6, TypeInvNext

LEMMA TransferEnd_MoveUnits ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
         /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
  <1>1. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
    BY TransferEnd_Values, FreeReclaim, UnitTypes DEF IndInv
  <1>2. AllocOwner' /\ SharedIff'
    BY TransferEnd_Values DEF IndInv, AllocOwner, SharedIff, NONE
  <1>3. SharedCached'
    BY TransferEnd_Values DEF IndInv, SharedCached, NONE
  <1>4. CacheShared'
    BY TransferEnd_Values, TransferEnd_Kept DEF IndInv, CacheShared
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4

LEMMA TransferEnd_MoveOwners ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  Tables' /\ OwnersOk' /\ DeferredOk'
  <1>1. /\ t.o \in Owners /\ t.to \in Groups /\ GroupEpoch[t.to] = OwnerEpoch[t.o]
        /\ t.slot \in Slots /\ SlotGroup[t.slot] = t.to
        /\ \A d \in deferred : t.slot \notin d.units
    BY TransferEnd_Facts, TransferEnd_Values
  <1>2. Tables'
    BY TransferEnd_Values, TransferEnd_Kept, TransferEnd_Valid DEF IndInv, Tables, Live
  <1>3. OwnersOk'
    <2>1. alloc'[t.slot] = alloc[t.slot] /\ alloc[t.slot] = t.o
      <3>1. t.slot \in AllUnits /\ oslot'[t.o] = t.slot
        BY <1>1, TransferEnd_Values DEF AllUnits
      <3> QED
        BY <1>1, <3>1, TransferEnd_Kept, TransferEnd_Facts, TransferEnd_Values
    <2> QED
      BY <1>1, <2>1, TransferEnd_Values, TransferEnd_Kept DEF IndInv, OwnersOk, Live
  <1>4. DeferredOk'
    BY <1>1, TransferEnd_Values, TransferEnd_Kept DEF IndInv, DeferredOk
  <1> QED
    BY <1>2, <1>3, <1>4

LEMMA TransferEnd_MoveWork ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  CopiesOk' /\ TasksOk' /\ TrialVisibility' /\ NoBad'
  <1>0. /\ t.kind = "transfer" /\ t.o \in Owners /\ Live(t.o)
        /\ ~\E c \in copies : c.o = t.o
        /\ \A x \in tasks : (x.kind = "transfer" /\ (x.o = t.o \/ x.slot = t.slot)) => x = t
        /\ \A x \in tasks :
             (x.kind = "transfer" /\ x # t /\ Mem(x.to) = Mem(t.to)) => x.writes \cap t.writes = {}
    BY TransferEnd_Facts, TransferEnd_Values DEF Live
  <1>1. \A o \in Owners : Moving(o)' => Moving(o)
    BY TransferEnd_Values DEF Moving
  <1>2. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies' PROVE CopyOk(c)'
      BY CopiesOkEach
    <2>2. c \in copies /\ CopyOk(c)
      BY TransferEnd_Values, CopiesOkEach DEF IndInv
    <2>3. c.o \in Owners => c.o # t.o
      BY <1>0, <2>2
    <2>4. (c.fill /\ c.kind = "load") =>
            /\ c.write \in Units
            /\ Mem(c.g) \notin valid'[c.write]
      <3>1. (c.fill /\ c.kind = "load") =>
              /\ c.write \in Units /\ Mem(c.g) \notin valid[c.write]
              /\ (Mem(t.to) = Mem(c.g) => c.write \notin t.writes)
        BY <1>0, <2>2, UnitTypes, WorkTypes DEF CopyOk, UnitTyped, WorkTyped, AllUnits
      <3> QED
        BY <3>1, TransferEnd_Valid
    <2> QED
      BY <1>1, <2>2, <2>3, <2>4, TransferEnd_Values, TransferEnd_Kept DEF CopyOk, Live, Mem
  <1>3. TasksOk'
    <2>1. SUFFICES ASSUME NEW x \in tasks' PROVE TaskOk(x)'
      BY TasksOkEach
    <2>2. x \in tasks /\ x # t /\ TaskOk(x)
      BY TransferEnd_Values, TasksOkEach DEF IndInv
    <2>3. x.kind = "transfer" =>
            /\ x.o # t.o /\ x.slot # t.slot
            /\ \A u \in x.writes : Mem(x.to) \notin valid'[u]
            /\ \A u \in TableUnits(x.o) \ x.writes : Mem(x.to) \in valid[u] => Mem(x.to) \in valid'[u]
      <3>1. x.kind = "transfer" => x.o # t.o /\ x.slot # t.slot
        BY <1>0, <2>2
      <3>2. x.kind = "transfer" => \A u \in x.writes : u \in Units /\ Mem(x.to) \notin valid[u]
        BY <2>2 DEF TaskOk
      <3>3. x.kind = "transfer" =>
              \A u \in x.writes : Mem(x.to) = Mem(t.to) => u \notin t.writes
        BY <1>0, <2>2
      <3>4. x.kind = "transfer" =>
              \A u \in TableUnits(x.o) :
                 ~(/\ u \in t.writes /\ alloc[u] = t.o
                   /\ ~\E d \in deferred : d.kind = "commit" /\ u \in d.units)
        <4>1. x.kind = "transfer" => \A u \in TableUnits(x.o) : alloc[u] = x.o \/ alloc[u] = "shared"
          BY <2>2, OwnerTypes DEF IndInv, Tables, TableUnits, TaskOk, OwnerTyped, Empty, Private,
                                   Shared
        <4> QED
          BY <1>0, <3>1, <4>1, <2>2 DEF TaskOk, NONE
      <3>5. x.kind = "transfer" => TableUnits(x.o) \subseteq Units
        BY <2>2, OwnerTypes DEF TaskOk, TableUnits, OwnerTyped, Empty, Private, Shared
      <3> QED
        BY <3>1, <3>2, <3>3, <3>4, <3>5, TransferEnd_Valid
    <2>4. \A u \in x.units \cup {x.slot, x.h} : u # NONE => alloc'[u] = alloc[u]
      BY TransferEnd_Kept
    <2>5. /\ x.kind = "transfer" => x.writes \subseteq x.units /\ x.slot # NONE
          /\ x.kind = "prefetch" => x.h # NONE
      BY <2>2 DEF TaskOk
    <2>6. x.kind = "transfer" => oslot'[x.o] = oslot[x.o] /\ \A o \in Owners : oslot'[o] # x.slot
      BY <2>2, <2>3, TransferEnd_Values DEF TaskOk
    <2> QED
      BY <1>1, <2>2, <2>3, <2>4, <2>5, <2>6, TransferEnd_Values DEF TaskOk, Live, TableUnits, Mem
  <1>4. TrialVisibility' /\ NoBad'
    BY TransferEnd_Values DEF IndInv, TrialVisibility, MadeBy, NoBad
  <1> QED
    BY <1>2, <1>3, <1>4

(* The executors' tables and the completions after the step.               *)
LEMMA TransferEnd_Plane ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  /\ \A g \in Groups, o \in Owners, p \in Rows :
              dbt'[g][o][p] = IF Eager /\ g = t.to /\ o = t.o THEN GivenRows(t)[p] ELSE dbt[g][o][p]
         /\ \A h \in handoff' : h \in handoff \/ h \in Given(t)
         /\ \A h \in handoff : h \in handoff'
         /\ \A p \in Rows : GivenRows(t)[p].u # NONE =>
              \E h \in Given(t) : h.p = p /\ GivenRows(t)[p] = [u |-> h.u, w |-> h.w]
  <1>0. PlaneTyped /\ t.o \in Owners /\ t.to \in Groups
    BY PlaneTypes, TransferEnd_Facts, TransferEnd_Values
  <1>1. \A g \in Groups, o \in Owners, p \in Rows :
          dbt'[g][o][p] = IF Eager /\ g = t.to /\ o = t.o THEN GivenRows(t)[p] ELSE dbt[g][o][p]
    BY <1>0, TransferEnd_Values DEF PlaneTyped
  <1>2. \A p \in Rows : GivenRows(t)[p].u # NONE =>
          \E h \in Given(t) : h.p = p /\ GivenRows(t)[p] = [u |-> h.u, w |-> h.w]
    BY TransferEnd_Rows DEF NoEntry
  <1> QED
    BY <1>1, <1>2, TransferEnd_Values

LEMMA TransferEnd_MovePlane ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
  <1>0. /\ t.kind = "transfer" /\ t.o \in Owners /\ t.to \in Groups /\ t.slot \in Slots
        /\ GroupEpoch[t.to] = OwnerEpoch[t.o] /\ OwnerEpoch[t.o] \notin lost
        /\ \A g \in Groups, p \in Rows : dbt[g][t.o][p].u = NONE
        /\ \A h \in handoff : h.o # t.o
        /\ \A x \in tasks : (x.kind = "transfer" /\ x.o = t.o) => x = t
        /\ ~\E d \in deferred : d.kind = "transfer" /\ d.o = t.o
    BY TransferEnd_Facts, TransferEnd_Values DEF IndInv, OwnersOk, Live
  <1>1. \A h \in Given(t) :
          /\ h.g = t.to /\ h.o = t.o /\ h.p \in Rows /\ h.u \in AllUnits
          /\ h.p = SLOT => h.u = t.slot
          /\ h.p \in Positions => table[t.o][h.p].u = h.u
          /\ (h.p \in Positions /\ h.w) =>
               /\ table[t.o][h.p].k = "private"
               /\ ~\E d \in deferred : d.kind = "commit" /\ h.u \in d.units
          /\ \A r \in requests : ~Names(r, h)
    BY <1>0, TransferEnd_Entries DEF Given
  <1>2. /\ Live(t.o)' /\ ogroup'[t.o] = t.to /\ oslot'[t.o] = t.slot
        /\ \A o \in Owners : o # t.o => ogroup'[o] = ogroup[o] /\ oslot'[o] = oslot[o]
    BY <1>0, TransferEnd_Values DEF Live
  <1>3. MovingUnmapped'
    <2>1. \A o \in Owners : Moving(o)' => Moving(o) /\ o # t.o
      BY <1>0, TransferEnd_Values DEF Moving
    <2> QED
      BY <1>0, <1>1, <2>1, TransferEnd_Plane, TransferEnd_Values DEF IndInv, MovingUnmapped, Moving
  <1>4. LostOk'
    BY <1>0, <1>1, TransferEnd_Plane, TransferEnd_Values DEF IndInv, LostOk
  <1>5. TablesMapped'
    <2>1. SUFFICES ASSUME NEW g \in Groups, NEW o \in Owners, NEW p \in Rows, dbt'[g][o][p].u # NONE
                   PROVE  (/\ Live(o) /\ ogroup[o] = g
                           /\ HoldsAt(o, p, dbt[g][o][p].u)
                           /\ (p \in Positions /\ dbt[g][o][p].w) =>
                                /\ table[o][p].k = "private"
                                /\ ~\E d \in deferred :
                                      d.kind = "commit" /\ dbt[g][o][p].u \in d.units)'
      BY DEF TablesMapped
    <2>2. CASE Eager /\ g = t.to /\ o = t.o
      <3>1. PICK h \in Given(t) : h.p = p /\ dbt'[g][o][p] = [u |-> h.u, w |-> h.w]
        BY <2>1, <2>2, TransferEnd_Plane
      <3>2. /\ Live(o)' /\ ogroup'[o] = g
            /\ HoldsAt(o, p, h.u)'
            /\ (p \in Positions /\ h.w) =>
                 /\ table'[o][p].k = "private"
                 /\ ~\E d \in deferred' : d.kind = "commit" /\ h.u \in d.units
        <4>1. h.p \in Rows /\ (h.p = SLOT => h.u = t.slot)
              /\ (h.p \in Positions => table[t.o][h.p].u = h.u)
          BY <1>1
        <4> QED
          BY <1>1, <1>2, <2>2, <3>1, <4>1, TransferEnd_Values DEF HoldsAt, Rows
      <3> QED
        BY <3>1, <3>2
    <2>3. CASE ~(Eager /\ g = t.to /\ o = t.o)
      <3>1. dbt'[g][o][p] = dbt[g][o][p] /\ o # t.o
        BY <1>0, <2>1, <2>3, TransferEnd_Plane
      <3>2. /\ Live(o) /\ ogroup[o] = g /\ HoldsAt(o, p, dbt[g][o][p].u)
            /\ (p \in Positions /\ dbt[g][o][p].w) =>
                 /\ table[o][p].k = "private"
                 /\ ~\E d \in deferred : d.kind = "commit" /\ dbt[g][o][p].u \in d.units
        BY <2>1, <3>1 DEF IndInv, TablesMapped
      <3> QED
        BY <1>2, <3>1, <3>2, TransferEnd_Values DEF HoldsAt, Live
    <2> QED
      BY <2>2, <2>3
  <1>6. HandoffOk'
    <2>1. SUFFICES ASSUME NEW h \in handoff'
                   PROVE  (/\ h.o \in Owners /\ h.g \in Groups /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
                           /\ (~\E r \in requests : Names(r, h)) =>
                                /\ Live(h.o) /\ ogroup[h.o] = h.g
                                /\ HoldsAt(h.o, h.p, h.u)
                                /\ (h.p \in Positions /\ h.w) =>
                                     /\ table[h.o][h.p].k = "private"
                                     /\ ~\E d \in deferred : d.kind = "commit" /\ h.u \in d.units)'
      BY DEF HandoffOk
    <2>2. CASE h \in handoff
      <3>1. h.o # t.o
        BY <1>0, <2>2
      <3>2. /\ h.o \in Owners /\ h.g \in Groups /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
            /\ (~\E r \in requests : Names(r, h)) =>
                 /\ Live(h.o) /\ ogroup[h.o] = h.g
                 /\ HoldsAt(h.o, h.p, h.u)
                 /\ (h.p \in Positions /\ h.w) =>
                      /\ table[h.o][h.p].k = "private"
                      /\ ~\E d \in deferred : d.kind = "commit" /\ h.u \in d.units
        BY <2>2 DEF IndInv, HandoffOk
      <3> QED
        BY <1>2, <3>1, <3>2, TransferEnd_Values DEF HoldsAt, Live
    <2>3. CASE h \in Given(t)
      <3>1. /\ h.o = t.o /\ h.g = t.to /\ h.o \in Owners /\ h.g \in Groups
            /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
        BY <1>0, <1>1, <2>3
      <3>2. /\ Live(h.o)' /\ ogroup'[h.o] = h.g
            /\ HoldsAt(h.o, h.p, h.u)'
            /\ (h.p \in Positions /\ h.w) =>
                 /\ table'[h.o][h.p].k = "private"
                 /\ ~\E d \in deferred' : d.kind = "commit" /\ h.u \in d.units
        BY <1>1, <1>2, <2>3, <3>1, TransferEnd_Values DEF HoldsAt, Rows
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <2>2, <2>3, TransferEnd_Plane
  <1>7. RequestsOk'
    <2>0. RequestTyped
      BY RequestTypes
    <2>1. \A r \in requests : r.kind # "finish" =>
            \A g \in Groups, p \in Rows :
              (dbt'[g][r.o][p].u # NONE /\ (TakesRow(r, r.o, p) \/ CommitsRow(r, r.o, p))) =>
                  /\ g \in r.sg
                  /\ g \in r.by => CommitsRow(r, r.o, p) /\ ~dbt'[g][r.o][p].w
      <3> SUFFICES ASSUME NEW r \in requests, r.kind # "finish", NEW g \in Groups, NEW p \in Rows,
                          dbt'[g][r.o][p].u # NONE, TakesRow(r, r.o, p) \/ CommitsRow(r, r.o, p)
                   PROVE  /\ g \in r.sg
                          /\ g \in r.by => CommitsRow(r, r.o, p) /\ ~dbt'[g][r.o][p].w
        OBVIOUS
      <3>1. r.o \in Owners
        BY <2>0 DEF RequestTyped
      <3>2. CASE Eager /\ g = t.to /\ r.o = t.o
        <4>1. PICK h \in Given(t) : h.p = p /\ dbt'[g][r.o][p] = [u |-> h.u, w |-> h.w]
          BY <3>1, <3>2, TransferEnd_Plane
        <4>2. h.o = r.o /\ ~Names(r, h)
          BY <1>1, <3>2, <4>1
        <4> QED
          BY <4>1, <4>2, TransferEnd_Names
      <3>3. CASE ~(Eager /\ g = t.to /\ r.o = t.o)
        <4>1. dbt'[g][r.o][p] = dbt[g][r.o][p]
          BY <3>1, <3>3, TransferEnd_Plane
        <4> QED
          BY <3>1, <4>1 DEF IndInv, RequestsOk
      <3> QED
        BY <3>2, <3>3
    <2> QED
      BY <2>1, TransferEnd_Values DEF IndInv, RequestsOk
  <1> QED
    BY <1>3, <1>4, <1>5, <1>6, <1>7

LEMMA TransferEnd_MoveBatches ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  BatchesOk'
  <1>1. SUFFICES ASSUME NEW b \in inflight' PROVE BatchOk(b)'
    BY BatchesOkEach
  <1>2. b \in inflight /\ BatchOk(b)
    BY TransferEnd_Values, BatchesOkEach DEF IndInv
  <1>3. /\ t.kind = "transfer" /\ t.o \in Owners /\ Live(t.o)
        /\ t.writes \subseteq Units
        /\ \A u \in t.writes : u \in TableUnits(t.o)
        /\ \A o \in Owners : o # t.o => ogroup'[o] = ogroup[o] /\ oslot'[o] = oslot[o]
    <2>1. t.kind = "transfer" /\ t.o \in Owners /\ Live(t.o)
      BY TransferEnd_Values, TransferEnd_Facts DEF Live
    <2>2. /\ t.writes \subseteq t.units \cap Units
          /\ t.units = TableUnits(t.o) \cup {oslot[t.o]}
          /\ oslot[t.o] \in Slots
      BY <2>1, TransferEnd_Facts DEF IndInv, OwnersOk
    <2> QED
      BY <2>1, <2>2, TransferEnd_Values
  (* A unit the batch reads stays covered and held.                        *)
  <1>4. ASSUME NEW o \in Owners, NEW u, Covered(b, o, u)
        PROVE  /\ Covered(b, o, u)' /\ Held(u)'
               /\ o = t.o => \E d \in deferred : d.kind = "release" /\ u \in d.units
    <2>1. CASE /\ Live(o) /\ ogroup[o] = b.g
               /\ \E p \in Rows :
                    /\ HoldsAt(o, p, u)
                    /\ \A r \in requests :
                         TakesRow(r, o, p) => b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
               /\ \A d \in deferred : (d.kind = "transfer" /\ d.o = o) => d.after[b.g] >= b.seq
               /\ ~\E x \in tasks : x.kind = "transfer" /\ x.o = o
      <3>1. o # t.o
        BY <1>3, <2>1
      <3>2. \A p \in Rows : HoldsAt(o, p, u)' = HoldsAt(o, p, u)
        BY <1>3, <3>1, TransferEnd_Values DEF HoldsAt
      <3>3. Covered(b, o, u)'
        BY <1>3, <2>1, <3>1, <3>2, TransferEnd_Values DEF Covered, Live
      <3>4. Held(u)'
        BY <1>3, <2>1, <3>1, TransferEnd_Values DEF HoldsAt, Rows, HeldIn
      <3> QED
        BY <3>1, <3>3, <3>4
    <2>2. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units /\ d.after[b.g] >= b.seq
      BY <2>2, TransferEnd_Values DEF Covered, HeldIn
    <2> QED
      BY <1>4, <2>1, <2>2 DEF Covered
  <1>5. \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits /\ x[2] \notin free
                           /\ alloc[x[2]] \in {x[1], "shared"} /\ Covered(b, x[1], x[2])
    BY <1>2, RequestTypes DEF BatchOk, RequestTyped
  <1>6. \A x \in b.reads : x[2] \in Units => Mem(b.g) \in valid'[x[2]]
    <2> SUFFICES ASSUME NEW x \in b.reads, x[2] \in Units PROVE Mem(b.g) \in valid'[x[2]]
      OBVIOUS
    <2>1. Mem(b.g) \in valid[x[2]]
      BY <1>2 DEF BatchOk
    <2>2. ~(x[2] \in t.writes /\ alloc[x[2]] = t.o)
      <3> SUFFICES ASSUME x[2] \in t.writes, alloc[x[2]] = t.o PROVE FALSE
        OBVIOUS
      <3>1. x[1] = t.o
        BY <1>3, <1>5 DEF NONE
      <3>2. PICK d \in deferred : d.kind = "release" /\ x[2] \in d.units
        BY <1>4, <1>5, <3>1
      <3>3. PICK p \in Positions : table[t.o][p].u = x[2]
        BY <1>3 DEF TableUnits
      <3>4. table[t.o][p].k = "private"
        BY <1>3, <3>3, OwnerTypes DEF IndInv, Tables, OwnerTyped, Empty, Private, Shared, NONE
      <3> QED
        BY <1>3, <3>2, <3>3, <3>4 DEF IndInv, DeferredOk
    <2> QED
      BY <2>1, <2>2, TransferEnd_Valid
  <1>7. \A x \in b.reads : Covered(b, x[1], x[2])' /\ x[2] \notin free' /\ alloc'[x[2]] = alloc[x[2]]
    <2>1. \A x \in b.reads : Covered(b, x[1], x[2])' /\ Held(x[2])'
      BY <1>4, <1>5
    <2> QED
      BY <1>5, <2>1, TransferEnd_Values
  <1> QED
    BY <1>2, <1>5, <1>6, <1>7, TransferEnd_Values DEF BatchOk, Mem

LEMMA TransferEnd_Inv ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, TransferEnd(t, ok), UNCHANGED rare
  PROVE  IndInv'
  <1>1. CASE ok /\ ostate[t.o] = "live"
    BY <1>1, TransferEnd_MoveTyped, TransferEnd_MoveUnits, TransferEnd_MoveOwners,
       TransferEnd_MoveWork, TransferEnd_MovePlane, TransferEnd_MoveBatches DEF IndInv
  <1>2. CASE ~(ok /\ ostate[t.o] = "live")
    BY <1>2, TransferEnd_Fail
  <1> QED
    BY <1>1, <1>2

-----------------------------------------------------------------------------

THEOREM Step8 ==
    IndInv /\ (/\ \/ \E d \in deferred : TransferDue(d)
                  \/ \E t \in tasks, ok \in BOOLEAN : TransferEnd(t, ok)
               /\ UNCHANGED rare)
    => IndInv'
  BY TransferDue_Inv, TransferEnd_Inv
=============================================================================
