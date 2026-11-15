------------------------------- MODULE KvStep6 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for the buffer pool's taking of a stamped  *)
(* request: TakeRelease, TakeDetach, TakeCommit, TakeFinish and            *)
(* TakeTransfer.                                                           *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, KvFrame, TLAPS

USE Sizes, Design
(* A reader's row and the readers of an owner's class are read through.   *)
USE DEF ReaderRow, ReadersOf

(* Taking a request keeps CrossingsOk when it adds no crossing, leaves the *)
(* tasks, copies, groups, caches and validity alone, attaches no owner,     *)
(* starts no transfer of an owner whose crossing waits or runs, and keeps  *)
(* the allocation of what a crossing holds.                                *)
LEMMA CrossingsTake ==
  ASSUME CrossingsOk,
         \A d \in deferred' : d.kind = "replicate" => d \in deferred,
         UNCHANGED <<tasks, copies, ogroup, lost, cache, vis, valid, trial>>,
         \A o \in Owners : ostate[o] # "new" => ostate'[o] # "new",
         \A o \in Owners :
           (\/ \E d \in deferred : d.kind = "replicate" /\ d.o = o
            \/ \E t \in tasks : t.kind = "replicate" /\ t.o = o) => (Moving(o)' => Moving(o)),
         \A d \in deferred : d.kind = "replicate" => \A u \in d.units : alloc'[u] = alloc[u],
         \A t \in tasks : t.kind = "replicate" => \A u \in t.writes : alloc'[u] = alloc[u]
  PROVE  CrossingsOk'
  <1>1. \A d \in deferred' : CrossWaitOk(d)'
    <2> SUFFICES ASSUME NEW d \in deferred', d.kind = "replicate" PROVE CrossWaitOk(d)'
      BY DEF CrossWaitOk
    <2>1. d \in deferred /\ CrossWaitOk(d)
      BY CrossingsOkEach
    <2>2. ~Moving(d.o)'
      BY <2>1 DEF CrossWaitOk
    <2> QED
      BY <2>1, <2>2 DEF CrossWaitOk, MadeBy
  <1>2. \A t \in tasks' : CrossTaskOk(t)'
    <2> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate" PROVE CrossTaskOk(t)'
      BY DEF CrossTaskOk
    <2>1. CrossTaskOk(t)
      BY CrossingsOkEach
    <2>2. ~Moving(t.o)'
      BY <2>1 DEF CrossTaskOk
    <2> QED
      BY <2>1, <2>2 DEF CrossTaskOk, Mem
  <1> QED
    BY <1>1, <1>2, CrossingsOkEach

-----------------------------------------------------------------------------
(* TakeFinish                                                              *)

(* The end of a copy that waits for its stamps.                            *)
FinishWait(r) ==
    [kind |-> "finish", units |-> {}, after |-> r.after,
     o |-> NONE, p |-> NONE, k |-> NONE, c |-> r.c, ok |-> r.ok,
     g |-> NONE, to |-> NONE]

LEMMA TakeFinish_Inv ==
  ASSUME IndInv, NEW r \in requests, TakeFinish(r), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ r.kind = "finish" /\ r.sg \subseteq r.by
        /\ requests' = requests \ {r}
        /\ \/ UNCHANGED deferred
           \/ /\ \E c \in copies : c.id = r.c /\ GroupEpoch[c.g] = r.e
              /\ ~\E d \in deferred : d.kind = "finish" /\ d.c = r.c
              /\ deferred' = deferred \cup {FinishWait(r)}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks,
                       trial, member, valid, vis, dbt, launched, inflight, handoff, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF TakeFinish, Submittable, StampGroups, FinishWait, On, AllRules, rare
  <1>2. \A h \in handoff : ~Names(r, h)
    BY <1>1 DEF Names
  <1>3. CASE UNCHANGED deferred
    BY <1>1, <1>2, <1>3, TakeIdle DEF pool, rare
  <1>4. CASE /\ \E c \in copies : c.id = r.c /\ GroupEpoch[c.g] = r.e
             /\ ~\E d \in deferred : d.kind = "finish" /\ d.c = r.c
             /\ deferred' = deferred \cup {FinishWait(r)}
    <2> DEFINE nd == FinishWait(r)
    <2>1. /\ nd.kind = "finish" /\ nd.units = {} /\ nd.after = r.after /\ nd.c = r.c
          /\ nd.o = NONE /\ nd.p = NONE /\ nd.k = NONE /\ nd.g = NONE /\ nd.to = NONE
      BY DEF FinishWait
    <2>2. \A u : Held(u)' = Held(u)
      <3>1. \A u : (\E d \in deferred' : u \in d.units) = (\E d \in deferred : u \in d.units)
        BY <1>4, <2>1
      <3> QED
        BY <1>1, <3>1 DEF Held, HeldIn
    <2>3. \A o \in Owners : Moving(o)' = Moving(o)
      BY <1>1, <1>4, <2>1 DEF Moving
    <2>4. TypeInv'
      <3>1. WorkTyped'
        <4>1. r.after \in Stamps /\ r.c \in 0 .. MaxCopies
          BY RequestTypes DEF RequestTyped
        <4> QED
          BY <1>1, <1>4, <2>1, <4>1, WorkTypes DEF WorkTyped, NONE
      <3>2. RequestTyped'
        BY <1>1, RequestTypes DEF RequestTyped
      <3> QED
        BY <1>1, <3>1, <3>2, UnitTypes, OwnerTypes, PlaneTypes, UnitTypedFrame, OwnerTypedFrame,
           PlaneTypedFrame, TypeInvNext
    <2>5. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
          /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ TrialVisibility' /\ NoBad'
      BY <1>1, <2>2, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
         SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame, TrialVisibilityFrame,
         NoBadFrame DEF IndInv, FreeClean, NoLeak, InRegion
    <2>6. RequestsOk' /\ LostOk'
      BY <1>1, <1>2, TakeCommon
    <2>7. DeferredOk'
      <3>1. SUFFICES ASSUME NEW d \in deferred' PROVE DeferOk(d)'
        BY DeferredOkEach
      <3>2. CASE d = nd
        <4>1. PICK c \in copies : c.id = r.c /\ GroupEpoch[c.g] = r.e
          BY <1>4
        <4>2. /\ c.seq > 0 => r.after[c.g] >= c.seq
              /\ c.seq = 0 => c.stopped
          BY <1>1, <4>1 DEF IndInv, RequestsOk
        <4> QED
          BY <1>1, <1>4, <2>1, <3>2, <4>1, <4>2 DEF DeferOk
      <3>3. CASE d # nd
        <4>1. d \in deferred /\ DeferOk(d)
          BY <1>4, <3>3, DeferredOkEach DEF IndInv
        <4>2. d.kind = "finish" => d.c # r.c
          BY <1>4, <4>1
        <4> QED
          BY <1>1, <1>4, <2>1, <4>1, <4>2 DEF DeferOk
      <3> QED
        BY <3>2, <3>3
    <2>8. CopiesOk' /\ TasksOk'
      <3>1. CopiesOk'
        <4>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
          BY <1>1, CopiesOkEach
        <4> QED
          BY <1>1, <2>3, CopiesOkEach DEF CopyOk, Live, Mem, IndInv
      <3>2. TasksOk'
        <4>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
          BY <1>1, TasksOkEach
        <4>2. TaskOk(t)
          BY TasksOkEach DEF IndInv
        <4> QED
          BY <1>1, <1>4, <2>1, <4>2 DEF TaskOk, Live, TableUnits, Mem
      <3> QED
        BY <3>1, <3>2
    <2>11. CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
      <3>1. CrossingsOk'
        BY <1>1, <1>4, <2>1, <2>3, CrossingsTake DEF IndInv
      <3>2. CopiesPlaced' /\ CrossingsPlaced'
        BY <1>1, TransfersFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
      <3> QED
        BY <3>1, <3>2
    <2>9. MovingUnmapped' /\ TablesMapped' /\ HandoffOk'
      <3>1. \A h \in handoff : (\E x \in requests' : Names(x, h)) = (\E x \in requests : Names(x, h))
        BY <1>1, <1>2
      <3>2. MovingUnmapped' /\ TablesMapped'
        BY <1>1, <1>4, <2>1 DEF IndInv, MovingUnmapped, TablesMapped, Live, HoldsAt
      <3>3. HandoffOk'
        BY <1>1, <1>4, <2>1, <3>1 DEF IndInv, HandoffOk, Live, HoldsAt
      <3> QED
        BY <3>2, <3>3
    <2>10. BatchesOk'
      <3>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
        BY <1>1, BatchesOkEach
      <3>2. BatchOk(b)
        BY BatchesOkEach DEF IndInv
      <3>3. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
        BY <1>1, <1>4, <2>1 DEF Covered, Live, HoldsAt
      <3>4. \A o \in Owners, p \in Positions :
              (\A x \in requests :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
              => \A x \in requests' :
                   FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
        BY <1>1, <1>2, TakeCommon
      <3> QED
        BY <1>1, <1>4, <2>1, <2>2, <3>2, <3>3, <3>4 DEF BatchOk, Mem, IndInv, FreeClean
    <2> QED
      BY <2>4, <2>5, <2>6, <2>7, <2>8, <2>9, <2>10, <2>11 DEF IndInv
  <1> QED
    BY <1>1, <1>3, <1>4

-----------------------------------------------------------------------------
(* TakeTransfer                                                            *)

(* A transfer that waits for its stamps.                                   *)
TransferWait(r) ==
    [kind |-> "transfer", units |-> {}, after |-> r.after,
     o |-> r.o, p |-> NONE, k |-> NONE, c |-> 0, ok |-> FALSE,
     g |-> NONE, to |-> r.to]

LEMMA TakeTransfer_Inv ==
  ASSUME IndInv, NEW r \in requests, TakeTransfer(r), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ r.kind = "transfer" /\ r.sg \subseteq r.by
        /\ requests' = requests \ {r}
        /\ handoff' = Kept(r)
        /\ \/ UNCHANGED deferred
           \/ /\ Live(r.o) /\ OwnerEpoch[r.o] = r.e /\ GroupEpoch[r.to] = r.e
              /\ ~Moving(r.o) /\ ~Replicating(r.o) /\ ~\E c \in copies : c.o = r.o
              /\ deferred' = deferred \cup {TransferWait(r)}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks,
                       trial, member, valid, vis, dbt, launched, inflight, alloc, sharedKey,
                       remote, lost, bad>>
    BY DEF TakeTransfer, Submittable, StampGroups, TransferWait, OwnSession, Live, On, AllRules,
           rare
  <1>2. handoff' \subseteq handoff /\ \A h \in handoff' : ~Names(r, h)
    BY <1>1 DEF Kept
  <1>3. CASE UNCHANGED deferred
    BY <1>1, <1>2, <1>3, TakeIdle DEF pool, rare
  <1>4. CASE /\ Live(r.o) /\ OwnerEpoch[r.o] = r.e /\ GroupEpoch[r.to] = r.e
             /\ ~Moving(r.o) /\ ~Replicating(r.o) /\ ~\E c \in copies : c.o = r.o
             /\ deferred' = deferred \cup {TransferWait(r)}
    <2> DEFINE nd == TransferWait(r)
    <2>0. r.o \in Owners /\ r.to \in Groups /\ r.after \in Stamps
      BY <1>1, RequestTypes DEF RequestTyped
    <2>1. /\ nd.kind = "transfer" /\ nd.units = {} /\ nd.after = r.after /\ nd.o = r.o
          /\ nd.to = r.to /\ nd.p = NONE /\ nd.k = NONE /\ nd.c = 0 /\ nd.g = NONE
      BY DEF TransferWait
    <2>2. \A u : Held(u)' = Held(u)
      <3>1. \A u : (\E d \in deferred' : u \in d.units) = (\E d \in deferred : u \in d.units)
        BY <1>4, <2>1
      <3> QED
        BY <1>1, <3>1 DEF Held, HeldIn
    <2>3. \A o \in Owners : Moving(o)' = (Moving(o) \/ o = r.o)
      BY <1>1, <1>4, <2>1 DEF Moving
    <2>4. /\ \A g \in Groups, p \in Rows : dbt[g][r.o][p].u = NONE
          /\ \A b \in inflight : b.g \in r.sg => b.g \in r.by
      BY <1>1, <1>4, <2>0, Submitted DEF TakesRow
    <2>5. \A h \in handoff' : h.o # r.o
      BY <1>1, <1>4 DEF Kept, Names
    <2>6. \A t \in tasks : t.kind = "transfer" => t.o # r.o
      BY <1>4 DEF Moving
    <2>7. \A h \in handoff' : (\E x \in requests' : Names(x, h)) = (\E x \in requests : Names(x, h))
      BY <1>1, <1>2
    <2>8. TypeInv'
      <3>1. WorkTyped'
        BY <1>1, <1>4, <2>0, <2>1, WorkTypes DEF WorkTyped, NONE, Stamps
      <3>2. RequestTyped'
        BY <1>1, <1>2, RequestTypes DEF RequestTyped
      <3> QED
        BY <1>1, <3>1, <3>2, UnitTypes, OwnerTypes, PlaneTypes, UnitTypedFrame, OwnerTypedFrame,
           PlaneTypedFrame, TypeInvNext
    <2>9. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
          /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ TrialVisibility' /\ NoBad'
      BY <1>1, <2>2, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
         SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame, TrialVisibilityFrame,
         NoBadFrame DEF IndInv, FreeClean, NoLeak, InRegion
    <2>10. RequestsOk' /\ LostOk'
      BY <1>1, <1>2, TakeCommon
    <2>11. DeferredOk'
      <3>1. SUFFICES ASSUME NEW d \in deferred' PROVE DeferOk(d)'
        BY DeferredOkEach
      <3>2. CASE d = nd
        BY <1>1, <1>4, <2>0, <2>1, <3>2 DEF DeferOk, Live
      <3>3. CASE d # nd
        <4>1. d \in deferred /\ DeferOk(d)
          BY <1>4, <3>3, DeferredOkEach DEF IndInv
        <4> QED
          BY <1>1, <1>4, <2>1, <4>1 DEF DeferOk
      <3> QED
        BY <3>2, <3>3
    <2>12. CopiesOk' /\ TasksOk'
      <3>1. CopiesOk'
        <4>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
          BY <1>1, CopiesOkEach
        <4>2. CopyOk(c) /\ c.o # r.o
          BY <1>4, CopiesOkEach DEF IndInv
        <4> QED
          BY <1>1, <2>3, <4>2 DEF CopyOk, Live, Mem
      <3>2. TasksOk'
        <4>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
          BY <1>1, TasksOkEach
        <4>2. TaskOk(t)
          BY TasksOkEach DEF IndInv
        <4> QED
          BY <1>1, <1>4, <2>1, <2>6, <4>2 DEF TaskOk, Live, TableUnits, Mem
      <3> QED
        BY <3>1, <3>2
    <2>16. CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
      <3>1. \A o \in Owners :
              (\/ \E d \in deferred : d.kind = "replicate" /\ d.o = o
               \/ \E t \in tasks : t.kind = "replicate" /\ t.o = o) => o # r.o
        BY <1>4 DEF Replicating
      <3>2. CrossingsOk'
        BY <1>1, <1>4, <2>1, <2>3, <3>1, CrossingsTake DEF IndInv
      <3>3. CopiesPlaced' /\ CrossingsPlaced'
        BY <1>1, TransfersFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
      <3> QED
        BY <3>2, <3>3
    <2>13. MovingUnmapped'
      <3>1. \A x \in deferred : x.kind = "transfer" => x.o # r.o
        BY <1>4 DEF Moving
      <3> QED
        BY <1>1, <1>2, <1>4, <2>1, <2>4, <2>5, <3>1 DEF IndInv, MovingUnmapped
    <2>14. TablesMapped' /\ HandoffOk'
      BY <1>1, <1>2, <1>4, <2>1, <2>7 DEF IndInv, TablesMapped, HandoffOk, Live, HoldsAt
    <2>15. BatchesOk'
      <3>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
        BY <1>1, BatchesOkEach
      <3>2. BatchOk(b)
        BY BatchesOkEach DEF IndInv
      <3>3. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
        <4> SUFFICES ASSUME NEW o \in Owners, NEW u, Covered(b, o, u) PROVE Covered(b, o, u)'
          OBVIOUS
        <4>1. CASE /\ Live(o) /\ (ogroup[o] = b.g \/ b.g \in ReadersOf(o))
                   /\ \E p \in Rows :
                        /\ HoldsAt(o, p, u)
                        /\ \A x \in requests :
                             TakesRow(x, o, p) =>
                                 b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
                   /\ \A d \in deferred : (d.kind = "transfer" /\ d.o = o) => d.after[b.g] >= b.seq
                   /\ ~\E t \in tasks : t.kind = "transfer" /\ t.o = o
          <5>1. o = r.o => r.after[b.g] >= b.seq
            BY <1>1, <1>4, <4>1 DEF TakesRow
          <5> QED
            BY <1>1, <1>4, <2>1, <4>1, <5>1 DEF Covered, Live, HoldsAt
        <4>2. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units /\ d.after[b.g] >= b.seq
          BY <1>4, <4>2 DEF Covered
        <4> QED
          BY <4>1, <4>2 DEF Covered
      <3>4. \A o \in Owners, p \in Positions :
              (\A x \in requests :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
              => \A x \in requests' :
                   FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
        BY <1>1, <1>2, TakeCommon
      <3> QED
        BY <1>1, <1>4, <2>1, <2>2, <3>2, <3>3, <3>4 DEF BatchOk, Mem, IndInv, FreeClean
    <2> QED
      BY <2>8, <2>9, <2>10, <2>11, <2>12, <2>13, <2>14, <2>15, <2>16 DEF IndInv
  <1> QED
    BY <1>1, <1>3, <1>4

-----------------------------------------------------------------------------
(* TakeCommit                                                              *)

(* A commit that waits for its stamps holding the block it was taken for.  *)
CommitWait(r) ==
    [kind |-> "commit", units |-> {table[r.o][r.p].u}, after |-> r.after,
     o |-> r.o, p |-> r.p, k |-> r.k, c |-> 0, ok |-> FALSE,
     g |-> ogroup[r.o], to |-> NONE]

LEMMA TakeCommit_Inv ==
  ASSUME IndInv, NEW r \in requests, TakeCommit(r), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ r.kind = "commit" /\ r.sg \subseteq r.by
        /\ requests' = requests \ {r}
        /\ handoff' = Kept(r)
        /\ \/ UNCHANGED deferred
           \/ /\ Live(r.o) /\ table[r.o][r.p].k = "private" /\ OwnerEpoch[r.o] = r.e
              /\ ~Moving(r.o)
              /\ ~\E d \in deferred : d.kind = "commit" /\ d.o = r.o /\ d.p = r.p
              /\ deferred' = deferred \cup {CommitWait(r)}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks,
                       trial, member, valid, vis, dbt, launched, inflight, alloc, sharedKey,
                       remote, lost, bad>>
    BY DEF TakeCommit, Submittable, StampGroups, CommitWait, OwnSession, Live, On, AllRules,
           rare
  <1>2. handoff' \subseteq handoff /\ \A h \in handoff' : ~Names(r, h)
    BY <1>1 DEF Kept
  <1>3. CASE UNCHANGED deferred
    BY <1>1, <1>2, <1>3, TakeIdle DEF pool, rare
  <1>4. CASE /\ Live(r.o) /\ table[r.o][r.p].k = "private" /\ OwnerEpoch[r.o] = r.e
             /\ ~Moving(r.o)
             /\ ~\E d \in deferred : d.kind = "commit" /\ d.o = r.o /\ d.p = r.p
             /\ deferred' = deferred \cup {CommitWait(r)}
    <2>1. PICK nd : nd = CommitWait(r)
      OBVIOUS
    <2> DEFINE u == table[r.o][r.p].u
    <2>2. /\ r.o \in Owners /\ r.p \in Positions /\ r.k \in Keys /\ r.after \in Stamps
          /\ ogroup[r.o] \in Groups /\ OwnerEpoch[r.o] \notin lost
      BY <1>1, <1>4, RequestTypes DEF RequestTyped, IndInv, OwnersOk
    <2>3. /\ nd.kind = "commit" /\ nd.units = {u} /\ nd.after = r.after /\ nd.o = r.o
          /\ nd.p = r.p /\ nd.k = r.k /\ nd.c = 0 /\ nd.g = ogroup[r.o] /\ nd.to = NONE
      BY <2>1 DEF CommitWait
    (* The block is private to the owner, at this position alone.          *)
    <2>4. /\ u \in Units /\ alloc[u] = r.o /\ Held(u)
          /\ \A o \in Owners, p \in Positions :
               table[o][p].u = u => o = r.o /\ p = r.p /\ table[o][p].k = "private"
          /\ \A c \in copies : c.write # u
          /\ \A e \in deferred : e.kind = "commit" => u \notin e.units
      <3>1. u \in Units /\ alloc[u] = r.o
        BY <1>4, <2>2 DEF IndInv, Tables
      <3>2. \A o \in Owners, p \in Positions :
              table[o][p].u = u => o = r.o /\ p = r.p /\ table[o][p].k = "private"
        <4> SUFFICES ASSUME NEW o \in Owners, NEW p \in Positions, table[o][p].u = u
                     PROVE  o = r.o /\ p = r.p /\ table[o][p].k = "private"
          OBVIOUS
        <4>1. table[o][p].k = "private"
          BY <2>2, <3>1, OwnerTypes DEF IndInv, Tables, OwnerTyped, Empty, Private, Shared, NONE
        <4>2. o = r.o
          BY <3>1, <4>1 DEF IndInv, Tables
        <4> QED
          BY <1>4, <2>2, <4>1, <4>2 DEF IndInv, Tables
      <3>3. \A c \in copies : c.write # u
        <4> SUFFICES ASSUME NEW c \in copies, c.write = u PROVE FALSE
          OBVIOUS
        <4>1. CopyOk(c)
          BY CopiesOkEach DEF IndInv
        <4>2. ~c.fill
          <5>1. c.fill /\ c.kind = "load" => alloc[cache[c.k]] = "shared"
            BY <4>1, UnitTypes, WorkTypes DEF CopyOk, IndInv, CacheShared, SharedIff, UnitTyped,
                                              WorkTyped, AllUnits
          <5> QED
            BY <2>2, <3>1, <4>1, <5>1 DEF CopyOk
        <4> QED
          BY <2>2, <4>1, <4>2 DEF CopyOk
      <3>4. \A e \in deferred : e.kind = "commit" => u \notin e.units
        BY <1>4, <2>2, <3>2 DEF IndInv, DeferredOk
      <3>5. Held(u)
        BY <2>2 DEF Held, HeldIn
      <3> QED
        BY <3>1, <3>2, <3>3, <3>4, <3>5
    <2>5. \A x : Held(x)' = Held(x)
      <3>1. \A x : (\E d \in deferred' : x \in d.units) = ((\E d \in deferred : x \in d.units) \/ x = u)
        BY <1>4, <2>1, <2>3
      <3>2. \A x : HeldIn(x, table, oslot, cache, hcache, deferred', copies, tasks)
                     = (HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks) \/ x = u)
        BY <3>1 DEF HeldIn
      <3>3. HeldIn(u, table, oslot, cache, hcache, deferred, copies, tasks)
        BY <2>2 DEF HeldIn
      <3>4. \A x : Held(x)' = HeldIn(x, table, oslot, cache, hcache, deferred', copies, tasks)
        BY <1>1 DEF Held
      <3>5. \A x : Held(x) = HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
        BY DEF Held
      <3> SUFFICES ASSUME NEW x PROVE Held(x)' = Held(x)
        OBVIOUS
      <3>6. CASE x = u
        BY <3>1, <3>3, <3>4, <3>5, <3>6 DEF HeldIn
      <3>7. CASE x # u
        <4>1. HeldIn(x, table, oslot, cache, hcache, deferred', copies, tasks)
                = HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
          BY <3>1, <3>7 DEF HeldIn
        <4> QED
          BY <3>4, <3>5, <4>1
      <3> QED
        BY <3>6, <3>7
    <2>6. \A o \in Owners : Moving(o)' = Moving(o)
      BY <1>1, <1>4, <2>1, <2>3 DEF Moving
    <2>7. \A h \in handoff' : (\E x \in requests' : Names(x, h)) = (\E x \in requests : Names(x, h))
      BY <1>1, <1>2
    <2>8. TypeInv'
      <3>1. WorkTyped'
        BY <1>1, <1>4, <2>1, <2>2, <2>3, <2>4, WorkTypes DEF WorkTyped, NONE, AllUnits
      <3>2. RequestTyped'
        BY <1>1, <1>2, RequestTypes DEF RequestTyped
      <3> QED
        BY <1>1, <3>1, <3>2, UnitTypes, OwnerTypes, PlaneTypes, UnitTypedFrame, OwnerTypedFrame,
           PlaneTypedFrame, TypeInvNext
    <2>9. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
          /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ TrialVisibility' /\ NoBad'
      BY <1>1, <2>5, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
         SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame, TrialVisibilityFrame,
         NoBadFrame DEF IndInv, FreeClean, NoLeak, InRegion
    <2>10. RequestsOk' /\ LostOk'
      BY <1>1, <1>2, TakeCommon
    <2>11. DeferredOk'
      <3>1. SUFFICES ASSUME NEW d \in deferred' PROVE DeferOk(d)'
        BY DeferredOkEach
      <3>2. CASE d = nd
        <4>1. \A e \in deferred' : (e.kind = "commit" /\ e.units = {u}) => e = nd
          BY <1>4, <2>1, <2>4
        <4> QED
          BY <1>1, <1>4, <2>1, <2>2, <2>3, <2>4, <3>2, <4>1 DEF DeferOk
      <3>3. CASE d # nd
        <4>1. d \in deferred /\ DeferOk(d)
          BY <1>4, <2>1, <3>3, DeferredOkEach DEF IndInv
        <4>2. d.kind = "commit" => d.units # nd.units
          BY <2>3, <2>4, <4>1
        <4>3. /\ \A e \in deferred' : e \in deferred \/ e = nd
              /\ nd.kind = "commit"
          BY <1>4, <2>1, <2>3
        <4> QED
          BY <1>1, <4>1, <4>2, <4>3 DEF DeferOk
      <3> QED
        BY <3>2, <3>3
    <2>12. CopiesOk' /\ TasksOk'
      <3>1. CopiesOk'
        <4>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
          BY <1>1, CopiesOkEach
        <4> QED
          BY <1>1, <2>6, CopiesOkEach DEF CopyOk, Live, Mem, IndInv
      <3>2. TasksOk'
        <4>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
          BY <1>1, TasksOkEach
        <4>2. TaskOk(t)
          BY TasksOkEach DEF IndInv
        <4> QED
          BY <1>1, <1>4, <2>1, <2>3, <2>4, <4>2 DEF TaskOk, Live, TableUnits, Mem
      <3> QED
        BY <3>1, <3>2
    <2>13. MovingUnmapped'
      BY <1>1, <1>2, <1>4, <2>1, <2>3 DEF IndInv, MovingUnmapped
    <2>17. CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
      <3>1. CrossingsOk'
        BY <1>1, <1>4, <2>1, <2>3, <2>6, CrossingsTake DEF IndInv
      <3>2. CopiesPlaced' /\ CrossingsPlaced'
        BY <1>1, TransfersFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
      <3> QED
        BY <3>1, <3>2
    (* No executor may write the block and no completion kept gives the    *)
    (* right to: the request named every such row.                         *)
    <2>14. TablesMapped'
      <3>1. \A g \in Groups, p \in Rows :
              (dbt[g][r.o][p].u = u /\ p \in Positions /\ dbt[g][r.o][p].w) => FALSE
        <4> SUFFICES ASSUME NEW g \in Groups, NEW p \in Rows, dbt[g][r.o][p].u = u,
                            p \in Positions, dbt[g][r.o][p].w
                     PROVE  FALSE
          OBVIOUS
        <4>1. table[r.o][p].u = u
          BY <2>2, <2>4 DEF IndInv, TablesMapped, HoldsAt
        <4>2. p = r.p
          BY <2>2, <2>4, <4>1
        <4> QED
          BY <1>1, <1>4, <2>2, <2>4, <4>2, Submitted DEF FreezesRow
      <3>2. \A g \in Groups, o \in Owners, p \in Rows :
              (dbt[g][o][p].u = u /\ dbt[g][o][p].u # NONE) => o = r.o
        BY <2>4, OwnerTypes DEF IndInv, TablesMapped, HoldsAt, Rows, OwnerTyped
      <3> QED
        BY <1>1, <1>4, <2>1, <2>3, <3>1, <3>2 DEF IndInv, TablesMapped, Live, HoldsAt
    <2>15. HandoffOk'
      <3>1. \A h \in handoff' :
              ((~\E x \in requests : Names(x, h)) /\ h.p \in Positions /\ h.w) => h.u # u
        <4> SUFFICES ASSUME NEW h \in handoff', ~\E x \in requests : Names(x, h),
                            h.p \in Positions, h.w, h.u = u
                     PROVE  FALSE
          OBVIOUS
        <4>1. h \in handoff /\ h.o \in Owners /\ HoldsAt(h.o, h.p, u)
          BY <1>2, <2>4, RequestTypes DEF IndInv, HandoffOk, RequestTyped
        <4>2. h.o = r.o /\ h.p = r.p
          BY <2>4, <4>1 DEF HoldsAt
        <4> QED
          BY <1>1, <1>2, <1>4, <4>2 DEF Names
      <3> QED
        BY <1>1, <1>2, <1>4, <2>1, <2>3, <2>7, <3>1 DEF IndInv, HandoffOk, Live, HoldsAt
    <2>16. BatchesOk'
      <3>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
        BY <1>1, BatchesOkEach
      <3>2. BatchOk(b)
        BY BatchesOkEach DEF IndInv
      <3>3. \A o \in Owners : \A x : Covered(b, o, x) => Covered(b, o, x)'
        BY <1>1, <1>4, <2>1, <2>3 DEF Covered, Live, HoldsAt
      <3>4. \A o \in Owners, p \in Positions :
              (\A x \in requests :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
              => \A x \in requests' :
                   FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
        BY <1>1, <1>2, TakeCommon
      (* A batch that writes the block was covered by the request.         *)
      <3>5. \A y \in b.writes : y[2] = u => r.after[b.g] >= b.seq
        <4> SUFFICES ASSUME NEW y \in b.writes, y[2] = u PROVE r.after[b.g] >= b.seq
          OBVIOUS
        <4>1. y[1] = r.o
          BY <2>4, <3>2 DEF BatchOk
        <4>2. FreezesRow(r, y[1], r.p) /\ table[y[1]][r.p].u = y[2]
          BY <1>1, <1>4, <4>1 DEF FreezesRow
        <4> QED
          BY <1>1, <2>2, <3>2, <4>2 DEF BatchOk
      <3> QED
        BY <1>1, <1>4, <2>1, <2>3, <2>5, <3>2, <3>3, <3>4, <3>5 DEF BatchOk, Mem, IndInv,
                                                                      FreeClean
    <2> QED
      BY <2>8, <2>9, <2>10, <2>11, <2>12, <2>13, <2>14, <2>15, <2>16, <2>17 DEF IndInv
  <1> QED
    BY <1>1, <1>3, <1>4

-----------------------------------------------------------------------------
(* TakeRelease                                                             *)

(* A release that waits for its stamps, holding what it takes away.        *)
ReleaseWait(units, after) ==
    [kind |-> "release", units |-> units, after |-> after,
     o |-> NONE, p |-> NONE, k |-> NONE, c |-> 0, ok |-> FALSE,
     g |-> NONE, to |-> NONE]

(* The blocks a release takes away and the owner's table without them.     *)
RelUnits(r) == {table[r.o][p].u : p \in r.ps} \ {NONE}
RelTable(r) ==
    [table EXCEPT ![r.o] = [p \in Positions |-> IF p \in r.ps THEN Empty ELSE table[r.o][p]]]

LEMMA TakeRelease_Effect ==
  ASSUME NEW r \in requests, TakeRelease(r)
  PROVE  /\ r.kind = "release" /\ r.sg \subseteq r.by
         /\ requests' = requests \ {r}
         /\ handoff' = Kept(r)
         /\ UNCHANGED <<dbt, launched, inflight>>
         /\ \/ UNCHANGED <<pool, alloc, sharedKey>>
            \/ /\ Live(r.o) /\ OwnerEpoch[r.o] = r.e /\ ~Moving(r.o)
               /\ \A p \in r.ps : table[r.o][p] # Empty
               /\ table' = RelTable(r) /\ oslot' = oslot
               /\ \/ /\ ~Reached(r.after) /\ RelUnits(r) # {}
                     /\ deferred' = deferred \cup {ReleaseWait(RelUnits(r), r.after)}
                     /\ UNCHANGED <<free, alloc, sharedKey>>
                  \/ /\ ~(~Reached(r.after) /\ RelUnits(r) # {})
                     /\ DropWith(RelUnits(r), RelTable(r), oslot, cache, hcache, deferred, copies,
                                 tasks)
                     /\ UNCHANGED deferred
               /\ UNCHANGED <<ostate, ogroup, cache, hcache, copies, ncopy, tasks, trial, member,
                              valid, vis>>
  BY DEF TakeRelease, Submittable, StampGroups, ReleaseUnits, ReleaseWait, RelUnits, RelTable,
         OwnSession, Live, On, AllRules

(* What a release taken with effect says of its owner's table, the        *)
(* executors' tables and the completions it keeps.                         *)
LEMMA TakeRelease_Facts ==
  ASSUME IndInv, NEW r \in requests, r.kind = "release", r.sg \subseteq r.by,
         Live(r.o), OwnerEpoch[r.o] = r.e, ~Moving(r.o), \A p \in r.ps : table[r.o][p] # Empty
  PROVE  /\ r.o \in Owners /\ r.ps \subseteq Positions /\ r.after \in Stamps
         /\ \A x \in Owners, p \in Positions :
              RelTable(r)[x][p] = IF x = r.o /\ p \in r.ps THEN Empty ELSE table[x][p]
         /\ RelUnits(r) = {table[r.o][p].u : p \in r.ps}
         /\ RelUnits(r) \subseteq Units
         /\ \A u \in RelUnits(r) : alloc[u] = r.o \/ alloc[u] = "shared"
         (* A private block taken away is held by no other table entry.    *)
         /\ \A x \in Owners, p \in Positions :
              (table[x][p].k = "private" /\ table[x][p].u \in RelUnits(r)) =>
                  x = r.o /\ p \in r.ps
         /\ \A g \in Groups, p \in r.ps : dbt[g][r.o][p].u = NONE
         /\ \A h \in Kept(r) : ~(h.o = r.o /\ h.p \in r.ps)
         /\ \A t \in tasks : t.kind = "transfer" => t.o # r.o
         /\ \A d \in deferred : d.kind = "transfer" => d.o # r.o
         /\ \A b \in inflight, p \in r.ps :
              (\A x \in requests :
                 TakesRow(x, r.o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
              => r.after[b.g] >= b.seq
  <1>0. OwnerTyped /\ RequestTyped
    BY OwnerTypes, RequestTypes
  <1>1. r.o \in Owners /\ r.ps \subseteq Positions /\ r.after \in Stamps
    BY <1>0 DEF RequestTyped
  <1>2. \A x \in Owners, p \in Positions :
          RelTable(r)[x][p] = IF x = r.o /\ p \in r.ps THEN Empty ELSE table[x][p]
    BY <1>0, <1>1 DEF RelTable, OwnerTyped
  <1>3. \A p \in r.ps : table[r.o][p].u \in Units /\ table[r.o][p].k # "none"
    BY <1>0, <1>1 DEF OwnerTyped, Empty, Private, Shared
  <1>4. RelUnits(r) = {table[r.o][p].u : p \in r.ps} /\ RelUnits(r) \subseteq Units
    BY <1>3 DEF RelUnits
  <1>5. \A p \in r.ps : alloc[table[r.o][p].u] = r.o \/ alloc[table[r.o][p].u] = "shared"
    BY <1>0, <1>1, <1>3 DEF IndInv, Tables, OwnerTyped, Empty, Private, Shared
  <1>6. \A x \in Owners, p \in Positions :
          (table[x][p].k = "private" /\ table[x][p].u \in RelUnits(r)) => x = r.o /\ p \in r.ps
    <2> SUFFICES ASSUME NEW x \in Owners, NEW p \in Positions, table[x][p].k = "private",
                        NEW q \in r.ps, table[x][p].u = table[r.o][q].u
                 PROVE  x = r.o /\ p \in r.ps
      BY <1>4
    <2>1. alloc[table[x][p].u] = x
      BY DEF IndInv, Tables
    <2>2. x = r.o
      BY <1>1, <1>5, <2>1 DEF IndInv, Tables
    <2>3. table[r.o][q].k # "shared"
      BY <1>1, <2>1, <2>2 DEF IndInv, Tables
    <2>4. table[r.o][q].k = "private"
      BY <1>0, <1>1, <1>3, <2>3 DEF OwnerTyped, Empty, Private, Shared
    <2> QED
      BY <1>1, <2>2, <2>4 DEF IndInv, Tables
  <1>7. \A g \in Groups, p \in r.ps : dbt[g][r.o][p].u = NONE
    BY <1>1, Submitted DEF TakesRow, Rows
  <1>8. \A h \in Kept(r) : ~(h.o = r.o /\ h.p \in r.ps)
    BY DEF Kept, Names
  <1>9. /\ \A t \in tasks : t.kind = "transfer" => t.o # r.o
        /\ \A d \in deferred : d.kind = "transfer" => d.o # r.o
    BY DEF Moving
  <1>10. \A b \in inflight, p \in r.ps :
           (\A x \in requests :
              TakesRow(x, r.o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
           => r.after[b.g] >= b.seq
    BY DEF TakesRow
  <1> QED
    BY <1>1, <1>2, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10

(* The release is taken with effect: what it takes away waits for its      *)
(* stamps, or is let go now when they are reached.                         *)
LEMMA TakeRelease_Taken ==
  ASSUME IndInv, NEW r \in requests, TakeRelease(r), UNCHANGED rare,
         Live(r.o), OwnerEpoch[r.o] = r.e, ~Moving(r.o), \A p \in r.ps : table[r.o][p] # Empty,
         table' = RelTable(r), oslot' = oslot,
         UNCHANGED <<ostate, ogroup, cache, hcache, copies, ncopy, tasks, trial, member, valid, vis>>,
         \/ /\ ~Reached(r.after) /\ RelUnits(r) # {}
            /\ deferred' = deferred \cup {ReleaseWait(RelUnits(r), r.after)}
            /\ UNCHANGED <<free, alloc, sharedKey>>
         \/ /\ ~(~Reached(r.after) /\ RelUnits(r) # {})
            /\ DropWith(RelUnits(r), RelTable(r), oslot, cache, hcache, deferred, copies, tasks)
            /\ UNCHANGED deferred
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. /\ r.kind = "release" /\ r.sg \subseteq r.by
        /\ requests' = requests \ {r}
        /\ handoff' = Kept(r)
        /\ UNCHANGED <<dbt, launched, inflight, remote, lost, bad>>
    BY TakeRelease_Effect DEF rare
  <1>2. PICK U : U = RelUnits(r)
    OBVIOUS
  <1>3. PICK NR : NR = ReleaseWait(U, r.after)
    OBVIOUS
  <1>4. /\ r.o \in Owners /\ r.ps \subseteq Positions /\ r.after \in Stamps
        /\ \A x \in Owners, p \in Positions :
             table'[x][p] = IF x = r.o /\ p \in r.ps THEN Empty ELSE table[x][p]
        /\ U = {table[r.o][p].u : p \in r.ps} /\ U \subseteq Units
        /\ \A u \in U : alloc[u] = r.o \/ alloc[u] = "shared"
        /\ \A x \in Owners, p \in Positions :
             (table[x][p].k = "private" /\ table[x][p].u \in U) => x = r.o /\ p \in r.ps
        /\ \A g \in Groups, p \in r.ps : dbt[g][r.o][p].u = NONE
        /\ \A h \in handoff' : ~(h.o = r.o /\ h.p \in r.ps)
        /\ \A t \in tasks : t.kind = "transfer" => t.o # r.o
        /\ \A b \in inflight, p \in r.ps :
             (\A x \in requests :
                TakesRow(x, r.o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
             => r.after[b.g] >= b.seq
    BY <1>1, <1>2, TakeRelease_Facts
  <1>5. /\ NR.kind = "release" /\ NR.units = U /\ NR.after = r.after
        /\ NR.o = NONE /\ NR.p = NONE /\ NR.k = NONE /\ NR.c = 0 /\ NR.g = NONE /\ NR.to = NONE
    BY <1>3 DEF ReleaseWait
  (* The owners' tables after the step hold what they held before, but the *)
  (* blocks taken away.                                                    *)
  <1>6. /\ \A x \in AllUnits :
             (\E o \in Owners, p \in Positions : table'[o][p].u = x) =>
                 \E o \in Owners, p \in Positions : table[o][p].u = x
        /\ \A x \in AllUnits :
             (\E o \in Owners, p \in Positions : table[o][p].u = x) =>
                 (\E o \in Owners, p \in Positions : table'[o][p].u = x) \/ x \in U
        /\ \A x \in U : \E p \in Positions : table[r.o][p].u = x
        /\ \A o \in Owners, p \in Positions : table'[o][p].u # NONE => table'[o][p] = table[o][p]
    BY <1>4 DEF Empty, AllUnits
  (* What the two ways of letting go have in common.                       *)
  <1>7. /\ \A x \in AllUnits : Held(x)' => Held(x)
        /\ \A x \in AllUnits : Held(x)' => alloc'[x] = alloc[x] /\ sharedKey'[x] = sharedKey[x]
        /\ \A x \in AllUnits : (Held(x)' /\ x \notin free) => x \notin free'
        /\ \A d \in deferred' : d \in deferred \/ d = NR
        /\ \A d \in deferred : d \in deferred'
        /\ (~Reached(r.after) /\ U # {}) => NR \in deferred'
        /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
        /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
        /\ UnitTyped' /\ WorkTyped'
    <2>1. \A x : Held(x)' = HeldIn(x, table', oslot, cache, hcache, deferred', copies, tasks)
      BY DEF Held
    <2>2. CASE /\ ~Reached(r.after) /\ RelUnits(r) # {}
               /\ deferred' = deferred \cup {ReleaseWait(RelUnits(r), r.after)}
               /\ UNCHANGED <<free, alloc, sharedKey>>
      <3>1. deferred' = deferred \cup {NR}
        BY <1>2, <1>3, <2>2
      <3>2. \A x \in AllUnits : Held(x)' = Held(x)
        <4>1. \A x : (\E d \in deferred' : x \in d.units)
                       = ((\E d \in deferred : x \in d.units) \/ x \in U)
          BY <1>5, <3>1
        <4>2. \A x \in U : \E o \in Owners, p \in Positions : table[o][p].u = x
          BY <1>4, <1>6
        <4>3. \A x \in AllUnits :
                HeldIn(x, table', oslot, cache, hcache, deferred', copies, tasks)
                  = HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
          <5> SUFFICES ASSUME NEW x \in AllUnits
                       PROVE  HeldIn(x, table', oslot, cache, hcache, deferred', copies, tasks)
                                = HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
            OBVIOUS
          <5>1. CASE x \in U
            BY <4>1, <4>2, <5>1 DEF HeldIn
          <5>2. CASE x \notin U
            BY <1>6, <4>1, <5>2 DEF HeldIn
          <5> QED
            BY <5>1, <5>2
        <4> QED
          BY <2>1, <4>3 DEF Held
      <3>3. UnitTyped'
        BY <1>0, <2>2 DEF UnitTyped
      <3>4. WorkTyped'
        BY <1>0, <1>4, <1>5, <3>1 DEF WorkTyped, NONE, AllUnits
      <3>5. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
            /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
        BY <1>0, <2>2, <3>2 DEF IndInv, FreeClean, NoLeak, InRegion, Outside, Regions, OneRegion,
                                AllocOwner, SharedIff, SharedCached, CacheShared, UnitTyped
      <3> QED
        BY <1>2, <2>2, <3>1, <3>2, <3>3, <3>4, <3>5
    <2>3. CASE /\ ~(~Reached(r.after) /\ RelUnits(r) # {})
               /\ DropWith(RelUnits(r), RelTable(r), oslot, cache, hcache, deferred, copies, tasks)
               /\ UNCHANGED deferred
      <3> DEFINE G == ReclaimedWith(U, RelTable(r), oslot, cache, hcache, deferred, copies, tasks)
      <3>1. /\ free' = free \cup G /\ alloc' = Reset(alloc, G) /\ sharedKey' = Reset(sharedKey, G)
            /\ G = {x \in U : ~Held(x)'}
        BY <1>2, <2>1, <2>3, ReclaimedHeld DEF DropWith, On, AllRules
      <3>2. /\ \A x \in AllUnits : Held(x)' => Held(x)
            /\ \A x \in AllUnits : Held(x) => Held(x)' \/ x \in G
            /\ \A x \in G : Held(x) /\ ~Held(x)'
            /\ G \subseteq AllUnits
        <4>1. \A x \in U : Held(x)
          BY <1>4, <1>6 DEF Held, HeldIn
        <4>2. \A x \in AllUnits :
                HeldIn(x, table', oslot, cache, hcache, deferred, copies, tasks)
                  => HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
          BY <1>6 DEF HeldIn
        <4>3. \A x \in AllUnits :
                HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
                  => HeldIn(x, table', oslot, cache, hcache, deferred, copies, tasks) \/ x \in U
          BY <1>6 DEF HeldIn
        <4>4. U \subseteq AllUnits
          BY <1>4 DEF AllUnits
        <4> QED
          BY <2>1, <2>3, <3>1, <4>1, <4>2, <4>3, <4>4 DEF Held
      <3>3. /\ \A v \in AllUnits : alloc'[v] = IF v \in G THEN NONE ELSE alloc[v]
            /\ \A v \in AllUnits : sharedKey'[v] = IF v \in G THEN NONE ELSE sharedKey[v]
        BY <3>1 DEF Reset
      <3>4. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
        BY <1>0, <2>3, <3>1, <3>2, <3>3, FreeReclaim DEF IndInv
      <3>5. \A k \in Keys : \A x \in AllUnits : (cache[k] = x \/ hcache[k] = x) => Held(x)'
        BY <2>1, <2>3 DEF HeldIn
      <3>6. AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
        BY <1>0, <2>3, <3>1, <3>3, <3>5 DEF IndInv, AllocOwner, SharedIff, SharedCached,
                                            CacheShared, UnitTyped, AllUnits, NONE
      <3>7. UnitTyped' /\ WorkTyped'
        BY <1>0, <2>3, <3>1, <3>2 DEF UnitTyped, WorkTyped, Reset
      <3>8. (~Reached(r.after) /\ U # {}) => FALSE
        BY <1>2, <2>3
      <3> QED
        BY <2>3, <3>1, <3>2, <3>3, <3>4, <3>6, <3>7, <3>8
    <2> QED
      BY <2>2, <2>3
  <1>8. \A x \in U : NR \in deferred' => Held(x)'
    BY <1>5 DEF Held, HeldIn
  <1>9. TypeInv'
    <2>1. OwnerTyped'
      BY <1>0, <1>4, <1>6 DEF OwnerTyped, RelTable, Empty, Private, Shared
    <2>2. RequestTyped'
      BY <1>0, <1>1 DEF RequestTyped, Kept
    <2>3. PlaneTyped'
      BY <1>0, <1>1, PlaneTypedFrame
    <2> QED
      BY <1>7, <2>1, <2>2, <2>3, TypeInvNext
  <1>10. Tables' /\ OwnersOk'
    <2>1. \A o \in Owners, p \in Positions :
            table'[o][p].u # NONE =>
                /\ table'[o][p].u \in AllUnits
                /\ \A x \in AllUnits : x = table'[o][p].u => Held(x)'
      BY <1>0, <1>6 DEF Held, HeldIn, OwnerTyped, Empty, Private, Shared, AllUnits
    <2>2. \A o \in Owners : oslot[o] # NONE =>
            oslot[o] \in AllUnits /\ \A x \in AllUnits : x = oslot[o] => Held(x)'
      BY <1>0 DEF Held, HeldIn, OwnerTyped, AllUnits
    <2>3. Tables'
      BY <1>4, <1>6, <1>7, <2>1 DEF IndInv, Tables, Live, Mem, Empty
    <2>4. OwnersOk'
      BY <1>1, <1>4, <1>7, <2>2 DEF IndInv, OwnersOk, Live, Empty
    <2> QED
      BY <2>3, <2>4
  <1>11. RequestsOk' /\ LostOk'
    BY <1>1, TakeCommon DEF Kept
  <1>12. DeferredOk'
    <2>1. SUFFICES ASSUME NEW d \in deferred' PROVE DeferOk(d)'
      BY DeferredOkEach
    <2>2. \A o \in Owners, p \in Positions :
            table'[o][p].k = "private" => table'[o][p] = table[o][p]
      BY <1>4 DEF Empty
    <2>3. CASE d \in deferred
      <3>1. DeferOk(d)
        BY <2>3, DeferredOkEach DEF IndInv
      <3>2. d.kind = "commit" => \A u \in d.units : u \in AllUnits /\ Held(u)'
        BY <1>0, <1>7, <2>3, <3>1 DEF DeferOk, Held, HeldIn, AllUnits
      <3> QED
        BY <1>1, <1>4, <1>5, <1>6, <1>7, <2>2, <3>1, <3>2 DEF DeferOk
    <2>4. CASE d = NR
      <3>1. /\ \A o \in Owners : oslot[o] \notin U
            /\ \A o \in Owners, p \in Positions : table'[o][p].k = "private" => table'[o][p].u \notin U
            /\ \A c \in copies : ~c.fill => c.write \notin U
        <4>1. \A o \in Owners : oslot[o] \notin U
          BY <1>0, <1>4 DEF OwnerTyped
        <4>2. \A o \in Owners, p \in Positions : table'[o][p].k = "private" => table'[o][p].u \notin U
          BY <1>4, <2>2 DEF Empty
        <4>3. \A c \in copies : ~c.fill => c.write \notin U
          BY <1>4, <1>6 DEF IndInv, CopiesOk
        <4> QED
          BY <4>1, <4>2, <4>3
      <3> QED
        BY <1>4, <1>5, <2>4, <3>1 DEF DeferOk
    <2> QED
      BY <1>7, <2>3, <2>4
  <1>13. CopiesOk' /\ TasksOk'
    <2>1. \A o \in Owners : Moving(o)' = Moving(o)
      BY <1>5, <1>7 DEF Moving
    <2>2. CopiesOk'
      <3>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
        BY CopiesOkEach
      <3>2. CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
        BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
      <3>3. Held(c.read)' /\ Held(c.write)'
        BY DEF Held, HeldIn
      <3>4. \A o \in Owners, p \in Positions : table'[o][p].u = c.write => table[o][p].u = c.write
        BY <1>4, <3>2 DEF Empty, AllUnits
      <3>5. c.kind \in {"clone", "load"} => table'[c.o][c.p] = Empty
        BY <1>4, <3>2 DEF CopyOk
      <3> QED
        BY <1>1, <1>7, <2>1, <3>2, <3>3, <3>4, <3>5 DEF CopyOk, Live, Mem
    <2>3. TasksOk'
      <3>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
        BY TasksOkEach
      <3>2. TaskOk(t)
        BY TasksOkEach DEF IndInv
      <3>3. \A u \in t.units \cup {t.slot, t.h} : u # NONE => u \in AllUnits /\ Held(u)'
        BY <1>0, <3>2 DEF TaskOk, WorkTyped, Held, HeldIn, AllUnits
      <3>4. t.kind = "transfer" =>
              /\ t.o # r.o /\ t.writes \subseteq t.units /\ t.slot # NONE
              /\ TableUnits(t.o)' = TableUnits(t.o)
        BY <1>4, <3>2 DEF TaskOk, TableUnits
      <3>5. t.kind = "prefetch" => t.h # NONE
        BY <3>2 DEF TaskOk
      <3>6. \A d \in deferred' : t.kind = "transfer" => t.slot \notin d.units
        BY <1>4, <1>5, <1>7, <3>2 DEF TaskOk
      <3> QED
        BY <1>7, <2>1, <3>2, <3>3, <3>4, <3>5, <3>6, <1>5 DEF TaskOk, Live, Mem
    <2> QED
      BY <2>2, <2>3
  <1>14. MovingUnmapped' /\ TablesMapped' /\ HandoffOk'
    <2>1. \A h \in handoff' : (\E x \in requests' : Names(x, h)) = (\E x \in requests : Names(x, h))
      BY <1>1, TakeCommon DEF Kept
    <2>3. \A x \in Owners, p \in Rows : \A u :
            ~(x = r.o /\ p \in r.ps) => HoldsAt(x, p, u)' = HoldsAt(x, p, u)
      BY <1>4 DEF HoldsAt, Rows
    <2>4. MovingUnmapped'
      BY <1>1, <1>5, <1>7 DEF IndInv, MovingUnmapped, Kept
    <2>5. TablesMapped'
      BY <1>1, <1>4, <1>5, <1>7, <2>3 DEF IndInv, TablesMapped, Live, Empty
    <2>6. HandoffOk'
      BY <1>0, <1>1, <1>4, <1>5, <1>7, <2>1, <2>3 DEF IndInv, HandoffOk, Live, Kept, Empty,
                                                         RequestTyped
    <2> QED
      BY <2>4, <2>5, <2>6
  <1>15. TrialVisibility' /\ NoBad'
    BY <1>1, <1>4 DEF IndInv, TrialVisibility, MadeBy, NoBad, Empty, Shared
  <1>17. CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A d \in deferred : d.kind = "replicate" => \A u \in d.units : alloc'[u] = alloc[u]
      <3> SUFFICES ASSUME NEW d \in deferred, d.kind = "replicate", NEW u \in d.units
                   PROVE  alloc'[u] = alloc[u]
        OBVIOUS
      <3>1. u \in AllUnits /\ d \in deferred'
        BY <1>0, <1>7 DEF WorkTyped
      <3>2. Held(u)'
        BY <3>1 DEF Held, HeldIn
      <3> QED
        BY <1>7, <3>1, <3>2
    <2>2. \A t \in tasks : t.kind = "replicate" => \A u \in t.writes : alloc'[u] = alloc[u]
      <3> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate", NEW u \in t.writes
                   PROVE  alloc'[u] = alloc[u]
        OBVIOUS
      <3>1. u \in t.units /\ u \in AllUnits /\ t.held
        BY <1>0, TasksHeld DEF IndInv, CrossingsOk, WorkTyped
      <3>2. Held(u)'
        BY <3>1 DEF Held, HeldIn
      <3> QED
        BY <1>7, <3>1, <3>2
    <2>3. \A o \in Owners : Moving(o)' => Moving(o)
      BY <1>5, <1>7 DEF Moving
    <2>4. CrossingsOk'
      BY <1>1, <1>4, <1>5, <1>7, <2>1, <2>2, <2>3, CrossingsTake DEF IndInv
    <2>5. CopiesPlaced' /\ CrossingsPlaced'
      BY <1>1, TransfersFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
    <2> QED
      BY <2>4, <2>5
  <1>16. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. ASSUME NEW o \in Owners, NEW u, Covered(b, o, u)
          PROVE  Covered(b, o, u)' /\ Held(u)'
      <3>1. CASE /\ Live(o) /\ (ogroup[o] = b.g \/ b.g \in ReadersOf(o))
                 /\ \E p \in Rows :
                      /\ HoldsAt(o, p, u)
                      /\ \A x \in requests :
                           TakesRow(x, o, p) =>
                               b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
                 /\ \A d \in deferred : (d.kind = "transfer" /\ d.o = o) => d.after[b.g] >= b.seq
                 /\ ~\E t \in tasks : t.kind = "transfer" /\ t.o = o
        <4>1. PICK p \in Rows :
                /\ HoldsAt(o, p, u)
                /\ \A x \in requests :
                     TakesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
          BY <3>1
        <4>2. CASE o = r.o /\ p \in r.ps
          <5>1. r.after[b.g] >= b.seq
            BY <1>4, <4>1, <4>2
          <5>2. ~Reached(r.after)
            BY <1>4, <5>1, NotReached
          <5>3. u \in U /\ U # {}
            BY <1>4, <4>1, <4>2 DEF HoldsAt
          <5>4. NR \in deferred' /\ Held(u)'
            BY <1>7, <1>8, <5>2, <5>3
          <5> QED
            BY <1>5, <5>1, <5>3, <5>4 DEF Covered
        <4>3. CASE ~(o = r.o /\ p \in r.ps)
          <5>1. HoldsAt(o, p, u)'
            BY <1>4, <4>1, <4>3 DEF HoldsAt, Rows
          <5>2. Covered(b, o, u)'
            BY <1>1, <1>5, <1>7, <3>1, <4>1, <5>1 DEF Covered, Live
          <5>3. Held(u)'
            BY <5>1 DEF HoldsAt, Held, HeldIn, Rows
          <5> QED
            BY <5>2, <5>3
        <4> QED
          BY <4>2, <4>3
      <3>2. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units /\ d.after[b.g] >= b.seq
        BY <1>7, <3>2 DEF Covered, Held, HeldIn
      <3> QED
        BY <2>3, <3>1, <3>2 DEF Covered
    <2>4. \A y \in b.reads : y[1] \in Owners /\ y[2] \in AllUnits /\ y[2] \notin free
                             /\ Covered(b, y[1], y[2])
      BY <1>0, <2>2 DEF BatchOk, RequestTyped
    <2>5. \A y \in b.reads :
            Covered(b, y[1], y[2])' /\ y[2] \notin free' /\ alloc'[y[2]] = alloc[y[2]]
      BY <1>7, <2>3, <2>4
    <2>6. \A o \in Owners, p \in Positions :
            (\A x \in requests :
               FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
            => \A x \in requests' :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
      BY <1>1, TakeCommon DEF Kept
    <2>7. \A y \in b.writes, p \in Positions : table'[y[1]][p].u = y[2] => table[y[1]][p].u = y[2]
      BY <1>4, <2>2, <2>4 DEF BatchOk, Empty, AllUnits
    <2> QED
      BY <1>1, <1>5, <1>7, <2>2, <2>5, <2>6, <2>7 DEF BatchOk, Mem
  <1> QED
    BY <1>7, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17 DEF IndInv

LEMMA TakeRelease_Inv ==
  ASSUME IndInv, NEW r \in requests, TakeRelease(r), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ r.kind = "release" /\ r.sg \subseteq r.by
        /\ requests' = requests \ {r}
        /\ handoff' = Kept(r)
        /\ UNCHANGED <<dbt, launched, inflight>>
    BY TakeRelease_Effect
  <1>2. CASE UNCHANGED <<pool, alloc, sharedKey>>
    BY <1>1, <1>2, TakeIdle DEF Kept
  <1> QED
    BY <1>1, <1>2, TakeRelease_Effect, TakeRelease_Taken

-----------------------------------------------------------------------------
(* TakeDetach                                                              *)

(* The blocks and slot row a detach takes away, and the owner's table and  *)
(* slot without them.                                                      *)
DetUnits(r) == ({table[r.o][p].u : p \in Positions} \cup {oslot[r.o]}) \ {NONE}
DetTable(r) == [table EXCEPT ![r.o] = [p \in Positions |-> Empty]]
DetSlot(r) == [oslot EXCEPT ![r.o] = NONE]

LEMMA TakeDetach_Effect ==
  ASSUME NEW r \in requests, TakeDetach(r)
  PROVE  /\ r.kind = "detach" /\ r.sg \subseteq r.by
         /\ requests' = requests \ {r}
         /\ handoff' = Kept(r)
         /\ UNCHANGED <<dbt, launched, inflight>>
         /\ \/ UNCHANGED <<pool, alloc, sharedKey>>
            \/ /\ Live(r.o) /\ OwnerEpoch[r.o] = r.e
               /\ table' = DetTable(r) /\ oslot' = DetSlot(r)
               /\ ostate' = [ostate EXCEPT ![r.o] = "gone"]
               /\ \/ /\ ~Reached(r.after) /\ DetUnits(r) # {}
                     /\ deferred' = deferred \cup {ReleaseWait(DetUnits(r), r.after)}
                     /\ UNCHANGED <<free, alloc, sharedKey>>
                  \/ /\ ~(~Reached(r.after) /\ DetUnits(r) # {})
                     /\ DropWith(DetUnits(r), DetTable(r), DetSlot(r), cache, hcache, deferred,
                                 copies, tasks)
                     /\ UNCHANGED deferred
               /\ UNCHANGED <<ogroup, cache, hcache, copies, ncopy, tasks, trial, member, valid,
                              vis>>
  <1>1. UNION {{c.read, c.write} : c \in copies \ copies} = {}
    OBVIOUS
  <1> QED
    BY <1>1 DEF TakeDetach, Submittable, StampGroups, ReleaseUnits, ReleaseWait, DetUnits,
                DetTable, DetSlot, OwnSession, Live, On, AllRules

(* What a detach taken with effect says of its owner, the executors'      *)
(* tables and the completions it keeps.                                    *)
LEMMA TakeDetach_Facts ==
  ASSUME IndInv, NEW r \in requests, r.kind = "detach", r.sg \subseteq r.by,
         Live(r.o), OwnerEpoch[r.o] = r.e
  PROVE  /\ r.o \in Owners /\ r.after \in Stamps
         /\ oslot[r.o] \in Slots /\ alloc[oslot[r.o]] = r.o
         /\ \A x \in Owners, p \in Positions :
              DetTable(r)[x][p] = IF x = r.o THEN Empty ELSE table[x][p]
         /\ \A x \in Owners : DetSlot(r)[x] = IF x = r.o THEN NONE ELSE oslot[x]
         /\ DetUnits(r) = TableUnits(r.o) \cup {oslot[r.o]}
         /\ TableUnits(r.o) \subseteq Units
         /\ \A u \in DetUnits(r) : alloc[u] = r.o \/ alloc[u] = "shared"
         /\ \A x \in Owners, p \in Positions :
              (table[x][p].k = "private" /\ table[x][p].u \in DetUnits(r)) => x = r.o
         /\ \A x \in Owners : x # r.o => oslot[x] \notin DetUnits(r)
         /\ \A g \in Groups, p \in Rows : dbt[g][r.o][p].u = NONE
         /\ \A h \in Kept(r) : h.o # r.o
         /\ \A b \in inflight, p \in Rows :
              (\A x \in requests :
                 TakesRow(x, r.o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
              => r.after[b.g] >= b.seq
  <1>0. OwnerTyped /\ RequestTyped
    BY OwnerTypes, RequestTypes
  <1>1. r.o \in Owners /\ r.after \in Stamps
    BY <1>0 DEF RequestTyped
  <1>2. oslot[r.o] \in Slots /\ alloc[oslot[r.o]] = r.o
    BY <1>1 DEF IndInv, OwnersOk
  <1>3. /\ \A x \in Owners, p \in Positions :
             DetTable(r)[x][p] = IF x = r.o THEN Empty ELSE table[x][p]
        /\ \A x \in Owners : DetSlot(r)[x] = IF x = r.o THEN NONE ELSE oslot[x]
    BY <1>0, <1>1 DEF DetTable, DetSlot, OwnerTyped
  <1>4. TableUnits(r.o) \subseteq Units /\ DetUnits(r) = TableUnits(r.o) \cup {oslot[r.o]}
    BY <1>0, <1>1, <1>2 DEF TableUnits, DetUnits, OwnerTyped, Empty, Private, Shared
  <1>5. \A u \in TableUnits(r.o) : alloc[u] = r.o \/ alloc[u] = "shared"
    BY <1>0, <1>1 DEF IndInv, Tables, TableUnits, OwnerTyped, Empty, Private, Shared
  <1>6. \A x \in Owners, p \in Positions :
          (table[x][p].k = "private" /\ table[x][p].u \in DetUnits(r)) => x = r.o
    <2> SUFFICES ASSUME NEW x \in Owners, NEW p \in Positions, table[x][p].k = "private",
                        table[x][p].u \in DetUnits(r)
                 PROVE  x = r.o
      OBVIOUS
    <2>1. alloc[table[x][p].u] = x /\ table[x][p].u \in Units
      BY DEF IndInv, Tables
    <2> QED
      BY <1>2, <1>4, <1>5, <2>1
  <1>7. \A x \in Owners : x # r.o => oslot[x] \notin DetUnits(r)
    <2> SUFFICES ASSUME NEW x \in Owners, x # r.o, oslot[x] \in DetUnits(r) PROVE FALSE
      OBVIOUS
    <2>1. oslot[x] \in Slots /\ oslot[x] = oslot[r.o]
      BY <1>0, <1>2, <1>4 DEF OwnerTyped
    <2>2. Live(x) /\ alloc[oslot[x]] = x
      BY <2>1 DEF IndInv, OwnersOk, Live
    <2> QED
      BY <1>2, <2>1, <2>2
  <1>8. \A g \in Groups, p \in Rows : dbt[g][r.o][p].u = NONE
    BY <1>1, Submitted DEF TakesRow
  <1>9. \A h \in Kept(r) : h.o # r.o
    BY DEF Kept, Names
  <1>10. \A b \in inflight, p \in Rows :
           (\A x \in requests :
              TakesRow(x, r.o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
           => r.after[b.g] >= b.seq
    BY DEF TakesRow
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10

(* The detach is taken with effect: the owner is gone, and what it held    *)
(* waits for the detach's stamps, or is let go now when they are reached.  *)
LEMMA TakeDetach_Taken ==
  ASSUME IndInv, NEW r \in requests, TakeDetach(r), UNCHANGED rare,
         Live(r.o), OwnerEpoch[r.o] = r.e,
         table' = DetTable(r), oslot' = DetSlot(r),
         ostate' = [ostate EXCEPT ![r.o] = "gone"],
         UNCHANGED <<ogroup, cache, hcache, copies, ncopy, tasks, trial, member, valid, vis>>,
         \/ /\ ~Reached(r.after) /\ DetUnits(r) # {}
            /\ deferred' = deferred \cup {ReleaseWait(DetUnits(r), r.after)}
            /\ UNCHANGED <<free, alloc, sharedKey>>
         \/ /\ ~(~Reached(r.after) /\ DetUnits(r) # {})
            /\ DropWith(DetUnits(r), DetTable(r), DetSlot(r), cache, hcache, deferred, copies,
                        tasks)
            /\ UNCHANGED deferred
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. /\ r.kind = "detach" /\ r.sg \subseteq r.by
        /\ requests' = requests \ {r}
        /\ handoff' = Kept(r)
        /\ UNCHANGED <<dbt, launched, inflight, remote, lost, bad>>
    BY TakeDetach_Effect DEF rare
  <1>2. PICK U : U = DetUnits(r)
    OBVIOUS
  <1>3. PICK NR : NR = ReleaseWait(U, r.after)
    OBVIOUS
  <1>4. /\ r.o \in Owners /\ r.after \in Stamps
        /\ oslot[r.o] \in Slots /\ alloc[oslot[r.o]] = r.o
        /\ \A x \in Owners, p \in Positions : table'[x][p] = IF x = r.o THEN Empty ELSE table[x][p]
        /\ \A x \in Owners : oslot'[x] = IF x = r.o THEN NONE ELSE oslot[x]
        /\ \A x \in Owners : ostate'[x] = IF x = r.o THEN "gone" ELSE ostate[x]
        /\ U = TableUnits(r.o) \cup {oslot[r.o]} /\ TableUnits(r.o) \subseteq Units
        /\ \A u \in U : alloc[u] = r.o \/ alloc[u] = "shared"
        /\ \A x \in Owners, p \in Positions :
             (table[x][p].k = "private" /\ table[x][p].u \in U) => x = r.o
        /\ \A x \in Owners : x # r.o => oslot[x] \notin U
        /\ \A g \in Groups, p \in Rows : dbt[g][r.o][p].u = NONE
        /\ \A h \in handoff' : h.o # r.o
        /\ \A b \in inflight, p \in Rows :
             (\A x \in requests :
                TakesRow(x, r.o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
             => r.after[b.g] >= b.seq
    BY <1>0, <1>1, <1>2, TakeDetach_Facts DEF OwnerTyped
  <1>5. /\ NR.kind = "release" /\ NR.units = U /\ NR.after = r.after
        /\ NR.o = NONE /\ NR.p = NONE /\ NR.k = NONE /\ NR.c = 0 /\ NR.g = NONE /\ NR.to = NONE
    BY <1>3 DEF ReleaseWait
  <1>6. /\ \A x \in Owners : x # r.o => Live(x)' = Live(x)
        /\ ~Live(r.o)'
        /\ \A x \in Owners : ostate'[x] # "new" <=> ostate[x] # "new" \/ x = r.o
    BY <1>4 DEF Live
  (* The owners' tables and slots after the step hold what they held      *)
  (* before, but what the owner held.                                      *)
  <1>7. /\ \A x \in AllUnits :
             (\E o \in Owners, p \in Positions : table'[o][p].u = x) =>
                 \E o \in Owners, p \in Positions : table[o][p].u = x
        /\ \A x \in AllUnits : (\E o \in Owners : oslot'[o] = x) => \E o \in Owners : oslot[o] = x
        /\ \A x \in AllUnits :
             (\/ \E o \in Owners, p \in Positions : table[o][p].u = x
              \/ \E o \in Owners : oslot[o] = x) =>
                 \/ \E o \in Owners, p \in Positions : table'[o][p].u = x
                 \/ \E o \in Owners : oslot'[o] = x
                 \/ x \in U
        /\ \A x \in U : (\E p \in Positions : table[r.o][p].u = x) \/ oslot[r.o] = x
        /\ \A o \in Owners, p \in Positions : table'[o][p].u # NONE => table'[o][p] = table[o][p]
        /\ \A o \in Owners : oslot'[o] # NONE => oslot'[o] = oslot[o]
    BY <1>4 DEF Empty, AllUnits, TableUnits
  <1>8. /\ \A x \in AllUnits : Held(x)' => Held(x)
        /\ \A x \in AllUnits : Held(x)' => alloc'[x] = alloc[x] /\ sharedKey'[x] = sharedKey[x]
        /\ \A x \in AllUnits : (Held(x)' /\ x \notin free) => x \notin free'
        /\ \A d \in deferred' : d \in deferred \/ d = NR
        /\ \A d \in deferred : d \in deferred'
        /\ (~Reached(r.after) /\ U # {}) => NR \in deferred'
        /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
        /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
        /\ UnitTyped' /\ WorkTyped'
    <2>1. \A x : Held(x)' = HeldIn(x, table', oslot', cache, hcache, deferred', copies, tasks)
      BY DEF Held
    <2>2. \A x \in U : Held(x)
      BY <1>4, <1>7 DEF Held, HeldIn
    <2>3. AllocOwner'
      <3>1. \A x \in AllUnits : alloc'[x] = alloc[x] \/ alloc'[x] = NONE
        <4>1. CASE UNCHANGED alloc
          BY <4>1
        <4>2. CASE ~UNCHANGED alloc
          <5>1. PICK G : alloc' = Reset(alloc, G)
            BY <4>2 DEF DropWith
          <5> QED
            BY <5>1 DEF Reset
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <1>6, <3>1 DEF IndInv, AllocOwner, NONE
    <2>4. CASE /\ ~Reached(r.after) /\ DetUnits(r) # {}
               /\ deferred' = deferred \cup {ReleaseWait(DetUnits(r), r.after)}
               /\ UNCHANGED <<free, alloc, sharedKey>>
      <3>1. deferred' = deferred \cup {NR}
        BY <1>2, <1>3, <2>4
      <3>2. \A x \in AllUnits : Held(x)' = Held(x)
        <4>1. \A x : (\E d \in deferred' : x \in d.units)
                       = ((\E d \in deferred : x \in d.units) \/ x \in U)
          BY <1>5, <3>1
        <4>2. \A x \in AllUnits :
                HeldIn(x, table', oslot', cache, hcache, deferred', copies, tasks)
                  = HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
          BY <1>4, <1>7, <4>1 DEF HeldIn
        <4> QED
          BY <2>1, <4>2 DEF Held
      <3>3. UnitTyped'
        BY <1>0, <2>4 DEF UnitTyped
      <3>4. WorkTyped'
        BY <1>0, <1>4, <1>5, <3>1 DEF WorkTyped, NONE, AllUnits
      <3>5. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
            /\ SharedIff' /\ SharedCached' /\ CacheShared'
        BY <1>0, <2>4, <3>2 DEF IndInv, FreeClean, NoLeak, InRegion, Outside, Regions, OneRegion,
                                SharedIff, SharedCached, CacheShared, UnitTyped
      <3> QED
        BY <1>2, <2>3, <2>4, <3>1, <3>2, <3>3, <3>4, <3>5
    <2>5. CASE /\ ~(~Reached(r.after) /\ DetUnits(r) # {})
               /\ DropWith(DetUnits(r), DetTable(r), DetSlot(r), cache, hcache, deferred, copies,
                           tasks)
               /\ UNCHANGED deferred
      <3> DEFINE G == ReclaimedWith(U, DetTable(r), DetSlot(r), cache, hcache, deferred, copies,
                                    tasks)
      <3>1. /\ free' = free \cup G /\ alloc' = Reset(alloc, G) /\ sharedKey' = Reset(sharedKey, G)
            /\ G = {x \in U : ~Held(x)'}
        BY <1>2, <2>1, <2>5, ReclaimedHeld DEF DropWith, On, AllRules
      <3>2. /\ \A x \in AllUnits : Held(x)' => Held(x)
            /\ \A x \in AllUnits : Held(x) => Held(x)' \/ x \in G
            /\ \A x \in G : Held(x) /\ ~Held(x)'
            /\ G \subseteq AllUnits
        <4>1. \A x \in AllUnits :
                HeldIn(x, table', oslot', cache, hcache, deferred, copies, tasks)
                  => HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
          BY <1>7 DEF HeldIn
        <4>2. \A x \in AllUnits :
                HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
                  => HeldIn(x, table', oslot', cache, hcache, deferred, copies, tasks) \/ x \in U
          BY <1>7 DEF HeldIn
        <4>3. U \subseteq AllUnits
          BY <1>4 DEF AllUnits
        <4> QED
          BY <2>1, <2>2, <2>5, <3>1, <4>1, <4>2, <4>3 DEF Held
      <3>3. /\ \A v \in AllUnits : alloc'[v] = IF v \in G THEN NONE ELSE alloc[v]
            /\ \A v \in AllUnits : sharedKey'[v] = IF v \in G THEN NONE ELSE sharedKey[v]
        BY <3>1 DEF Reset
      <3>4. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
        BY <1>0, <2>5, <3>1, <3>2, <3>3, FreeReclaim DEF IndInv
      <3>5. \A k \in Keys : \A x \in AllUnits : (cache[k] = x \/ hcache[k] = x) => Held(x)'
        BY <2>1, <2>5 DEF HeldIn
      <3>6. SharedIff' /\ SharedCached' /\ CacheShared'
        BY <1>0, <2>5, <3>1, <3>3, <3>5 DEF IndInv, SharedIff, SharedCached, CacheShared,
                                            UnitTyped, AllUnits, NONE
      <3>7. UnitTyped' /\ WorkTyped'
        BY <1>0, <2>5, <3>1, <3>2 DEF UnitTyped, WorkTyped, Reset
      <3>8. (~Reached(r.after) /\ U # {}) => FALSE
        BY <1>2, <2>5
      <3> QED
        BY <2>3, <2>5, <3>1, <3>2, <3>3, <3>4, <3>6, <3>7, <3>8
    <2> QED
      BY <2>4, <2>5
  <1>9. \A x \in U : NR \in deferred' => Held(x)'
    BY <1>5 DEF Held, HeldIn
  <1>10. TypeInv'
    <2>1. OwnerTyped'
      BY <1>0, <1>4, <1>7 DEF OwnerTyped, DetTable, DetSlot, Empty, Private, Shared
    <2>2. RequestTyped'
      BY <1>0, <1>1 DEF RequestTyped, Kept
    <2>3. PlaneTyped'
      BY <1>0, <1>1, PlaneTypedFrame
    <2> QED
      BY <1>8, <2>1, <2>2, <2>3, TypeInvNext
  <1>11. Tables' /\ OwnersOk'
    <2>1. \A o \in Owners, p \in Positions :
            table'[o][p].u # NONE =>
                /\ table'[o][p].u \in AllUnits
                /\ \A x \in AllUnits : x = table'[o][p].u => Held(x)'
      BY <1>0, <1>7 DEF Held, HeldIn, OwnerTyped, Empty, Private, Shared, AllUnits
    <2>2. \A o \in Owners : oslot'[o] # NONE =>
            oslot'[o] \in AllUnits /\ \A x \in AllUnits : x = oslot'[o] => Held(x)'
      BY <1>0, <1>7 DEF Held, HeldIn, OwnerTyped, AllUnits
    <2>3. Tables'
      BY <1>4, <1>6, <1>7, <1>8, <2>1 DEF IndInv, Tables, Mem, Empty
    <2>4. OwnersOk'
      BY <1>1, <1>4, <1>6, <1>7, <1>8, <2>2 DEF IndInv, OwnersOk, Empty
    <2> QED
      BY <2>3, <2>4
  <1>12. RequestsOk' /\ LostOk'
    BY <1>1, TakeCommon DEF Kept
  <1>13. DeferredOk'
    <2>1. SUFFICES ASSUME NEW d \in deferred' PROVE DeferOk(d)'
      BY DeferredOkEach
    <2>2. \A o \in Owners, p \in Positions :
            table'[o][p].k = "private" => table'[o][p] = table[o][p] /\ o # r.o
      BY <1>4 DEF Empty
    <2>3. CASE d \in deferred
      <3>1. DeferOk(d)
        BY <2>3, DeferredOkEach DEF IndInv
      <3>2. d.kind = "commit" => \A u \in d.units : u \in AllUnits /\ Held(u)'
        BY <1>0, <1>8, <2>3, <3>1 DEF DeferOk, Held, HeldIn, AllUnits
      <3>3. d.kind = "release" => \A o \in Owners : oslot'[o] \notin d.units
        BY <1>4, <3>1 DEF DeferOk
      <3> QED
        BY <1>1, <1>4, <1>5, <1>6, <1>7, <1>8, <2>2, <3>1, <3>2, <3>3 DEF DeferOk
    <2>4. CASE d = NR
      <3>1. /\ \A o \in Owners : oslot'[o] \notin U
            /\ \A o \in Owners, p \in Positions : table'[o][p].k = "private" => table'[o][p].u \notin U
            /\ \A c \in copies : ~c.fill => c.write \notin U
            /\ U \subseteq Units \cup Slots
        <4>1. \A o \in Owners : oslot'[o] \notin U
          BY <1>4 DEF AllUnits
        <4>2. \A o \in Owners, p \in Positions : table'[o][p].k = "private" => table'[o][p].u \notin U
          BY <1>4, <2>2
        <4>3. \A c \in copies : ~c.fill => c.write \notin U
          <5> SUFFICES ASSUME NEW c \in copies, ~c.fill, c.write \in U PROVE FALSE
            OBVIOUS
          <5>1. CopyOk(c)
            BY CopiesOkEach DEF IndInv
          <5>2. c.write # oslot[r.o]
            BY <1>4, <5>1 DEF CopyOk
          <5>3. c.write \in TableUnits(r.o)
            BY <1>4, <5>2
          <5> QED
            BY <1>4, <5>1, <5>3 DEF CopyOk, TableUnits
        <4> QED
          BY <1>4, <4>1, <4>2, <4>3
      <3> QED
        BY <1>4, <1>5, <2>4, <3>1 DEF DeferOk
    <2> QED
      BY <1>8, <2>3, <2>4
  <1>14. CopiesOk' /\ TasksOk'
    <2>1. \A o \in Owners : Moving(o)' = Moving(o)
      BY <1>5, <1>8 DEF Moving
    <2>2. CopiesOk'
      <3>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
        BY CopiesOkEach
      <3>2. CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
        BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
      <3>3. Held(c.read)' /\ Held(c.write)'
        BY DEF Held, HeldIn
      <3>4. \A o \in Owners, p \in Positions : table'[o][p].u = c.write => table[o][p].u = c.write
        BY <1>4, <3>2 DEF Empty, AllUnits
      <3>5. c.kind \in {"clone", "load"} => table'[c.o][c.p] = Empty
        BY <1>4, <3>2 DEF CopyOk
      <3>6. c.o \in Owners => (ostate'[c.o] # "new" /\ (Live(c.o)' => Live(c.o)))
        BY <1>6, <3>2 DEF CopyOk
      <3> QED
        BY <1>1, <1>8, <2>1, <3>2, <3>3, <3>4, <3>5, <3>6 DEF CopyOk, Mem
    <2>3. TasksOk'
      <3>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
        BY TasksOkEach
      <3>2. TaskOk(t)
        BY TasksOkEach DEF IndInv
      <3>3. \A u \in t.units \cup {t.slot, t.h} : u # NONE => u \in AllUnits /\ Held(u)'
        BY <1>0, <3>2 DEF TaskOk, WorkTyped, Held, HeldIn, AllUnits
      <3>4. t.kind = "transfer" =>
              /\ t.writes \subseteq t.units /\ t.slot # NONE /\ t.slot \in Slots
              /\ \A o \in Owners : oslot'[o] # t.slot
              /\ t.slot \notin U
              /\ (Live(t.o)' => t.o # r.o /\ Live(t.o) /\ TableUnits(t.o)' = TableUnits(t.o)
                                /\ oslot'[t.o] = oslot[t.o])
        BY <1>4, <1>6, <3>2 DEF TaskOk, TableUnits
      <3>5. t.kind = "prefetch" => t.h # NONE
        BY <3>2 DEF TaskOk
      <3>6. \A d \in deferred' : t.kind = "transfer" => t.slot \notin d.units
        BY <1>5, <1>8, <3>2, <3>4 DEF TaskOk
      <3> QED
        BY <1>8, <2>1, <3>2, <3>3, <3>4, <3>5, <3>6, <1>5 DEF TaskOk, Mem
    <2> QED
      BY <2>2, <2>3
  <1>15. MovingUnmapped' /\ TablesMapped' /\ HandoffOk'
    <2>1. \A h \in handoff' : (\E x \in requests' : Names(x, h)) = (\E x \in requests : Names(x, h))
      BY <1>1, TakeCommon DEF Kept
    <2>2. \A x \in Owners, p \in Rows : \A u :
            x # r.o => HoldsAt(x, p, u)' = HoldsAt(x, p, u)
      BY <1>4 DEF HoldsAt, Rows
    <2>3. MovingUnmapped'
      BY <1>1, <1>5, <1>8 DEF IndInv, MovingUnmapped, Kept
    <2>4. TablesMapped'
      BY <1>1, <1>4, <1>5, <1>6, <1>8, <2>2 DEF IndInv, TablesMapped, Empty
    (* A completion kept is of another owner, whose rows, validity and     *)
    (* state the detach leaves alone.                                       *)
    <2>5. HandoffOk'
      <3> SUFFICES ASSUME NEW y \in handoff'
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
      <3>1. y \in handoff /\ y.o # r.o /\ y.o \in Owners /\ y.p \in Rows
        BY <1>0, <1>1, <1>4 DEF Kept, RequestTyped
      <3>2. (\E x \in requests' : Names(x, y)) = (\E x \in requests : Names(x, y))
        BY <2>1
      <3>3. /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
            /\ (~\E x \in requests : Names(x, y)) =>
                 /\ ostate[y.o] = "live"
                 /\ ogroup[y.o] = y.g \/ ReaderRow(y.g, y.o, y.p, y.u, y.w)
                 /\ HoldsAt(y.o, y.p, y.u)
                 /\ (y.p \in Positions /\ y.w) =>
                      /\ table[y.o][y.p].k = "private"
                      /\ ~\E d \in deferred : d.kind \in {"commit", "replicate"} /\ y.u \in d.units
                      /\ ~\E t \in tasks : t.kind = "replicate" /\ y.u \in t.units
        BY <3>1 DEF IndInv, HandoffOk, Live
      <3>4. /\ ostate'[y.o] = ostate[y.o]
            /\ HoldsAt(y.o, y.p, y.u)' = HoldsAt(y.o, y.p, y.u)
            /\ y.p \in Positions => table'[y.o][y.p] = table[y.o][y.p]
            /\ ReaderRow(y.g, y.o, y.p, y.u, y.w)' = ReaderRow(y.g, y.o, y.p, y.u, y.w)
        BY <1>4, <2>2, <3>1
      <3>5. \A d \in deferred' : d.kind \in {"commit", "replicate"} => d \in deferred
        BY <1>5, <1>8
      <3> QED
        BY <3>2, <3>3, <3>4, <3>5
    <2> QED
      BY <2>3, <2>4, <2>5
  <1>16. TrialVisibility' /\ NoBad'
    BY <1>1, <1>4 DEF IndInv, TrialVisibility, MadeBy, NoBad, Empty, Shared
  <1>18. CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A d \in deferred : d.kind = "replicate" => \A u \in d.units : alloc'[u] = alloc[u]
      <3> SUFFICES ASSUME NEW d \in deferred, d.kind = "replicate", NEW u \in d.units
                   PROVE  alloc'[u] = alloc[u]
        OBVIOUS
      <3>1. u \in AllUnits /\ d \in deferred'
        BY <1>0, <1>8 DEF WorkTyped
      <3>2. Held(u)'
        BY <3>1 DEF Held, HeldIn
      <3> QED
        BY <1>8, <3>1, <3>2
    <2>2. \A t \in tasks : t.kind = "replicate" => \A u \in t.writes : alloc'[u] = alloc[u]
      <3> SUFFICES ASSUME NEW t \in tasks, t.kind = "replicate", NEW u \in t.writes
                   PROVE  alloc'[u] = alloc[u]
        OBVIOUS
      <3>1. u \in t.units /\ u \in AllUnits /\ t.held
        BY <1>0, TasksHeld DEF IndInv, CrossingsOk, WorkTyped
      <3>2. Held(u)'
        BY <3>1 DEF Held, HeldIn
      <3> QED
        BY <1>8, <3>1, <3>2
    <2>3. \A o \in Owners : Moving(o)' => Moving(o)
      BY <1>5, <1>8 DEF Moving
    <2>4. \A o \in Owners : ostate[o] # "new" => ostate'[o] # "new"
      BY <1>6
    <2>5. CrossingsOk'
      BY <1>1, <1>5, <1>8, <2>1, <2>2, <2>3, <2>4, CrossingsTake DEF IndInv
    <2>6. CopiesPlaced' /\ CrossingsPlaced'
      BY <1>1, TransfersFrame, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
    <2> QED
      BY <2>5, <2>6
  <1>17. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. ASSUME NEW o \in Owners, NEW u \in AllUnits, Covered(b, o, u)
          PROVE  Covered(b, o, u)' /\ Held(u)'
      <3>1. CASE /\ Live(o) /\ (ogroup[o] = b.g \/ b.g \in ReadersOf(o))
                 /\ \E p \in Rows :
                      /\ HoldsAt(o, p, u)
                      /\ \A x \in requests :
                           TakesRow(x, o, p) =>
                               b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
                 /\ \A d \in deferred : (d.kind = "transfer" /\ d.o = o) => d.after[b.g] >= b.seq
                 /\ ~\E t \in tasks : t.kind = "transfer" /\ t.o = o
        <4>1. PICK p \in Rows :
                /\ HoldsAt(o, p, u)
                /\ \A x \in requests :
                     TakesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
          BY <3>1
        <4>2. CASE o = r.o
          <5>1. r.after[b.g] >= b.seq
            BY <1>4, <4>1, <4>2
          <5>2. ~Reached(r.after)
            BY <1>4, <5>1, NotReached
          <5>3. u \in U /\ U # {}
            <6>1. u # NONE
              BY DEF AllUnits
            <6> QED
              BY <1>4, <4>1, <4>2, <6>1 DEF HoldsAt, TableUnits, Rows
          <5>4. NR \in deferred' /\ Held(u)'
            BY <1>8, <1>9, <5>2, <5>3
          <5> QED
            BY <1>5, <5>1, <5>3, <5>4 DEF Covered
        <4>3. CASE o # r.o
          <5>1. HoldsAt(o, p, u)'
            BY <1>4, <4>1, <4>3 DEF HoldsAt, Rows
          <5>2. Covered(b, o, u)'
            BY <1>1, <1>5, <1>6, <1>8, <3>1, <4>1, <4>3, <5>1 DEF Covered
          <5>3. Held(u)'
            BY <5>1 DEF HoldsAt, Held, HeldIn, Rows
          <5> QED
            BY <5>2, <5>3
        <4> QED
          BY <4>2, <4>3
      <3>2. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units /\ d.after[b.g] >= b.seq
        BY <1>8, <3>2 DEF Covered, Held, HeldIn
      <3> QED
        BY <2>3, <3>1, <3>2 DEF Covered
    <2>4. \A y \in b.reads : y[1] \in Owners /\ y[2] \in AllUnits /\ y[2] \notin free
                             /\ Covered(b, y[1], y[2])
      BY <1>0, <2>2 DEF BatchOk, RequestTyped
    <2>5. \A y \in b.reads :
            Covered(b, y[1], y[2])' /\ y[2] \notin free' /\ alloc'[y[2]] = alloc[y[2]]
      BY <1>8, <2>3, <2>4
    <2>6. \A o \in Owners, p \in Positions :
            (\A x \in requests :
               FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
            => \A x \in requests' :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
      BY <1>1, TakeCommon DEF Kept
    <2>7. \A y \in b.writes, p \in Positions : table'[y[1]][p].u = y[2] => table[y[1]][p].u = y[2]
      BY <1>4, <2>2, <2>4 DEF BatchOk, Empty, AllUnits
    <2> QED
      BY <1>1, <1>5, <1>8, <2>2, <2>5, <2>6, <2>7 DEF BatchOk, Mem
  <1> QED
    BY <1>8, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18 DEF IndInv

LEMMA TakeDetach_Inv ==
  ASSUME IndInv, NEW r \in requests, TakeDetach(r), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ r.kind = "detach" /\ r.sg \subseteq r.by
        /\ requests' = requests \ {r}
        /\ handoff' = Kept(r)
        /\ UNCHANGED <<dbt, launched, inflight>>
    BY TakeDetach_Effect
  <1>2. CASE UNCHANGED <<pool, alloc, sharedKey>>
    BY <1>1, <1>2, TakeIdle DEF Kept
  <1> QED
    BY <1>1, <1>2, TakeDetach_Effect, TakeDetach_Taken

-----------------------------------------------------------------------------

THEOREM Step6 ==
    IndInv /\ (/\ \E r \in requests :
                    TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r) \/ TakeTransfer(r)
               /\ UNCHANGED rare)
    => IndInv'
  BY TakeRelease_Inv, TakeDetach_Inv, TakeCommit_Inv, TakeFinish_Inv, TakeTransfer_Inv
=============================================================================
