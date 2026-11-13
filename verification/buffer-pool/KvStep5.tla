------------------------------- MODULE KvStep5 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for the actions of the executor's batches: *)
(* Launch, EndBatch and ExecFinish.                                        *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, KvFrame, TLAPS

USE Sizes, Design

-----------------------------------------------------------------------------
(* EndBatch                                                                *)

LEMMA EndBatch_Inv ==
  ASSUME IndInv, NEW b \in inflight, EndBatch(b), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ inflight' = inflight \ {b}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       tasks, valid, vis, trial, member, dbt, launched, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF EndBatch, pool, rare
  <1>2. TypeInv'
    <2>1. RequestTyped'
      BY <1>1, RequestTypes DEF RequestTyped
    <2> QED
      BY <1>1, <2>1, UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, UnitTypedFrame,
         OwnerTypedFrame, PlaneTypedFrame, WorkTypedFrame, TypeInvNext
  <1>3. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
        /\ TasksOk' /\ MovingUnmapped' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
        /\ TrialVisibility' /\ NoBad'
    BY <1>1, FreeCleanFrame, NoLeakFrame, OutsideFrame, RegionsFrame, AllocOwnerFrame,
       SharedIffFrame, SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame,
       DeferredOkFrame, CopiesOkFrame, TasksOkFrame, MovingUnmappedFrame, TablesMappedFrame,
       HandoffOkFrame, RequestsOkFrame, TrialVisibilityFrame, NoBadFrame, TransfersFrame DEF IndInv
  <1>4. LostOk'
    BY <1>1 DEF IndInv, LostOk
  <1>5. BatchesOk'
    <2>1. \A x, o, u : Covered(x, o, u)' = Covered(x, o, u)
      BY <1>1, CoveredFrame, TransfersFrame
    <2> QED
      BY <1>1, <2>1 DEF IndInv, BatchesOk, Mem
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5 DEF IndInv

-----------------------------------------------------------------------------
(* Launch                                                                  *)

(* The values Launch(g) computes: the rows its batch addresses, the copies *)
(* that start with it, a copy once started with batch n, and the batch.    *)
LaunchRows(g) == {r \in Owners \X Rows : dbt[g][r[1]][r[2]].u # NONE}
LaunchStart(g) == {c \in copies : c.g = g /\ c.seq = 0 /\ ~c.stopped}
Started(g, n, c) == IF c.g = g /\ c.seq = 0 /\ ~c.stopped THEN [c EXCEPT !.seq = n] ELSE c
NewBatch(g) ==
    [g |-> g, seq |-> launched[g] + 1,
     reads |-> {<<r[1], dbt[g][r[1]][r[2]].u>> : r \in LaunchRows(g)},
     writes |-> {<<r[1], dbt[g][r[1]][r[2]].u>> :
                   r \in {x \in LaunchRows(g) : dbt[g][x[1]][x[2]].w}},
     cr |-> {c.read : c \in LaunchStart(g)},
     cw |-> {c.write : c \in LaunchStart(g)}]

LEMMA Launch_Effect ==
  ASSUME NEW g \in Groups, Launch(g)
  PROVE  /\ GroupEpoch[g] \notin lost /\ launched[g] < MaxBatch
         /\ inflight' = inflight \cup {NewBatch(g)}
         /\ copies' = {Started(g, launched[g] + 1, c) : c \in copies}
         /\ launched' = [launched EXCEPT ![g] = launched[g] + 1]
         /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks,
                        trial, member, valid, vis, dbt, handoff, requests, alloc, sharedKey>>
  BY DEF Launch, LaunchRows, LaunchStart, Started, NewBatch

LEMMA Launch_Copies ==
  ASSUME IndInv, NEW g, NEW n
  PROVE  \A c \in copies :
           /\ Started(g, n, c).id = c.id /\ Started(g, n, c).kind = c.kind
           /\ Started(g, n, c).o = c.o /\ Started(g, n, c).p = c.p /\ Started(g, n, c).k = c.k
           /\ Started(g, n, c).g = c.g /\ Started(g, n, c).read = c.read
           /\ Started(g, n, c).write = c.write /\ Started(g, n, c).fill = c.fill
           /\ Started(g, n, c).done = c.done /\ Started(g, n, c).stopped = c.stopped
           /\ Started(g, n, c).seq = IF c.g = g /\ c.seq = 0 /\ ~c.stopped THEN n ELSE c.seq
  BY WorkTypes DEF WorkTyped, Started

(* The steps' values of launched, pointwise.                               *)
LEMMA Launch_Launched ==
  ASSUME IndInv, NEW g \in Groups, Launch(g)
  PROVE  /\ \A x \in Groups : launched'[x] = IF x = g THEN launched[g] + 1 ELSE launched[x]
         /\ \A x \in Groups : launched[x] <= launched'[x]
         /\ launched' \in [Groups -> 0 .. MaxBatch]
         /\ launched[g] \in 0 .. MaxBatch
  BY Launch_Effect, PlaneTypes DEF PlaneTyped

LEMMA Launch_Held ==
  ASSUME IndInv, NEW g \in Groups, Launch(g)
  PROVE  /\ \A u : Held(u)' = Held(u)
         /\ \A x, o, u : Covered(x, o, u)' = Covered(x, o, u)
  <1>1. \A u : (\E c \in copies' : u = c.read \/ u = c.write)
                 = (\E c \in copies : u = c.read \/ u = c.write)
    BY Launch_Effect, Launch_Copies
  <1>2. UNCHANGED <<table, oslot, cache, hcache, deferred, tasks>>
    BY Launch_Effect
  <1>3. \A u : Held(u)' = Held(u)
    BY <1>1, <1>2, Zenon DEF Held, HeldIn
  <1>4. \A x, o, u : Covered(x, o, u)' = Covered(x, o, u)
    BY Launch_Effect, CoveredFrame, TransfersFrame
  <1> QED
    BY <1>3, <1>4

LEMMA Launch_Typed ==
  ASSUME IndInv, NEW g \in Groups, Launch(g), UNCHANGED rare
  PROVE  TypeInv'
  <1>1. UnitTyped' /\ OwnerTyped'
    BY Launch_Effect, UnitTypes, OwnerTypes, UnitTypedFrame, OwnerTypedFrame
  <1>2. PlaneTyped'
    BY Launch_Effect, Launch_Launched, PlaneTypes DEF PlaneTyped, rare
  <1>3. WorkTyped'
    <2>1. \A c \in copies :
            LET x == Started(g, launched[g] + 1, c) IN
              x = [id |-> x.id, kind |-> x.kind, o |-> x.o, p |-> x.p, k |-> x.k, g |-> x.g,
                   read |-> x.read, write |-> x.write, fill |-> x.fill, seq |-> x.seq,
                   done |-> x.done, stopped |-> x.stopped]
      BY WorkTypes DEF WorkTyped, Started
    <2> QED
      BY <2>1, Launch_Effect, Launch_Launched, Launch_Copies, WorkTypes DEF WorkTyped
  <1>4. RequestTyped'
    <2>1. /\ NewBatch(g).reads \subseteq Owners \X AllUnits
          /\ NewBatch(g).cr \subseteq AllUnits /\ NewBatch(g).cw \subseteq AllUnits
      BY PlaneTypes, WorkTypes DEF PlaneTyped, WorkTyped, NewBatch, LaunchRows, LaunchStart, Rows
    <2> QED
      BY <2>1, Launch_Effect, RequestTypes DEF RequestTyped
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, TypeInvNext

LEMMA Launch_Frame ==
  ASSUME IndInv, NEW g \in Groups, Launch(g), UNCHANGED rare
  PROVE  /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
         /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ MovingUnmapped'
         /\ TablesMapped' /\ HandoffOk' /\ NoBad'
  <1>1. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks,
                    trial, member, valid, vis, dbt, handoff, requests, alloc, sharedKey,
                    remote, lost, bad>>
    BY Launch_Effect DEF rare
  <1>2. FreeClean' /\ NoLeak'
    BY <1>1, Launch_Held DEF IndInv, FreeClean, NoLeak, InRegion
  <1> QED
    BY <1>1, <1>2, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame, SharedCachedFrame,
       CacheSharedFrame, TablesFrame, OwnersOkFrame, MovingUnmappedFrame, TablesMappedFrame,
       HandoffOkFrame, NoBadFrame, TransfersFrame DEF IndInv

LEMMA Launch_LostOk ==
  ASSUME IndInv, NEW g \in Groups, Launch(g), UNCHANGED rare
  PROVE  LostOk'
  BY Launch_Effect DEF IndInv, LostOk, NewBatch, rare

LEMMA Launch_DeferredOk ==
  ASSUME IndInv, NEW g \in Groups, Launch(g), UNCHANGED rare
  PROVE  DeferredOk'
  BY Launch_Copies, Launch_Effect DEF IndInv, DeferredOk, rare

LEMMA Launch_CopiesOk ==
  ASSUME IndInv, NEW g \in Groups, Launch(g), UNCHANGED rare
  PROVE  CopiesOk'
  <1>1. PICK n : n = launched[g] + 1
    OBVIOUS
  <1> DEFINE F(c) == Started(g, n, c)
  <1>2. \A x \in copies' : \E c \in copies : x = F(c)
    BY <1>1, Launch_Effect
  <1>3. \A c \in copies :
          /\ SameCopy(F(c), c)
          /\ F(c).seq = IF c.g = g /\ c.seq = 0 /\ ~c.stopped THEN n ELSE c.seq
          /\ F(c).done = c.done
    BY Launch_Copies DEF SameCopy
  <1>4. \A c \in copies : F(c).seq <= launched'[c.g] /\ (F(c).done => F(c).seq > 0)
    <2>1. \A c \in copies : /\ c.g \in Groups /\ c.seq <= launched[c.g] /\ (c.done => c.seq > 0)
                            /\ c.seq \in 0 .. MaxBatch /\ launched[c.g] \in 0 .. MaxBatch
      BY WorkTypes, PlaneTypes DEF IndInv, CopiesOk, WorkTyped, PlaneTyped
    <2> QED
      BY <1>1, <2>1, <1>3, Launch_Launched
  <1>5. \A x, y \in copies' : x.id = y.id => x = y
    BY <1>2, <1>3 DEF IndInv, CopiesOk, SameCopy
  <1>6. /\ \A x \in copies' : \E c \in copies : SameCopy(x, c)
        /\ \A x \in copies' : x.seq <= launched'[x.g] /\ (x.done => x.seq > 0)
    BY <1>2, <1>3, <1>4 DEF SameCopy
  <1>7. UNCHANGED <<ncopy, lost, alloc, cache, valid, tasks, table, ostate, ogroup, deferred>>
    BY Launch_Effect DEF rare
  <1> QED
    BY <1>5, <1>6, <1>7, CopiesMap DEF IndInv

LEMMA Launch_TasksOk ==
  ASSUME IndInv, NEW g \in Groups, Launch(g), UNCHANGED rare
  PROVE  TasksOk' /\ TrialVisibility'
  BY Launch_Copies, Launch_Effect DEF IndInv, TasksOk, Live, TableUnits, Mem, TrialVisibility,
                                      MadeBy

LEMMA Launch_RequestsOk ==
  ASSUME IndInv, NEW g \in Groups, Launch(g), UNCHANGED rare
  PROVE  RequestsOk'
  <1>1. \A r \in requests : \A x \in Groups : r.after[x] <= launched'[x]
    BY Launch_Launched, RequestTypes, PlaneTypes DEF IndInv, RequestsOk, RequestTyped, PlaneTyped,
                                                     Stamps
  <1>2. \A r \in requests : r.kind = "finish" =>
          \A x \in copies' :
            (x.id = r.c /\ GroupEpoch[x.g] = r.e) =>
                /\ x.g \in r.sg
                /\ x.g \in r.by =>
                     /\ x.seq > 0 => r.after[x.g] >= x.seq
                     /\ x.seq = 0 => x.stopped
    BY Launch_Copies, Launch_Effect DEF IndInv, RequestsOk
  <1> QED
    BY <1>1, <1>2, Launch_Effect DEF IndInv, RequestsOk

LEMMA Launch_BatchesOk ==
  ASSUME IndInv, NEW g \in Groups, Launch(g), UNCHANGED rare
  PROVE  BatchesOk'
  <1>0. PICK nb : nb = NewBatch(g)
    OBVIOUS
  <1>1. /\ inflight' = inflight \cup {nb}
        /\ nb.g = g /\ nb.seq = launched[g] + 1
        /\ nb.reads = {<<r[1], dbt[g][r[1]][r[2]].u>> : r \in LaunchRows(g)}
        /\ nb.writes = {<<r[1], dbt[g][r[1]][r[2]].u>> :
                          r \in {x \in LaunchRows(g) : dbt[g][x[1]][x[2]].w}}
        /\ nb.cr = {c.read : c \in LaunchStart(g)}
        /\ nb.cw = {c.write : c \in LaunchStart(g)}
    BY <1>0, Launch_Effect DEF NewBatch
  <1>2. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks,
                    trial, member, valid, vis, dbt, handoff, requests, alloc, sharedKey,
                    remote, lost, bad>>
    BY Launch_Effect DEF rare
  (* The rows the new batch addresses.                                     *)
  <1>3. ASSUME NEW o \in Owners, NEW p \in Rows, dbt[g][o][p].u # NONE
        PROVE  /\ dbt[g][o][p].u \notin free
               /\ alloc[dbt[g][o][p].u] \in {o, "shared"}
               /\ dbt[g][o][p].u \in Units => Mem(g) \in valid[dbt[g][o][p].u]
               /\ Covered(nb, o, dbt[g][o][p].u)
               /\ dbt[g][o][p].w =>
                    /\ alloc[dbt[g][o][p].u] = o
                    /\ \A d \in deferred :
                         (d.kind = "commit" /\ dbt[g][o][p].u \in d.units) => d.after[g] >= nb.seq
                    /\ \A r \in requests, q \in Positions :
                         (CommitsRow(r, o, q) /\ table[o][q].u = dbt[g][o][p].u) =>
                             g \in r.sg /\ (g \in r.by => r.after[g] >= nb.seq)
    <2> DEFINE u == dbt[g][o][p].u
    <2>1. /\ Live(o) /\ ogroup[o] = g /\ HoldsAt(o, p, u)
          /\ (p \in Positions /\ dbt[g][o][p].w) =>
               /\ table[o][p].k = "private"
               /\ ~\E d \in deferred : d.kind = "commit" /\ u \in d.units
      BY <1>3 DEF IndInv, TablesMapped
    <2>2. /\ ~\E d \in deferred : d.kind = "transfer" /\ d.o = o
          /\ ~\E t \in tasks : t.kind = "transfer" /\ t.o = o
      BY <1>3 DEF IndInv, MovingUnmapped
    <2>3. \A r \in requests : TakesRow(r, o, p) => g \in r.sg /\ g \notin r.by
      BY <1>3 DEF IndInv, RequestsOk, TakesRow, CommitsRow
    <2>4. Covered(nb, o, u)
      BY <1>1, <2>1, <2>2, <2>3 DEF Covered
    <2>5. Held(u) /\ u \notin free
      BY <2>1 DEF HoldsAt, Held, HeldIn, Rows, IndInv, FreeClean
    <2>6. CASE p = SLOT
      <3>1. u = oslot[o] /\ u \in Slots /\ alloc[u] = o
        BY <2>1, <2>6 DEF HoldsAt, IndInv, OwnersOk
      <3>2. \A d \in deferred : d.kind = "commit" => u \notin d.units
        BY <3>1 DEF IndInv, DeferredOk
      <3>3. \A q \in Positions : table[o][q].u # u
        BY <3>1, OwnerTypes DEF OwnerTyped, Empty, Private, Shared
      <3> QED
        BY <2>4, <2>5, <3>1, <3>2, <3>3
    <2>7. CASE p \in Positions
      <3>1. /\ table[o][p].u = u /\ u \in Units
            /\ table[o][p].k = "private" => alloc[u] = o
            /\ table[o][p].k # "private" => alloc[u] = "shared"
            /\ Mem(ogroup[o]) \in valid[u]
        BY <1>3, <2>1, <2>7, OwnerTypes DEF HoldsAt, IndInv, Tables, OwnerTyped, Empty, Private,
                                            Shared
      <3>2. ASSUME dbt[g][o][p].w, NEW r \in requests, NEW q \in Positions,
                   CommitsRow(r, o, q), table[o][q].u = u
            PROVE  g \in r.sg /\ (g \in r.by => r.after[g] >= nb.seq)
        <4>1. q = p
          BY <2>1, <2>7, <3>1, <3>2 DEF IndInv, Tables
        <4> QED
          BY <1>3, <3>2, <4>1 DEF IndInv, RequestsOk, CommitsRow
      <3> QED
        BY <2>1, <2>4, <2>5, <2>7, <3>1, <3>2
    <2> QED
      BY <2>6, <2>7 DEF Rows
  <1>4. /\ \A x \in nb.reads :
             /\ x[1] \in Owners /\ x[2] \notin free' /\ alloc'[x[2]] \in {x[1], "shared"}
             /\ (x[2] \in Units => Mem(nb.g) \in valid'[x[2]])
             /\ Covered(nb, x[1], x[2])'
        /\ \A x \in nb.writes :
             /\ alloc'[x[2]] = x[1]
             /\ \A d \in deferred' :
                  (d.kind = "commit" /\ x[2] \in d.units) => d.after[nb.g] >= nb.seq
             /\ \A r \in requests', q \in Positions :
                  (CommitsRow(r, x[1], q) /\ table'[x[1]][q].u = x[2]) =>
                      nb.g \in r.sg /\ (nb.g \in r.by => r.after[nb.g] >= nb.seq)
    BY <1>1, <1>2, <1>3, Launch_Held DEF LaunchRows
  <1>5. /\ \A u \in nb.cw : \E c \in copies' : c.write = u /\ c.g = nb.g /\ c.seq = nb.seq
        /\ \A u \in nb.cr : \E c \in copies' : c.read = u /\ c.g = nb.g /\ c.seq = nb.seq
    BY <1>1, Launch_Copies, Launch_Effect DEF LaunchStart
  <1>6. /\ nb.g \in Groups /\ GroupEpoch[nb.g] \notin lost'
        /\ nb.seq \in 1 .. launched'[nb.g]
        /\ \A x \in inflight' : (x.g = nb.g /\ x.seq = nb.seq) => x = nb
        /\ nb.writes \subseteq nb.reads
    <2>1. \A x \in inflight : x.g = g => x.seq <= launched[g]
      BY DEF IndInv, BatchesOk
    <2> QED
      BY <1>1, <1>2, <2>1, Launch_Effect, Launch_Launched DEF LaunchRows
  <1>7. \A b \in inflight :
          /\ b.seq \in 1 .. launched'[b.g]
          /\ \A x \in inflight' : (x.g = b.g /\ x.seq = b.seq) => x = b
          /\ \A u \in b.cw : \E c \in copies' : c.write = u /\ c.g = b.g /\ c.seq = b.seq
          /\ \A u \in b.cr : \E c \in copies' : c.read = u /\ c.g = b.g /\ c.seq = b.seq
    <2>1. \A b \in inflight : b.g \in Groups /\ b.seq \in 1 .. launched[b.g]
      BY DEF IndInv, BatchesOk
    <2>2. \A b \in inflight : \A c \in copies : c.seq = b.seq => Started(g, launched[g] + 1, c) = c
      BY <2>1 DEF Started
    <2> QED
      BY <1>1, <2>1, <2>2, Launch_Effect, Launch_Launched DEF IndInv, BatchesOk
  <1> QED
    BY <1>1, <1>2, <1>4, <1>5, <1>6, <1>7, Launch_Held DEF IndInv, BatchesOk

LEMMA Launch_Inv ==
  ASSUME IndInv, NEW g \in Groups, Launch(g), UNCHANGED rare
  PROVE  IndInv'
  BY Launch_Typed, Launch_Frame, Launch_LostOk, Launch_DeferredOk, Launch_CopiesOk,
     Launch_TasksOk, Launch_RequestsOk, Launch_BatchesOk DEF IndInv

-----------------------------------------------------------------------------
(* ExecFinish                                                              *)

(* The copy once its Finish is returned, and the Finish.                   *)
Finished(c) == [c EXCEPT !.done = TRUE]
FinishOf(c, ok) ==
    [Request("finish", GroupEpoch[c.g], NONE, {}, NONE, NONE, c.id, ok)
        EXCEPT !.after[c.g] = launched[c.g], !.by = {c.g}, !.sg = {c.g}]

LEMMA ExecFinish_Effect ==
  ASSUME NEW c \in copies, NEW ok \in BOOLEAN, ExecFinish(c, ok)
  PROVE  /\ c.seq > 0 /\ ~c.done /\ ~c.stopped
         /\ copies' = (copies \ {c}) \cup {Finished(c)}
         /\ requests' = requests \cup {FinishOf(c, ok)}
         /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks,
                        trial, member, valid, vis, dbt, launched, inflight, handoff, alloc,
                        sharedKey>>
  BY DEF ExecFinish, Finished, FinishOf

LEMMA ExecFinish_Values ==
  ASSUME IndInv, NEW c \in copies, NEW ok \in BOOLEAN
  PROVE  /\ Finished(c).id = c.id /\ Finished(c).kind = c.kind
         /\ Finished(c).o = c.o /\ Finished(c).p = c.p /\ Finished(c).k = c.k
         /\ Finished(c).g = c.g /\ Finished(c).read = c.read
         /\ Finished(c).write = c.write /\ Finished(c).fill = c.fill
         /\ Finished(c).seq = c.seq /\ Finished(c).stopped = c.stopped
         /\ Finished(c).done = TRUE
         /\ FinishOf(c, ok).kind = "finish" /\ FinishOf(c, ok).e = GroupEpoch[c.g]
         /\ FinishOf(c, ok).o = NONE /\ FinishOf(c, ok).ps = {} /\ FinishOf(c, ok).p = NONE
         /\ FinishOf(c, ok).k = NONE /\ FinishOf(c, ok).c = c.id /\ FinishOf(c, ok).to = NONE
         /\ FinishOf(c, ok).after = [x \in Groups |-> IF x = c.g THEN launched[c.g] ELSE 0]
         /\ FinishOf(c, ok).by = {c.g} /\ FinishOf(c, ok).sg = {c.g}
  <1>1. /\ c = [id |-> c.id, kind |-> c.kind, o |-> c.o, p |-> c.p, k |-> c.k, g |-> c.g,
              read |-> c.read, write |-> c.write, fill |-> c.fill, seq |-> c.seq,
              done |-> c.done, stopped |-> c.stopped]
        /\ c.g \in Groups
    BY WorkTypes DEF WorkTyped, IndInv, CopiesOk
  <1> QED
    BY <1>1 DEF Finished, FinishOf, Request, NoStamp

(* The copies after the step, as the image of a map that keeps every field *)
(* but done.                                                               *)
LEMMA ExecFinish_Copies ==
  ASSUME IndInv, NEW c \in copies, NEW ok \in BOOLEAN, ExecFinish(c, ok)
  PROVE  /\ copies' = {IF x = c THEN Finished(c) ELSE x : x \in copies}
         /\ \A x \in copies' : x \in copies \/ x = Finished(c)
         /\ Finished(c) \in copies'
         /\ \A x \in copies : x # c => x \in copies'
  BY ExecFinish_Effect

LEMMA ExecFinish_Held ==
  ASSUME IndInv, NEW c \in copies, NEW ok \in BOOLEAN, ExecFinish(c, ok)
  PROVE  /\ \A u : Held(u)' = Held(u)
         /\ \A x, o, u : Covered(x, o, u)' = Covered(x, o, u)
  <1>1. \A u : (\E x \in copies' : u = x.read \/ u = x.write)
                 = (\E x \in copies : u = x.read \/ u = x.write)
    BY ExecFinish_Copies, ExecFinish_Values
  <1>2. UNCHANGED <<table, oslot, cache, hcache, deferred, tasks>>
    BY ExecFinish_Effect
  <1>5. \A u : Held(u)' = Held(u)
    BY <1>1, <1>2, Zenon DEF Held, HeldIn
  <1>3. \A o, p : ~TakesRow(FinishOf(c, ok), o, p)
    BY ExecFinish_Values DEF TakesRow
  <1>4. \A x, o, u : Covered(x, o, u)' = Covered(x, o, u)
    BY <1>3, ExecFinish_Effect DEF Covered, Live, HoldsAt
  <1> QED
    BY <1>5, <1>4

LEMMA ExecFinish_Typed ==
  ASSUME IndInv, NEW c \in copies, NEW ok \in BOOLEAN, ExecFinish(c, ok),
         UNCHANGED rare
  PROVE  TypeInv'
  <1>1. UnitTyped' /\ OwnerTyped' /\ PlaneTyped'
    BY ExecFinish_Effect, UnitTypes, OwnerTypes, PlaneTypes, UnitTypedFrame, OwnerTypedFrame,
       PlaneTypedFrame DEF rare
  <1>2. WorkTyped'
    <2>1. LET x == Finished(c) IN
            x = [id |-> x.id, kind |-> x.kind, o |-> x.o, p |-> x.p, k |-> x.k, g |-> x.g,
                 read |-> x.read, write |-> x.write, fill |-> x.fill, seq |-> x.seq,
                 done |-> x.done, stopped |-> x.stopped]
      BY WorkTypes DEF WorkTyped, Finished
    <2> QED
      BY <2>1, ExecFinish_Copies, ExecFinish_Values, ExecFinish_Effect, WorkTypes DEF WorkTyped
  <1>3. RequestTyped'
    <2>1. /\ c.g \in Groups /\ c.id \in 1 .. ncopy /\ ncopy \in 0 .. MaxCopies
          /\ launched[c.g] \in 0 .. MaxBatch
      BY PlaneTypes DEF IndInv, CopiesOk, PlaneTyped
    <2>2. LET r == FinishOf(c, ok) IN
            /\ r.e \in Epochs /\ r.after \in Stamps /\ r.ps \subseteq Positions
            /\ r.o \in Owners \cup {NONE} /\ r.p \in Positions \cup {NONE}
            /\ r.k \in Keys \cup {NONE}
            /\ r.c \in 0 .. MaxCopies /\ r.to \in Groups \cup {NONE}
            /\ r.kind # "finish" => r.o \in Owners
            /\ r.kind = "commit" => r.p \in Positions /\ r.k \in Keys
            /\ r.kind = "transfer" => r.to \in Groups
            /\ r = [kind |-> r.kind, e |-> r.e, o |-> r.o, ps |-> r.ps, p |-> r.p, k |-> r.k,
                    c |-> r.c, ok |-> r.ok, to |-> r.to, after |-> r.after, by |-> r.by, sg |-> r.sg]
      BY <2>1, ExecFinish_Values DEF Stamps, FinishOf, Request, NoStamp
    <2> QED
      BY <2>2, ExecFinish_Effect, RequestTypes DEF RequestTyped
  <1> QED
    BY <1>1, <1>2, <1>3, TypeInvNext

LEMMA ExecFinish_Frame ==
  ASSUME IndInv, NEW c \in copies, NEW ok \in BOOLEAN, ExecFinish(c, ok),
         UNCHANGED rare
  PROVE  /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
         /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ MovingUnmapped'
         /\ TablesMapped' /\ NoBad'
  <1>1. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks,
                    trial, member, valid, vis, dbt, launched, inflight, handoff, alloc, sharedKey,
                    remote, lost, bad>>
    BY ExecFinish_Effect DEF rare
  <1>2. FreeClean' /\ NoLeak'
    BY <1>1, ExecFinish_Held DEF IndInv, FreeClean, NoLeak, InRegion
  <1> QED
    BY <1>1, <1>2, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame, SharedCachedFrame,
       CacheSharedFrame, TablesFrame, OwnersOkFrame, MovingUnmappedFrame, TablesMappedFrame,
       NoBadFrame, TransfersFrame DEF IndInv

LEMMA ExecFinish_CopiesOk ==
  ASSUME IndInv, NEW c \in copies, NEW ok \in BOOLEAN, ExecFinish(c, ok),
         UNCHANGED rare
  PROVE  CopiesOk' /\ DeferredOk' /\ TasksOk' /\ TrialVisibility'
  <1> DEFINE F(x) == IF x = c THEN Finished(c) ELSE x
  <1>1. copies' = {F(x) : x \in copies}
    BY ExecFinish_Copies
  <1>2. \A x \in copies :
          /\ F(x).id = x.id /\ F(x).kind = x.kind /\ F(x).o = x.o /\ F(x).p = x.p
          /\ F(x).k = x.k /\ F(x).g = x.g /\ F(x).read = x.read /\ F(x).write = x.write
          /\ F(x).fill = x.fill /\ F(x).seq = x.seq /\ F(x).stopped = x.stopped
          /\ F(x).seq <= launched'[x.g]
          /\ F(x).done => F(x).seq > 0
    <2>1. \A x \in copies : x.seq <= launched[x.g] /\ (x.done => x.seq > 0)
      BY DEF IndInv, CopiesOk
    <2> QED
      BY <2>1, ExecFinish_Values, ExecFinish_Effect
  <1>3. UNCHANGED <<ncopy, lost, alloc, cache, valid, tasks, table, ostate, ogroup, deferred>>
    BY ExecFinish_Effect DEF rare
  <1>4. CopiesOk'
    <2>1. /\ \A x \in copies' : \E y \in copies : SameCopy(x, y)
          /\ \A x \in copies' : x.seq <= launched'[x.g] /\ (x.done => x.seq > 0)
      BY <1>1, <1>2 DEF SameCopy
    <2>2. \A x, y \in copies' : x.id = y.id => x = y
      BY <1>1, <1>2 DEF IndInv, CopiesOk
    <2> QED
      BY <1>3, <2>1, <2>2, CopiesMap DEF IndInv
  <1>5. DeferredOk'
    BY <1>1, <1>2, ExecFinish_Effect DEF IndInv, DeferredOk, rare
  <1>6. TasksOk' /\ TrialVisibility'
    BY <1>1, <1>2, ExecFinish_Effect DEF IndInv, TasksOk, Live, TableUnits, Mem, TrialVisibility,
                                         MadeBy
  <1> QED
    BY <1>4, <1>5, <1>6

LEMMA ExecFinish_Requests ==
  ASSUME IndInv, NEW c \in copies, NEW ok \in BOOLEAN, ExecFinish(c, ok),
         UNCHANGED rare
  PROVE  RequestsOk' /\ HandoffOk' /\ LostOk' /\ BatchesOk'
  <1>1. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks,
                    trial, member, valid, vis, dbt, launched, inflight, handoff, alloc, sharedKey,
                    remote, lost, bad>>
    BY ExecFinish_Effect DEF rare
  <1>2. /\ c.g \in Groups /\ c.id \in 1 .. ncopy /\ c.seq <= launched[c.g]
        /\ GroupEpoch[c.g] \notin lost
        /\ \A x \in copies : x.id = c.id => x = c
    BY DEF IndInv, CopiesOk
  <1>3. \A o, p : ~CommitsRow(FinishOf(c, ok), o, p) /\ ~TakesRow(FinishOf(c, ok), o, p)
    BY ExecFinish_Values DEF CommitsRow, TakesRow
  <1>4. \A x \in copies' : \E y \in copies :
          /\ x.id = y.id /\ x.g = y.g /\ x.seq = y.seq /\ x.stopped = y.stopped
          /\ x.read = y.read /\ x.write = y.write
          /\ (x.id = c.id => y = c)
    BY <1>2, ExecFinish_Copies, ExecFinish_Values
  <1>5. \A y \in copies : \E x \in copies' :
          x.id = y.id /\ x.g = y.g /\ x.seq = y.seq /\ x.read = y.read /\ x.write = y.write
    BY ExecFinish_Copies, ExecFinish_Values
  <1>6. RequestsOk'
    <2>1. LET r == FinishOf(c, ok) IN
            /\ r.by \subseteq r.sg /\ r.sg \subseteq Groups
            /\ \A x \in r.sg : GroupEpoch[x] = r.e
            /\ \A x \in Groups : r.after[x] <= launched[x] /\ (x \notin r.by => r.after[x] = 0)
            /\ r.kind \in {"release", "detach", "commit", "transfer", "finish"}
            /\ r.c \in 1 .. ncopy
            /\ \A x \in copies' :
                 (x.id = r.c /\ GroupEpoch[x.g] = r.e) =>
                     /\ x.g \in r.sg
                     /\ x.g \in r.by =>
                          /\ x.seq > 0 => r.after[x.g] >= x.seq
                          /\ x.seq = 0 => x.stopped
      BY <1>2, <1>4, ExecFinish_Values, ExecFinish_Effect, PlaneTypes, WorkTypes
         DEF PlaneTyped, WorkTyped
    <2>2. \A r \in requests : r.kind = "finish" =>
            \A x \in copies' :
              (x.id = r.c /\ GroupEpoch[x.g] = r.e) =>
                  /\ x.g \in r.sg
                  /\ x.g \in r.by =>
                       /\ x.seq > 0 => r.after[x.g] >= x.seq
                       /\ x.seq = 0 => x.stopped
      BY <1>4 DEF IndInv, RequestsOk
    <2> QED
      BY <1>1, <2>1, <2>2, ExecFinish_Effect, ExecFinish_Values DEF IndInv, RequestsOk
  <1>7. HandoffOk'
    BY <1>1, ExecFinish_Effect DEF IndInv, HandoffOk, Live, HoldsAt
  <1>8. LostOk'
    BY <1>1, <1>2, ExecFinish_Effect, ExecFinish_Values DEF IndInv, LostOk
  <1>9. BatchesOk'
    BY <1>1, <1>3, <1>5, ExecFinish_Held, ExecFinish_Effect DEF IndInv, BatchesOk, Mem
  <1> QED
    BY <1>6, <1>7, <1>8, <1>9

LEMMA ExecFinish_Inv ==
  ASSUME IndInv, NEW c \in copies, NEW ok \in BOOLEAN, ExecFinish(c, ok),
         UNCHANGED rare
  PROVE  IndInv'
  BY ExecFinish_Typed, ExecFinish_Frame, ExecFinish_CopiesOk, ExecFinish_Requests DEF IndInv

-----------------------------------------------------------------------------

THEOREM Step5 ==
    IndInv /\ (/\ \/ \E g \in Groups : Launch(g)
                  \/ \E b \in inflight : EndBatch(b)
                  \/ \E c \in copies, ok \in BOOLEAN : ExecFinish(c, ok)
               /\ UNCHANGED rare)
    => IndInv'
  BY Launch_Inv, EndBatch_Inv, ExecFinish_Inv
=============================================================================
