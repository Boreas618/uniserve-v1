------------------------------- MODULE KvStep4 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for the stamped requests and the           *)
(* completions: Issue, ExecStamp, ExecMap and DropCompletion.              *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, KvFrame, TLAPS

USE Sizes, Design
(* A reader's row and the readers of an owner's class are read through.   *)
USE DEF ReaderRow, ReadersOf

(* A group that reads an owner's class is of the owner's session.          *)
LEMMA ReaderEpoch ==
  ASSUME NEW o \in Owners, ogroup[o] \in Groups, GroupEpoch[ogroup[o]] = OwnerEpoch[o],
         NEW g \in ReadersOf(o)
  PROVE  g \in Groups /\ GroupEpoch[g] = OwnerEpoch[o]
  OBVIOUS

(* Every conjunct but TypeInv, LostOk, HandoffOk, RequestsOk and BatchesOk *)
(* reads neither the requests nor the executors' tables nor the            *)
(* completions not yet handed on.                                          *)
LEMMA PlaneFrame ==
  ASSUME IndInv,
         UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                     tasks, valid, vis, trial, member, launched, inflight, alloc, sharedKey,
                     remote, lost, bad>>
  PROVE  /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
         /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
         /\ TasksOk' /\ TrialVisibility' /\ NoBad' /\ CrossingsOk' /\ CopiesPlaced'
         /\ CrossingsPlaced' /\ ImportsPlaced'
  BY HandoffTasksFrame, FreeCleanFrame, NoLeakFrame, OutsideFrame, RegionsFrame, AllocOwnerFrame,
     SharedIffFrame, SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame,
     DeferredOkFrame, CopiesOkFrame, TasksOkFrame, TrialVisibilityFrame, NoBadFrame,
     CrossingsOkFrame, CopiesPlacedFrame, CrossingsPlacedFrame, ImportsPlacedFrame DEF IndInv

-----------------------------------------------------------------------------
(* Issue                                                                   *)

(* A request as Issue makes it: a record of its fields, of an epoch that   *)
(* is not lost, stamped by no group yet, to be stamped by the groups of    *)
(* its session that may hold what it concerns.                             *)
IssuedOk(x) ==
    /\ x = [kind |-> x.kind, e |-> x.e, o |-> x.o, ps |-> x.ps, p |-> x.p, k |-> x.k,
            c |-> x.c, ok |-> x.ok, to |-> x.to, after |-> x.after, by |-> x.by, sg |-> x.sg]
    /\ x.kind \in {"release", "detach", "commit", "lend", "replicate", "finish"}
    /\ x.e \in Epochs /\ x.e \notin lost
    /\ x.after = NoStamp /\ x.by = {}
    /\ x.sg = {g \in Groups :
                 /\ GroupEpoch[g] = x.e
                 /\ IF x.kind = "finish"
                    THEN \E c \in copies : c.id = x.c /\ c.g = g
                    ELSE ogroup[x.o] = g \/ g \in ReadersOf(x.o)}
    /\ x.ps \subseteq Positions /\ x.p \in Positions \cup {NONE} /\ x.k \in Keys \cup {NONE}
    /\ x.to \in Groups \cup {NONE}
    /\ x.kind # "finish" => x.o \in Owners /\ x.c = 0
    /\ x.kind = "finish" => x.o = NONE /\ x.c \in 1 .. ncopy
    /\ x.kind = "commit" => x.p \in Positions /\ x.k \in Keys
    /\ x.kind = "replicate" => x.p \in Positions /\ x.to \in Groups

LEMMA Issue_Effect ==
  ASSUME Issue
  PROVE  /\ \E x : IssuedOk(x) /\ requests' = requests \cup {x}
         /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                        tasks, valid, vis, trial, member, dbt, launched, inflight, handoff, alloc,
                        sharedKey>>
  <1>1. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                    tasks, valid, vis, trial, member, dbt, launched, inflight, handoff, alloc,
                    sharedKey>>
    BY DEF Issue, pool
  <1>2. PICK e \in Epochs \ lost :
          \/ \E o \in Owners :
               \/ \E ps \in (SUBSET Positions) \ {{}} :
                    requests' = requests \cup {Issued(Request("release", e, o, ps, NONE, NONE, 0, FALSE))}
               \/ requests' = requests \cup {Issued(Request("detach", e, o, {}, NONE, NONE, 0, FALSE))}
               \/ \E p \in Positions, k \in Keys :
                    requests' = requests \cup {Issued(Request("commit", e, o, {}, p, k, 0, FALSE))}
               \/ /\ Handoffs
                  /\ requests' = requests \cup
                       {Issued(Request("lend", e, o, {}, NONE, NONE, 0, FALSE))}
               \/ \E p \in Positions, g \in Groups :
                    /\ Crosses
                    /\ requests' = requests \cup
                         {Issued([Request("replicate", e, o, {}, p, NONE, 0, FALSE) EXCEPT !.to = g])}
          \/ \E c \in 1 .. ncopy :
               /\ "H13" \notin Hazards
               /\ requests' = requests \cup
                    {Issued(Request("finish", e, NONE, {}, NONE, NONE, c, FALSE))}
    BY DEF Issue
  <1>3. \E x : IssuedOk(x) /\ requests' = requests \cup {x}
    <2>1. CASE \E o \in Owners, ps \in (SUBSET Positions) \ {{}} :
                 requests' = requests \cup {Issued(Request("release", e, o, ps, NONE, NONE, 0, FALSE))}
      <3>1. PICK o \in Owners, ps \in (SUBSET Positions) \ {{}} :
              requests' = requests \cup {Issued(Request("release", e, o, ps, NONE, NONE, 0, FALSE))}
        BY <2>1
      <3>2. IssuedOk(Issued(Request("release", e, o, ps, NONE, NONE, 0, FALSE)))
        BY DEF IssuedOk, Issued, Request, NoStamp
      <3> QED
        BY <3>1, <3>2
    <2>2. CASE \E o \in Owners :
                 requests' = requests \cup {Issued(Request("detach", e, o, {}, NONE, NONE, 0, FALSE))}
      <3>1. PICK o \in Owners :
              requests' = requests \cup {Issued(Request("detach", e, o, {}, NONE, NONE, 0, FALSE))}
        BY <2>2
      <3>2. IssuedOk(Issued(Request("detach", e, o, {}, NONE, NONE, 0, FALSE)))
        BY DEF IssuedOk, Issued, Request, NoStamp
      <3> QED
        BY <3>1, <3>2
    <2>3. CASE \E o \in Owners, p \in Positions, k \in Keys :
                 requests' = requests \cup {Issued(Request("commit", e, o, {}, p, k, 0, FALSE))}
      <3>1. PICK o \in Owners, p \in Positions, k \in Keys :
              requests' = requests \cup {Issued(Request("commit", e, o, {}, p, k, 0, FALSE))}
        BY <2>3
      <3>2. IssuedOk(Issued(Request("commit", e, o, {}, p, k, 0, FALSE)))
        BY DEF IssuedOk, Issued, Request, NoStamp
      <3> QED
        BY <3>1, <3>2
    <2>4. CASE \E o \in Owners :
                 requests' = requests \cup
                   {Issued(Request("lend", e, o, {}, NONE, NONE, 0, FALSE))}
      <3>1. PICK o \in Owners :
              requests' = requests \cup
                {Issued(Request("lend", e, o, {}, NONE, NONE, 0, FALSE))}
        BY <2>4
      <3>2. IssuedOk(Issued(Request("lend", e, o, {}, NONE, NONE, 0, FALSE)))
        BY DEF IssuedOk, Issued, Request, NoStamp
      <3> QED
        BY <3>1, <3>2
    <2>6. CASE \E o \in Owners, p \in Positions, g \in Groups :
                 requests' = requests \cup
                   {Issued([Request("replicate", e, o, {}, p, NONE, 0, FALSE) EXCEPT !.to = g])}
      <3>1. PICK o \in Owners, p \in Positions, g \in Groups :
              requests' = requests \cup
                {Issued([Request("replicate", e, o, {}, p, NONE, 0, FALSE) EXCEPT !.to = g])}
        BY <2>6
      <3>2. IssuedOk(Issued([Request("replicate", e, o, {}, p, NONE, 0, FALSE) EXCEPT !.to = g]))
        BY DEF IssuedOk, Issued, Request, NoStamp
      <3> QED
        BY <3>1, <3>2
    <2>5. CASE \E c \in 1 .. ncopy :
                 requests' = requests \cup {Issued(Request("finish", e, NONE, {}, NONE, NONE, c, FALSE))}
      <3>1. PICK c \in 1 .. ncopy :
              requests' = requests \cup {Issued(Request("finish", e, NONE, {}, NONE, NONE, c, FALSE))}
        BY <2>5
      <3>2. IssuedOk(Issued(Request("finish", e, NONE, {}, NONE, NONE, c, FALSE)))
        BY DEF IssuedOk, Issued, Request, NoStamp
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <1>2, <2>1, <2>2, <2>3, <2>4, <2>5, <2>6
  <1> QED
    BY <1>1, <1>3

LEMMA Issue_Inv ==
  ASSUME IndInv, Issue, UNCHANGED rare
  PROVE  IndInv'
  <1>1. PICK x : IssuedOk(x) /\ requests' = requests \cup {x}
    BY Issue_Effect
  <1>2. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                    tasks, valid, vis, trial, member, dbt, launched, inflight, handoff, alloc,
                    sharedKey, remote, lost, bad>>
    BY Issue_Effect DEF rare
  <1>3. TypeInv'
    <2>1. RequestTyped'
      <3>1. /\ x.e \in Epochs /\ x.after \in Stamps /\ x.ps \subseteq Positions
            /\ x.o \in Owners \cup {NONE} /\ x.p \in Positions \cup {NONE} /\ x.k \in Keys \cup {NONE}
            /\ x.c \in 0 .. MaxCopies /\ x.to \in Groups \cup {NONE}
            /\ x.kind # "finish" => x.o \in Owners
            /\ x.kind = "commit" => x.p \in Positions /\ x.k \in Keys
            /\ x.kind = "replicate" => x.p \in Positions /\ x.to \in Groups
        BY <1>1, PlaneTypes DEF IssuedOk, PlaneTyped, NoStamp, Stamps, NONE
      <3> QED
        BY <1>1, <1>2, <3>1, RequestTypes DEF RequestTyped, IssuedOk
    <2> QED
      BY <1>2, <2>1, UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, UnitTypedFrame,
         OwnerTypedFrame, PlaneTypedFrame, WorkTypedFrame, TypeInvNext
  <1>4. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
        /\ TasksOk' /\ TrialVisibility' /\ NoBad' /\ CrossingsOk' /\ CopiesPlaced'
        /\ CrossingsPlaced' /\ ImportsPlaced'
    BY <1>2, PlaneFrame
  <1>5. LendingUnmapped' /\ TablesMapped'
    BY <1>2, HandoffTasksFrame, LendingUnmappedFrame, TablesMappedFrame DEF IndInv
  <1>6. LostOk' /\ HandoffOk'
    BY <1>1, <1>2 DEF IndInv, LostOk, HandoffOk, IssuedOk, Live, HoldsAt
  (* The new request names entries only in the table of the group its      *)
  (* owner is attached to, which stamps it.                                *)
  <1>7. RequestsOk'
    <2>1. /\ x.by \subseteq x.sg /\ x.sg \subseteq Groups
          /\ \A g \in x.sg : GroupEpoch[g] = x.e
          /\ \A g \in Groups : x.after[g] <= launched[g] /\ (g \notin x.by => x.after[g] = 0)
          /\ x.kind \in {"release", "detach", "commit", "lend", "replicate", "finish"}
      BY <1>1, PlaneTypes DEF IssuedOk, NoStamp, PlaneTyped
    <2>2. x.kind # "finish" =>
            \A g \in Groups, p \in Rows :
              (dbt[g][x.o][p].u # NONE /\ (TakesRow(x, x.o, p) \/ FreezesRow(x, x.o, p))) =>
                  /\ g \in x.sg
                  /\ g \in x.by => FreezesRow(x, x.o, p) /\ ~dbt[g][x.o][p].w
      <3> SUFFICES ASSUME x.kind # "finish", NEW g \in Groups, NEW p \in Rows,
                          dbt[g][x.o][p].u # NONE, TakesRow(x, x.o, p) \/ FreezesRow(x, x.o, p)
                   PROVE  g \in x.sg
        BY <1>1 DEF IssuedOk
      <3>1. x.o \in Owners /\ x.e = OwnerEpoch[x.o]
        BY <1>1 DEF IssuedOk, TakesRow, FreezesRow
      <3>2. /\ Live(x.o) /\ ogroup[x.o] \in Groups /\ GroupEpoch[ogroup[x.o]] = OwnerEpoch[x.o]
            /\ ogroup[x.o] = g \/ g \in ReadersOf(x.o)
        BY <3>1 DEF IndInv, TablesMapped, OwnersOk, Live
      <3>3. GroupEpoch[g] = OwnerEpoch[x.o]
        BY <3>1, <3>2, ReaderEpoch
      <3> QED
        BY <1>1, <3>1, <3>2, <3>3 DEF IssuedOk
    <2>3. x.kind = "finish" =>
            /\ x.c \in 1 .. ncopy
            /\ \A c \in copies :
                 (c.id = x.c /\ GroupEpoch[c.g] = x.e) =>
                     /\ c.g \in x.sg
                     /\ c.g \in x.by =>
                          /\ c.seq > 0 => x.after[c.g] >= c.seq
                          /\ c.seq = 0 => c.stopped
      BY <1>1 DEF IssuedOk, IndInv, CopiesOk
    <2> QED
      BY <1>1, <1>2, <2>1, <2>2, <2>3 DEF IndInv, RequestsOk
  (* The new request is stamped by the group of a batch that addresses     *)
  (* what it names, so the batch stays covered.                            *)
  <1>8. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>2, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. ASSUME NEW o \in Owners, Live(o), ogroup[o] = b.g \/ b.g \in ReadersOf(o), NEW p \in Rows,
                 TakesRow(x, o, p)
          PROVE  b.g \in x.sg /\ b.g \notin x.by
      <3>1. x.o = o /\ x.e = OwnerEpoch[o] /\ x.kind # "finish"
        BY <2>3 DEF TakesRow
      <3>2. ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o]
        BY <2>3 DEF IndInv, OwnersOk
      <3>3. GroupEpoch[b.g] = OwnerEpoch[o] /\ b.g \in Groups
        BY <2>2, <2>3, <3>2, ReaderEpoch DEF BatchOk
      <3> QED
        BY <1>1, <2>3, <3>1, <3>3 DEF IssuedOk
    <2>4. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
      <3> SUFFICES ASSUME NEW o \in Owners, NEW u, Covered(b, o, u) PROVE Covered(b, o, u)'
        OBVIOUS
      <3>1. CASE /\ Live(o) /\ (ogroup[o] = b.g \/ b.g \in ReadersOf(o))
                 /\ \E p \in Rows :
                      /\ HoldsAt(o, p, u)
                      /\ \A r \in requests :
                           TakesRow(r, o, p) =>
                               b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
                 /\ \A d \in deferred : (d.kind = "lend" /\ d.o = o) => d.after[b.g] >= b.seq
                 /\ ~\E t \in tasks : t.kind = "lend" /\ t.o = o
        BY <1>1, <1>2, <2>3, <3>1 DEF Covered, Live, HoldsAt
      <3>2. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units /\ d.after[b.g] >= b.seq
        BY <1>2, <3>2 DEF Covered
      <3> QED
        BY <3>1, <3>2 DEF Covered
    (* A batch writes a private block of its owner, which no release holds, *)
    (* so the owner is attached to the batch's group.                       *)
    <2>5. ASSUME NEW y \in b.writes, NEW p \in Positions, FreezesRow(x, y[1], p),
                 table[y[1]][p].u = y[2]
          PROVE  b.g \in x.sg /\ b.g \notin x.by
      <3>1. y \in b.reads /\ alloc[y[2]] = y[1] /\ y[1] \in Owners /\ Covered(b, y[1], y[2])
        BY <2>2 DEF BatchOk
      <3>2. table[y[1]][p].k = "private"
        <4>1. y[2] \in AllUnits /\ y[2] # NONE
          BY <3>1, RequestTypes DEF RequestTyped, AllUnits
        <4> QED
          BY <2>5, <3>1, <4>1, OwnerTypes DEF IndInv, Tables, OwnerTyped, Empty, Private, Shared
      <3>3. ~\E d \in deferred : d.kind = "release" /\ y[2] \in d.units
        BY <2>5, <3>1, <3>2 DEF IndInv, DeferredOk
      <3>4. Live(y[1]) /\ (ogroup[y[1]] = b.g \/ b.g \in ReadersOf(y[1]))
        BY <3>1, <3>3 DEF Covered
      <3>5. x.o = y[1] /\ x.e = OwnerEpoch[y[1]] /\ x.kind \in {"commit", "replicate"}
        BY <2>5 DEF FreezesRow
      <3>6. ogroup[y[1]] \in Groups /\ GroupEpoch[ogroup[y[1]]] = OwnerEpoch[y[1]]
        BY <3>1, <3>4 DEF IndInv, OwnersOk
      <3>7. GroupEpoch[b.g] = OwnerEpoch[y[1]] /\ b.g \in Groups
        BY <2>2, <3>1, <3>4, <3>6, ReaderEpoch DEF BatchOk
      <3> QED
        BY <1>1, <3>4, <3>5, <3>7 DEF IssuedOk
    <2> QED
      BY <1>1, <1>2, <2>2, <2>4, <2>5 DEF BatchOk, Mem
  <1> QED
    BY <1>3, <1>4, <1>5, <1>6, <1>7, <1>8 DEF IndInv

-----------------------------------------------------------------------------
(* ExecStamp                                                               *)

(* The values ExecStamp(g, r) computes: r stamped by g, the rows of g's    *)
(* table for r's owner once g applies r, and a copy given up.              *)
Stamped(g, r) == [r EXCEPT !.after[g] = launched[g], !.by = r.by \cup {g}]
StampRows(g, r) ==
    [p \in Rows |->
       CASE r.kind = "release" /\ p \in r.ps -> NoEntry
         [] r.kind \in {"detach", "lend"} -> NoEntry
         [] r.kind \in {"commit", "replicate"} /\ p = r.p -> [u |-> dbt[g][r.o][p].u, w |-> FALSE]
         [] OTHER -> dbt[g][r.o][p]]
Stop(n, c) == IF c.id = n THEN [c EXCEPT !.stopped = TRUE] ELSE c

LEMMA ExecStamp_Effect ==
  ASSUME NEW g \in Groups, NEW r \in requests, ExecStamp(g, r)
  PROVE  /\ g \in r.sg /\ g \notin r.by
         /\ \/ /\ r.kind = "finish" /\ \E c \in copies : c.id = r.c /\ c.done
               /\ requests' = requests \ {r}
               /\ UNCHANGED <<dbt, copies>>
            \/ /\ r.kind = "finish" /\ ~\E c \in copies : c.id = r.c /\ c.done
               /\ requests' = (requests \ {r}) \cup {Stamped(g, r)}
               /\ copies' = {Stop(r.c, c) : c \in copies}
               /\ UNCHANGED dbt
            \/ /\ r.kind # "finish"
               /\ requests' = (requests \ {r}) \cup {Stamped(g, r)}
               /\ dbt' = [dbt EXCEPT ![g][r.o] = StampRows(g, r)]
               /\ UNCHANGED copies
         /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks,
                        trial, member, valid, vis, launched, inflight, handoff, alloc, sharedKey>>
  BY DEF ExecStamp, StampGroups, Stamped, StampRows, Stop

(* A request stamped by g differs from it in its stamp of g and in who     *)
(* stamped it; it names what the request names.                            *)
LEMMA ExecStamp_Values ==
  ASSUME IndInv, NEW g \in Groups, NEW r \in requests
  PROVE  LET x == Stamped(g, r) IN
         /\ x.kind = r.kind /\ x.e = r.e /\ x.o = r.o /\ x.ps = r.ps /\ x.p = r.p /\ x.k = r.k
         /\ x.c = r.c /\ x.ok = r.ok /\ x.to = r.to /\ x.sg = r.sg
         /\ x.by = r.by \cup {g}
         /\ x.after = [y \in Groups |-> IF y = g THEN launched[g] ELSE r.after[y]]
         /\ x = [kind |-> x.kind, e |-> x.e, o |-> x.o, ps |-> x.ps, p |-> x.p, k |-> x.k,
                 c |-> x.c, ok |-> x.ok, to |-> x.to, after |-> x.after, by |-> x.by, sg |-> x.sg]
         /\ \A h : Names(x, h) = Names(r, h)
         /\ \A o, p : TakesRow(x, o, p) = TakesRow(r, o, p) /\ FreezesRow(x, o, p) = FreezesRow(r, o, p)
  <1>1. /\ r = [kind |-> r.kind, e |-> r.e, o |-> r.o, ps |-> r.ps, p |-> r.p, k |-> r.k,
                c |-> r.c, ok |-> r.ok, to |-> r.to, after |-> r.after, by |-> r.by, sg |-> r.sg]
        /\ r.after \in [Groups -> 0 .. MaxBatch]
    BY RequestTypes DEF RequestTyped, Stamps
  <1>2. LET x == Stamped(g, r) IN
        /\ x.kind = r.kind /\ x.e = r.e /\ x.o = r.o /\ x.ps = r.ps /\ x.p = r.p /\ x.k = r.k
        /\ x.c = r.c /\ x.ok = r.ok /\ x.to = r.to /\ x.sg = r.sg
        /\ x.by = r.by \cup {g}
        /\ x.after = [y \in Groups |-> IF y = g THEN launched[g] ELSE r.after[y]]
        /\ x = [kind |-> x.kind, e |-> x.e, o |-> x.o, ps |-> x.ps, p |-> x.p, k |-> x.k,
                c |-> x.c, ok |-> x.ok, to |-> x.to, after |-> x.after, by |-> x.by, sg |-> x.sg]
    BY <1>1 DEF Stamped
  <1> QED
    BY <1>2 DEF Names, TakesRow, FreezesRow

(* A copy given up differs from it in stopped alone.                       *)
LEMMA ExecStamp_Stop ==
  ASSUME IndInv, NEW n
  PROVE  \A c \in copies :
           /\ SameCopy(Stop(n, c), c)
           /\ Stop(n, c).seq = c.seq /\ Stop(n, c).done = c.done
           /\ Stop(n, c).stopped = IF c.id = n THEN TRUE ELSE c.stopped
           /\ Stop(n, c) = [id |-> Stop(n, c).id, kind |-> Stop(n, c).kind, o |-> Stop(n, c).o,
                            p |-> Stop(n, c).p, k |-> Stop(n, c).k, g |-> Stop(n, c).g,
                            read |-> Stop(n, c).read, write |-> Stop(n, c).write,
                            fill |-> Stop(n, c).fill, parts |-> Stop(n, c).parts,
                            seq |-> Stop(n, c).seq, done |-> Stop(n, c).done,
                            stopped |-> Stop(n, c).stopped]
  BY WorkTypes DEF WorkTyped, Stop, SameCopy

(* Replacing r by r stamped by g keeps every batch's cover: a group's      *)
(* stamp is the last batch it launched.                                    *)
LEMMA ExecStamp_Covered ==
  ASSUME IndInv, NEW g \in Groups, NEW r \in requests,
         requests' = (requests \ {r}) \cup {Stamped(g, r)} \/ requests' = requests \ {r},
         UNCHANGED <<ostate, ogroup, oslot, table, deferred, tasks, launched>>
  PROVE  /\ \A b \in inflight, o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
         /\ \A b \in inflight, o \in Owners, p \in Positions :
              (\A x \in requests :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
              => \A x \in requests' :
                   FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
  <1>1. \A b \in inflight : b.g \in Groups /\ b.seq \in 1 .. launched[b.g]
    BY DEF IndInv, BatchesOk
  <1>2. /\ \A b \in inflight :
             LET x == Stamped(g, r) IN
             /\ b.g \in x.sg <=> b.g \in r.sg
             /\ b.g \in x.by =>
                  \/ b.g \in r.by /\ x.after[b.g] = r.after[b.g]
                  \/ x.after[b.g] >= b.seq
        /\ \A o, p : TakesRow(Stamped(g, r), o, p) = TakesRow(r, o, p)
        /\ \A o, p : FreezesRow(Stamped(g, r), o, p) = FreezesRow(r, o, p)
    <2>1. \A b \in inflight : launched[b.g] \in Nat /\ b.seq \in Nat
      BY <1>1, PlaneTypes DEF PlaneTyped
    <2> QED
      BY <1>1, <2>1, ExecStamp_Values
  <1>3. \A b \in inflight, o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
    <2> SUFFICES ASSUME NEW b \in inflight, NEW o \in Owners, NEW u, Covered(b, o, u)
                 PROVE  Covered(b, o, u)'
      OBVIOUS
    <2>1. CASE /\ Live(o) /\ (ogroup[o] = b.g \/ b.g \in ReadersOf(o))
               /\ \E p \in Rows :
                    /\ HoldsAt(o, p, u)
                    /\ \A x \in requests :
                         TakesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
               /\ \A d \in deferred : (d.kind = "lend" /\ d.o = o) => d.after[b.g] >= b.seq
               /\ ~\E t \in tasks : t.kind = "lend" /\ t.o = o
      BY <1>2, <2>1 DEF Covered, Live, HoldsAt
    <2>2. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units /\ d.after[b.g] >= b.seq
      BY <2>2 DEF Covered
    <2> QED
      BY <2>1, <2>2 DEF Covered
  <1> QED
    BY <1>2, <1>3

(* A give-up for a copy whose Finish was returned is discarded.            *)
LEMMA ExecStamp_Drop ==
  ASSUME IndInv, NEW g \in Groups, NEW r \in requests, ExecStamp(g, r), UNCHANGED rare,
         r.kind = "finish", requests' = requests \ {r}, UNCHANGED <<dbt, copies>>
  PROVE  IndInv'
  <1>1. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                    tasks, valid, vis, trial, member, dbt, launched, inflight, handoff, alloc,
                    sharedKey, remote, lost, bad>>
    BY ExecStamp_Effect DEF rare
  <1>2. TypeInv'
    <2>1. RequestTyped'
      BY <1>1, RequestTypes DEF RequestTyped
    <2> QED
      BY <1>1, <2>1, UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, UnitTypedFrame,
         OwnerTypedFrame, PlaneTypedFrame, WorkTypedFrame, TypeInvNext
  <1>3. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
        /\ TasksOk' /\ TrialVisibility' /\ NoBad' /\ CrossingsOk' /\ CopiesPlaced'
        /\ CrossingsPlaced' /\ ImportsPlaced'
    BY <1>1, PlaneFrame
  <1>4. LendingUnmapped' /\ TablesMapped'
    BY <1>1, HandoffTasksFrame, LendingUnmappedFrame, TablesMappedFrame DEF IndInv
  <1>5. \A h : (\E x \in requests' : Names(x, h)) = (\E x \in requests : Names(x, h))
    BY DEF Names
  <1>6. LostOk' /\ HandoffOk' /\ RequestsOk'
    BY <1>1, <1>5 DEF IndInv, LostOk, HandoffOk, RequestsOk, Live, HoldsAt
  <1>7. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
      BY <1>1, ExecStamp_Covered
    <2>4. \A o \in Owners, p \in Positions :
            (\A x \in requests :
               FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
            => \A x \in requests' :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
      BY <1>1, ExecStamp_Covered
    <2> QED
      BY <1>1, <2>2, <2>3, <2>4 DEF BatchOk, Mem
  <1> QED
    BY <1>2, <1>3, <1>4, <1>6, <1>7 DEF IndInv

(* A copy is given up: it runs with no further batch.                      *)
LEMMA ExecStamp_Give ==
  ASSUME IndInv, NEW g \in Groups, NEW r \in requests, ExecStamp(g, r), UNCHANGED rare,
         r.kind = "finish", ~\E c \in copies : c.id = r.c /\ c.done,
         requests' = (requests \ {r}) \cup {Stamped(g, r)},
         copies' = {Stop(r.c, c) : c \in copies}, UNCHANGED dbt
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. PICK x : x = Stamped(g, r)
    OBVIOUS
  <1>2. /\ requests' = (requests \ {r}) \cup {x}
        /\ g \in r.sg /\ g \notin r.by
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks,
                       valid, vis, trial, member, dbt, launched, inflight, handoff, alloc, sharedKey,
                       remote, lost, bad>>
    BY <1>1, ExecStamp_Effect DEF rare
  <1>3. /\ x.kind = r.kind /\ x.e = r.e /\ x.o = r.o /\ x.ps = r.ps /\ x.p = r.p /\ x.k = r.k
        /\ x.c = r.c /\ x.ok = r.ok /\ x.to = r.to /\ x.sg = r.sg
        /\ x.by = r.by \cup {g}
        /\ x.after = [y \in Groups |-> IF y = g THEN launched[g] ELSE r.after[y]]
        /\ x = [kind |-> x.kind, e |-> x.e, o |-> x.o, ps |-> x.ps, p |-> x.p, k |-> x.k,
                c |-> x.c, ok |-> x.ok, to |-> x.to, after |-> x.after, by |-> x.by, sg |-> x.sg]
        /\ \A h : Names(x, h) = Names(r, h)
    BY <1>1, ExecStamp_Values
  <1>4. /\ \A y \in copies' : \E c \in copies : y = Stop(r.c, c)
        /\ \A c \in copies : Stop(r.c, c) \in copies'
        /\ \A c \in copies :
             /\ SameCopy(Stop(r.c, c), c)
             /\ Stop(r.c, c).seq = c.seq /\ Stop(r.c, c).done = c.done
             /\ Stop(r.c, c).stopped = IF c.id = r.c THEN TRUE ELSE c.stopped
             /\ Stop(r.c, c) =
                  [id |-> Stop(r.c, c).id, kind |-> Stop(r.c, c).kind, o |-> Stop(r.c, c).o,
                   p |-> Stop(r.c, c).p, k |-> Stop(r.c, c).k, g |-> Stop(r.c, c).g,
                   read |-> Stop(r.c, c).read, write |-> Stop(r.c, c).write,
                   fill |-> Stop(r.c, c).fill, parts |-> Stop(r.c, c).parts,
                   seq |-> Stop(r.c, c).seq, done |-> Stop(r.c, c).done,
                   stopped |-> Stop(r.c, c).stopped]
    BY ExecStamp_Stop
  <1>5. /\ \A y \in copies' : \E c \in copies : SameCopy(y, c) /\ y.seq = c.seq /\ y.done = c.done
                                                /\ (c.stopped => y.stopped)
        /\ \A c \in copies : \E y \in copies' : SameCopy(y, c) /\ y.seq = c.seq /\ y.done = c.done
                                                /\ (c.stopped => y.stopped)
        /\ \A y, z \in copies' : y.id = z.id => y = z
    <2>1. \A c, d \in copies : c.id = d.id => c = d
      BY DEF IndInv, CopiesOk
    <2> QED
      BY <1>4, <2>1 DEF SameCopy
  <1>6. \A u : Held(u)' = Held(u)
    <2>1. \A u : (\E y \in copies' : u = y.read \/ u = y.write)
                   = (\E c \in copies : u = c.read \/ u = c.write)
      BY <1>5 DEF SameCopy
    <2> QED
      <3>1. \A u : HeldIn(u, table, oslot, cache, hcache, deferred, copies', tasks)
                     = HeldIn(u, table, oslot, cache, hcache, deferred, copies, tasks)
        BY <2>1 DEF HeldIn
      <3> QED
        BY <1>2, <3>1 DEF Held
  <1>7. TypeInv'
    <2>1. WorkTyped'
      BY <1>0, <1>2, <1>4 DEF WorkTyped, SameCopy
    <2>2. RequestTyped'
      <3>1. x.after \in Stamps
        BY <1>0, <1>3 DEF PlaneTyped, RequestTyped, Stamps
      <3> QED
        BY <1>0, <1>2, <1>3, <3>1 DEF RequestTyped
    <2> QED
      BY <1>2, <2>1, <2>2, UnitTypes, OwnerTypes, PlaneTypes, UnitTypedFrame, OwnerTypedFrame,
         PlaneTypedFrame, TypeInvNext
  <1>8. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ LendingUnmapped'
        /\ TablesMapped' /\ NoBad'
    BY <1>2, <1>6, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
       SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame, LendingUnmappedFrame,
       TablesMappedFrame, NoBadFrame, HandoffTasksFrame DEF IndInv, FreeClean, NoLeak, InRegion
  <1>9. CopiesOk'
    <2>1. \A y \in copies' : y.seq <= launched'[y.g] /\ (y.done => y.seq > 0)
      BY <1>2, <1>5 DEF IndInv, CopiesOk, SameCopy
    <2> QED
      BY <1>2, <1>5, <2>1, CopiesMap DEF IndInv
  <1>10. DeferredOk'
    <2>1. SUFFICES ASSUME NEW d \in deferred PROVE DeferOk(d)'
      BY <1>2, DeferredOkEach
    <2>2. DeferOk(d)
      BY DeferredOkEach DEF IndInv
    <2> QED
      BY <1>2, <1>5, <2>2 DEF DeferOk, SameCopy, Importing
  <1>11. /\ TasksOk' /\ TrialVisibility' /\ CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced'
         /\ ImportsPlaced'
    <2>1. TasksOk'
      <3>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
        BY <1>2, TasksOkEach
      <3>2. TaskOk(t)
        BY TasksOkEach DEF IndInv
      <3>3. \A o, p : CopyFor(o, p)' = CopyFor(o, p)
        <4>1. \A y \in copies' : \E c \in copies : SameCopy(y, c)
          BY <1>5
        <4>2. \A c \in copies : \E y \in copies' : SameCopy(y, c)
          BY <1>5
        <4> QED
          BY <4>1, <4>2 DEF CopyFor, SameCopy
      <3>4. \A u : (\E c \in copies' : c.read = u \/ c.write = u)
                     = (\E c \in copies : c.read = u \/ c.write = u)
        <4>1. \A y \in copies' : \E c \in copies : SameCopy(y, c)
          BY <1>5
        <4>2. \A c \in copies : \E y \in copies' : SameCopy(y, c)
          BY <1>5
        <4> QED
          BY <4>1, <4>2 DEF SameCopy
      <3> QED
        BY <1>2, <3>2, <3>3, <3>4, TaskOkCopies
    <2>2. TrialVisibility'
      BY <1>2, <1>5 DEF IndInv, TrialVisibility, SameCopy, MadeBy
    <2>3. \A o \in Owners : Lending(o)' = Lending(o)
      BY <1>2 DEF Lending
    <2>4. CrossingsOk'
      <3>1. \A d \in deferred : CrossWaitOk(d)'
        <4> SUFFICES ASSUME NEW d \in deferred PROVE CrossWaitOk(d)'
          OBVIOUS
        <4>1. CrossWaitOk(d)
          BY CrossingsOkEach DEF IndInv
        <4> QED
          BY <1>2, <1>5, <2>3, <4>1 DEF CrossWaitOk, SameCopy, MadeBy
      <3>2. \A t \in tasks : CrossTaskOk(t)'
        <4> SUFFICES ASSUME NEW t \in tasks PROVE CrossTaskOk(t)'
          OBVIOUS
        <4>1. CrossTaskOk(t)
          BY CrossingsOkEach DEF IndInv
        <4> QED
          BY <1>2, <1>5, <2>3, <4>1 DEF CrossTaskOk, SameCopy, Mem
      <3> QED
        BY <1>2, <3>1, <3>2, CrossingsOkEach
    <2>5. CopiesPlaced'
      <3>1. SUFFICES ASSUME NEW y \in copies' PROVE CopyPlaced(y)'
        BY PlacedEach
      <3>2. PICK c \in copies : SameCopy(y, c)
        BY <1>5
      <3>3. CopyPlaced(c)
        BY PlacedEach DEF IndInv
      <3> QED
        BY <1>2, <3>2, <3>3 DEF CopyPlaced, SameCopy, Mem
    <2>6. CrossingsPlaced'
      BY <1>2, HandoffTasksFrame, CrossingsPlacedFrame DEF IndInv
    <2>7. ImportsPlaced'
      BY <1>2, HandoffTasksFrame, ImportsPlacedFrame DEF IndInv
    <2> QED
      BY <2>1, <2>2, <2>4, <2>5, <2>6, <2>7
  <1>12. \A h : (\E y \in requests' : Names(y, h)) = (\E y \in requests : Names(y, h))
    BY <1>2, <1>3
  <1>13. LostOk' /\ HandoffOk'
    BY <1>2, <1>3, <1>12 DEF IndInv, LostOk, HandoffOk, Live, HoldsAt
  <1>14. RequestsOk'
    <2>1. /\ r.c \in 1 .. ncopy /\ r.by \subseteq r.sg /\ r.sg \subseteq Groups
          /\ \A y \in r.sg : GroupEpoch[y] = r.e
          /\ \A y \in Groups : r.after[y] <= launched[y] /\ (y \notin r.by => r.after[y] = 0)
          /\ \A c \in copies :
               (c.id = r.c /\ GroupEpoch[c.g] = r.e) =>
                   /\ c.g \in r.sg
                   /\ c.g \in r.by =>
                        /\ c.seq > 0 => r.after[c.g] >= c.seq
                        /\ c.seq = 0 => c.stopped
      BY DEF IndInv, RequestsOk
    <2>2. \A c \in copies : c.g \in Groups /\ c.seq <= launched[c.g] /\ c.seq \in Nat
                            /\ launched[c.g] \in Nat
      BY <1>0 DEF IndInv, CopiesOk, PlaneTyped, WorkTyped
    <2>3. /\ x.by \subseteq x.sg /\ x.sg \subseteq Groups
          /\ \A y \in x.sg : GroupEpoch[y] = x.e
          /\ \A y \in Groups : x.after[y] <= launched[y] /\ (y \notin x.by => x.after[y] = 0)
          /\ x.kind \in {"release", "detach", "commit", "lend", "replicate", "finish"}
          /\ x.kind = "finish" => x.c \in 1 .. ncopy
          /\ x.kind = "finish" =>
               \A y \in copies' :
                 (y.id = x.c /\ GroupEpoch[y.g] = x.e) =>
                     /\ y.g \in x.sg
                     /\ y.g \in x.by =>
                          /\ y.seq > 0 => x.after[y.g] >= y.seq
                          /\ y.seq = 0 => y.stopped
      <3>1. \A y \in copies' : y.id = r.c => y.stopped
        BY <1>4 DEF SameCopy
      <3> QED
        BY <1>0, <1>2, <1>3, <1>5, <2>1, <2>2, <3>1 DEF SameCopy, PlaneTyped
    <2>4. \A y \in requests \ {r} : y.kind = "finish" =>
            \A c \in copies' :
              (c.id = y.c /\ GroupEpoch[c.g] = y.e) =>
                  /\ c.g \in y.sg
                  /\ c.g \in y.by =>
                       /\ c.seq > 0 => y.after[c.g] >= c.seq
                       /\ c.seq = 0 => c.stopped
      BY <1>5 DEF IndInv, RequestsOk, SameCopy
    <2> QED
      BY <1>2, <1>3, <2>3, <2>4 DEF IndInv, RequestsOk, TakesRow, FreezesRow
  <1>15. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>2, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
      BY <1>2, ExecStamp_Covered
    <2>4. \A o \in Owners, p \in Positions :
            (\A y \in requests :
               FreezesRow(y, o, p) => b.g \in y.sg /\ (b.g \in y.by => y.after[b.g] >= b.seq))
            => \A y \in requests' :
                 FreezesRow(y, o, p) => b.g \in y.sg /\ (b.g \in y.by => y.after[b.g] >= b.seq)
      BY <1>2, ExecStamp_Covered
    <2> QED
      BY <1>2, <1>5, <2>2, <2>3, <2>4 DEF BatchOk, Mem, SameCopy
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>13, <1>14, <1>15 DEF IndInv

(* The executor applies a release, a detach, a lend or a commit.            *)
LEMMA ExecStamp_Apply ==
  ASSUME IndInv, NEW g \in Groups, NEW r \in requests, ExecStamp(g, r), UNCHANGED rare,
         r.kind # "finish",
         requests' = (requests \ {r}) \cup {Stamped(g, r)},
         dbt' = [dbt EXCEPT ![g][r.o] = StampRows(g, r)], UNCHANGED copies
  PROVE  IndInv'
  <1>0. PlaneTyped /\ RequestTyped
    BY PlaneTypes, RequestTypes
  <1>1. PICK x : x = Stamped(g, r)
    OBVIOUS
  <1>2. /\ requests' = (requests \ {r}) \cup {x}
        /\ g \in r.sg /\ g \notin r.by /\ r.o \in Owners
        /\ GroupEpoch[g] = r.e /\ r.e \notin lost
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       tasks, valid, vis, trial, member, launched, inflight, handoff, alloc,
                       sharedKey, remote, lost, bad>>
    BY <1>0, <1>1, ExecStamp_Effect DEF rare, RequestTyped, IndInv, RequestsOk, LostOk
  <1>3. /\ x.kind = r.kind /\ x.e = r.e /\ x.o = r.o /\ x.ps = r.ps /\ x.p = r.p /\ x.k = r.k
        /\ x.c = r.c /\ x.ok = r.ok /\ x.to = r.to /\ x.sg = r.sg
        /\ x.by = r.by \cup {g}
        /\ x.after = [y \in Groups |-> IF y = g THEN launched[g] ELSE r.after[y]]
        /\ x = [kind |-> x.kind, e |-> x.e, o |-> x.o, ps |-> x.ps, p |-> x.p, k |-> x.k,
                c |-> x.c, ok |-> x.ok, to |-> x.to, after |-> x.after, by |-> x.by, sg |-> x.sg]
        /\ \A h : Names(x, h) = Names(r, h)
        /\ \A o, p : TakesRow(x, o, p) = TakesRow(r, o, p) /\ FreezesRow(x, o, p) = FreezesRow(r, o, p)
    BY <1>1, ExecStamp_Values
  (* g's table loses the rows r takes away and the right to write the row  *)
  (* r commits; every other table is unchanged.                            *)
  <1>4. /\ \A y \in Groups, o \in Owners, p \in Rows :
             dbt'[y][o][p] = IF y = g /\ o = r.o THEN StampRows(g, r)[p] ELSE dbt[y][o][p]
        /\ \A p \in Rows :
             /\ \/ StampRows(g, r)[p].u = NONE
                \/ /\ StampRows(g, r)[p].u = dbt[g][r.o][p].u
                   /\ StampRows(g, r)[p].w => dbt[g][r.o][p].w
             /\ TakesRow(r, r.o, p) => StampRows(g, r)[p].u = NONE
             /\ FreezesRow(r, r.o, p) => ~StampRows(g, r)[p].w
             /\ StampRows(g, r)[p].u \in AllUnits \cup {NONE}
        /\ DOMAIN StampRows(g, r) = Rows
    BY <1>0, <1>2 DEF PlaneTyped, StampRows, NoEntry, TakesRow, FreezesRow
  <1>5. \A y \in Groups, o \in Owners, p \in Rows :
          \/ dbt'[y][o][p].u = NONE
          \/ dbt'[y][o][p].u = dbt[y][o][p].u /\ (dbt'[y][o][p].w => dbt[y][o][p].w)
    BY <1>4
  <1>6. TypeInv'
    <2>1. PlaneTyped'
      BY <1>0, <1>2, <1>4 DEF PlaneTyped
    <2>2. RequestTyped'
      <3>1. x.after \in Stamps
        BY <1>0, <1>3 DEF PlaneTyped, RequestTyped, Stamps
      <3> QED
        BY <1>0, <1>2, <1>3, <3>1 DEF RequestTyped
    <2> QED
      BY <1>2, <2>1, <2>2, UnitTypes, OwnerTypes, WorkTypes, UnitTypedFrame, OwnerTypedFrame,
         WorkTypedFrame, TypeInvNext
  <1>7. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
        /\ TasksOk' /\ TrialVisibility' /\ NoBad' /\ CrossingsOk' /\ CopiesPlaced'
        /\ CrossingsPlaced' /\ ImportsPlaced'
    BY <1>2, PlaneFrame
  <1>8. LendingUnmapped' /\ TablesMapped'
    BY <1>2, <1>5 DEF IndInv, LendingUnmapped, Lending, TablesMapped, Live, HoldsAt
  <1>9. \A h : (\E y \in requests' : Names(y, h)) = (\E y \in requests : Names(y, h))
    BY <1>2, <1>3
  <1>10. LostOk' /\ HandoffOk'
    BY <1>2, <1>3, <1>4, <1>9 DEF IndInv, LostOk, HandoffOk, Live, HoldsAt
  <1>11. RequestsOk'
    <2>1. /\ x.by \subseteq x.sg /\ x.sg \subseteq Groups
          /\ \A y \in x.sg : GroupEpoch[y] = x.e
          /\ \A y \in Groups : x.after[y] <= launched[y] /\ (y \notin x.by => x.after[y] = 0)
          /\ x.kind \in {"release", "detach", "commit", "lend", "replicate", "finish"}
          /\ x.kind # "finish"
      BY <1>0, <1>2, <1>3 DEF IndInv, RequestsOk, PlaneTyped
    <2>2. \A y \in Groups, p \in Rows :
            (dbt'[y][x.o][p].u # NONE /\ (TakesRow(x, x.o, p) \/ FreezesRow(x, x.o, p))) =>
                /\ y \in x.sg
                /\ y \in x.by => FreezesRow(x, x.o, p) /\ ~dbt'[y][x.o][p].w
      <3> SUFFICES ASSUME NEW y \in Groups, NEW p \in Rows, dbt'[y][x.o][p].u # NONE,
                          TakesRow(x, x.o, p) \/ FreezesRow(x, x.o, p)
                   PROVE  /\ y \in x.sg
                          /\ y \in x.by => FreezesRow(x, x.o, p) /\ ~dbt'[y][x.o][p].w
        OBVIOUS
      <3>1. CASE y = g
        BY <1>2, <1>3, <1>4, <3>1
      <3>2. CASE y # g
        <4>1. dbt'[y][r.o][p] = dbt[y][r.o][p]
          BY <1>2, <1>4, <3>2
        <4> QED
          BY <1>2, <1>3, <3>2, <4>1 DEF IndInv, RequestsOk
      <3> QED
        BY <3>1, <3>2
    <2>3. \A y \in requests \ {r} : y.kind # "finish" =>
            \A z \in Groups, p \in Rows :
              (dbt'[z][y.o][p].u # NONE /\ (TakesRow(y, y.o, p) \/ FreezesRow(y, y.o, p))) =>
                  /\ z \in y.sg
                  /\ z \in y.by => FreezesRow(y, y.o, p) /\ ~dbt'[z][y.o][p].w
      <3> SUFFICES ASSUME NEW y \in requests \ {r}, y.kind # "finish", NEW z \in Groups,
                          NEW p \in Rows, dbt'[z][y.o][p].u # NONE,
                          TakesRow(y, y.o, p) \/ FreezesRow(y, y.o, p)
                   PROVE  /\ z \in y.sg
                          /\ z \in y.by => FreezesRow(y, y.o, p) /\ ~dbt'[z][y.o][p].w
        OBVIOUS
      <3>1. y.o \in Owners
        BY <1>0 DEF RequestTyped
      <3>2. dbt'[z][y.o][p].u = dbt[z][y.o][p].u /\ (dbt'[z][y.o][p].w => dbt[z][y.o][p].w)
        BY <1>5, <3>1
      <3> QED
        BY <3>2 DEF IndInv, RequestsOk
    <2> QED
      BY <1>2, <1>3, <2>1, <2>2, <2>3 DEF IndInv, RequestsOk
  <1>12. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>2, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A o \in Owners : \A u : Covered(b, o, u) => Covered(b, o, u)'
      BY <1>2, ExecStamp_Covered
    <2>4. \A o \in Owners, p \in Positions :
            (\A y \in requests :
               FreezesRow(y, o, p) => b.g \in y.sg /\ (b.g \in y.by => y.after[b.g] >= b.seq))
            => \A y \in requests' :
                 FreezesRow(y, o, p) => b.g \in y.sg /\ (b.g \in y.by => y.after[b.g] >= b.seq)
      BY <1>2, ExecStamp_Covered
    <2> QED
      BY <1>2, <2>2, <2>3, <2>4 DEF BatchOk, Mem
  <1> QED
    BY <1>6, <1>7, <1>8, <1>10, <1>11, <1>12 DEF IndInv

LEMMA ExecStamp_Inv ==
  ASSUME IndInv, NEW g \in Groups, NEW r \in requests, ExecStamp(g, r), UNCHANGED rare
  PROVE  IndInv'
  BY ExecStamp_Effect, ExecStamp_Drop, ExecStamp_Give, ExecStamp_Apply

-----------------------------------------------------------------------------
(* ExecMap and DropCompletion                                              *)

(* A completion leaves the ones not yet handed on; ExecMap writes its row. *)
LEMMA Handoff_Common ==
  ASSUME IndInv, NEW h \in handoff, handoff' = handoff \ {h}, UNCHANGED rare,
         UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                     tasks, valid, vis, trial, member, launched, inflight, requests, alloc,
                     sharedKey>>
  PROVE  /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
         /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
         /\ TasksOk' /\ TrialVisibility' /\ NoBad' /\ BatchesOk' /\ HandoffOk'
         /\ UnitTyped' /\ OwnerTyped' /\ WorkTyped' /\ RequestTyped'
         /\ CrossingsOk' /\ CopiesPlaced' /\ CrossingsPlaced' /\ ImportsPlaced'
  <1>1. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                    tasks, valid, vis, trial, member, launched, inflight, requests, alloc,
                    sharedKey, remote, lost, bad>>
    BY DEF rare
  <1>2. BatchesOk'
    BY <1>1, HandoffTasksFrame, BatchesOkFrame DEF IndInv
  <1>3. HandoffOk'
    BY <1>1 DEF IndInv, HandoffOk, Live, HoldsAt
  <1>4. UnitTyped' /\ OwnerTyped' /\ WorkTyped' /\ RequestTyped'
    BY <1>1, UnitTypes, OwnerTypes, WorkTypes, RequestTypes, UnitTypedFrame, OwnerTypedFrame,
       WorkTypedFrame DEF RequestTyped
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, PlaneFrame

LEMMA DropCompletion_Inv ==
  ASSUME IndInv, NEW h \in handoff, DropCompletion(h), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ handoff' = handoff \ {h}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       tasks, valid, vis, trial, member, dbt, launched, inflight, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF DropCompletion, pool, rare
  <1>2. TypeInv'
    BY <1>1, Handoff_Common, PlaneTypes, PlaneTypedFrame, TypeInvNext
  <1>3. LendingUnmapped' /\ LostOk' /\ TablesMapped' /\ RequestsOk'
    BY <1>1, HandoffTasksFrame, TablesMappedFrame, RequestsOkFrame
       DEF IndInv, LendingUnmapped, Lending, LostOk
  <1> QED
    BY <1>1, <1>2, <1>3, Handoff_Common DEF IndInv

LEMMA ExecMap_Inv ==
  ASSUME IndInv, NEW h \in handoff, ExecMap(h), UNCHANGED rare
  PROVE  IndInv'
  <1>0. PlaneTyped /\ RequestTyped
    BY PlaneTypes, RequestTypes
  <1>1. /\ ~\E r \in requests : Names(r, h)
        /\ handoff' = handoff \ {h}
        /\ dbt' = [dbt EXCEPT ![h.g][h.o][h.p] = [u |-> h.u, w |-> h.w]]
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       tasks, valid, vis, trial, member, launched, inflight, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY DEF ExecMap, pool, rare
  <1>2. /\ h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows /\ h.u \in AllUnits
        /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
        /\ Live(h.o) /\ (ogroup[h.o] = h.g \/ ReaderRow(h.g, h.o, h.p, h.u, h.w))
        /\ HoldsAt(h.o, h.p, h.u)
        /\ (h.p \in Positions /\ h.w) =>
             /\ table[h.o][h.p].k = "private"
             /\ ~\E d \in deferred : d.kind \in {"commit", "replicate"} /\ h.u \in d.units
             /\ ~\E t \in tasks : t.kind = "replicate" /\ h.u \in t.units
    BY <1>0, <1>1 DEF IndInv, HandoffOk, RequestTyped
  <1>3. /\ GroupEpoch[h.g] \notin lost
        /\ ~Lending(h.o)
    BY <1>2 DEF IndInv, LostOk, LendingUnmapped
  (* The row the completion names is written; every other row is unchanged. *)
  <1>4. \A y \in Groups, o \in Owners, p \in Rows :
          dbt'[y][o][p] = IF y = h.g /\ o = h.o /\ p = h.p THEN [u |-> h.u, w |-> h.w]
                                                             ELSE dbt[y][o][p]
    BY <1>0, <1>1, <1>2 DEF PlaneTyped
  <1>5. TypeInv'
    <2>1. PlaneTyped'
      BY <1>0, <1>1, <1>2, <1>4 DEF PlaneTyped
    <2> QED
      BY <1>1, <2>1, Handoff_Common, TypeInvNext
  <1>6. LendingUnmapped'
    BY <1>1, <1>3, <1>4 DEF IndInv, LendingUnmapped, Lending
  <1>7. LostOk'
    BY <1>1, <1>3, <1>4 DEF IndInv, LostOk
  <1>8. TablesMapped'
    BY <1>1, <1>2, <1>4 DEF IndInv, TablesMapped, Live, HoldsAt
  (* No request under way names the row, so none takes it or commits it.    *)
  <1>9. RequestsOk'
    <2>1. \A r \in requests : ~(TakesRow(r, h.o, h.p) \/ FreezesRow(r, h.o, h.p))
      BY <1>1, NamesRow
    <2>2. \A r \in requests : r.kind # "finish" => r.o \in Owners
      BY <1>0 DEF RequestTyped
    <2> QED
      BY <1>1, <1>4, <2>1, <2>2 DEF IndInv, RequestsOk
  <1> QED
    BY <1>1, <1>5, <1>6, <1>7, <1>8, <1>9, Handoff_Common DEF IndInv

-----------------------------------------------------------------------------

THEOREM Step4 ==
    IndInv /\ (/\ \/ Issue
                  \/ \E g \in Groups, r \in requests : ExecStamp(g, r)
                  \/ \E h \in handoff : ExecMap(h) \/ DropCompletion(h)
               /\ UNCHANGED rare)
    => IndInv'
  BY Issue_Inv, ExecStamp_Inv, ExecMap_Inv, DropCompletion_Inv
=============================================================================
