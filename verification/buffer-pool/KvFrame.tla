------------------------------- MODULE KvFrame -------------------------------
(***************************************************************************)
(* Facts the proofs of the inductive step share: each conjunct of IndInv  *)
(* holds after a step that leaves every variable it reads unchanged;       *)
(* TypeInv in five parts; CopiesOk, DeferredOk, TasksOk and BatchesOk item *)
(* by item; what steps that reclaim units or change copies field by field  *)
(* keep; and what IndInv says of a cached block, of a block a handler      *)
(* obtains, and of a completion a handler gives.                           *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, TLAPS

(* The transfer tasks, the crossings' tasks, and the units the tasks hold:  *)
(* several conjuncts read the tasks only through them.                      *)
Transfers == {t \in tasks : t.kind = "transfer"}
Crossings == {t \in tasks : t.kind = "replicate"}
TaskUnits == UNION {t.units : t \in tasks}

LEMMA TransfersFrame ==
  ASSUME UNCHANGED tasks
  PROVE  Transfers' = Transfers /\ Crossings' = Crossings /\ TaskUnits' \subseteq TaskUnits
  BY DEF Transfers, Crossings, TaskUnits

LEMMA TypeInvFrame ==
  ASSUME TypeInv,
         UNCHANGED <<free, member, ostate, ogroup, oslot, table, cache, hcache, ncopy, valid, vis,
                     trial, launched, alloc, remote, lost, sharedKey, dbt, deferred, copies, tasks,
                     requests, inflight, handoff>>
  PROVE  TypeInv'
  BY DEF TypeInv, TypeOK, Stamps

LEMMA FreeCleanFrame ==
  ASSUME FreeClean,
         UNCHANGED <<free, alloc, sharedKey, table, oslot, cache, hcache, deferred, copies, tasks>>
  PROVE  FreeClean'
  BY DEF FreeClean, Held, HeldIn

LEMMA NoLeakFrame ==
  ASSUME NoLeak,
         UNCHANGED <<member, free, table, oslot, cache, hcache, deferred, copies, tasks>>
  PROVE  NoLeak'
  BY DEF NoLeak, InRegion, Held, HeldIn

LEMMA OutsideFrame ==
  ASSUME Outside, UNCHANGED <<member, alloc, sharedKey>>
  PROVE  Outside'
  BY DEF Outside

LEMMA RegionsFrame ==
  ASSUME Regions, UNCHANGED <<member, free>>
  PROVE  Regions'
  BY DEF Regions, OneRegion

LEMMA AllocOwnerFrame ==
  ASSUME AllocOwner, UNCHANGED <<alloc, ostate>>
  PROVE  AllocOwner'
  BY DEF AllocOwner

LEMMA SharedIffFrame ==
  ASSUME SharedIff, UNCHANGED <<alloc, sharedKey>>
  PROVE  SharedIff'
  BY DEF SharedIff

LEMMA SharedCachedFrame ==
  ASSUME SharedCached, UNCHANGED <<alloc, sharedKey, cache, hcache>>
  PROVE  SharedCached'
  BY DEF SharedCached

LEMMA CacheSharedFrame ==
  ASSUME CacheShared, UNCHANGED <<cache, hcache, sharedKey>>
  PROVE  CacheShared'
  BY DEF CacheShared

LEMMA TablesFrame ==
  ASSUME Tables, UNCHANGED <<table, alloc, ostate, ogroup, valid>>
  PROVE  Tables'
  BY DEF Tables, Live, Mem

LEMMA OwnersOkFrame ==
  ASSUME OwnersOk, UNCHANGED <<ostate, ogroup, lost, trial, oslot, alloc, table>>
  PROVE  OwnersOk'
  BY DEF OwnersOk, Live

LEMMA DeferredOkFrame ==
  ASSUME DeferredOk, UNCHANGED <<deferred, oslot, table, copies, lost, alloc, ostate>>
  PROVE  DeferredOk'
  BY DEF DeferredOk

LEMMA CopiesOkFrame ==
  ASSUME CopiesOk,
         UNCHANGED <<copies, ncopy, lost, launched, alloc, cache, hcache, valid, table, ostate,
                     ogroup, deferred>>,
         Transfers' = Transfers, TaskUnits' \subseteq TaskUnits
  PROVE  CopiesOk'
  <1>1. \A q : Moving(q)' = Moving(q)
    BY DEF Moving, Transfers
  <1>2. \A u : (\A t \in tasks : u \notin t.units) => \A t \in tasks' : u \notin t.units
    BY DEF TaskUnits
  <1> QED
    BY <1>1, <1>2 DEF CopiesOk, Live, Mem, Transfers

LEMMA TasksOkFrame ==
  ASSUME TasksOk,
         UNCHANGED <<tasks, alloc, oslot, deferred, copies, valid, table, ostate, hcache, vis>>
  PROVE  TasksOk'
  BY DEF TasksOk, Live, TableUnits, Mem

LEMMA MovingUnmappedFrame ==
  ASSUME MovingUnmapped, UNCHANGED <<deferred, dbt, handoff>>, Transfers' = Transfers
  PROVE  MovingUnmapped'
  BY DEF MovingUnmapped, Transfers

LEMMA LostOkFrame ==
  ASSUME LostOk, UNCHANGED <<lost, dbt, inflight, handoff, requests>>
  PROVE  LostOk'
  BY DEF LostOk

LEMMA TablesMappedFrame ==
  ASSUME TablesMapped, UNCHANGED <<dbt, ostate, ogroup, oslot, table, deferred, valid>>,
         Crossings' = Crossings
  PROVE  TablesMapped'
  <1>1. \A u : (\E t \in tasks' : t.kind = "replicate" /\ u \in t.units)
                 <=> (\E t \in tasks : t.kind = "replicate" /\ u \in t.units)
    BY DEF Crossings
  <1> QED
    BY <1>1 DEF TablesMapped, Live, HoldsAt, ReaderRow, ReadersOf, Mem

LEMMA HandoffOkFrame ==
  ASSUME HandoffOk, UNCHANGED <<handoff, requests, ostate, ogroup, oslot, table, deferred, valid>>,
         Crossings' = Crossings
  PROVE  HandoffOk'
  <1>1. \A u : (\E t \in tasks' : t.kind = "replicate" /\ u \in t.units)
                 <=> (\E t \in tasks : t.kind = "replicate" /\ u \in t.units)
    BY DEF Crossings
  <1> QED
    BY <1>1 DEF HandoffOk, Live, HoldsAt, ReaderRow, ReadersOf, Mem

LEMMA RequestsOkFrame ==
  ASSUME RequestsOk, UNCHANGED <<requests, launched, dbt, ncopy, copies>>
  PROVE  RequestsOk'
  BY DEF RequestsOk

(* A batch's cover reads the owners, their tables, the requests, the       *)
(* deferred work and which owners have a transfer task.                    *)
LEMMA CoveredFrame ==
  ASSUME UNCHANGED <<ostate, ogroup, oslot, table, requests, deferred>>, Transfers' = Transfers
  PROVE  \A b, o, u : Covered(b, o, u)' = Covered(b, o, u)
  <1>1. \A o : (\E t \in tasks' : t.kind = "transfer" /\ t.o = o)
                 <=> (\E t \in tasks : t.kind = "transfer" /\ t.o = o)
    BY DEF Transfers
  <1> QED
    BY <1>1 DEF Covered, Live, HoldsAt, ReadersOf

LEMMA BatchesOkFrame ==
  ASSUME BatchesOk,
         UNCHANGED <<inflight, launched, free, alloc, valid, ostate, ogroup, oslot, table, requests,
                     deferred, copies, lost>>,
         Transfers' = Transfers
  PROVE  BatchesOk'
  <1>1. \A b, o, u : Covered(b, o, u)' = Covered(b, o, u)
    BY CoveredFrame
  <1> QED
    BY <1>1 DEF BatchesOk, Mem

LEMMA TrialVisibilityFrame ==
  ASSUME TrialVisibility, UNCHANGED <<cache, vis, table, copies, hcache, tasks, trial>>
  PROVE  TrialVisibility'
  BY DEF TrialVisibility, MadeBy

LEMMA NoBadFrame ==
  ASSUME NoBad, UNCHANGED bad
  PROVE  NoBad'
  BY DEF NoBad

LEMMA CrossingsOkFrame ==
  ASSUME CrossingsOk,
         UNCHANGED <<deferred, tasks, copies, ostate, ogroup, lost, alloc, cache, vis, valid, trial>>
  PROVE  CrossingsOk'
  BY DEF CrossingsOk, Moving, ReadersOf, Mem, MadeBy

LEMMA CopiesPlacedFrame ==
  ASSUME CopiesPlaced, UNCHANGED <<copies, valid>>
  PROVE  CopiesPlaced'
  BY DEF CopiesPlaced, Mem

LEMMA CrossingsPlacedFrame ==
  ASSUME CrossingsPlaced, UNCHANGED <<ogroup, valid, inflight>>, Crossings' = Crossings
  PROVE  CrossingsPlaced'
  BY DEF CrossingsPlaced, Crossings, Mem

-----------------------------------------------------------------------------
(* TypeInv in five parts: the units, the owners, the control plane's       *)
(* functions, the deferred work with the copies and tasks, and the         *)
(* requests with the batches and completions; a copy, a task and a request *)
(* are records of their fields alone.                                      *)

UnitTyped ==
    /\ free \subseteq AllUnits /\ member \subseteq Units
    /\ alloc \in [AllUnits -> Owners \cup {NONE, "shared", "copy", "task"}]
    /\ sharedKey \in [AllUnits -> Keys \cup {NONE}]
    /\ cache \in [Keys -> Units \cup {NONE}] /\ hcache \in [Keys -> HostUnits \cup {NONE}]
    /\ valid \in [Units \cup HostUnits -> SUBSET (Memories \cup HostParts)]
    /\ \A u \in Units : valid[u] \subseteq Memories
    /\ \A h \in HostUnits : valid[h] \subseteq HostParts
(* The values valid takes: the memories that hold a device block, the host *)
(* parts that hold a host block.                                           *)
ValidTyped ==
    /\ valid \in [Units \cup HostUnits -> SUBSET (Memories \cup HostParts)]
    /\ \A u \in Units : valid[u] \subseteq Memories
    /\ \A h \in HostUnits : valid[h] \subseteq HostParts
OwnerTyped ==
    /\ ostate \in [Owners -> {"new", "live", "gone"}]
    /\ ogroup \in [Owners -> Groups \cup {NONE}]
    /\ oslot \in [Owners -> Slots \cup {NONE}]
    /\ DOMAIN table = Owners
    /\ \A o \in Owners : DOMAIN table[o] = Positions
    /\ \A o \in Owners, p \in Positions :
         \/ table[o][p] = Empty
         \/ table[o][p] = Private(table[o][p].u) /\ table[o][p].u \in Units
         \/ table[o][p] = Shared(table[o][p].u) /\ table[o][p].u \in Units
PlaneTyped ==
    /\ launched \in [Groups -> 0 .. MaxBatch] /\ ncopy \in 0 .. MaxCopies
    /\ DOMAIN dbt = Groups
    /\ \A g \in Groups : DOMAIN dbt[g] = Owners
    /\ \A g \in Groups, o \in Owners : DOMAIN dbt[g][o] = Rows
    /\ \A g \in Groups, o \in Owners, p \in Rows : dbt[g][o][p].u \in AllUnits \cup {NONE}
    /\ vis \in [Keys -> [d : Epochs \cup {NONE}, h : Epochs \cup {NONE}]]
    /\ trial \in [Epochs -> {"no", "yes", "rolled"}]
    /\ remote \subseteq Keys /\ lost \subseteq Epochs
WorkTyped ==
    /\ \A d \in deferred :
         /\ d.units \subseteq AllUnits /\ d.after \in Stamps
         /\ d.o \in Owners \cup {NONE} /\ d.p \in Positions \cup {NONE}
         /\ d.k \in Keys \cup {NONE}
         /\ d.c \in 0 .. MaxCopies /\ d.g \in Groups \cup {NONE} /\ d.to \in Groups \cup {NONE}
    /\ \A c \in copies :
         /\ c.seq \in 0 .. MaxBatch /\ c.k \in Keys
         /\ c.o \in Owners \cup {NONE} /\ c.p \in Positions \cup {NONE}
         /\ c.read \in AllUnits /\ c.write \in AllUnits /\ c.parts \subseteq HostParts
    /\ \A t \in tasks :
         /\ t.units \subseteq AllUnits /\ t.writes \subseteq AllUnits
         /\ t.k \in Keys \cup {NONE} /\ t.h \in HostUnits \cup {NONE}
         /\ t.slot \in Slots \cup {NONE} /\ t.o \in Owners \cup {NONE}
         /\ t.to \in Groups \cup {NONE}
         /\ t.p \in Positions \cup {NONE} /\ t.src \in {NONE, "remote", "peer"}
         /\ t.kind \in {"transfer", "replicate"} <=> t.k = NONE
         /\ t.kind = "prefetch" => t.to \in Groups
    /\ \A c \in copies :
         c = [id |-> c.id, kind |-> c.kind, o |-> c.o, p |-> c.p, k |-> c.k, g |-> c.g,
              read |-> c.read, write |-> c.write, fill |-> c.fill, parts |-> c.parts,
              seq |-> c.seq, done |-> c.done, stopped |-> c.stopped]
    /\ \A t \in tasks :
         t = [kind |-> t.kind, k |-> t.k, h |-> t.h, src |-> t.src, ok |-> t.ok,
              expired |-> t.expired, p |-> t.p, o |-> t.o, to |-> t.to, slot |-> t.slot,
              held |-> t.held, units |-> t.units, writes |-> t.writes]
RequestTyped ==
    /\ \A r \in requests :
         /\ r.e \in Epochs /\ r.after \in Stamps /\ r.ps \subseteq Positions
         /\ r.o \in Owners \cup {NONE} /\ r.p \in Positions \cup {NONE}
         /\ r.k \in Keys \cup {NONE}
         /\ r.c \in 0 .. MaxCopies /\ r.to \in Groups \cup {NONE}
         /\ r.kind # "finish" => r.o \in Owners
         /\ r.kind = "commit" => r.p \in Positions /\ r.k \in Keys
         /\ r.kind = "transfer" => r.to \in Groups
         /\ r.kind = "replicate" => r.p \in Positions /\ r.to \in Groups
    /\ \A b \in inflight :
         /\ b.reads \subseteq Owners \X AllUnits
         /\ b.cr \subseteq AllUnits /\ b.cw \subseteq AllUnits
    /\ \A h \in handoff :
         h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows /\ h.u \in AllUnits
    /\ \A r \in requests :
         r = [kind |-> r.kind, e |-> r.e, o |-> r.o, ps |-> r.ps, p |-> r.p, k |-> r.k,
              c |-> r.c, ok |-> r.ok, to |-> r.to, after |-> r.after, by |-> r.by, sg |-> r.sg]

LEMMA UnitTypes == IndInv => UnitTyped
  BY DEF IndInv, TypeInv, TypeOK, UnitTyped

LEMMA OwnerTypes == IndInv => OwnerTyped
  <1> SUFFICES ASSUME IndInv PROVE OwnerTyped
    OBVIOUS
  <1>1. TypeInv
    BY DEF IndInv
  <1>2. \A o \in Owners, p \in Positions :
          table[o][p] \in {Empty} \cup {Private(u) : u \in Units} \cup {Shared(u) : u \in Units}
    BY <1>1 DEF TypeInv
  <1>3. \A o \in Owners, p \in Positions :
          \/ table[o][p] = Empty
          \/ table[o][p] = Private(table[o][p].u) /\ table[o][p].u \in Units
          \/ table[o][p] = Shared(table[o][p].u) /\ table[o][p].u \in Units
    BY <1>2 DEF Empty, Private, Shared
  <1> QED
    BY <1>1, <1>3 DEF TypeInv, TypeOK, OwnerTyped

LEMMA PlaneTypes == IndInv => PlaneTyped
  BY DEF IndInv, TypeInv, TypeOK, PlaneTyped

LEMMA WorkTypes == IndInv => WorkTyped
  BY DEF IndInv, TypeInv, WorkTyped

LEMMA RequestTypes == IndInv => RequestTyped
  BY DEF IndInv, TypeInv, RequestTyped

LEMMA TypeInvNext ==
  ASSUME UnitTyped', OwnerTyped', PlaneTyped', WorkTyped', RequestTyped'
  PROVE  TypeInv'
  <1>1. \A o \in Owners, p \in Positions :
          /\ table'[o][p] \in {Empty} \cup {Private(u) : u \in Units} \cup {Shared(u) : u \in Units}
          /\ table'[o][p].k \in {"none", "private", "shared"}
          /\ table'[o][p].u \in Units \cup {NONE}
    BY DEF OwnerTyped, Empty, Private, Shared
  <1>2. \A k \in Keys : vis'[k].d \in Epochs \cup {NONE} /\ vis'[k].h \in Epochs \cup {NONE}
    BY DEF PlaneTyped
  <1>3. TypeOK'
    BY <1>1, <1>2 DEF TypeOK, UnitTyped, OwnerTyped, PlaneTyped
  <1> QED
    BY <1>1, <1>3 DEF TypeInv, OwnerTyped, PlaneTyped, WorkTyped, RequestTyped

LEMMA UnitTypedFrame ==
  ASSUME UnitTyped, UNCHANGED <<free, member, alloc, sharedKey, cache, hcache, valid>>
  PROVE  UnitTyped'
  BY DEF UnitTyped

LEMMA OwnerTypedFrame ==
  ASSUME OwnerTyped, UNCHANGED <<ostate, ogroup, oslot, table>>
  PROVE  OwnerTyped'
  BY DEF OwnerTyped

LEMMA PlaneTypedFrame ==
  ASSUME PlaneTyped, UNCHANGED <<launched, ncopy, dbt, vis, trial, remote, lost>>
  PROVE  PlaneTyped'
  BY DEF PlaneTyped

LEMMA WorkTypedFrame ==
  ASSUME WorkTyped, UNCHANGED <<deferred, copies, tasks>>
  PROVE  WorkTyped'
  BY DEF WorkTyped

LEMMA RequestTypedFrame ==
  ASSUME RequestTyped, UNCHANGED <<requests, inflight, handoff>>
  PROVE  RequestTyped'
  BY DEF RequestTyped

-----------------------------------------------------------------------------
(* DeferredOk, TasksOk and BatchesOk item by item.                         *)

DeferOk(d) ==
    /\ d.kind \in {"release", "commit", "finish", "transfer", "replicate"}
    /\ d.kind = "release" =>
         /\ d.units \subseteq Units \cup Slots
         /\ \A o \in Owners :
              /\ oslot[o] \notin d.units
              /\ \A p \in Positions : table[o][p].k = "private" => table[o][p].u \notin d.units
         /\ \A c \in copies : ~c.fill => c.write \notin d.units
    /\ d.kind \in {"finish", "transfer"} => d.units = {}
    /\ d.kind = "commit" =>
         /\ d.o \in Owners /\ d.p \in Positions /\ d.k \in Keys
         /\ OwnerEpoch[d.o] \notin lost
         /\ \E u \in Units : d.units = {u} /\ alloc[u] = d.o
         /\ \A o \in Owners, p \in Positions :
              table[o][p].u \in d.units => o = d.o /\ p = d.p /\ table[o][p].k = "private"
         /\ \A e \in deferred : (e.kind = "commit" /\ e.units = d.units) => e = d
         /\ \A c \in copies : c.write \notin d.units
    /\ d.kind = "finish" =>
         /\ \E c \in copies :
              /\ c.id = d.c
              /\ c.seq > 0 => d.after[c.g] >= c.seq
              /\ c.seq = 0 => c.stopped
         /\ \A e \in deferred : (e.kind = "finish" /\ e.c = d.c) => e = d
    /\ d.kind = "transfer" =>
         /\ d.o \in Owners /\ d.to \in Groups /\ GroupEpoch[d.to] = OwnerEpoch[d.o]
         /\ ostate[d.o] # "new"

LEMMA DeferredOkEach ==
  /\ DeferredOk = \A d \in deferred : DeferOk(d)
  /\ DeferredOk' = \A d \in deferred' : DeferOk(d)'
  BY DEF DeferredOk, DeferOk

TaskOk(t) ==
    /\ t.held
    /\ t.kind \in {"transfer", "prefetch", "put", "replicate"}
    /\ t.kind = "transfer" =>
         /\ t.h = NONE
         /\ t.o \in Owners /\ t.to \in Groups /\ GroupEpoch[t.to] = OwnerEpoch[t.o]
         /\ t.slot \in Slots /\ alloc[t.slot] = t.o /\ SlotGroup[t.slot] = t.to
         /\ \A o \in Owners : oslot[o] # t.slot
         /\ \A d \in deferred : t.slot \notin d.units
         /\ \A c \in copies : c.read # t.slot /\ c.write # t.slot
         /\ \A x \in tasks : t.slot \notin x.units
         /\ t.units \subseteq Units \cup Slots
         /\ t.writes \subseteq t.units \cap Units
         /\ \A u \in t.writes :
              Mem(t.to) \notin valid[u] /\ (alloc[u] = t.o \/ alloc[u] = "shared")
         /\ \A x \in tasks : (x.kind = "transfer" /\ (x.o = t.o \/ x.slot = t.slot)) => x = t
         /\ \A x \in tasks :
              (x.kind = "transfer" /\ x # t /\ Mem(x.to) = Mem(t.to)) => x.writes \cap t.writes = {}
         /\ ~\E d \in deferred : d.kind = "transfer" /\ d.o = t.o
         /\ Live(t.o) =>
              /\ t.units = TableUnits(t.o) \cup {oslot[t.o]}
              /\ ~\E c \in copies : c.o = t.o
              /\ \A u \in TableUnits(t.o) \ t.writes : Mem(t.to) \in valid[u]
    /\ t.kind \in {"prefetch", "put"} => t.units = {} /\ t.writes = {} /\ t.slot = NONE
    /\ t.kind = "prefetch" =>
         /\ t.h \in HostUnits /\ alloc[t.h] = "task"
         /\ \A x \in tasks : (x.kind = "prefetch" /\ x.h = t.h) => x = t
    /\ t.kind = "put" => t.h \in HostUnits /\ hcache[t.k] = t.h /\ vis[t.k].h = NONE

LEMMA TasksOkEach ==
  /\ TasksOk = \A t \in tasks : TaskOk(t)
  /\ TasksOk' = \A t \in tasks' : TaskOk(t)'
  BY DEF TasksOk, TaskOk

BatchOk(b) ==
    /\ b.g \in Groups /\ GroupEpoch[b.g] \notin lost
    /\ b.seq \in 1 .. launched[b.g]
    /\ \A x \in inflight : (x.g = b.g /\ x.seq = b.seq) => x = b
    /\ b.writes \subseteq b.reads
    /\ \A x \in b.reads :
         /\ x[1] \in Owners
         /\ x[2] \notin free
         /\ alloc[x[2]] \in {x[1], "shared"}
         /\ x[2] \in Units => Mem(b.g) \in valid[x[2]]
         /\ Covered(b, x[1], x[2])
    /\ \A x \in b.writes :
         /\ alloc[x[2]] = x[1]
         /\ \A d \in deferred :
              (d.kind \in {"commit", "replicate"} /\ x[2] \in d.units) =>
                  d.after[b.g] >= b.seq
         /\ \A r \in requests, p \in Positions :
              (FreezesRow(r, x[1], p) /\ table[x[1]][p].u = x[2]) =>
                  b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
    /\ \A u \in b.cw : \E c \in copies : c.write = u /\ c.g = b.g /\ c.seq = b.seq
    /\ \A u \in b.cr : \E c \in copies : c.read = u /\ c.g = b.g /\ c.seq = b.seq

LEMMA BatchesOkEach ==
  /\ BatchesOk = \A b \in inflight : BatchOk(b)
  /\ BatchesOk' = \A b \in inflight' : BatchOk(b)'
  BY DEF BatchesOk, BatchOk

(* CrossingsOk crossing by crossing, and P8's formulas item by item.      *)
CrossWaitOk(d) ==
    d.kind = "replicate" =>
        /\ d.o \in Owners /\ d.p \in Positions /\ d.to \in Groups
        /\ ostate[d.o] # "new" /\ OwnerEpoch[d.o] \notin lost
        /\ d.to \in ReadersOf(d.o)
        /\ \E u \in Units : d.units = {u} /\ (alloc[u] = d.o \/ alloc[u] = "shared")
        /\ ~Moving(d.o)
        /\ \A c \in copies : ~c.fill => c.write \notin d.units
        /\ \A k \in Keys : cache[k] \in d.units => vis[k].d = MadeBy(OwnerEpoch[d.o])
CrossTaskOk(t) ==
    t.kind = "replicate" =>
        /\ t.h = NONE /\ t.slot = NONE
        /\ t.o \in Owners /\ t.p \in Positions /\ t.to \in Groups
        /\ ostate[t.o] # "new" /\ t.to \in ReadersOf(t.o)
        /\ \E u \in Units : t.units = {u} /\ t.writes = {u}
        /\ \A u \in t.writes :
             Mem(t.to) \notin valid[u] /\ (alloc[u] = t.o \/ alloc[u] = "shared")
        /\ ~Moving(t.o)
        /\ \A x \in tasks :
             (x.kind \in {"transfer", "replicate"} /\ x # t /\ Mem(x.to) = Mem(t.to)) =>
                 x.writes \cap t.writes = {}
        /\ \A c \in copies :
             (c.fill /\ c.kind = "load" /\ Mem(c.g) = Mem(t.to)) => c.write \notin t.writes

LEMMA CrossingsOkEach ==
  /\ CrossingsOk = ((\A d \in deferred : CrossWaitOk(d)) /\ (\A t \in tasks : CrossTaskOk(t)))
  /\ CrossingsOk' = ((\A d \in deferred' : CrossWaitOk(d)') /\ (\A t \in tasks' : CrossTaskOk(t)'))
  BY DEF CrossingsOk, CrossWaitOk, CrossTaskOk

CopyPlaced(c) ==
    /\ c.read \in Units => Mem(c.g) \in valid[c.read]
    /\ c.read \in HostUnits => GroupParts[c.g] \subseteq valid[c.read]
    /\ (c.fill /\ c.write \in HostUnits) => c.parts \cap valid[c.write] = {}
CrossingPlaced(t) ==
    t.kind = "replicate" =>
        \A u \in t.units :
            /\ Mem(ogroup[t.o]) \in valid[u]
            /\ \A b \in inflight : \A w \in b.writes : w[2] # u

LEMMA PlacedEach ==
  /\ CopiesPlaced = \A c \in copies : CopyPlaced(c)
  /\ CopiesPlaced' = \A c \in copies' : CopyPlaced(c)'
  /\ CrossingsPlaced = \A t \in tasks : CrossingPlaced(t)
  /\ CrossingsPlaced' = \A t \in tasks' : CrossingPlaced(t)'
  BY DEF CopiesPlaced, CopyPlaced, CrossingsPlaced, CrossingPlaced

-----------------------------------------------------------------------------
(* CopiesOk copy by copy.                                                  *)

CopyOk(c) ==
    /\ c.id \in 1 .. ncopy
    /\ \A x \in copies : x.id = c.id => x = c
    /\ c.g \in Groups /\ GroupEpoch[c.g] \notin lost
    /\ c.seq <= launched[c.g]
    /\ c.done => c.seq > 0
    /\ c.kind \in {"clone", "load", "store", "save", "restore"}
    /\ IF c.kind = "save" THEN c.read \in Slots /\ alloc[c.read] = c.o
                          ELSE alloc[c.read] = "shared"
    /\ c.o \in Owners => OwnerEpoch[c.o] = GroupEpoch[c.g]
    /\ ~c.fill =>
         /\ c.kind \in {"clone", "load", "store", "save"}
         /\ c.kind \in {"clone", "load"} => c.write \in Units /\ alloc[c.write] = c.o
         /\ c.kind \in {"clone", "load"} => \A t \in tasks : c.write \notin t.units
         /\ c.kind \in {"store", "save"} => c.write \in HostUnits /\ alloc[c.write] = "copy"
         /\ \A o \in Owners, p \in Positions : table[o][p].u # c.write
         /\ \A x \in copies : x.write = c.write => x = c
    /\ c.fill =>
         \/ /\ c.kind = "load"
            /\ cache[c.k] = c.write
            /\ Mem(c.g) \notin valid[c.write]
            /\ \A x \in copies : (x.write = c.write /\ Mem(x.g) = Mem(c.g)) => x = c
            /\ \A t \in tasks :
                 (t.kind = "transfer" /\ Mem(t.to) = Mem(c.g)) => c.write \notin t.writes
         \/ c.kind = "restore" /\ c.write \in Slots /\ alloc[c.write] = c.o
         \/ /\ c.kind = "store"
            /\ c.write \in HostUnits /\ hcache[c.k] = c.write
            /\ \A x \in copies : x.write = c.write => x = c
    /\ c.kind \in {"clone", "load"} =>
         /\ c.o \in Owners /\ c.p \in Positions
         /\ table[c.o][c.p] = Empty
         /\ \A x \in copies : (x.o = c.o /\ x.p = c.p) => x = c
    /\ c.kind \in {"save", "restore"} => c.o \in Owners
    /\ c.o \in Owners => ostate[c.o] # "new"
    /\ (c.o \in Owners /\ Live(c.o)) => ogroup[c.o] = c.g /\ ~Moving(c.o)

LEMMA CopiesOkEach ==
  /\ CopiesOk = \A c \in copies : CopyOk(c)
  /\ CopiesOk' = \A c \in copies' : CopyOk(c)'
  BY DEF CopiesOk, CopyOk

(* Whether copy x has the fields of copy c, but seq, done and stopped.      *)
SameCopy(x, c) ==
    /\ x.id = c.id /\ x.kind = c.kind /\ x.o = c.o /\ x.p = c.p /\ x.k = c.k /\ x.g = c.g
    /\ x.read = c.read /\ x.write = c.write /\ x.fill = c.fill /\ x.parts = c.parts

(* A step that changes copies field by field, keeping every field but seq, *)
(* done and stopped, keeps CopiesOk when the new seq and done agree.       *)
LEMMA CopiesMap ==
  ASSUME CopiesOk,
         \A x \in copies' : \E c \in copies : SameCopy(x, c),
         \A x, y \in copies' : x.id = y.id => x = y,
         \A x \in copies' : x.seq <= launched'[x.g] /\ (x.done => x.seq > 0),
         UNCHANGED <<ncopy, lost, alloc, cache, hcache, valid, tasks, table, ostate, ogroup,
                     deferred>>
  PROVE  CopiesOk'
  <1>1. SUFFICES ASSUME NEW x \in copies' PROVE CopyOk(x)'
    BY CopiesOkEach
  <1>2. PICK c \in copies : SameCopy(x, c)
    OBVIOUS
  <1>3. CopyOk(c)
    BY CopiesOkEach
  <1>4. \A y \in copies' : \E d \in copies : SameCopy(y, d) /\ (d = c => y = x)
    BY <1>2 DEF SameCopy
  <1>5. \A o \in Owners : Moving(o)' = Moving(o)
    BY DEF Moving
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5 DEF CopyOk, SameCopy, Live, Mem

-----------------------------------------------------------------------------
(* Steps that return units to the free list: G is what they reclaim, and   *)
(* alloc and sharedKey are reset on G alone.                               *)

LEMMA FreeReclaim ==
  ASSUME FreeClean, NoLeak, Outside, Regions, UnitTyped, NEW G, G \subseteq AllUnits,
         free' = free \cup G, member' = member,
         \A v \in AllUnits : alloc'[v] = IF v \in G THEN NONE ELSE alloc[v],
         \A v \in AllUnits : sharedKey'[v] = IF v \in G THEN NONE ELSE sharedKey[v],
         \A v \in AllUnits : Held(v)' => Held(v),
         \A v \in AllUnits : Held(v) => Held(v)' \/ v \in G,
         \A v \in G : Held(v) /\ ~Held(v)'
  PROVE  FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
  <1>1. FreeClean'
    BY DEF FreeClean, UnitTyped
  <1>2. NoLeak'
    BY DEF NoLeak, InRegion
  <1>3. Outside'
    BY DEF Outside, AllUnits
  <1>4. \A v \in G : v \in Units => v \in member
    BY DEF NoLeak, InRegion, AllUnits
  <1>5. Regions'
    BY <1>4 DEF Regions, OneRegion
  <1> QED
    BY <1>1, <1>2, <1>3, <1>5

(* A step that frees no unit a batch in flight reads and keeps the         *)
(* allocation of every such unit keeps BatchesOk.                          *)
LEMMA BatchesKeep ==
  ASSUME BatchesOk, NEW G,
         \A b \in inflight : \A x \in b.reads : x[2] \notin G /\ alloc'[x[2]] = alloc[x[2]],
         free' \subseteq free \cup G,
         UNCHANGED <<inflight, launched, valid, ostate, ogroup, oslot, table, requests, deferred,
                     tasks, copies, lost>>
  PROVE  BatchesOk'
  <1>1. \A b, o, u : Covered(b, o, u)' = Covered(b, o, u)
    BY CoveredFrame, TransfersFrame
  <1>2. \A b \in inflight : b.writes \subseteq b.reads
    BY DEF BatchesOk
  <1> QED
    BY <1>1, <1>2 DEF BatchesOk, Mem
(* The tasks stay as IndInv describes them across a step that leaves them, *)
(* the owners, the deferred work and the copies alone and changes no       *)
(* allocation, validity, table entry or host copy they read.               *)
LEMMA TasksKeep ==
  ASSUME TasksOk,
         UNCHANGED <<tasks, deferred, copies, oslot, ostate, vis>>,
         \A t \in tasks : t.kind = "transfer" =>
           /\ alloc'[t.slot] = alloc[t.slot]
           /\ \A u \in t.writes : alloc'[u] = alloc[u] /\ valid'[u] = valid[u]
           /\ ostate[t.o] = "live" =>
                /\ TableUnits(t.o)' = TableUnits(t.o)
                /\ \A u \in TableUnits(t.o) : valid'[u] = valid[u],
         \A t \in tasks : t.kind = "prefetch" => alloc'[t.h] = alloc[t.h],
         \A t \in tasks : t.kind = "put" => hcache'[t.k] = hcache[t.k]
  PROVE  TasksOk'
  <1> SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
    BY TasksOkEach
  <1>1. TaskOk(t)
    BY TasksOkEach
  <1> QED
    BY <1>1 DEF TaskOk, Live

(* A unit a batch in flight reads is held by its owner's table or slot, or *)
(* by a release that waits.                                                *)
LEMMA ReadHeld ==
  ASSUME IndInv, NEW b \in inflight, NEW x \in b.reads
  PROVE  \/ \E p \in Positions : table[x[1]][p].u = x[2]
         \/ oslot[x[1]] = x[2]
         \/ \E d \in deferred : d.kind = "release" /\ x[2] \in d.units
  <1>1. Covered(b, x[1], x[2])
    BY DEF IndInv, BatchesOk
  <1> QED
    BY <1>1 DEF Covered, HoldsAt, Rows
-----------------------------------------------------------------------------
(* What follows is proved with the configuration's assumptions and the    *)
(* rules in force.                                                        *)

USE Sizes, Design DEF On, AllRules

-----------------------------------------------------------------------------
(* What IndInv says about a task and about a cached block.                 *)

LEMMA TasksHeld ==
  ASSUME IndInv
  PROVE  \A t \in tasks : t.held
  BY DEF IndInv, TasksOk

LEMMA DeviceCached ==
  ASSUME IndInv, NEW k \in Keys, cache[k] # NONE
  PROVE  /\ cache[k] \in Units /\ cache[k] \in member /\ cache[k] \notin free
         /\ alloc[cache[k]] = "shared" /\ sharedKey[cache[k]] = k
         /\ \A j \in Keys : cache[j] = cache[k] => j = k
  <1>1. cache[k] \in Units /\ sharedKey[cache[k]] = k
    BY DEF IndInv, CacheShared
  <1>2. alloc[cache[k]] = "shared"
    BY <1>1 DEF IndInv, SharedIff, AllUnits
  <1>3. \A j \in Keys : cache[j] = cache[k] => j = k
    BY <1>1 DEF IndInv, CacheShared
  <1>4. Held(cache[k])
    BY DEF Held, HeldIn
  <1>5. cache[k] \notin free
    BY <1>4 DEF IndInv, FreeClean
  <1>6. cache[k] \in member
    BY <1>1, <1>4 DEF IndInv, NoLeak, InRegion, AllUnits
  <1> QED
    BY <1>1, <1>2, <1>3, <1>5, <1>6

LEMMA HostCached ==
  ASSUME IndInv, NEW k \in Keys, hcache[k] # NONE
  PROVE  /\ hcache[k] \in HostUnits /\ hcache[k] \notin free
         /\ alloc[hcache[k]] = "shared" /\ sharedKey[hcache[k]] = k
         /\ \A j \in Keys : hcache[j] = hcache[k] => j = k
  <1>1. hcache[k] \in HostUnits /\ sharedKey[hcache[k]] = k
    BY DEF IndInv, CacheShared
  <1>2. alloc[hcache[k]] = "shared"
    BY <1>1 DEF IndInv, SharedIff, AllUnits
  <1>3. \A j \in Keys : hcache[j] = hcache[k] => j = k
    BY <1>1 DEF IndInv, CacheShared
  <1>4. Held(hcache[k])
    BY DEF Held, HeldIn
  <1>5. hcache[k] \notin free
    BY <1>4 DEF IndInv, FreeClean
  <1> QED
    BY <1>1, <1>2, <1>3, <1>5

-----------------------------------------------------------------------------
(* A device unit a handler obtains for a new private block of owner o: a   *)
(* free unit of o's region, or an evictable one that leaves the cache.     *)
(* Nothing but the cache holds it, no batch in flight reads it, and it is  *)
(* in its region.                                                          *)

Unheld(u) ==
  /\ \A o \in Owners, p \in Positions : table[o][p].u # u
  /\ \A o \in Owners : oslot[o] # u
  /\ \A k \in Keys : hcache[k] # u
  /\ \A d \in deferred : u \notin d.units
  /\ \A c \in copies : c.read # u /\ c.write # u
  /\ \A t \in tasks : u \notin t.units /\ u # t.slot /\ u # t.h

LEMMA ObtainFacts ==
  ASSUME IndInv, NEW o \in Owners, Live(o), NEW u \in RegionUnits(o),
         Obtain(u, RegionUnits(o), OwnerEpoch[o])
  PROVE  /\ u \in Units /\ u \in member
         /\ alloc[u] \in {NONE, "shared"}
         /\ Unheld(u)
         /\ cache' = [k \in Keys |-> IF cache[k] = u THEN NONE ELSE cache[k]]
         /\ \A b \in inflight : \A x \in b.reads : x[2] # u
  <1>1. u \in Units /\ u # NONE /\ u \notin Slots /\ u \notin HostUnits
    BY DEF RegionUnits
  <1>2. /\ \A q \in Owners : oslot[q] # u
        /\ \A j \in Keys : hcache[j] # u
        /\ \A t \in tasks : u # t.slot /\ u # t.h
    <2>1. oslot \in [Owners -> Slots \cup {NONE}] /\ hcache \in [Keys -> HostUnits \cup {NONE}]
      BY DEF IndInv, TypeInv, TypeOK
    <2>2. \A t \in tasks : t.slot \in Slots \cup {NONE} /\ t.h \in HostUnits \cup {NONE}
      BY DEF IndInv, TypeInv
    <2> QED
      BY <1>1, <2>1, <2>2
  <1>3. /\ u \in member
        /\ alloc[u] \in {NONE, "shared"}
        /\ Unheld(u)
        /\ cache' = [k \in Keys |-> IF cache[k] = u THEN NONE ELSE cache[k]]
    <2>1. CASE u \in free /\ cache' = cache
      <3>1. alloc[u] = NONE /\ ~Held(u)
        BY <2>1 DEF IndInv, FreeClean
      <3>2. u \in member
        BY <1>1, <2>1 DEF IndInv, Regions
      <3>3. Unheld(u) /\ \A k \in Keys : cache[k] # u
        BY <3>1, TasksHeld DEF Held, HeldIn, Unheld
      <3>4. cache \in [Keys -> Units \cup {NONE}]
        BY DEF IndInv, TypeInv, TypeOK
      <3>5. cache' = [k \in Keys |-> IF cache[k] = u THEN NONE ELSE cache[k]]
        BY <2>1, <3>3, <3>4
      <3> QED
        BY <3>1, <3>2, <3>3, <3>5
    <2>2. CASE /\ free \cap RegionUnits(o) = {}
               /\ EvictableFor(u, OwnerEpoch[o])
               /\ cache' = [k \in Keys |-> IF cache[k] = u THEN NONE ELSE cache[k]]
      <3>1. PICK k \in Keys : cache[k] = u
        BY <2>2 DEF EvictableFor
      <3>2. alloc[u] = "shared" /\ u \in member
        BY <1>1, <3>1, DeviceCached
      <3>3. /\ ~\E q \in Owners, p \in Positions : table[q][p] = Shared(u)
            /\ ~\E d \in deferred : u \in d.units
            /\ ~\E c \in copies : u = c.read \/ u = c.write
            /\ ~\E t \in tasks : t.held /\ u \in t.units
        <4>1. EvictableFor(u, OwnerEpoch[o])
          BY <2>2
        <4>2. ~PinnedFor(u, OwnerEpoch[o])
          BY <4>1 DEF EvictableFor
        <4>3. ~\E q \in Owners, p \in Positions : table[q][p] = Shared(u)
          BY <4>2 DEF PinnedFor
        <4> QED
          BY <4>1, <4>3 DEF EvictableFor
      <3>4. \A q \in Owners, p \in Positions : table[q][p].u # u
        <4> SUFFICES ASSUME NEW q \in Owners, NEW p \in Positions, table[q][p].u = u
                     PROVE  FALSE
          OBVIOUS
        <4>1. table[q][p] \in {Empty} \cup {Private(v) : v \in Units} \cup {Shared(v) : v \in Units}
          BY DEF IndInv, TypeInv
        <4>2. table[q][p].k = "private" => alloc[table[q][p].u] = q
          BY DEF IndInv, Tables
        <4>3. CASE table[q][p] = Empty
          BY <4>3, <1>1 DEF Empty
        <4>4. CASE \E v \in Units : table[q][p] = Private(v)
          BY <4>4, <4>2, <3>2 DEF Private
        <4>5. CASE \E v \in Units : table[q][p] = Shared(v)
          BY <4>5, <3>3 DEF Shared
        <4> QED
          BY <4>1, <4>3, <4>4, <4>5
      <3>5. \A t \in tasks : u \notin t.units
        BY <3>3, TasksHeld
      <3> QED
        BY <1>2, <2>2, <3>2, <3>3, <3>4, <3>5 DEF Unheld
    <2> QED
      BY <2>1, <2>2 DEF Obtain
  <1>4. \A b \in inflight : \A x \in b.reads : x[2] # u
    <2> SUFFICES ASSUME NEW b \in inflight, NEW x \in b.reads, x[2] = u PROVE FALSE
      OBVIOUS
    <2>1. Covered(b, x[1], u)
      BY DEF IndInv, BatchesOk
    <2>2. CASE \E p \in Rows : HoldsAt(x[1], p, u)
      <3>1. PICK p \in Rows : HoldsAt(x[1], p, u)
        BY <2>2
      <3>2. x[1] \in Owners
        BY DEF IndInv, BatchesOk
      <3> QED
        BY <3>1, <3>2, <1>3 DEF HoldsAt, Unheld, Rows
    <2>3. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units
      BY <2>3, <1>3 DEF Unheld
    <2> QED
      BY <2>1, <2>2, <2>3 DEF Covered
  <1> QED
    BY <1>1, <1>3, <1>4
-----------------------------------------------------------------------------
(* What Give(h) does to the executors' tables and to the completions not  *)
(* yet handed on: h waits to be handed on, or its row is written now      *)
(* unless a stamped request under way names it.                           *)

Unnamed(h) == ~\E r \in requests : Names(r, h)

LEMMA GiveRow ==
  ASSUME NEW h, Give(h), h.g \in Groups, h.o \in Owners, h.p \in Rows,
         DOMAIN dbt = Groups,
         \A g \in Groups : DOMAIN dbt[g] = Owners,
         \A g \in Groups, o \in Owners : DOMAIN dbt[g][o] = Rows
  PROVE  /\ DOMAIN dbt' = Groups
         /\ \A g \in Groups : DOMAIN dbt'[g] = Owners
         /\ \A g \in Groups, o \in Owners : DOMAIN dbt'[g][o] = Rows
         /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {h}
         /\ \A g \in Groups, o \in Owners, p \in Rows :
              dbt'[g][o][p] = IF Eager /\ g = h.g /\ o = h.o /\ p = h.p /\ Unnamed(h)
                              THEN [u |-> h.u, w |-> h.w]
                              ELSE dbt[g][o][p]
  <1>1. CASE Eager /\ Unnamed(h)
    <2>1. dbt' = [dbt EXCEPT ![h.g][h.o][h.p] = [u |-> h.u, w |-> h.w]] /\ handoff' = handoff
      BY <1>1 DEF Give, Unnamed
    <2> QED
      BY <1>1, <2>1
  <1>2. CASE Eager /\ ~Unnamed(h)
    <2>1. dbt' = dbt /\ handoff' = handoff
      BY <1>2 DEF Give, Unnamed
    <2> QED
      BY <1>2, <2>1
  <1>3. CASE ~Eager
    <2>1. dbt' = dbt /\ handoff' = handoff \cup {h}
      BY <1>3 DEF Give
    <2> QED
      BY <1>3, <2>1
  <1> QED
    BY <1>1, <1>2, <1>3

(* A completion names a row a stamped request takes away or commits.      *)
LEMMA NamesRow ==
  ASSUME NEW r, NEW h
  PROVE  Names(r, h) <=> TakesRow(r, h.o, h.p) \/ FreezesRow(r, h.o, h.p)
  BY DEF Names, TakesRow, FreezesRow

LEMMA TypeFacts ==
  ASSUME IndInv
  PROVE  TypeOK
  BY DEF IndInv, TypeInv

-----------------------------------------------------------------------------
(* TypeInv after a step that changes the owners' tables, the executors'   *)
(* tables and the completions not yet handed on, and leaves the deferred   *)
(* work, the copies, the tasks, the requests and the batches alone.        *)
LEMMA RowTypes ==
  ASSUME TypeInv, TypeOK',
         DOMAIN table' = Owners,
         \A q \in Owners : DOMAIN table'[q] = Positions,
         \A q \in Owners, r \in Positions :
           table'[q][r] \in {Empty} \cup {Private(v) : v \in Units} \cup {Shared(v) : v \in Units},
         DOMAIN dbt' = Groups,
         \A x \in Groups : DOMAIN dbt'[x] = Owners,
         \A x \in Groups, q \in Owners : DOMAIN dbt'[x][q] = Rows,
         \A x \in Groups, q \in Owners, r \in Rows : dbt'[x][q][r].u \in AllUnits \cup {NONE},
         \A h \in handoff' : h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows /\ h.u \in AllUnits,
         UNCHANGED <<vis, trial, remote, lost, deferred, copies, tasks, requests, inflight>>
  PROVE  TypeInv'
  BY DEF TypeInv, Stamps

-----------------------------------------------------------------------------
(* A batch's cover of a unit of owner q stays when the requests and the   *)
(* deferred work stay, q gains no transfer task, and a live q stays live  *)
(* in its group holding what it held.                                     *)
LEMMA CoverStays ==
  ASSUME NEW b, NEW q, NEW v, Covered(b, q, v),
         UNCHANGED <<requests, deferred>>,
         (\E t \in tasks' : t.kind = "transfer" /\ t.o = q)
           => (\E t \in tasks : t.kind = "transfer" /\ t.o = q),
         ostate[q] = "live" =>
           /\ ostate'[q] = "live" /\ ogroup'[q] = ogroup[q]
           /\ \A r \in Rows : HoldsAt(q, r, v) => HoldsAt(q, r, v)'
  PROVE  Covered(b, q, v)'
  <1>1. CASE /\ ostate[q] = "live" /\ (ogroup[q] = b.g \/ b.g \in ReadersOf(q))
             /\ \E p \in Rows :
                  /\ HoldsAt(q, p, v)
                  /\ \A r \in requests :
                       TakesRow(r, q, p) => b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
             /\ \A d \in deferred : (d.kind = "transfer" /\ d.o = q) => d.after[b.g] >= b.seq
             /\ ~\E t \in tasks : t.kind = "transfer" /\ t.o = q
    <2>1. PICK p \in Rows :
            /\ HoldsAt(q, p, v)
            /\ \A r \in requests :
                 TakesRow(r, q, p) => b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
      BY <1>1
    <2>2. HoldsAt(q, p, v)'
      BY <1>1, <2>1
    <2>3. ReadersOf(q)' = ReadersOf(q)
      BY <1>1 DEF ReadersOf
    <2> QED
      BY <1>1, <2>1, <2>2, <2>3 DEF Covered, Live
  <1>2. CASE \E d \in deferred : d.kind = "release" /\ v \in d.units /\ d.after[b.g] >= b.seq
    BY <1>2 DEF Covered
  <1> QED
    BY <1>1, <1>2 DEF Covered, Live

-----------------------------------------------------------------------------
(* The copies stay as IndInv describes them across a step that leaves     *)
(* them, the owners and the deferred work alone and changes no            *)
(* allocation, validity, cache entry or table entry they read.            *)

LEMMA CopiesStay ==
  ASSUME CopiesOk,
         UNCHANGED <<copies, ncopy, lost, launched, ostate, ogroup, deferred>>,
         Transfers' = Transfers, TaskUnits' \subseteq TaskUnits,
         \A c \in copies : alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write],
         \A c \in copies : c.fill /\ c.kind = "load" =>
                             cache'[c.k] = cache[c.k] /\ valid'[c.write] = valid[c.write],
         \A c \in copies : c.fill /\ c.kind = "store" => hcache'[c.k] = hcache[c.k],
         \A c \in copies : ~c.fill => \A q \in Owners, r \in Positions : table'[q][r].u # c.write,
         \A c \in copies : c.kind \in {"clone", "load"} => table'[c.o][c.p] = table[c.o][c.p]
  PROVE  CopiesOk'
  <1>1. \A q : Moving(q)' = Moving(q)
    BY DEF Moving, Transfers
  <1> SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
    BY CopiesOkEach
  <1>2. CopyOk(c)
    BY CopiesOkEach
  <1>3. \A x : (\A t \in tasks' : (t.kind = "transfer" /\ Mem(t.to) = Mem(c.g)) => x \notin t.writes)
                 <=> (\A t \in tasks : (t.kind = "transfer" /\ Mem(t.to) = Mem(c.g)) => x \notin t.writes)
    BY DEF Transfers
  <1>4. \A u : (\A t \in tasks : u \notin t.units) => \A t \in tasks' : u \notin t.units
    BY DEF TaskUnits
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4 DEF CopyOk, Live

-----------------------------------------------------------------------------
(* What taking a stamped request shares: the request leaves the requests  *)
(* under way, and the completions it names leave those not handed on.     *)

(* A stamp no batch in flight has passed is not reached.                   *)
LEMMA NotReached ==
  ASSUME IndInv, NEW after \in Stamps, NEW b \in inflight, after[b.g] >= b.seq
  PROVE  ~Reached(after)
  <1>1. b.g \in Groups /\ b.seq \in Nat /\ after[b.g] \in Nat
    BY DEF IndInv, BatchesOk, Stamps
  <1> QED
    BY <1>1 DEF Reached

(* What a submitted request says of the executors' tables: no group still *)
(* holds a row it takes away, nor may write the row it commits.            *)
LEMMA Submitted ==
  ASSUME IndInv, NEW r \in requests, r.sg \subseteq r.by, r.kind # "finish"
  PROVE  /\ r.o \in Owners
         /\ \A g \in Groups, p \in Rows : TakesRow(r, r.o, p) => dbt[g][r.o][p].u = NONE
         /\ \A g \in Groups, p \in Rows :
              (FreezesRow(r, r.o, p) /\ dbt[g][r.o][p].u # NONE) => ~dbt[g][r.o][p].w
         /\ \A b \in inflight : b.g \in r.sg => b.g \in r.by
  <1>1. r.o \in Owners
    BY RequestTypes DEF RequestTyped
  <1>2. \A g \in Groups, p \in Rows :
          (dbt[g][r.o][p].u # NONE /\ (TakesRow(r, r.o, p) \/ FreezesRow(r, r.o, p))) =>
              /\ g \in r.sg
              /\ g \in r.by => FreezesRow(r, r.o, p) /\ ~dbt[g][r.o][p].w
    BY DEF IndInv, RequestsOk
  <1>3. \A p : ~(TakesRow(r, r.o, p) /\ FreezesRow(r, r.o, p))
    BY DEF TakesRow, FreezesRow
  <1> QED
    BY <1>1, <1>2, <1>3

(* With r gone from the requests and the completions it names dropped,     *)
(* RequestsOk and LostOk hold, a completion kept is named as before, and a *)
(* batch's write stays covered by every commit request under way.          *)
LEMMA TakeCommon ==
  ASSUME IndInv, NEW r \in requests, requests' = requests \ {r},
         handoff' \subseteq handoff, \A h \in handoff' : ~Names(r, h),
         UNCHANGED <<dbt, launched, inflight, ncopy, copies, lost>>
  PROVE  /\ RequestsOk' /\ LostOk'
         /\ \A h \in handoff' : (\E x \in requests' : Names(x, h)) = (\E x \in requests : Names(x, h))
         /\ \A b \in inflight, o \in Owners, p \in Positions :
              (\A x \in requests :
                 FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq))
              => \A x \in requests' :
                   FreezesRow(x, o, p) => b.g \in x.sg /\ (b.g \in x.by => x.after[b.g] >= b.seq)
  <1>1. RequestsOk'
    BY DEF IndInv, RequestsOk
  <1>2. LostOk'
    BY DEF IndInv, LostOk
  <1> QED
    BY <1>1, <1>2

(* The request is refused, or taken without effect on the buffer pool.    *)
LEMMA TakeIdle ==
  ASSUME IndInv, NEW r \in requests, requests' = requests \ {r},
         handoff' \subseteq handoff, \A h \in handoff' : ~Names(r, h),
         UNCHANGED <<pool, dbt, launched, inflight, alloc, sharedKey, rare>>
  PROVE  IndInv'
  <1>1. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                    tasks, valid, vis, trial, member, dbt, launched, inflight, alloc, sharedKey,
                    remote, lost, bad>>
    BY DEF pool, rare
  <1>2. TypeInv'
    <2>1. RequestTyped'
      BY <1>1, RequestTypes DEF RequestTyped
    <2> QED
      BY <1>1, <2>1, UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, UnitTypedFrame,
         OwnerTypedFrame, PlaneTypedFrame, WorkTypedFrame, TypeInvNext
  <1>3. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions' /\ AllocOwner' /\ SharedIff'
        /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk' /\ DeferredOk' /\ CopiesOk'
        /\ TasksOk' /\ TrialVisibility' /\ NoBad' /\ TablesMapped' /\ CrossingsOk'
        /\ CopiesPlaced' /\ CrossingsPlaced'
    BY <1>1, TransfersFrame, FreeCleanFrame, NoLeakFrame, OutsideFrame, RegionsFrame,
       AllocOwnerFrame, SharedIffFrame, SharedCachedFrame, CacheSharedFrame, TablesFrame,
       OwnersOkFrame, DeferredOkFrame, CopiesOkFrame, TasksOkFrame, TrialVisibilityFrame,
       NoBadFrame, TablesMappedFrame, CrossingsOkFrame, CopiesPlacedFrame, CrossingsPlacedFrame
       DEF IndInv
  <1>4. RequestsOk' /\ LostOk'
    BY <1>1, TakeCommon
  <1>5. MovingUnmapped'
    BY <1>1 DEF IndInv, MovingUnmapped
  <1>6. HandoffOk'
    <2>1. \A h \in handoff' : (\E x \in requests' : Names(x, h)) = (\E x \in requests : Names(x, h))
      BY <1>1, TakeCommon
    <2> QED
      BY <1>1, <2>1 DEF IndInv, HandoffOk, Live, HoldsAt, ReaderRow, ReadersOf, Mem
  <1>7. BatchesOk'
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
      BY <1>1, TakeCommon
    <2> QED
      BY <1>1, <2>2, <2>3, <2>4 DEF BatchOk, Mem
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF IndInv

-----------------------------------------------------------------------------
(* A deferred item, and the copy it ends, retire: what they held is let go *)
(* when nothing else holds it. A release that comes due covers no batch in *)
(* flight, and no batch in flight runs the copy that ends. A crossing that *)
(* retires lets go of a block its owner no longer holds where it named it.*)

LEMMA Retire ==
  ASSUME IndInv, NEW d \in deferred, NEW X, X \subseteq copies, NEW U,
         d.kind \in {"release", "commit", "finish", "replicate"},
         U = d.units \cup UNION {{c.read, c.write} : c \in X},
         \A c \in X : c.id = d.c /\ d.kind = "finish",
         d.kind = "release" => \A b \in inflight : d.after[b.g] < b.seq,
         \A c \in X : \A b \in inflight : ~(c.g = b.g /\ c.seq = b.seq),
         deferred' = deferred \ {d}, copies' = copies \ X,
         DropWith(U, table, oslot, cache, hcache, deferred', copies', tasks),
         ValidTyped',
         \A u \in Units \cup HostUnits : valid'[u] = valid[u] \/ (u \in U /\ ~Held(u)'),
         UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, ncopy, tasks, trial, member, vis,
                     dbt, launched, inflight, handoff, requests, remote, lost, bad>>
  PROVE  IndInv'
  <1>0. UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1> DEFINE G == ReclaimedWith(U, table, oslot, cache, hcache, deferred', copies', tasks)
  <1>1. /\ \A x : Held(x)' = HeldIn(x, table, oslot, cache, hcache, deferred', copies', tasks)
        /\ G = {x \in U : ~Held(x)'}
        /\ free' = free \cup G /\ alloc' = Reset(alloc, G) /\ sharedKey' = Reset(sharedKey, G)
    BY DEF Held, DropWith, ReclaimedWith, On, AllRules
  <1>2. /\ U \subseteq AllUnits
        /\ \A x \in U : Held(x)
    BY <1>0 DEF WorkTyped, Held, HeldIn
  <1>3. /\ \A x \in AllUnits : Held(x)' => Held(x)
        /\ \A x \in AllUnits : Held(x) => Held(x)' \/ x \in G
        /\ \A x \in G : Held(x) /\ ~Held(x)'
        /\ G \subseteq AllUnits
    <2>1. \A x : HeldIn(x, table, oslot, cache, hcache, deferred', copies', tasks)
                   => HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
      BY DEF HeldIn
    <2>2. \A x : HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
                   => HeldIn(x, table, oslot, cache, hcache, deferred', copies', tasks) \/ x \in U
      BY DEF HeldIn
    <2> QED
      BY <1>1, <1>2, <2>1, <2>2 DEF Held
  <1>4. /\ \A v \in AllUnits : alloc'[v] = IF v \in G THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v \in G THEN NONE ELSE sharedKey[v]
        /\ \A v \in AllUnits : Held(v)' => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
        /\ \A v \in Units \cup HostUnits : Held(v)' => valid'[v] = valid[v]
    BY <1>1 DEF Reset
  (* What stays held after the step.                                       *)
  <1>5. /\ \A o \in Owners, p \in Positions : \A x \in AllUnits : table[o][p].u = x => Held(x)'
        /\ \A o \in Owners : \A x \in AllUnits : oslot[o] = x => Held(x)'
        /\ \A k \in Keys : \A x \in AllUnits : (cache[k] = x \/ hcache[k] = x) => Held(x)'
        /\ \A e \in deferred' : \A x \in e.units : Held(x)'
        /\ \A c \in copies' : Held(c.read)' /\ Held(c.write)'
        /\ \A t \in tasks : t.held => \A x \in t.units \cup {t.slot, t.h} : Held(x)'
    BY <1>1 DEF HeldIn
  <1>6. /\ \A e \in deferred' : e \in deferred /\ e # d
        /\ \A c \in copies' : c \in copies /\ c \notin X
        /\ \A o \in Owners : Moving(o)' => Moving(o)
    BY DEF Moving
  <1>7. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>3 DEF UnitTyped, ValidTyped, Reset
    <2>2. WorkTyped' /\ RequestTyped'
      BY <1>0, <1>6 DEF WorkTyped, RequestTyped
    <2> QED
      BY <2>1, <2>2, OwnerTypes, PlaneTypes, OwnerTypedFrame, PlaneTypedFrame, TypeInvNext
  <1>8. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
    BY <1>0, <1>1, <1>3, <1>4, FreeReclaim DEF IndInv
  <1>9. AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. AllocOwner' /\ SharedIff'
      BY <1>4 DEF IndInv, AllocOwner, SharedIff, NONE
    <2>2. \A v \in AllUnits : alloc[v] = "shared" /\ v \notin G =>
            cache[sharedKey[v]] = v \/ hcache[sharedKey[v]] = v
      BY DEF IndInv, SharedCached
    <2>3. SharedCached'
      BY <1>4, <2>2 DEF SharedCached, NONE
    <2>4. CacheShared'
      BY <1>0, <1>4, <1>5 DEF IndInv, CacheShared, UnitTyped, AllUnits
    <2> QED
      BY <2>1, <2>3, <2>4
  <1>10. Tables' /\ OwnersOk'
    <2>1. \A o \in Owners, p \in Positions :
            table[o][p].u # NONE =>
              /\ table[o][p].u \in AllUnits
              /\ alloc'[table[o][p].u] = alloc[table[o][p].u]
              /\ valid'[table[o][p].u] = valid[table[o][p].u]
      BY <1>0, <1>4, <1>5 DEF OwnerTyped, Empty, Private, Shared, AllUnits
    <2>2. \A o \in Owners : oslot[o] # NONE => alloc'[oslot[o]] = alloc[oslot[o]]
      BY <1>0, <1>4, <1>5 DEF OwnerTyped, AllUnits
    <2> QED
      BY <2>1, <2>2 DEF IndInv, Tables, OwnersOk, Live, Mem
  <1>11. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred' PROVE DeferOk(e)'
      BY DeferredOkEach
    <2>2. e \in deferred /\ e # d /\ DeferOk(e) /\ DeferOk(d)
      BY <1>6, DeferredOkEach DEF IndInv
    <2>3. e.kind = "commit" => \A u \in e.units : alloc'[u] = alloc[u]
      BY <1>0, <1>4, <1>5 DEF WorkTyped
    <2>4. e.kind = "finish" => \A c \in copies : c.id = e.c => c \in copies'
      BY <2>2 DEF DeferOk
    <2> QED
      BY <1>6, <2>2, <2>3, <2>4 DEF DeferOk
  <1>12. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies' PROVE CopyOk(c)'
      BY CopiesOkEach
    <2>2. c \in copies /\ CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
      BY <1>0, <1>6, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. /\ alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
          /\ c.write \in Units => valid'[c.write] = valid[c.write]
      BY <1>4, <1>5, <2>2 DEF AllUnits
    <2>4. /\ c.id \in 1 .. ncopy'
          /\ \A x \in copies' : x.id = c.id => x = c
          /\ c.g \in Groups /\ GroupEpoch[c.g] \notin lost'
          /\ c.seq <= launched'[c.g]
          /\ c.done => c.seq > 0
          /\ c.kind \in {"clone", "load", "store", "save", "restore"}
          /\ IF c.kind = "save" THEN c.read \in Slots /\ alloc'[c.read] = c.o
                                ELSE alloc'[c.read] = "shared"
          /\ c.o \in Owners => OwnerEpoch[c.o] = GroupEpoch[c.g]
      BY <1>6, <2>2, <2>3 DEF CopyOk
    <2>5. ~c.fill =>
            /\ c.kind \in {"clone", "load", "store", "save"}
            /\ c.kind \in {"clone", "load"} => c.write \in Units /\ alloc'[c.write] = c.o
            /\ c.kind \in {"clone", "load"} => \A t \in tasks' : c.write \notin t.units
            /\ c.kind \in {"store", "save"} => c.write \in HostUnits /\ alloc'[c.write] = "copy"
            /\ \A o \in Owners, p \in Positions : table'[o][p].u # c.write
            /\ \A x \in copies' : x.write = c.write => x = c
      BY <1>6, <2>2, <2>3 DEF CopyOk
    <2>6. c.fill =>
            \/ /\ c.kind = "load"
               /\ cache'[c.k] = c.write
               /\ Mem(c.g) \notin valid'[c.write]
               /\ \A x \in copies' : (x.write = c.write /\ Mem(x.g) = Mem(c.g)) => x = c
               /\ \A t \in tasks' :
                    (t.kind = "transfer" /\ Mem(t.to) = Mem(c.g)) => c.write \notin t.writes
            \/ c.kind = "restore" /\ c.write \in Slots /\ alloc'[c.write] = c.o
            \/ /\ c.kind = "store"
               /\ c.write \in HostUnits /\ hcache'[c.k] = c.write
               /\ \A x \in copies' : x.write = c.write => x = c
      <3>1. c.fill /\ c.kind = "load" => c.write \in Units
        BY <1>0, <2>2 DEF CopyOk, UnitTyped, WorkTyped, AllUnits
      <3> QED
        BY <1>6, <2>2, <2>3, <3>1 DEF CopyOk
    <2>7. /\ c.kind \in {"clone", "load"} =>
               /\ c.o \in Owners /\ c.p \in Positions
               /\ table'[c.o][c.p] = Empty
               /\ \A x \in copies' : (x.o = c.o /\ x.p = c.p) => x = c
          /\ c.kind \in {"save", "restore"} => c.o \in Owners
          /\ c.o \in Owners => ostate'[c.o] # "new"
          /\ (c.o \in Owners /\ Live(c.o)') => ogroup'[c.o] = c.g /\ ~Moving(c.o)'
      BY <1>6, <2>2 DEF CopyOk, Live
    <2> QED
      BY <2>4, <2>5, <2>6, <2>7 DEF CopyOk
  <1>13. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. TaskOk(t)
      BY TasksOkEach DEF IndInv
    <2>3. \A u \in t.units \cup {t.slot, t.h} : u # NONE =>
            alloc'[u] = alloc[u] /\ (u \in Units => valid'[u] = valid[u])
      BY <1>0, <1>4, <1>5, <2>2 DEF TaskOk, WorkTyped, AllUnits
    <2>4. t.kind = "transfer" =>
            /\ t.writes \subseteq t.units /\ t.slot # NONE /\ t.writes \subseteq Units
            /\ (Live(t.o) => \A u \in TableUnits(t.o) : valid'[u] = valid[u])
      <3>1. \A o \in Owners : \A u \in TableUnits(o) : u \in Units /\ Held(u)'
        BY <1>0, <1>5 DEF TableUnits, OwnerTyped, Empty, Private, Shared, AllUnits
      <3> QED
        BY <1>4, <2>2, <3>1 DEF TaskOk
    <2>5. t.kind = "prefetch" => t.h # NONE
      BY <2>2 DEF TaskOk
    <2> QED
      BY <1>6, <2>2, <2>3, <2>4, <2>5 DEF TaskOk, Live, TableUnits, Mem
  <1>14. /\ MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
         /\ TrialVisibility' /\ NoBad'
    (* A block a table holds stays held, so its validity stays.            *)
    <2>0. \A o \in Owners, p \in Positions :
            table[o][p].u # NONE => valid'[table[o][p].u] = valid[table[o][p].u]
      BY <1>0, <1>4, <1>5 DEF OwnerTyped, Empty, Private, Shared, AllUnits
    <2>1. MovingUnmapped' /\ TablesMapped' /\ HandoffOk'
      <3>1. \A u : (\E t \in tasks' : t.kind = "replicate" /\ u \in t.units)
                     <=> (\E t \in tasks : t.kind = "replicate" /\ u \in t.units)
        OBVIOUS
      <3>2. MovingUnmapped'
        BY <1>6 DEF IndInv, MovingUnmapped, Moving
      <3>3. TablesMapped'
        BY <1>6, <2>0, <3>1 DEF IndInv, TablesMapped, Live, HoldsAt, ReaderRow, ReadersOf, Mem
      <3>4. HandoffOk'
        <4> SUFFICES ASSUME NEW y \in handoff
                     PROVE  /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
                            /\ (~\E x \in requests : Names(x, y)) =>
                                 /\ ostate[y.o] = "live"
                                 /\ \/ ogroup[y.o] = y.g
                                    \/ /\ y.g \in ReadersOf(y.o) /\ y.p \in Positions /\ ~y.w
                                       /\ Mem(y.g) \in valid'[y.u]
                                 /\ IF y.p = SLOT THEN oslot[y.o] = y.u ELSE table[y.o][y.p].u = y.u
                                 /\ (y.p \in Positions /\ y.w) =>
                                      /\ table[y.o][y.p].k = "private"
                                      /\ ~\E e \in deferred' :
                                            e.kind \in {"commit", "replicate"} /\ y.u \in e.units
                                      /\ ~\E z \in tasks : z.kind = "replicate" /\ y.u \in z.units
          BY DEF HandoffOk, Live, HoldsAt, ReaderRow, ReadersOf
        <4>1. /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
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
          BY DEF IndInv, HandoffOk, Live, HoldsAt, ReaderRow
        <4>2. (y.p \in Positions /\ table[y.o][y.p].u = y.u /\ y.o \in Owners)
                => valid'[y.u] = valid[y.u]
          <5>1. y.u \in AllUnits
            BY <1>0 DEF RequestTyped
          <5> QED
            BY <2>0, <5>1 DEF AllUnits
        <4> QED
          BY <1>6, <4>1, <4>2
      <3> QED
        BY <3>2, <3>3, <3>4
    <2>2. LostOk' /\ NoBad'
      BY DEF IndInv, LostOk, NoBad
    <2>3. RequestsOk' /\ TrialVisibility'
      BY <1>6 DEF IndInv, RequestsOk, TrialVisibility, MadeBy
    <2> QED
      BY <2>1, <2>2, <2>3
  <1>15. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY BatchesOkEach
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
                 /\ \A e \in deferred : (e.kind = "transfer" /\ e.o = o) => e.after[b.g] >= b.seq
                 /\ ~\E t \in tasks : t.kind = "transfer" /\ t.o = o
        <4>1. Covered(b, o, u)'
          BY <1>6, <3>1 DEF Covered, Live, HoldsAt, ReadersOf
        <4>2. Held(u)'
          BY <1>5, <3>1 DEF HoldsAt, Rows
        <4> QED
          BY <4>1, <4>2
      <3>2. CASE \E e \in deferred : e.kind = "release" /\ u \in e.units /\ e.after[b.g] >= b.seq
        <4>1. PICK e \in deferred : e.kind = "release" /\ u \in e.units /\ e.after[b.g] >= b.seq
          BY <3>2
        <4>2. e # d
          <5>1. b.g \in Groups /\ b.seq \in Nat
            BY <2>2 DEF BatchOk
          <5>2. e.after \in Stamps
            BY <1>0, <4>1 DEF WorkTyped
          <5>3. e.after[b.g] \in Nat
            BY <5>1, <5>2 DEF Stamps
          <5> QED
            BY <4>1, <5>1, <5>3
        <4>3. e \in deferred'
          BY <4>1, <4>2
        <4> QED
          BY <1>5, <4>1, <4>3 DEF Covered
      <3> QED
        BY <2>3, <3>1, <3>2 DEF Covered
    <2>4. \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits /\ x[2] \notin free
                             /\ Covered(b, x[1], x[2])
      BY <1>0, <2>2 DEF BatchOk, RequestTyped
    <2>5. \A x \in b.reads :
            /\ Covered(b, x[1], x[2])' /\ x[2] \notin free' /\ alloc'[x[2]] = alloc[x[2]]
            /\ x[2] \in Units => valid'[x[2]] = valid[x[2]]
      BY <1>1, <1>4, <2>3, <2>4
    <2>6. /\ \A x \in b.cw : \E c \in copies' : c.write = x /\ c.g = b.g /\ c.seq = b.seq
          /\ \A x \in b.cr : \E c \in copies' : c.read = x /\ c.g = b.g /\ c.seq = b.seq
      BY <2>2 DEF BatchOk
    <2> QED
      BY <1>6, <2>2, <2>5, <2>6 DEF BatchOk, Mem
  (* The crossings, and where copies and crossings read: what they hold   *)
  (* stays held, keeps its allocation and stays valid where it was.       *)
  <1>16. CrossingsOk'
    <2>1. \A e \in deferred' : CrossWaitOk(e)'
      <3> SUFFICES ASSUME NEW e \in deferred' PROVE CrossWaitOk(e)'
        OBVIOUS
      <3>1. e \in deferred /\ CrossWaitOk(e)
        BY <1>6, CrossingsOkEach DEF IndInv
      <3>2. \A x \in e.units : alloc'[x] = alloc[x]
        BY <1>0, <1>4, <1>5 DEF WorkTyped
      <3> QED
        BY <1>6, <3>1, <3>2 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>2. \A t \in tasks' : CrossTaskOk(t)'
      <3> SUFFICES ASSUME NEW t \in tasks PROVE CrossTaskOk(t)'
        OBVIOUS
      <3>1. CrossTaskOk(t) /\ t.held
        BY CrossingsOkEach DEF IndInv, TasksOk
      <3>2. t.kind = "replicate" =>
              \A x \in t.writes : alloc'[x] = alloc[x] /\ valid'[x] = valid[x]
        <4>1. t.kind = "replicate" => t.writes \subseteq t.units /\ t.units \subseteq Units
          BY <3>1 DEF CrossTaskOk
        <4> QED
          BY <1>4, <1>5, <3>1, <4>1 DEF AllUnits
      <3> QED
        BY <1>6, <3>1, <3>2 DEF CrossTaskOk, ReadersOf, Mem
    <2> QED
      BY <2>1, <2>2, CrossingsOkEach
  <1>17. CopiesPlaced' /\ CrossingsPlaced'
    <2>1. \A c \in copies' : CopyPlaced(c)'
      <3> SUFFICES ASSUME NEW c \in copies' PROVE CopyPlaced(c)'
        OBVIOUS
      <3>1. c \in copies /\ CopyPlaced(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
        BY <1>0, <1>6, PlacedEach DEF IndInv, WorkTyped
      <3>2. /\ c.read \in Units \cup HostUnits => valid'[c.read] = valid[c.read]
            /\ c.write \in Units \cup HostUnits => valid'[c.write] = valid[c.write]
        BY <1>4, <1>5, <3>1
      <3> QED
        BY <3>1, <3>2 DEF CopyPlaced, Mem
    <2>2. \A t \in tasks' : CrossingPlaced(t)'
      <3> SUFFICES ASSUME NEW t \in tasks PROVE CrossingPlaced(t)'
        OBVIOUS
      <3>1. CrossingPlaced(t) /\ t.held /\ (t.kind = "replicate" => t.units \subseteq Units)
        BY PlacedEach, CrossingsOkEach DEF IndInv, TasksOk, CrossTaskOk
      <3>2. t.kind = "replicate" => \A x \in t.units : valid'[x] = valid[x]
        BY <1>4, <1>5, <3>1 DEF AllUnits
      <3> QED
        BY <3>1, <3>2 DEF CrossingPlaced, Mem
    <2> QED
      BY <2>1, <2>2, PlacedEach
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17 DEF IndInv

-----------------------------------------------------------------------------
(* The executors' tables and the completions stay as IndInv describes them *)
(* across a step that changes validity only on host units.               *)
LEMMA HostValidMapped ==
  ASSUME IndInv,
         UNCHANGED <<dbt, ostate, ogroup, oslot, table, deferred, handoff, requests>>,
         Crossings' = Crossings,
         \A u \in Units : valid'[u] = valid[u]
  PROVE  TablesMapped' /\ HandoffOk'
  <1>0. OwnerTyped
    BY OwnerTypes
  <1>1. \A o \in Owners, p \in Positions : table[o][p].u # NONE => table[o][p].u \in Units
    BY <1>0 DEF OwnerTyped, Empty, Private, Shared
  <1>2. \A u : (\E t \in tasks' : t.kind = "replicate" /\ u \in t.units)
                 <=> (\E t \in tasks : t.kind = "replicate" /\ u \in t.units)
    BY DEF Crossings
  (* A reader's row names a block a table holds: a device block.           *)
  <1>3. \A g \in Groups, o \in Owners, p \in Rows : \A u, w :
          (HoldsAt(o, p, u) /\ u # NONE /\ ReaderRow(g, o, p, u, w)) => ReaderRow(g, o, p, u, w)'
    <2> SUFFICES ASSUME NEW g \in Groups, NEW o \in Owners, NEW p \in Rows, NEW u, NEW w,
                        HoldsAt(o, p, u), u # NONE, ReaderRow(g, o, p, u, w)
                 PROVE  ReaderRow(g, o, p, u, w)'
      OBVIOUS
    <2>1. p \in Positions /\ table[o][p].u = u
      BY DEF ReaderRow, HoldsAt
    <2>2. u \in Units
      BY <1>1, <2>1
    <2> QED
      BY <2>2 DEF ReaderRow, ReadersOf, Mem
  <1>4. TablesMapped'
    <2> SUFFICES ASSUME NEW g \in Groups, NEW o \in Owners, NEW p \in Rows, dbt[g][o][p].u # NONE
                 PROVE  (/\ Live(o)
                         /\ ogroup[o] = g \/ ReaderRow(g, o, p, dbt[g][o][p].u, dbt[g][o][p].w)
                         /\ HoldsAt(o, p, dbt[g][o][p].u)
                         /\ (p \in Positions /\ dbt[g][o][p].w) =>
                              /\ table[o][p].k = "private"
                              /\ ~\E d \in deferred :
                                    d.kind \in {"commit", "replicate"} /\ dbt[g][o][p].u \in d.units
                              /\ ~\E t \in tasks :
                                    t.kind = "replicate" /\ dbt[g][o][p].u \in t.units)'
      BY DEF TablesMapped
    <2>1. /\ Live(o)
          /\ ogroup[o] = g \/ ReaderRow(g, o, p, dbt[g][o][p].u, dbt[g][o][p].w)
          /\ HoldsAt(o, p, dbt[g][o][p].u)
          /\ (p \in Positions /\ dbt[g][o][p].w) =>
               /\ table[o][p].k = "private"
               /\ ~\E d \in deferred :
                     d.kind \in {"commit", "replicate"} /\ dbt[g][o][p].u \in d.units
               /\ ~\E t \in tasks : t.kind = "replicate" /\ dbt[g][o][p].u \in t.units
      BY DEF IndInv, TablesMapped
    <2> QED
      BY <2>1, <1>2, <1>3 DEF Live, HoldsAt
  <1>5. HandoffOk'
    <2> SUFFICES ASSUME NEW h \in handoff
                 PROVE  (/\ h.o \in Owners /\ h.g \in Groups /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
                         /\ (~\E r \in requests : Names(r, h)) =>
                              /\ Live(h.o)
                              /\ ogroup[h.o] = h.g \/ ReaderRow(h.g, h.o, h.p, h.u, h.w)
                              /\ HoldsAt(h.o, h.p, h.u)
                              /\ (h.p \in Positions /\ h.w) =>
                                   /\ table[h.o][h.p].k = "private"
                                   /\ ~\E d \in deferred :
                                         d.kind \in {"commit", "replicate"} /\ h.u \in d.units
                                   /\ ~\E t \in tasks :
                                         t.kind = "replicate" /\ h.u \in t.units)'
      BY DEF HandoffOk
    <2>1. /\ h.o \in Owners /\ h.g \in Groups /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
          /\ h.p \in Rows /\ h.u \in AllUnits
          /\ (~\E r \in requests : Names(r, h)) =>
               /\ Live(h.o)
               /\ ogroup[h.o] = h.g \/ ReaderRow(h.g, h.o, h.p, h.u, h.w)
               /\ HoldsAt(h.o, h.p, h.u)
               /\ (h.p \in Positions /\ h.w) =>
                    /\ table[h.o][h.p].k = "private"
                    /\ ~\E d \in deferred : d.kind \in {"commit", "replicate"} /\ h.u \in d.units
                    /\ ~\E t \in tasks : t.kind = "replicate" /\ h.u \in t.units
      BY RequestTypes DEF IndInv, HandoffOk, RequestTyped
    <2>2. h.u # NONE
      BY <2>1 DEF AllUnits
    <2>3. (~\E r \in requests' : Names(r, h)) = (~\E r \in requests : Names(r, h))
      OBVIOUS
    <2> QED
      BY <2>1, <2>2, <2>3, <1>2, <1>3 DEF Live, HoldsAt
  <1> QED
    BY <1>4, <1>5

-----------------------------------------------------------------------------
(* Filling an empty position of a live owner's table, as Allocate, Pin    *)
(* and the end of a clone or a load do: what it means for the executors'   *)
(* tables, the completions and the stamped requests.                       *)

LEMMA FillPlane ==
  ASSUME IndInv, NEW o \in Owners, NEW p \in Positions, NEW v \in Units, NEW wr \in BOOLEAN,
         ostate[o] = "live", table[o][p] = Empty, ~Moving(o),
         \A q \in Owners, r \in Positions : (q # o \/ r # p) => table'[q][r] = table[q][r],
         table'[o][p].u = v,
         wr => /\ table'[o][p].k = "private" /\ \A d \in deferred : v \notin d.units
               /\ \A t \in tasks' : v \notin t.units,
         Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> v, w |-> wr]),
         UNCHANGED <<ostate, ogroup, oslot, requests, lost, inflight, launched, ncopy>>,
         \A d \in deferred' : d \in deferred,
         \A t \in tasks' : t.kind \in {"transfer", "replicate"} => t \in tasks,
         \A c \in copies' : c \in copies,
         \* A block a table holds stays valid in every memory it was valid in.
         \A u \in Units : (\E q \in Owners, r \in Positions : table[q][r].u = u) =>
                           valid[u] \subseteq valid'[u]
  PROVE  MovingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
  <1> DEFINE h == [g |-> ogroup[o], o |-> o, p |-> p, u |-> v, w |-> wr]
  <1>1. ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o] /\ OwnerEpoch[o] \notin lost
    BY DEF IndInv, OwnersOk, Live
  <1>2. /\ handoff' = IF Eager THEN handoff ELSE handoff \cup {h}
        /\ \A x \in Groups, q \in Owners, r \in Rows :
             dbt'[x][q][r] = IF Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h)
                             THEN [u |-> h.u, w |-> h.w]
                             ELSE dbt[x][q][r]
    <2>1. h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows
      BY <1>1 DEF Rows
    <2> QED
      BY <2>1, GiveRow, PlaneTypes DEF PlaneTyped
  <1>3. \A x \in Groups, r \in Rows : dbt[x][o][r].u # NONE => r # p
    BY DEF IndInv, TablesMapped, HoldsAt, Empty, Rows
  <1>4. \A y \in handoff : (Unnamed(y) /\ y.o = o) => y.p # p
    <2> SUFFICES ASSUME NEW y \in handoff, Unnamed(y), y.o = o, y.p = p PROVE FALSE
      OBVIOUS
    <2>1. y.u \in AllUnits /\ table[o][p].u = y.u
      BY DEF IndInv, HandoffOk, HoldsAt, Unnamed, TypeInv
    <2> QED
      BY <2>1 DEF Empty, AllUnits
  <1>5. \A q \in Owners, r \in Rows :
          (q # o \/ r # p) => IF r = SLOT THEN oslot'[q] = oslot[q] ELSE table'[q][r] = table[q][r]
    BY DEF Rows
  <1>6. MovingUnmapped'
    <2> SUFFICES ASSUME NEW q \in Owners,
                        \/ \E t \in tasks : t.kind = "transfer" /\ t.o = q
                        \/ \E d \in deferred : d.kind = "transfer" /\ d.o = q
                 PROVE  /\ \A x \in Groups, r \in Rows : dbt'[x][q][r].u = NONE
                        /\ \A y \in handoff' : y.o # q
                        /\ \A d, e \in deferred :
                             (d.kind = "transfer" /\ e.kind = "transfer" /\ d.o = q /\ e.o = q) => d = e
      BY DEF MovingUnmapped
    <2>1. q # o
      BY DEF Moving
    <2> QED
      BY <2>1, <1>2 DEF IndInv, MovingUnmapped
  <1>7. LostOk'
    BY <1>1, <1>2 DEF IndInv, LostOk
  (* A reader's row of a block a table holds stays a reader's row; a row   *)
  (* that is writable names a block no crossing holds.                     *)
  <1>11. \A x \in Groups, q \in Owners, r \in Rows : \A u, w :
           (r \in Positions /\ u = table[q][r].u /\ u \in Units /\ ReaderRow(x, q, r, u, w))
               => ReaderRow(x, q, r, u, w)'
    BY DEF ReaderRow, ReadersOf, Mem
  <1>12. \A u : (\E t \in tasks' : t.kind = "replicate" /\ u \in t.units)
                  => (\E t \in tasks : t.kind = "replicate" /\ u \in t.units)
    OBVIOUS
  <1>8. TablesMapped'
    <2> SUFFICES ASSUME NEW x \in Groups, NEW q \in Owners, NEW r \in Rows, dbt'[x][q][r].u # NONE
                 PROVE  /\ ostate'[q] = "live"
                        /\ \/ ogroup'[q] = x
                           \/ ReaderRow(x, q, r, dbt[x][q][r].u, dbt[x][q][r].w)'
                        /\ IF r = SLOT THEN oslot'[q] = dbt'[x][q][r].u
                                       ELSE table'[q][r].u = dbt'[x][q][r].u
                        /\ (r \in Positions /\ dbt'[x][q][r].w) =>
                             /\ table'[q][r].k = "private"
                             /\ ~\E d \in deferred' :
                                   d.kind \in {"commit", "replicate"} /\ dbt'[x][q][r].u \in d.units
                             /\ ~\E t \in tasks' : t.kind = "replicate" /\ dbt'[x][q][r].u \in t.units
      BY DEF TablesMapped, Live, HoldsAt
    <2>1. CASE Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h)
      <3>1. dbt'[x][q][r] = [u |-> v, w |-> wr] /\ x = ogroup[o] /\ q = o /\ r = p
        BY <2>1, <1>2
      <3> QED
        BY <3>1, SLOT \notin Positions
    <2>2. CASE ~(Eager /\ x = h.g /\ q = h.o /\ r = h.p /\ Unnamed(h))
      <3>1. dbt'[x][q][r] = dbt[x][q][r]
        BY <2>2, <1>2
      <3>2. /\ ostate[q] = "live"
            /\ ogroup[q] = x \/ ReaderRow(x, q, r, dbt[x][q][r].u, dbt[x][q][r].w)
            /\ HoldsAt(q, r, dbt[x][q][r].u)
            /\ (r \in Positions /\ dbt[x][q][r].w) =>
                 /\ table[q][r].k = "private"
                 /\ ~\E d \in deferred :
                       d.kind \in {"commit", "replicate"} /\ dbt[x][q][r].u \in d.units
                 /\ ~\E t \in tasks : t.kind = "replicate" /\ dbt[x][q][r].u \in t.units
        BY <3>1 DEF IndInv, TablesMapped, Live
      <3>3. q # o \/ r # p
        BY <3>1, <1>3
      <3>4. IF r = SLOT THEN oslot'[q] = oslot[q] ELSE table'[q][r] = table[q][r]
        BY <3>3, <1>5
      <3>5. ReaderRow(x, q, r, dbt[x][q][r].u, dbt[x][q][r].w)
              => ReaderRow(x, q, r, dbt[x][q][r].u, dbt[x][q][r].w)'
        <4> SUFFICES ASSUME ReaderRow(x, q, r, dbt[x][q][r].u, dbt[x][q][r].w)
                     PROVE  ReaderRow(x, q, r, dbt[x][q][r].u, dbt[x][q][r].w)'
          OBVIOUS
        <4>1. r \in Positions /\ dbt[x][q][r].u = table[q][r].u
          BY <3>2 DEF ReaderRow, HoldsAt
        <4>2. dbt[x][q][r].u \in Units
          <5>1. dbt[x][q][r].u # NONE
            BY <3>1
          <5> QED
            BY <4>1, <5>1, OwnerTypes DEF OwnerTyped, Empty, Private, Shared
        <4>3. valid[dbt[x][q][r].u] \subseteq valid'[dbt[x][q][r].u]
          BY <4>1, <4>2
        <4>4. ReadersOf(q)' = ReadersOf(q)
          BY DEF ReadersOf
        <4> QED
          BY <3>1, <4>1, <4>3, <4>4 DEF ReaderRow, Mem
      <3> QED
        BY <3>1, <3>2, <3>4, <3>5, <1>12 DEF HoldsAt, Rows
    <2> QED
      BY <2>1, <2>2
  <1>9. HandoffOk'
    <2> SUFFICES ASSUME NEW y \in handoff'
                 PROVE  /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
                        /\ Unnamed(y) =>
                             /\ ostate'[y.o] = "live"
                             /\ ogroup'[y.o] = y.g \/ ReaderRow(y.g, y.o, y.p, y.u, y.w)'
                             /\ IF y.p = SLOT THEN oslot'[y.o] = y.u ELSE table'[y.o][y.p].u = y.u
                             /\ (y.p \in Positions /\ y.w) =>
                                  /\ table'[y.o][y.p].k = "private"
                                  /\ ~\E d \in deferred' :
                                        d.kind \in {"commit", "replicate"} /\ y.u \in d.units
                                  /\ ~\E t \in tasks' : t.kind = "replicate" /\ y.u \in t.units
      BY DEF HandoffOk, Live, Unnamed, HoldsAt
    <2>1. CASE y = h
      BY <2>1, <1>1, SLOT \notin Positions
    <2>2. CASE y # h
      <3>1. y \in handoff
        BY <2>2, <1>2
      <3>2. /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
            /\ Unnamed(y) =>
                 /\ ostate[y.o] = "live" /\ (ogroup[y.o] = y.g \/ ReaderRow(y.g, y.o, y.p, y.u, y.w))
                 /\ HoldsAt(y.o, y.p, y.u)
                 /\ (y.p \in Positions /\ y.w) =>
                      /\ table[y.o][y.p].k = "private"
                      /\ ~\E d \in deferred : d.kind \in {"commit", "replicate"} /\ y.u \in d.units
                      /\ ~\E t \in tasks : t.kind = "replicate" /\ y.u \in t.units
        BY <3>1 DEF IndInv, HandoffOk, Live, Unnamed
      <3>3. y.p \in Rows /\ y.u \in AllUnits
        BY <3>1 DEF IndInv, TypeInv
      <3>4. Unnamed(y) => (y.o # o \/ y.p # p)
        BY <3>1, <1>4
      <3>5. Unnamed(y) =>
              IF y.p = SLOT THEN oslot'[y.o] = oslot[y.o] ELSE table'[y.o][y.p] = table[y.o][y.p]
        BY <3>2, <3>3, <3>4, <1>5
      <3>6. (Unnamed(y) /\ ReaderRow(y.g, y.o, y.p, y.u, y.w)) => ReaderRow(y.g, y.o, y.p, y.u, y.w)'
        <4> SUFFICES ASSUME Unnamed(y), ReaderRow(y.g, y.o, y.p, y.u, y.w)
                     PROVE  ReaderRow(y.g, y.o, y.p, y.u, y.w)'
          OBVIOUS
        <4>1. y.p \in Positions /\ y.u = table[y.o][y.p].u /\ y.o \in Owners
          BY <3>2 DEF ReaderRow, HoldsAt
        <4>2. y.u \in Units
          <5>1. y.u # NONE
            BY <3>3 DEF AllUnits
          <5> QED
            BY <4>1, <5>1, OwnerTypes DEF OwnerTyped, Empty, Private, Shared
        <4> QED
          BY <4>1, <4>2, <3>2, <3>3, <1>11
      <3> QED
        BY <3>2, <3>3, <3>5, <3>6, <1>12 DEF HoldsAt, Unnamed, Rows
    <2> QED
      BY <2>1, <2>2
  <1>10. RequestsOk'
    <2>1. \A r \in requests :
            r.kind # "finish" =>
              \A x \in Groups, q \in Rows :
                (dbt'[x][r.o][q].u # NONE /\ (TakesRow(r, r.o, q) \/ FreezesRow(r, r.o, q))) =>
                  /\ x \in r.sg
                  /\ x \in r.by => FreezesRow(r, r.o, q) /\ ~dbt'[x][r.o][q].w
      <3> SUFFICES ASSUME NEW r \in requests, r.kind # "finish", NEW x \in Groups, NEW q \in Rows,
                          dbt'[x][r.o][q].u # NONE, TakesRow(r, r.o, q) \/ FreezesRow(r, r.o, q)
                   PROVE  /\ x \in r.sg
                          /\ x \in r.by => FreezesRow(r, r.o, q) /\ ~dbt'[x][r.o][q].w
        OBVIOUS
      <3>1. r.o \in Owners
        BY DEF IndInv, TypeInv
      <3>2. CASE Eager /\ x = h.g /\ r.o = h.o /\ q = h.p /\ Unnamed(h)
        <4>1. Names(r, h)
          BY <3>2, NamesRow
        <4> QED
          BY <3>2, <4>1 DEF Unnamed
      <3>3. CASE ~(Eager /\ x = h.g /\ r.o = h.o /\ q = h.p /\ Unnamed(h))
        <4>1. dbt'[x][r.o][q] = dbt[x][r.o][q]
          BY <3>1, <3>3, <1>2
        <4> QED
          BY <3>1, <4>1 DEF IndInv, RequestsOk
      <3> QED
        BY <3>2, <3>3
    <2> QED
      BY <2>1 DEF IndInv, RequestsOk
  <1> QED
    BY <1>6, <1>7, <1>8, <1>9, <1>10
=============================================================================
