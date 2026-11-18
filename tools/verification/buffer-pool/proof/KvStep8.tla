------------------------------- MODULE KvStep8 -------------------------------
(***************************************************************************)
(* The inductive step of IndInv for the handoff: LendDue, Import, Borrow,  *)
(* GiveBack, ReleaseBorrow and ImportEnd.                                  *)
(***************************************************************************)
EXTENDS KvInductive, KvProof, KvFrame, TLAPS

USE Sizes, Design

(* What IndInv says of the tasks and the units in general.                 *)
LEMMA TaskFacts ==
  ASSUME IndInv
  PROVE  /\ \A t \in tasks : TaskOk(t)
         /\ \A t \in tasks : t.units \subseteq AllUnits /\ t.writes \subseteq AllUnits
         /\ \A t \in tasks : Reads(t) \subseteq t.units
         /\ \A t \in tasks : t.h \in HostUnits \cup {NONE}
         /\ \A o \in Owners : TableUnits(o) \subseteq Units
  <1>1. \A t \in tasks : TaskOk(t)
    BY TasksOkEach DEF IndInv
  <1>2. \A o \in Owners : TableUnits(o) \subseteq Units
    BY OwnerTypes DEF OwnerTyped, TableUnits, Empty, Private, Shared
  <1> QED
    BY <1>1, <1>2, WorkTypes DEF WorkTyped, Reads

(* A task that stays keeps TaskOk when what each of its clauses reads keeps *)
(* the values the clause needs.                                             *)
LEMMA TaskKeep ==
  ASSUME NEW t, TaskOk(t),
         \A u \in t.units \cup {t.h} : u # NONE => alloc'[u] = alloc[u],
         t.kind = "put" => hcache'[t.k] = hcache[t.k] /\ vis'[t.k] = vis[t.k],
         t.o \in Owners /\ ostate[t.o] # "new" => ostate'[t.o] # "new",
         t.kind = "lend" =>
           /\ ostate'[t.o] = "live"
           /\ \A x \in tasks' : (x.kind = "lend" /\ x.o = t.o) => x = t
           /\ ~\E d \in deferred' : d.kind = "lend" /\ d.o = t.o
           /\ ~Importing(t.o)',
         (t.kind = "borrow" /\ ostate'[t.o] = "live") =>
           /\ \E x \in tasks' : x.kind = "lend" /\ x.o = t.o
           /\ t.units = TableUnits(t.o)',
         t.kind = "import" =>
           /\ \A u \in t.writes :
                /\ \A o \in Owners, p \in Positions : table'[o][p].u # u
                /\ \A k \in Keys : cache'[k] # u
                /\ \A d \in deferred' : u \notin d.units
                /\ \A c \in copies' : c.read # u /\ c.write # u
                /\ \A x \in tasks' : x # t => u \notin x.units
           /\ \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t
           /\ ~Lending(t.o)'
           /\ ostate'[t.o] = "live" =>
                \A x \in t.places : table'[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])',
         \A u \in Reads(t) :
           alloc[u] \in Owners =>
             (ostate'[alloc[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc[u]),
         t.kind = "prefetch" => \A x \in tasks' : (x.kind = "prefetch" /\ x.h = t.h) => x = t
  PROVE  TaskOk(t)'
  <1>1. /\ Reads(t) \subseteq t.units
        /\ t.kind = "import" => t.writes \subseteq t.units
        /\ t.kind = "prefetch" => t.h # NONE
    BY DEF TaskOk, Reads
  <1>2. /\ t.held
        /\ t.kind \in {"import", "lend", "borrow", "prefetch", "put", "replicate"}
        /\ t.kind \in {"import", "lend", "borrow"} =>
             /\ t.h = NONE /\ t.p = NONE /\ t.to = NONE
             /\ t.o \in Owners /\ ostate'[t.o] # "new"
             /\ t.units \subseteq Units
        /\ t.kind \in {"lend", "borrow"} => t.writes = {} /\ t.places = {}
        /\ t.kind \in {"prefetch", "put"} => t.units = {} /\ t.writes = {} /\ t.places = {}
    BY DEF TaskOk
  <1>3. t.kind = "lend" =>
          /\ ostate'[t.o] = "live" /\ t.units = {}
          /\ \A x \in tasks' : (x.kind = "lend" /\ x.o = t.o) => x = t
          /\ ~\E d \in deferred' : d.kind = "lend" /\ d.o = t.o
          /\ ~Importing(t.o)'
    BY DEF TaskOk
  <1>4. t.kind = "import" =>
          /\ t.writes = {x[2] : x \in t.places} /\ t.writes \subseteq t.units
          /\ \A x, y \in t.places : (x[1] = y[1] \/ x[2] = y[2]) => x = y
          /\ \A u \in t.writes : alloc'[u] = t.o
    <2> SUFFICES ASSUME t.kind = "import"
                 PROVE  /\ t.writes = {x[2] : x \in t.places} /\ t.writes \subseteq t.units
                        /\ \A x, y \in t.places : (x[1] = y[1] \/ x[2] = y[2]) => x = y
                        /\ \A u \in t.writes : alloc'[u] = t.o
      OBVIOUS
    <2>1. \A u \in t.writes : u # NONE /\ alloc[u] = t.o
      <3>1. t.units \subseteq Units
        BY DEF TaskOk
      <3> QED
        BY <1>1, <3>1 DEF TaskOk
    <2> QED
      BY <1>1, <2>1 DEF TaskOk
  <1>5. \A u \in Reads(t) :
          \/ alloc'[u] = "shared"
          \/ alloc'[u] \in Owners
             /\ (ostate'[alloc'[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc'[u])
    <2> SUFFICES ASSUME NEW u \in Reads(t)
                 PROVE  \/ alloc'[u] = "shared"
                        \/ alloc'[u] \in Owners
                           /\ (ostate'[alloc'[u]] = "live" =>
                                 \E x \in tasks' : x.kind = "lend" /\ x.o = alloc'[u])
      OBVIOUS
    <2>1. u \in t.units /\ u # NONE
      <3>1. t.kind \in {"import", "borrow"} /\ u \in t.units
        BY DEF Reads
      <3>2. t.units \subseteq Units
        BY <3>1 DEF TaskOk
      <3> QED
        BY <3>1, <3>2
    <2>2. alloc'[u] = alloc[u]
      BY <2>1
    <2>3. \/ alloc[u] = "shared"
          \/ alloc[u] \in Owners /\ (ostate[alloc[u]] = "live" => \E x \in tasks : x.kind = "lend" /\ x.o = alloc[u])
      BY DEF TaskOk, Live
    <2>4. alloc[u] \in Owners =>
            (ostate'[alloc[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc[u])
      OBVIOUS
    <2> QED
      BY <2>2, <2>3, <2>4
  <1>6. /\ t.kind = "prefetch" =>
             /\ t.h \in HostUnits /\ alloc'[t.h] = "task"
             /\ \A x \in tasks' : (x.kind = "prefetch" /\ x.h = t.h) => x = t
        /\ t.kind = "put" => t.h \in HostUnits /\ hcache'[t.k] = t.h /\ vis'[t.k].h = NONE
    BY <1>1 DEF TaskOk
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5, <1>6 DEF TaskOk, Live

(* A new import's TaskOk, from its clauses.                                 *)
LEMMA ImportOk ==
  ASSUME NEW t, t.kind = "import", t.held,
         t.h = NONE, t.p = NONE, t.to = NONE, t.o \in Owners, ostate'[t.o] # "new",
         t.units \subseteq Units,
         t.writes = {x[2] : x \in t.places}, t.writes \subseteq t.units,
         \A x, y \in t.places : (x[1] = y[1] \/ x[2] = y[2]) => x = y,
         \A u \in t.writes :
           /\ alloc'[u] = t.o
           /\ \A o \in Owners, p \in Positions : table'[o][p].u # u
           /\ \A k \in Keys : cache'[k] # u
           /\ \A d \in deferred' : u \notin d.units
           /\ \A c \in copies' : c.read # u /\ c.write # u
           /\ \A x \in tasks' : x # t => u \notin x.units,
         \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t,
         ~Lending(t.o)',
         ostate'[t.o] = "live" => \A x \in t.places : table'[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])',
         \A u \in t.units \ t.writes :
           \/ alloc'[u] = "shared"
           \/ alloc'[u] \in Owners
              /\ (ostate'[alloc'[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc'[u])
  PROVE  TaskOk(t)'
  <1>1. Reads(t) = t.units \ t.writes
    BY DEF Reads
  <1>2. /\ t.kind \in {"import", "lend", "borrow", "prefetch", "put", "replicate"}
        /\ ~(t.kind \in {"lend", "borrow", "prefetch", "put"})
    OBVIOUS
  <1> QED
    BY <1>1, <1>2 DEF TaskOk, Live

(* A new borrow's TaskOk, from its clauses.                                 *)
LEMMA BorrowOk ==
  ASSUME NEW t, t.kind = "borrow", t.held,
         t.h = NONE, t.p = NONE, t.to = NONE, t.o \in Owners, ostate'[t.o] = "live",
         t.units \subseteq Units, t.writes = {}, t.places = {},
         \E x \in tasks' : x.kind = "lend" /\ x.o = t.o,
         t.units = TableUnits(t.o)',
         \A u \in Reads(t) :
           \/ alloc'[u] = "shared"
           \/ alloc'[u] \in Owners
              /\ (ostate'[alloc'[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc'[u])
  PROVE  TaskOk(t)'
  <1>1. /\ t.kind \in {"import", "lend", "borrow", "prefetch", "put", "replicate"}
        /\ ~(t.kind \in {"import", "lend", "prefetch", "put"})
    OBVIOUS
  <1> QED
    BY <1>1 DEF TaskOk, Live

-----------------------------------------------------------------------------
(* LendDue                                                                 *)

(* The lend that stands once LendDue(d) takes effect.                      *)
LendTask(o) == [Task("lend", NONE, NONE) EXCEPT !.o = o]

LEMMA LendTaskFields ==
  ASSUME NEW o
  PROVE  LendTask(o) = [kind |-> "lend", k |-> NONE, h |-> NONE,
                        expired |-> FALSE, p |-> NONE, o |-> o, to |-> NONE, held |-> TRUE,
                        units |-> {}, writes |-> {}, places |-> {}]
  BY DEF LendTask, Task

LEMMA LendDue_Effect ==
  ASSUME NEW d \in deferred, LendDue(d)
  PROVE  /\ d.kind = "lend" /\ Reached(d.after)
         /\ deferred' = deferred \ {d}
         /\ tasks' = IF ostate[d.o] = "live" THEN tasks \cup {LendTask(d.o)} ELSE tasks
         /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, valid,
                        vis, trial, member, dbt, launched, inflight, handoff, requests, alloc,
                        sharedKey>>
  BY DEF LendDue, LendTask, plane, On, AllRules

(* What IndInv says of a lend that comes due.                              *)
LEMMA LendDue_Facts ==
  ASSUME IndInv, NEW d \in deferred, LendDue(d)
  PROVE  /\ d.o \in Owners /\ ostate[d.o] # "new" /\ d.units = {}
         /\ ~Importing(d.o)
         /\ Lending(d.o)
         /\ \A x \in deferred : (x.kind = "lend" /\ x.o = d.o) => x = d
         /\ \A t \in tasks : t.kind = "lend" => t.o # d.o
         /\ \A b \in inflight : d.after[b.g] < b.seq
  <1>1. d.kind = "lend" /\ Reached(d.after)
    BY LendDue_Effect
  <1>2. /\ d.o \in Owners /\ ostate[d.o] # "new" /\ d.units = {}
        /\ ~Importing(d.o)
    BY <1>1 DEF IndInv, DeferredOk
  <1>3. Lending(d.o)
    BY <1>1 DEF Lending
  <1>4. \A x \in deferred : (x.kind = "lend" /\ x.o = d.o) => x = d
    BY <1>1, <1>2, <1>3 DEF IndInv, LendingUnmapped
  <1>5. \A t \in tasks : t.kind = "lend" => t.o # d.o
    <2> SUFFICES ASSUME NEW t \in tasks, t.kind = "lend", t.o = d.o PROVE FALSE
      OBVIOUS
    <2>1. ~\E x \in deferred : x.kind = "lend" /\ x.o = t.o
      BY TaskFacts DEF TaskOk
    <2> QED
      BY <1>1, <2>1
  <1>6. \A b \in inflight : d.after[b.g] < b.seq
    BY <1>1 DEF Reached, IndInv, BatchesOk
  <1> QED
    BY <1>2, <1>3, <1>4, <1>5, <1>6

(* The lend's taking effect changes no unit's holders, makes no owner lent *)
(* that was not, and no owner import.                                      *)
LEMMA LendDue_Same ==
  ASSUME IndInv, NEW d \in deferred, LendDue(d)
  PROVE  /\ \A u \in AllUnits : Held(u)' = Held(u)
         /\ \A q : Lending(q)' => Lending(q)
         /\ \A q : Importing(q)' = Importing(q)
         /\ \A x \in deferred' : x \in deferred /\ x # d
         /\ \A t \in tasks' : t \in tasks \/ t = LendTask(d.o)
         /\ \A t \in tasks : t \in tasks'
         /\ Crossings' = Crossings
         /\ \A u : (\E t \in tasks' : u \in Reads(t)) <=> (\E t \in tasks : u \in Reads(t))
         /\ \A t \in tasks' : t.kind \in {"import", "borrow", "replicate", "prefetch", "put"}
                                => t \in tasks
  <1>1. /\ deferred' = deferred \ {d}
        /\ tasks' = IF ostate[d.o] = "live" THEN tasks \cup {LendTask(d.o)} ELSE tasks
        /\ UNCHANGED <<table, oslot, cache, hcache, copies>>
    BY LendDue_Effect
  <1>2. d.units = {} /\ Lending(d.o)
    BY LendDue_Facts
  <1>3. LendTask(d.o).kind = "lend" /\ LendTask(d.o).units = {} /\ LendTask(d.o).h = NONE
        /\ LendTask(d.o).o = d.o /\ LendTask(d.o).writes = {}
    BY LendTaskFields
  <1>4. \A u \in AllUnits : Held(u)' = Held(u)
    <2> SUFFICES ASSUME NEW u \in AllUnits PROVE Held(u)' = Held(u)
      OBVIOUS
    <2>1. u # NONE
      BY DEF AllUnits
    <2> QED
      BY <1>1, <1>2, <1>3, <2>1 DEF Held, HeldIn
  <1>5. \A q : Lending(q)' => Lending(q)
    BY <1>1, <1>2, <1>3 DEF Lending
  <1>6. \A q : Importing(q)' = Importing(q)
    BY <1>1, <1>3 DEF Importing
  <1>7. Crossings' = Crossings
    BY <1>1, <1>3 DEF Crossings
  <1>8. \A u : (\E t \in tasks' : u \in Reads(t)) <=> (\E t \in tasks : u \in Reads(t))
    BY <1>1, <1>3 DEF Reads
  <1> QED
    BY <1>1, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8

LEMMA LendDue_Inv ==
  ASSUME IndInv, NEW d \in deferred, LendDue(d), UNCHANGED rare
  PROVE  IndInv'
  <1>0. /\ UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. /\ d.kind = "lend" /\ Reached(d.after)
        /\ deferred' = deferred \ {d}
        /\ tasks' = IF ostate[d.o] = "live" THEN tasks \cup {LendTask(d.o)} ELSE tasks
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, valid,
                       vis, trial, member, dbt, launched, inflight, handoff, requests, alloc,
                       sharedKey, remote, lost, bad>>
    BY LendDue_Effect DEF rare
  <1>2. /\ d.o \in Owners /\ ostate[d.o] # "new" /\ d.units = {}
        /\ ~Importing(d.o)
        /\ Lending(d.o)
        /\ \A x \in deferred : (x.kind = "lend" /\ x.o = d.o) => x = d
        /\ \A t \in tasks : t.kind = "lend" => t.o # d.o
        /\ \A b \in inflight : d.after[b.g] < b.seq
    BY LendDue_Facts
  <1>3. /\ \A u \in AllUnits : Held(u)' = Held(u)
        /\ \A q : Lending(q)' => Lending(q)
        /\ \A q : Importing(q)' = Importing(q)
        /\ \A x \in deferred' : x \in deferred /\ x # d
        /\ \A t \in tasks' : t \in tasks \/ t = LendTask(d.o)
        /\ \A t \in tasks : t \in tasks'
        /\ Crossings' = Crossings
        /\ \A u : (\E t \in tasks' : u \in Reads(t)) <=> (\E t \in tasks : u \in Reads(t))
        /\ \A t \in tasks' : t.kind \in {"import", "borrow", "replicate", "prefetch", "put"}
                               => t \in tasks
    BY LendDue_Same
  <1>4. LendTask(d.o) = [kind |-> "lend", k |-> NONE, h |-> NONE,
                         expired |-> FALSE, p |-> NONE, o |-> d.o, to |-> NONE, held |-> TRUE,
                         units |-> {}, writes |-> {}, places |-> {}]
    BY LendTaskFields
  <1>5. TypeInv'
    <2>1. WorkTyped'
      <3>1. \A t \in tasks' :
              /\ t.units \subseteq AllUnits /\ t.writes \subseteq AllUnits
              /\ t.k \in Keys \cup {NONE} /\ t.h \in HostUnits \cup {NONE}
              /\ t.o \in Owners \cup {NONE}
              /\ t.to \in Groups \cup {NONE}
              /\ t.p \in Positions \cup {NONE}
              /\ t.places \subseteq Positions \X Units
              /\ t.kind \in {"import", "lend", "borrow", "replicate"} <=> t.k = NONE
              /\ t.kind = "prefetch" => t.to \in Groups
              /\ t = [kind |-> t.kind, k |-> t.k, h |-> t.h,
                      expired |-> t.expired, p |-> t.p, o |-> t.o, to |-> t.to, held |-> t.held,
                      units |-> t.units, writes |-> t.writes, places |-> t.places]
        BY <1>0, <1>2, <1>3, <1>4 DEF WorkTyped
      <3> QED
        BY <1>0, <1>1, <1>3, <3>1 DEF WorkTyped
    <2> QED
      BY <1>0, <1>1, <2>1, UnitTypedFrame, OwnerTypedFrame, PlaneTypedFrame, RequestTypedFrame,
         TypeInvNext
  <1>6. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
        /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk'
    <2>1. FreeClean' /\ NoLeak'
      BY <1>1, <1>3, UnitTypes DEF IndInv, FreeClean, NoLeak, InRegion, UnitTyped
    <2> QED
      BY <1>1, <2>1, OutsideFrame, RegionsFrame, AllocOwnerFrame, SharedIffFrame,
         SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame DEF IndInv
  <1>7. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred' PROVE DeferOk(e)'
      BY DeferredOkEach
    <2>2. e \in deferred /\ e # d /\ DeferOk(e)
      BY <1>3, DeferredOkEach DEF IndInv
    <2> QED
      BY <1>1, <1>3, <2>2 DEF DeferOk
  <1>8. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY <1>1, CopiesOkEach
    <2>2. CopyOk(c)
      BY CopiesOkEach DEF IndInv
    <2>3. \A u : (\A t \in tasks : u \notin t.units) => \A t \in tasks' : u \notin t.units
      BY <1>3, <1>4
    <2> QED
      BY <1>1, <1>3, <2>2, <2>3 DEF CopyOk, Live, Mem
  <1>9. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. \A x \in tasks : TaskOk(x)
      BY TaskFacts
    <2>3. CASE t = LendTask(d.o)
      <3>1. ostate[d.o] = "live"
        <4>1. CASE ostate[d.o] = "live"
          BY <4>1
        <4>2. CASE ostate[d.o] # "live"
          <5>1. t \in tasks
            BY <1>1, <4>2
          <5> QED
            BY <2>3, <5>1, <1>2, <1>4
        <4> QED
          BY <4>1, <4>2
      <3>2. \A x \in tasks' : (x.kind = "lend" /\ x.o = d.o) => x = t
        BY <1>2, <1>3, <2>3
      <3>3. ~\E e \in deferred' : e.kind = "lend" /\ e.o = d.o
        BY <1>2, <1>3
      <3>4. Reads(t) = {}
        BY <2>3, <1>4 DEF Reads
      <3> QED
        BY <1>1, <1>2, <1>3, <1>4, <2>3, <3>1, <3>2, <3>3, <3>4 DEF TaskOk, Live
    <2>4. CASE t # LendTask(d.o)
      <3>1. t \in tasks /\ TaskOk(t)
        BY <1>3, <2>2, <2>4
      <3>2. t.kind = "lend" => t.o # d.o
        BY <1>2, <3>1
      <3>4. t.kind = "import" =>
              /\ \A u \in t.writes :
                   /\ \A o \in Owners, p \in Positions : table'[o][p].u # u
                   /\ \A k \in Keys : cache'[k] # u
                   /\ \A e \in deferred' : u \notin e.units
                   /\ \A c \in copies' : c.read # u /\ c.write # u
                   /\ \A x \in tasks' : x # t => u \notin x.units
              /\ \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t
              /\ ~Lending(t.o)'
              /\ ostate'[t.o] = "live" =>
                   \A x \in t.places : table'[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])'
        <4> SUFFICES ASSUME t.kind = "import"
                     PROVE  /\ \A u \in t.writes :
                                 /\ \A o \in Owners, p \in Positions : table'[o][p].u # u
                                 /\ \A k \in Keys : cache'[k] # u
                                 /\ \A e \in deferred' : u \notin e.units
                                 /\ \A c \in copies' : c.read # u /\ c.write # u
                                 /\ \A x \in tasks' : x # t => u \notin x.units
                            /\ \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t
                            /\ ~Lending(t.o)'
                            /\ ostate'[t.o] = "live" =>
                                 \A x \in t.places : table'[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])'
          OBVIOUS
        <4>1. /\ \A u \in t.writes :
                   /\ \A o \in Owners, p \in Positions : table[o][p].u # u
                   /\ \A k \in Keys : cache[k] # u
                   /\ \A e \in deferred : u \notin e.units
                   /\ \A c \in copies : c.read # u /\ c.write # u
                   /\ \A x \in tasks : x # t => u \notin x.units
              /\ \A x \in tasks : (x.kind = "import" /\ x.o = t.o) => x = t
              /\ ~Lending(t.o)
              /\ ostate[t.o] = "live" =>
                   \A x \in t.places : table[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])
          BY <3>1 DEF TaskOk, Live
        <4>2. LendTask(d.o).units = {} /\ LendTask(d.o).kind = "lend"
          BY <1>4
        <4>3. \A o, p : CopyFor(o, p)' = CopyFor(o, p)
          BY <1>1 DEF CopyFor
        <4> QED
          BY <4>1, <4>2, <4>3, <1>1, <1>3
      <3>5. t.kind = "lend" =>
              /\ ostate'[t.o] = "live"
              /\ \A x \in tasks' : (x.kind = "lend" /\ x.o = t.o) => x = t
              /\ ~\E e \in deferred' : e.kind = "lend" /\ e.o = t.o
              /\ ~Importing(t.o)'
        <4>1. t.kind = "lend" =>
                /\ ostate[t.o] = "live"
                /\ \A x \in tasks : (x.kind = "lend" /\ x.o = t.o) => x = t
                /\ ~\E e \in deferred : e.kind = "lend" /\ e.o = t.o
                /\ ~Importing(t.o)
          BY <3>1 DEF TaskOk, Live
        <4> QED
          BY <4>1, <3>2, <1>1, <1>3, <1>4
      <3>6. \A q : (\E x \in tasks : x.kind = "lend" /\ x.o = q) => (\E x \in tasks' : x.kind = "lend" /\ x.o = q)
        BY <1>3
      <3>7. (t.kind = "borrow" /\ ostate'[t.o] = "live") =>
              /\ \E x \in tasks' : x.kind = "lend" /\ x.o = t.o
              /\ t.units = TableUnits(t.o)'
        <4>1. (t.kind = "borrow" /\ ostate[t.o] = "live") =>
                /\ \E x \in tasks : x.kind = "lend" /\ x.o = t.o
                /\ t.units = TableUnits(t.o)
          BY <3>1 DEF TaskOk, Live
        <4> QED
          BY <4>1, <3>6, <1>1 DEF TableUnits
      <3>8. \A u \in Reads(t) :
              alloc[u] \in Owners =>
                (ostate'[alloc[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc[u])
        <4>1. \A u \in Reads(t) :
                alloc[u] \in Owners =>
                  (ostate[alloc[u]] = "live" => \E x \in tasks : x.kind = "lend" /\ x.o = alloc[u])
          BY <3>1 DEF TaskOk, Live
        <4> QED
          BY <4>1, <3>6, <1>1
      <3>9. t.kind = "prefetch" => \A x \in tasks' : (x.kind = "prefetch" /\ x.h = t.h) => x = t
        <4>1. t.kind = "prefetch" => \A x \in tasks : (x.kind = "prefetch" /\ x.h = t.h) => x = t
          BY <3>1 DEF TaskOk
        <4> QED
          BY <4>1, <1>3, <1>4
      <3> QED
        BY <3>1, <3>4, <3>5, <3>7, <3>8, <3>9, <1>1, TaskKeep
    <2> QED
      BY <2>3, <2>4
  <1>10. LendingUnmapped'
    <2> SUFFICES ASSUME NEW q \in Owners, Lending(q)'
                 PROVE  /\ \A g \in Groups, p \in Rows : dbt'[g][q][p].u = NONE
                        /\ \A h \in handoff' : h.o # q
                        /\ \A x, y \in deferred' :
                             (x.kind = "lend" /\ y.kind = "lend" /\ x.o = q /\ y.o = q) => x = y
      BY DEF LendingUnmapped
    <2>1. Lending(q)
      BY <1>3
    <2> QED
      BY <1>1, <1>3, <2>1 DEF IndInv, LendingUnmapped
  <1>11. LostOk' /\ RequestsOk' /\ NoBad' /\ TrialVisibility'
    <2>1. TrialVisibility'
      <3>1. \A t \in tasks' : t.kind \in {"put", "replicate"} => t \in tasks
        BY <1>3
      <3> QED
        BY <1>1, <3>1 DEF IndInv, TrialVisibility, MadeBy
    <2> QED
      BY <1>1, <2>1, LostOkFrame, RequestsOkFrame, NoBadFrame DEF IndInv
  <1>12. TablesMapped' /\ HandoffOk'
    <2>1. \A x \in deferred' : x \in deferred
      BY <1>3
    <2>2. \A u : (\E t \in tasks' : t.kind = "replicate" /\ u \in t.units)
                   <=> (\E t \in tasks : t.kind = "replicate" /\ u \in t.units)
      BY <1>3 DEF Crossings
    <2>3. TablesMapped'
      BY <1>1, <2>1, <2>2 DEF IndInv, TablesMapped, Live, HoldsAt, ReaderRow, ReadersOf, Mem
    <2>4. HandoffOk'
      BY <1>1, <2>1, <2>2 DEF IndInv, HandoffOk, Live, HoldsAt, ReaderRow, ReadersOf, Mem
    <2> QED
      BY <2>3, <2>4
  <1>13. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. d.after[b.g] < b.seq /\ b.g \in Groups /\ b.seq \in Nat /\ d.after[b.g] \in Nat
      BY <1>0, <1>2, <2>2 DEF BatchOk, WorkTyped, Stamps
    <2>4. ASSUME NEW o \in Owners, NEW u, Covered(b, o, u) PROVE Covered(b, o, u)'
      <3>1. CASE /\ Live(o) /\ (ogroup[o] = b.g \/ b.g \in ReadersOf(o))
                 /\ \E p \in Rows :
                      /\ HoldsAt(o, p, u)
                      /\ \A r \in requests :
                           TakesRow(r, o, p) => b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
                 /\ \A e \in deferred : (e.kind = "lend" /\ e.o = o) => e.after[b.g] >= b.seq
                 /\ ~\E t \in tasks : t.kind = "lend" /\ t.o = o
        <4>1. o # d.o
          BY <3>1, <1>1, <2>3
        <4>2. ~\E t \in tasks' : t.kind = "lend" /\ t.o = o
          BY <3>1, <4>1, <1>3, <1>4
        <4> QED
          BY <1>1, <1>3, <3>1, <4>2 DEF Covered, Live, HoldsAt, ReadersOf
      <3>2. CASE \E e \in deferred : e.kind = "release" /\ u \in e.units /\ e.after[b.g] >= b.seq
        BY <1>1, <3>2 DEF Covered
      <3> QED
        BY <2>4, <3>1, <3>2 DEF Covered
    <2>5. \A x \in b.reads : x[1] \in Owners
      BY <2>2 DEF BatchOk
    <2>6. \A x \in deferred' : x \in deferred
      BY <1>3
    <2> QED
      BY <1>1, <2>2, <2>4, <2>5, <2>6 DEF BatchOk, Mem
  <1>14. CrossingsOk'
    <2>1. \A e \in deferred' : CrossWaitOk(e)'
      <3> SUFFICES ASSUME NEW e \in deferred' PROVE CrossWaitOk(e)'
        OBVIOUS
      <3>1. e \in deferred /\ CrossWaitOk(e)
        BY <1>3, CrossingsOkEach DEF IndInv
      <3> QED
        BY <1>1, <1>3, <3>1 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>2. \A t \in tasks' : CrossTaskOk(t)'
      <3> SUFFICES ASSUME NEW t \in tasks' PROVE CrossTaskOk(t)'
        OBVIOUS
      <3>1. CASE t.kind = "replicate"
        <4>1. t \in tasks /\ CrossTaskOk(t)
          BY <1>3, <3>1, CrossingsOkEach DEF IndInv
        <4>2. \A x \in tasks' : x.kind = "replicate" => x \in tasks
          BY <1>3
        <4> QED
          BY <1>1, <1>3, <4>1, <4>2 DEF CrossTaskOk, ReadersOf, Mem
      <3>2. CASE t.kind # "replicate"
        BY <3>2 DEF CrossTaskOk
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <1>1, <2>1, <2>2, CrossingsOkEach
  <1>15. CopiesPlaced' /\ CrossingsPlaced' /\ ImportsPlaced'
    <2>1. CopiesPlaced'
      BY <1>1, CopiesPlacedFrame DEF IndInv
    <2>2. CrossingsPlaced'
      BY <1>1, <1>3, CrossingsPlacedFrame DEF IndInv
    <2>3. ImportsPlaced'
      <3>1. \A t \in tasks' : t.kind \in {"import", "borrow"} => t \in tasks
        BY <1>3
      <3> QED
        BY <1>1, <3>1 DEF IndInv, ImportsPlaced
    <2> QED
      BY <2>1, <2>2, <2>3
  <1> QED
    BY <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15 DEF IndInv

-----------------------------------------------------------------------------
(* A unit a handler takes for a new block of owner o, free or evictable:   *)
(* nothing but the cache holds it, and no batch in flight reads it.        *)

LEMMA TakeUnit ==
  ASSUME IndInv, NEW o \in Owners, NEW u \in RegionUnits(o),
         u \in free \/ EvictableFor(u, OwnerEpoch[o])
  PROVE  /\ u \in Units /\ u \in member /\ alloc[u] \in {NONE, "shared"} /\ Unheld(u)
         /\ (u \in free => \A k \in Keys : cache[k] # u)
         /\ \A b \in inflight : \A x \in b.reads : x[2] # u
  <1>1. u \in Units /\ u # NONE /\ u \notin Slots /\ u \notin HostUnits
    BY DEF RegionUnits
  <1>2. /\ \A q \in Owners : oslot[q] # u
        /\ \A j \in Keys : hcache[j] # u
        /\ \A t \in tasks : u # t.h
    <2>1. oslot \in [Owners -> Slots \cup {NONE}] /\ hcache \in [Keys -> HostUnits \cup {NONE}]
      BY DEF IndInv, TypeInv, TypeOK
    <2>2. \A t \in tasks : t.h \in HostUnits \cup {NONE}
      BY TaskFacts
    <2> QED
      BY <1>1, <2>1, <2>2
  <1>3. /\ u \in member
        /\ alloc[u] \in {NONE, "shared"}
        /\ Unheld(u)
        /\ (u \in free => \A k \in Keys : cache[k] # u)
    <2>1. CASE u \in free
      <3>1. alloc[u] = NONE /\ ~Held(u)
        BY <2>1 DEF IndInv, FreeClean
      <3>2. u \in member
        BY <1>1, <2>1 DEF IndInv, Regions
      <3>3. Unheld(u) /\ \A k \in Keys : cache[k] # u
        BY <3>1, TasksHeld DEF Held, HeldIn, Unheld
      <3> QED
        BY <3>1, <3>2, <3>3
    <2>2. CASE u \notin free
      <3>0. EvictableFor(u, OwnerEpoch[o])
        BY <2>2
      <3>1. PICK k \in Keys : cache[k] = u
        BY <3>0 DEF EvictableFor
      <3>2. alloc[u] = "shared" /\ u \in member
        BY <1>1, <3>1, DeviceCached
      <3>3. /\ ~\E q \in Owners, p \in Positions : table[q][p] = Shared(u)
            /\ ~\E d \in deferred : u \in d.units
            /\ ~\E c \in copies : u = c.read \/ u = c.write
            /\ ~\E t \in tasks : t.held /\ u \in t.units
        <4>1. ~PinnedFor(u, OwnerEpoch[o])
          BY <3>0 DEF EvictableFor, On, AllRules
        <4>2. ~\E q \in Owners, p \in Positions : table[q][p] = Shared(u)
          BY <4>1 DEF PinnedFor
        <4> QED
          BY <3>0, <4>2 DEF EvictableFor, On, AllRules
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
      BY <2>1, <2>2
  <1>4. \A b \in inflight : \A x \in b.reads : x[2] # u
    <2> SUFFICES ASSUME NEW b \in inflight, NEW x \in b.reads, x[2] = u PROVE FALSE
      OBVIOUS
    <2>1. Covered(b, x[1], u) /\ x[1] \in Owners
      BY DEF IndInv, BatchesOk
    <2>2. CASE \E p \in Rows : HoldsAt(x[1], p, u)
      <3>1. PICK p \in Rows : HoldsAt(x[1], p, u)
        BY <2>2
      <3> QED
        BY <2>1, <3>1, <1>3 DEF HoldsAt, Unheld, Rows
    <2>3. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units
      BY <2>3, <1>3 DEF Unheld
    <2> QED
      BY <2>1, <2>2, <2>3 DEF Covered
  <1> QED
    BY <1>1, <1>3, <1>4

(* A batch in flight writes no block of a lent owner's table: its rows left *)
(* every executor's table, and the batches that wrote them have ended.      *)
LEMMA LentUnwritten ==
  ASSUME IndInv, NEW l \in Owners, \E t \in tasks : t.kind = "lend" /\ t.o = l,
         NEW p \in Positions, table[l][p] # Empty
  PROVE  \A b \in inflight : \A x \in b.writes : x[2] # table[l][p].u
  <1> SUFFICES ASSUME NEW b \in inflight, NEW x \in b.writes, x[2] = table[l][p].u PROVE FALSE
    OBVIOUS
  <1>1. /\ x \in b.reads /\ x[1] \in Owners /\ alloc[x[2]] = x[1]
        /\ Covered(b, x[1], x[2])
    BY DEF IndInv, BatchesOk
  <1>2. table[l][p] = Private(table[l][p].u) \/ table[l][p] = Shared(table[l][p].u)
    BY OwnerTypes DEF OwnerTyped
  <1>3. CASE table[l][p] = Shared(table[l][p].u)
    <2>1. alloc[table[l][p].u] = "shared"
      BY <1>3 DEF IndInv, Tables, Shared
    <2> QED
      BY <1>1, <2>1
  <1>4. CASE table[l][p] = Private(table[l][p].u)
    <2>1. alloc[table[l][p].u] = l
      BY <1>4 DEF IndInv, Tables, Private
    <2>2. x[1] = l
      BY <1>1, <2>1
    <2>3. ~\E d \in deferred : d.kind = "release" /\ x[2] \in d.units
      BY <1>4 DEF IndInv, DeferredOk, Private
    <2> QED
      BY <1>1, <2>2, <2>3 DEF Covered
  <1> QED
    BY <1>2, <1>3, <1>4

-----------------------------------------------------------------------------
(* Import                                                                  *)

(* The values Import(o, ps, l) computes for the units w it takes.          *)
Taken(ps, w) == {w[p] : p \in ps}
ReadUnits(ps, l) == IF l = NONE THEN {} ELSE {table[l][p].u : p \in ps}
ImportTask(o, ps, l, w) ==
    [Task("import", NONE, NONE)
       EXCEPT !.o = o, !.units = ReadUnits(ps, l) \cup Taken(ps, w), !.writes = Taken(ps, w),
              !.places = {<<p, w[p]>> : p \in ps}]

LEMMA ImportTaskFields ==
  ASSUME NEW o, NEW ps, NEW l, NEW w
  PROVE  ImportTask(o, ps, l, w) =
           [kind |-> "import", k |-> NONE, h |-> NONE,
            expired |-> FALSE, p |-> NONE, o |-> o, to |-> NONE, held |-> TRUE,
            units |-> ReadUnits(ps, l) \cup Taken(ps, w), writes |-> Taken(ps, w),
            places |-> {<<p, w[p]>> : p \in ps}]
  BY DEF ImportTask, Task

(* How the import takes its units.                                         *)
Takes(o, ps, w) ==
    /\ \A p, q \in ps : p # q => w[p] # w[q]
    /\ \/ Taken(ps, w) \subseteq free /\ cache' = cache
       \/ /\ free \cap RegionUnits(o) \subseteq Taken(ps, w)
          /\ \A u \in Taken(ps, w) \ free : EvictableFor(u, OwnerEpoch[o])
          /\ cache' = [k \in Keys |-> IF cache[k] \in Taken(ps, w) THEN NONE ELSE cache[k]]
    /\ free' = free \ Taken(ps, w)
    /\ alloc' = [u \in AllUnits |-> IF u \in Taken(ps, w) THEN o ELSE alloc[u]]
    /\ sharedKey' = [u \in AllUnits |-> IF u \in Taken(ps, w) THEN NONE ELSE sharedKey[u]]

LEMMA Import_Effect ==
  ASSUME NEW o \in Owners, NEW ps \in SUBSET Positions, NEW l \in Owners \cup {NONE},
         Import(o, ps, l)
  PROVE  /\ ostate[o] = "live"
         /\ ~Moving(o)
         /\ \A p \in ps : table[o][p] = Empty /\ ~CopyFor(o, p)
         /\ l # NONE => /\ \E t \in tasks : t.kind = "lend" /\ t.o = l
                        /\ \A p \in ps : table[l][p] # Empty
         /\ \E w \in [ps -> RegionUnits(o)] :
              /\ Takes(o, ps, w)
              /\ tasks' = tasks \cup {ImportTask(o, ps, l, w)}
         /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, copies, ncopy, trial,
                        member, valid, vis, dbt, launched, inflight, handoff, requests>>
  BY DEF Import, Takes, Taken, ReadUnits, ImportTask, plane, On, AllRules

(* The units the import takes, and those it reads.                         *)
LEMMA Import_Units ==
  ASSUME IndInv, NEW o \in Owners, ostate[o] = "live", NEW ps \in SUBSET Positions,
         NEW l \in Owners \cup {NONE},
         l # NONE => /\ \E t \in tasks : t.kind = "lend" /\ t.o = l
                     /\ \A p \in ps : table[l][p] # Empty,
         NEW w \in [ps -> RegionUnits(o)], Takes(o, ps, w)
  PROVE  /\ Taken(ps, w) \subseteq Units /\ Taken(ps, w) \subseteq member
         /\ \A u \in Taken(ps, w) : alloc[u] \in {NONE, "shared"} /\ Unheld(u)
         /\ cache' = [k \in Keys |-> IF cache[k] \in Taken(ps, w) THEN NONE ELSE cache[k]]
         /\ \A b \in inflight : \A x \in b.reads : x[2] \notin Taken(ps, w)
         /\ ReadUnits(ps, l) \subseteq Units
         /\ \A u \in ReadUnits(ps, l) : Held(u) /\ u \notin Taken(ps, w)
         /\ \A u \in ReadUnits(ps, l) :
              \/ alloc[u] = "shared"
              \/ alloc[u] = l /\ l \in Owners /\ \E t \in tasks : t.kind = "lend" /\ t.o = l
         /\ \A u \in ReadUnits(ps, l) : \A b \in inflight : \A x \in b.writes : x[2] # u
         /\ \A u \in ReadUnits(ps, l) : \A c \in copies : ~c.fill => c.write # u
  <1>1. \A u \in Taken(ps, w) :
          /\ u \in Units /\ u \in member /\ alloc[u] \in {NONE, "shared"} /\ Unheld(u)
          /\ (u \in free => \A k \in Keys : cache[k] # u)
          /\ \A b \in inflight : \A x \in b.reads : x[2] # u
    <2> SUFFICES ASSUME NEW u \in Taken(ps, w)
                 PROVE  /\ u \in Units /\ u \in member /\ alloc[u] \in {NONE, "shared"} /\ Unheld(u)
                        /\ (u \in free => \A k \in Keys : cache[k] # u)
                        /\ \A b \in inflight : \A x \in b.reads : x[2] # u
      OBVIOUS
    <2>1. u \in RegionUnits(o) /\ (u \in free \/ EvictableFor(u, OwnerEpoch[o]))
      BY DEF Takes, Taken
    <2> QED
      BY <2>1, TakeUnit
  <1>2. cache' = [k \in Keys |-> IF cache[k] \in Taken(ps, w) THEN NONE ELSE cache[k]]
    <2>1. CASE Taken(ps, w) \subseteq free /\ cache' = cache
      <3>1. \A k \in Keys : cache[k] \notin Taken(ps, w)
        BY <2>1, <1>1
      <3>2. cache \in [Keys -> Units \cup {NONE}]
        BY DEF IndInv, TypeInv, TypeOK
      <3> QED
        BY <2>1, <3>1, <3>2
    <2> QED
      BY <2>1 DEF Takes
  <1>3. ASSUME NEW u \in ReadUnits(ps, l)
        PROVE  /\ u \in Units /\ Held(u) /\ u \notin Taken(ps, w)
               /\ \/ alloc[u] = "shared"
                  \/ alloc[u] = l /\ l \in Owners /\ \E t \in tasks : t.kind = "lend" /\ t.o = l
               /\ \A b \in inflight : \A x \in b.writes : x[2] # u
               /\ \A c \in copies : ~c.fill => c.write # u
    <2>1. l # NONE /\ l \in Owners
      BY DEF ReadUnits
    <2>2. PICK p \in ps : u = table[l][p].u
      BY <2>1 DEF ReadUnits
    <2>0. p \in Positions
      BY <2>2
    <2>3. table[l][p] # Empty
      BY <2>0, <2>1, <2>2
    <2>4. (table[l][p] = Private(u) \/ table[l][p] = Shared(u)) /\ u \in Units
      <3>1. \/ table[l][p] = Empty
            \/ table[l][p] = Private(table[l][p].u) /\ table[l][p].u \in Units
            \/ table[l][p] = Shared(table[l][p].u) /\ table[l][p].u \in Units
        BY <2>0, <2>1, OwnerTypes DEF OwnerTyped
      <3> QED
        BY <3>1, <2>2, <2>3
    <2>5. Held(u)
      <3>1. \E q \in Owners, r \in Positions : table[q][r].u = u
        BY <2>0, <2>1, <2>2
      <3> QED
        BY <3>1 DEF Held, HeldIn
    <2>6. u \notin Taken(ps, w)
      <3>1. \A v \in Taken(ps, w) : \A q \in Owners, r \in Positions : table[q][r].u # v
        BY <1>1 DEF Unheld
      <3> QED
        BY <3>1, <2>0, <2>1, <2>2
    <2>7. alloc[u] = "shared" \/ alloc[u] = l
      <3>1. /\ table[l][p].k = "private" => alloc[table[l][p].u] = l
            /\ table[l][p].k = "shared" => alloc[table[l][p].u] = "shared"
        BY <2>0, <2>1 DEF IndInv, Tables
      <3> QED
        BY <3>1, <2>2, <2>4 DEF Private, Shared
    <2>8. \A b \in inflight : \A x \in b.writes : x[2] # u
      BY <2>0, <2>1, <2>2, <2>3, LentUnwritten
    <2>9. \A c \in copies : ~c.fill => c.write # u
      <3>1. \A c \in copies : ~c.fill => \A q \in Owners, r \in Positions : table[q][r].u # c.write
        BY DEF IndInv, CopiesOk
      <3> QED
        BY <3>1, <2>0, <2>1, <2>2
    <2> QED
      BY <2>1, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9
  <1> QED
    BY <1>1, <1>2, <1>3 DEF Taken

LEMMA Import_Inv ==
  ASSUME IndInv, NEW o \in Owners, NEW ps \in SUBSET Positions, NEW l \in Owners \cup {NONE},
         Import(o, ps, l), UNCHANGED rare
  PROVE  IndInv'
  <1>0. /\ UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. /\ ostate[o] = "live"
        /\ ~Moving(o)
        /\ \A p \in ps : table[o][p] = Empty /\ ~CopyFor(o, p)
        /\ l # NONE => /\ \E t \in tasks : t.kind = "lend" /\ t.o = l
                       /\ \A p \in ps : table[l][p] # Empty
        /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, copies, ncopy, trial,
                       member, valid, vis, dbt, launched, inflight, handoff, requests, remote,
                       lost, bad>>
    BY Import_Effect DEF rare
  <1>2. PICK w \in [ps -> RegionUnits(o)] :
          /\ Takes(o, ps, w)
          /\ tasks' = tasks \cup {ImportTask(o, ps, l, w)}
    BY Import_Effect
  <1> DEFINE W == Taken(ps, w)
             R == ReadUnits(ps, l)
             T == ImportTask(o, ps, l, w)
  <1>3. /\ W \subseteq Units /\ W \subseteq member
        /\ \A u \in W : alloc[u] \in {NONE, "shared"} /\ Unheld(u)
        /\ cache' = [k \in Keys |-> IF cache[k] \in W THEN NONE ELSE cache[k]]
        /\ \A b \in inflight : \A x \in b.reads : x[2] \notin W
        /\ R \subseteq Units
        /\ \A u \in R : Held(u) /\ u \notin W
        /\ \A u \in R :
             \/ alloc[u] = "shared"
             \/ alloc[u] = l /\ l \in Owners /\ \E t \in tasks : t.kind = "lend" /\ t.o = l
        /\ \A u \in R : \A b \in inflight : \A x \in b.writes : x[2] # u
        /\ \A u \in R : \A c \in copies : ~c.fill => c.write # u
    BY <1>1, <1>2, Import_Units
  <1>4. /\ free' = free \ W
        /\ alloc' = [u \in AllUnits |-> IF u \in W THEN o ELSE alloc[u]]
        /\ sharedKey' = [u \in AllUnits |-> IF u \in W THEN NONE ELSE sharedKey[u]]
        /\ \A p, q \in ps : p # q => w[p] # w[q]
    BY <1>2 DEF Takes
  <1>5. T = [kind |-> "import", k |-> NONE, h |-> NONE,
             expired |-> FALSE, p |-> NONE, o |-> o, to |-> NONE, held |-> TRUE,
             units |-> R \cup W, writes |-> W, places |-> {<<p, w[p]>> : p \in ps}]
    BY ImportTaskFields
  <1>6. /\ ~Lending(o) /\ ~Importing(o)
        /\ \A q : Lending(q)' = Lending(q)
        /\ Crossings' = Crossings
        /\ \A q : (\E t \in tasks' : t.kind = "lend" /\ t.o = q)
                    <=> (\E t \in tasks : t.kind = "lend" /\ t.o = q)
        /\ \A t \in tasks' : t \in tasks \/ t = T
        /\ \A t \in tasks : t \in tasks'
        /\ T \notin tasks
    <2>1. ~Lending(o) /\ ~Importing(o)
      BY <1>1 DEF Moving
    <2>2. T \notin tasks
      BY <2>1, <1>5 DEF Importing
    <2> QED
      BY <1>1, <1>2, <1>5, <2>1, <2>2 DEF Lending, Crossings
  (* The holders after the step: those before, the cache less the keys of  *)
  (* the units taken, and the task, which holds those units.               *)
  <1>7. /\ \A u \in AllUnits : u \notin W => (Held(u)' = Held(u))
        /\ \A u \in W : Held(u)'
    <2>1. \A u \in AllUnits : u \notin W =>
            (\E k \in Keys : cache'[k] = u \/ hcache'[k] = u) = (\E k \in Keys : cache[k] = u \/ hcache[k] = u)
      <3> SUFFICES ASSUME NEW u \in AllUnits, u \notin W
                   PROVE  (\E k \in Keys : cache'[k] = u \/ hcache'[k] = u)
                            = (\E k \in Keys : cache[k] = u \/ hcache[k] = u)
        OBVIOUS
      <3>1. u # NONE
        BY DEF AllUnits
      <3>2. \A k \in Keys : cache'[k] = IF cache[k] \in W THEN NONE ELSE cache[k]
        BY <1>3
      <3>3. \A k \in Keys : (cache'[k] = u) = (cache[k] = u)
        BY <3>1, <3>2
      <3>4. hcache' = hcache
        BY <1>1
      <3> QED
        BY <3>3, <3>4
    <2>2. \A u \in R \cup W : Held(u)'
      <3> SUFFICES ASSUME NEW u \in R \cup W PROVE Held(u)'
        OBVIOUS
      <3>1. T \in tasks' /\ T.held /\ u \in T.units
        BY <1>2, <1>5
      <3> QED
        BY <3>1 DEF Held, HeldIn
    <2>3. \A u \in AllUnits : u \notin W => (Held(u)' = Held(u))
      <3> SUFFICES ASSUME NEW u \in AllUnits, u \notin W PROVE Held(u)' = Held(u)
        OBVIOUS
      <3>1. CASE u \in R
        <4>1. Held(u) /\ Held(u)'
          BY <3>1, <1>3, <2>2
        <4> QED
          BY <4>1
      <3>2. CASE u \notin R
        <4>1. u # NONE
          BY DEF AllUnits
        <4>2. (\E t \in tasks' : t.held /\ (u \in t.units \/ u = t.h))
                = (\E t \in tasks : t.held /\ (u \in t.units \/ u = t.h))
          <5>1. tasks' = tasks \cup {T} /\ T.units = R \cup W /\ T.h = NONE
            BY <1>2, <1>5
          <5> QED
            BY <5>1, <4>1, <3>2
        <4>3. (\E k \in Keys : cache'[k] = u \/ hcache'[k] = u) = (\E k \in Keys : cache[k] = u \/ hcache[k] = u)
          BY <2>1
        <4>4. UNCHANGED <<table, oslot, deferred, copies>>
          BY <1>1
        <4> QED
          BY <4>2, <4>3, <4>4 DEF Held, HeldIn
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <2>2, <2>3
  <1>8. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>3, <1>4 DEF UnitTyped
    <2>2. WorkTyped'
      <3>1. /\ T.units \subseteq AllUnits /\ T.writes \subseteq AllUnits
            /\ T.places \subseteq Positions \X Units
        BY <1>3, <1>5 DEF AllUnits, Taken
      <3>2. T = [kind |-> T.kind, k |-> T.k, h |-> T.h,
                 expired |-> T.expired, p |-> T.p, o |-> T.o, to |-> T.to, held |-> T.held,
                 units |-> T.units, writes |-> T.writes, places |-> T.places]
        BY <1>5
      <3> QED
        BY <1>0, <1>1, <1>5, <1>6, <3>1, <3>2 DEF WorkTyped
    <2> QED
      BY <1>0, <1>1, <2>1, <2>2, OwnerTypedFrame, PlaneTypedFrame, RequestTypedFrame,
         TypeInvNext
  <1>9. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
    <2>1. FreeClean'
      BY <1>4, <1>7, UnitTypes DEF IndInv, FreeClean, UnitTyped
    <2>2. NoLeak'
      BY <1>1, <1>3, <1>4, <1>7, UnitTypes DEF IndInv, NoLeak, InRegion, UnitTyped
    <2>3. Outside'
      BY <1>1, <1>3, <1>4 DEF IndInv, Outside
    <2>4. Regions'
      BY <1>1, <1>4 DEF IndInv, Regions, OneRegion
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4
  <1>10. AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. AllocOwner' /\ SharedIff'
      BY <1>1, <1>4 DEF IndInv, AllocOwner, SharedIff
    <2>2. SharedCached'
      <3> SUFFICES ASSUME NEW u \in AllUnits, alloc'[u] = "shared"
                   PROVE  cache'[sharedKey'[u]] = u \/ hcache'[sharedKey'[u]] = u
        BY DEF SharedCached
      <3>1. u \notin W /\ alloc[u] = "shared" /\ sharedKey'[u] = sharedKey[u]
        BY <1>4
      <3>2. sharedKey[u] \in Keys
        <4>1. sharedKey[u] # NONE
          BY <3>1 DEF IndInv, SharedIff
        <4>2. sharedKey \in [AllUnits -> Keys \cup {NONE}]
          BY UnitTypes DEF UnitTyped
        <4> QED
          BY <4>1, <4>2
      <3>3. cache[sharedKey[u]] = u \/ hcache[sharedKey[u]] = u
        BY <3>1 DEF IndInv, SharedCached
      <3> QED
        BY <1>1, <1>3, <3>1, <3>2, <3>3
    <2>3. CacheShared'
      <3>1. \A k \in Keys : cache'[k] # NONE => cache'[k] = cache[k] /\ cache[k] \notin W
        BY <1>3
      <3>2. \A k \in Keys : hcache[k] # NONE => hcache[k] \in HostUnits /\ hcache[k] \notin W
        BY <1>3 DEF IndInv, CacheShared
      <3>3. \A k \in Keys : cache[k] # NONE => cache[k] \in AllUnits
        BY DEF IndInv, CacheShared, AllUnits
      <3> QED
        BY <1>1, <1>4, <3>1, <3>2, <3>3 DEF IndInv, CacheShared, AllUnits
    <2> QED
      BY <2>1, <2>2, <2>3
  <1>11. Tables' /\ OwnersOk'
    <2>1. \A q \in Owners, p \in Positions : table[q][p].u # NONE =>
            table[q][p].u \in AllUnits /\ table[q][p].u \notin W
      BY <1>0, <1>3 DEF OwnerTyped, Unheld, Empty, Private, Shared, AllUnits
    <2>2. \A q \in Owners : oslot[q] # NONE => oslot[q] \in AllUnits /\ oslot[q] \notin W
      BY <1>0, <1>3 DEF OwnerTyped, Unheld, AllUnits
    <2>3. \A q \in Owners, p \in Positions : table[q][p].k # "none" => table[q][p].u # NONE
      BY <1>0 DEF OwnerTyped, Empty, Private, Shared
    <2> QED
      BY <1>1, <1>4, <2>1, <2>2, <2>3 DEF IndInv, Tables, OwnersOk, Live, Mem
  <1>12. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred PROVE DeferOk(e)'
      BY <1>1, DeferredOkEach
    <2>2. DeferOk(e)
      BY DeferredOkEach DEF IndInv
    <2>3. \A u \in e.units : u \in AllUnits /\ u \notin W
      BY <1>0, <1>3 DEF WorkTyped, Unheld
    <2>4. e.kind = "lend" => e.o # o
      BY <1>6 DEF Lending
    <2>5. \A q : q # o => Importing(q)' = Importing(q)
      BY <1>6, <1>5 DEF Importing
    <2> QED
      BY <1>1, <1>4, <2>2, <2>3, <2>4, <2>5 DEF DeferOk
  <1>13. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY <1>1, CopiesOkEach
    <2>2. CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
      BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. c.read \notin W /\ c.write \notin W
      BY <1>3 DEF Unheld
    <2>4. ~c.fill => c.write \notin R
      BY <1>3
    <2>5. (c.fill /\ c.kind = "load") => cache'[c.k] = cache[c.k]
      <3> SUFFICES ASSUME c.fill, c.kind = "load" PROVE cache'[c.k] = cache[c.k]
        OBVIOUS
      <3>1. cache[c.k] = c.write /\ c.k \in Keys
        BY <2>2, <1>0 DEF CopyOk, WorkTyped
      <3> QED
        BY <1>3, <2>3, <3>1
    <2>6. \A u : (\A t \in tasks : u \notin t.units) /\ u \notin R /\ u \notin W
                   => \A t \in tasks' : u \notin t.units
      BY <1>6, <1>5
    <2> QED
      BY <1>1, <1>4, <1>6, <2>2, <2>3, <2>4, <2>5, <2>6 DEF CopyOk, Live, Mem
  <1>14. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. \A x \in tasks : TaskOk(x)
      BY TaskFacts
    <2>3. CASE t = T
      <3>1. /\ T.held /\ T.kind = "import" /\ T.h = NONE /\ T.p = NONE /\ T.to = NONE
            /\ T.o = o /\ o \in Owners /\ ostate[o] # "new"
            /\ T.units = R \cup W /\ T.units \subseteq Units
            /\ T.writes = W /\ T.writes = {x[2] : x \in T.places} /\ T.writes \subseteq T.units
        BY <1>1, <1>3, <1>5 DEF Taken
      <3>2. \A x, y \in T.places : (x[1] = y[1] \/ x[2] = y[2]) => x = y
        BY <1>4, <1>5
      <3>3. \A u \in W :
              /\ alloc'[u] = o
              /\ \A q \in Owners, p \in Positions : table'[q][p].u # u
              /\ \A k \in Keys : cache'[k] # u
              /\ \A d \in deferred' : u \notin d.units
              /\ \A c \in copies' : c.read # u /\ c.write # u
              /\ \A x \in tasks' : x # T => u \notin x.units
        <4>1. W \subseteq AllUnits
          BY <1>3 DEF AllUnits
        <4> QED
          BY <1>1, <1>3, <1>4, <1>6, <4>1 DEF Unheld
      <3>4. \A x \in tasks' : (x.kind = "import" /\ x.o = o) => x = T
        BY <1>6 DEF Importing
      <3>5. ~Lending(o)'
        BY <1>6
      <3>6. \A x \in T.places : table'[o][x[1]] = Empty /\ ~CopyFor(o, x[1])'
        BY <1>1, <1>5 DEF CopyFor
      <3>7. Reads(T) = R
        BY <1>3, <1>5 DEF Reads
      <3>8. \A u \in R :
              \/ alloc'[u] = "shared"
              \/ alloc'[u] \in Owners
                 /\ (ostate'[alloc'[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc'[u])
        <4> SUFFICES ASSUME NEW u \in R
                     PROVE  \/ alloc'[u] = "shared"
                            \/ alloc'[u] \in Owners
                               /\ (ostate'[alloc'[u]] = "live" =>
                                     \E x \in tasks' : x.kind = "lend" /\ x.o = alloc'[u])
          OBVIOUS
        <4>1. u \in AllUnits /\ u \notin W
          BY <1>3 DEF AllUnits
        <4>2. alloc'[u] = alloc[u]
          BY <4>1, <1>4
        <4>3. \/ alloc[u] = "shared"
              \/ alloc[u] \in Owners /\ \E x \in tasks : x.kind = "lend" /\ x.o = alloc[u]
          BY <1>3
        <4>4. \A x \in tasks : x \in tasks'
          BY <1>6
        <4> QED
          BY <4>2, <4>3, <4>4
      <3>9. ostate'[o] # "new" /\ T.o = o /\ T.kind = "import"
        BY <3>1, <1>1
      <3>10. \A u \in T.units \ T.writes :
               \/ alloc'[u] = "shared"
               \/ alloc'[u] \in Owners
                  /\ (ostate'[alloc'[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc'[u])
        BY <3>1, <3>7, <3>8 DEF Reads
      <3> QED
        BY <2>3, <3>1, <3>2, <3>3, <3>4, <3>5, <3>6, <3>9, <3>10, ImportOk
    <2>4. CASE t # T
      <3>1. t \in tasks /\ TaskOk(t)
        BY <1>6, <2>2, <2>4
      <3>2. \A u \in t.units : u \in AllUnits /\ u \notin W /\ alloc'[u] = alloc[u]
        <4>1. t.units \subseteq AllUnits
          BY <3>1, TaskFacts
        <4> QED
          BY <4>1, <1>3, <1>4, <3>1 DEF Unheld
      <3>3. t.kind = "import" =>
              /\ \A u \in t.writes : \A x \in tasks' : x # t => u \notin x.units
              /\ \A u \in t.writes : \A k \in Keys : cache'[k] # u
              /\ \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t
              /\ ~Lending(t.o)'
        <4> SUFFICES ASSUME t.kind = "import"
                     PROVE  /\ \A u \in t.writes : \A x \in tasks' : x # t => u \notin x.units
                            /\ \A u \in t.writes : \A k \in Keys : cache'[k] # u
                            /\ \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t
                            /\ ~Lending(t.o)'
          OBVIOUS
        <4>1. \A u \in t.writes : u \notin R /\ u \notin W /\ \A k \in Keys : cache[k] # u
          <5>1. \A u \in t.writes : \A q \in Owners, p \in Positions : table[q][p].u # u
            BY <3>1 DEF TaskOk
          <5>2. \A u \in t.writes : u \notin R
            BY <5>1 DEF ReadUnits
          <5> QED
            BY <5>2, <3>1, <3>2 DEF TaskOk
        <4>2. t.o # o
          BY <1>6, <3>1 DEF Importing
        <4> QED
          BY <1>3, <1>5, <1>6, <3>1, <4>1, <4>2 DEF TaskOk
      <3>4. t.kind = "lend" => t.o # o /\ ~Importing(t.o)'
        <4>1. t.kind = "lend" => t.o # o
          BY <1>6, <3>1 DEF Lending
        <4>2. t.kind = "lend" => ~Importing(t.o)
          BY <3>1 DEF TaskOk
        <4> QED
          BY <4>1, <4>2, <1>5, <1>6 DEF Importing
      <3>5. \A x \in tasks' : x.kind = "lend" => x \in tasks
        BY <1>5, <1>6
      <3>6. \A u \in t.units \cup {t.h} : u # NONE => alloc'[u] = alloc[u]
        <4>1. t.h \in HostUnits \cup {NONE}
          BY <3>1, TaskFacts
        <4>2. t.h # NONE => t.h \in AllUnits /\ t.h \notin W
          BY <4>1, <1>3 DEF AllUnits
        <4> QED
          BY <3>2, <4>2, <1>4
      <3>7. t.kind = "import" =>
              /\ \A u \in t.writes :
                   /\ \A q \in Owners, p \in Positions : table'[q][p].u # u
                   /\ \A d \in deferred' : u \notin d.units
                   /\ \A c \in copies' : c.read # u /\ c.write # u
              /\ ostate'[t.o] = "live" =>
                   \A x \in t.places : table'[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])'
        <4>1. t.kind = "import" =>
                /\ \A u \in t.writes :
                     /\ \A q \in Owners, p \in Positions : table[q][p].u # u
                     /\ \A d \in deferred : u \notin d.units
                     /\ \A c \in copies : c.read # u /\ c.write # u
                /\ ostate[t.o] = "live" =>
                     \A x \in t.places : table[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])
          BY <3>1 DEF TaskOk, Live
        <4>2. \A q, p : CopyFor(q, p)' = CopyFor(q, p)
          BY <1>1 DEF CopyFor
        <4> QED
          BY <4>1, <4>2, <1>1
      <3>8. t.kind = "lend" =>
              /\ ostate'[t.o] = "live"
              /\ \A x \in tasks' : (x.kind = "lend" /\ x.o = t.o) => x = t
              /\ ~\E d \in deferred' : d.kind = "lend" /\ d.o = t.o
        <4>1. t.kind = "lend" =>
                /\ ostate[t.o] = "live"
                /\ \A x \in tasks : (x.kind = "lend" /\ x.o = t.o) => x = t
                /\ ~\E d \in deferred : d.kind = "lend" /\ d.o = t.o
          BY <3>1 DEF TaskOk, Live
        <4> QED
          BY <4>1, <3>5, <1>1
      <3>9. /\ (t.kind = "borrow" /\ ostate'[t.o] = "live") =>
                 /\ \E x \in tasks' : x.kind = "lend" /\ x.o = t.o
                 /\ t.units = TableUnits(t.o)'
            /\ \A u \in Reads(t) :
                 alloc[u] \in Owners =>
                   (ostate'[alloc[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc[u])
            /\ t.kind = "prefetch" => \A x \in tasks' : (x.kind = "prefetch" /\ x.h = t.h) => x = t
        <4>1. /\ (t.kind = "borrow" /\ ostate[t.o] = "live") =>
                   /\ \E x \in tasks : x.kind = "lend" /\ x.o = t.o
                   /\ t.units = TableUnits(t.o)
              /\ \A u \in Reads(t) :
                   alloc[u] \in Owners =>
                     (ostate[alloc[u]] = "live" => \E x \in tasks : x.kind = "lend" /\ x.o = alloc[u])
              /\ t.kind = "prefetch" => \A x \in tasks : (x.kind = "prefetch" /\ x.h = t.h) => x = t
          BY <3>1 DEF TaskOk, Live
        <4>2. \A x \in tasks : x \in tasks'
          BY <1>6
        <4>3. \A x \in tasks' : x.kind # "import" => x \in tasks
          BY <1>5, <1>6
        <4> QED
          BY <4>1, <4>2, <4>3, <1>1 DEF TableUnits
      <3> QED
        BY <3>1, <3>3, <3>4, <3>6, <3>7, <3>8, <3>9, <1>1, TaskKeep
    <2> QED
      BY <2>3, <2>4
  <1>15. LendingUnmapped' /\ LostOk' /\ RequestsOk' /\ NoBad'
    <2>1. LendingUnmapped'
      BY <1>1, <1>6 DEF IndInv, LendingUnmapped
    <2> QED
      BY <1>1, <2>1, LostOkFrame, RequestsOkFrame, NoBadFrame DEF IndInv
  <1>16. TablesMapped' /\ HandoffOk'
    BY <1>1, <1>6, TablesMappedFrame, HandoffOkFrame DEF IndInv
  <1>17. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A x \in b.reads : x[2] \in AllUnits /\ x[2] \notin W /\ alloc'[x[2]] = alloc[x[2]]
      BY <1>0, <1>3, <1>4, <2>2 DEF RequestTyped
    <2>4. \A q, u : Covered(b, q, u)' = Covered(b, q, u)
      BY <1>1, <1>6 DEF Covered, Live, HoldsAt, ReadersOf
    <2> QED
      BY <1>1, <1>4, <2>2, <2>3, <2>4 DEF BatchOk, Mem
  <1>18. CrossingsOk'
    <2>1. \A e \in deferred : CrossWaitOk(e)'
      <3> SUFFICES ASSUME NEW e \in deferred PROVE CrossWaitOk(e)'
        OBVIOUS
      <3>1. CrossWaitOk(e)
        BY CrossingsOkEach DEF IndInv
      <3>2. \A u \in e.units : u \in AllUnits /\ u \notin W
        BY <1>0, <1>3 DEF WorkTyped, Unheld
      <3>3. \A k \in Keys : cache'[k] \in e.units => cache[k] = cache'[k]
        BY <1>3, <3>2 DEF AllUnits
      <3> QED
        BY <1>1, <1>4, <1>6, <3>1, <3>2, <3>3 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>2. \A t \in tasks' : CrossTaskOk(t)'
      <3> SUFFICES ASSUME NEW t \in tasks' PROVE CrossTaskOk(t)'
        OBVIOUS
      <3>1. CASE t.kind = "replicate"
        <4>1. t \in tasks /\ CrossTaskOk(t)
          BY <3>1, <1>5, <1>6, CrossingsOkEach DEF IndInv
        <4>2. \A u \in t.writes : u \in AllUnits /\ u \notin W
          <5>1. t.writes \subseteq t.units /\ t.units \subseteq Units
            BY <4>1, <3>1 DEF CrossTaskOk
          <5> QED
            BY <5>1, <4>1, <1>3 DEF Unheld, AllUnits
        <4>3. \A x \in tasks' : x.kind = "replicate" => x \in tasks
          BY <1>5, <1>6
        <4> QED
          BY <1>1, <1>4, <1>6, <4>1, <4>2, <4>3 DEF CrossTaskOk, ReadersOf, Mem
      <3>2. CASE t.kind # "replicate"
        BY <3>2 DEF CrossTaskOk
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <1>1, <2>1, <2>2, CrossingsOkEach
  <1>19. TrialVisibility'
    <2>1. \A k \in Keys : cache'[k] # NONE => cache'[k] = cache[k]
      BY <1>3
    <2>2. \A x \in tasks' : x.kind \in {"put", "replicate"} => x \in tasks
      BY <1>5, <1>6
    <2>3. \A k \in Keys : cache'[k] = NONE \/ cache'[k] = cache[k]
      BY <2>1
    <2>4. \A x \in tasks : x.kind = "replicate" => x.units \subseteq Units
      BY CrossingsOkEach DEF IndInv, CrossTaskOk
    <2> SUFFICES ASSUME NEW k \in Keys
                 PROVE  (/\ (cache[k] # NONE /\ vis[k].d # NONE) =>
                              /\ \A o \in Owners, p \in Positions :
                                   table[o][p] = Shared(cache[k]) => OwnerEpoch[o] = vis[k].d
                              /\ \A c \in copies : c.read = cache[k] => GroupEpoch[c.g] = vis[k].d
                         /\ (hcache[k] # NONE /\ vis[k].h # NONE) =>
                              /\ \A c \in copies : c.read = hcache[k] => GroupEpoch[c.g] = vis[k].h
                              /\ \A t \in tasks : t.kind = "put" => t.h # hcache[k]
                         /\ \A c \in copies :
                              (c.kind = "load" /\ c.fill /\ c.k = k) => vis[k].d = MadeBy(GroupEpoch[c.g])
                         /\ \A c \in copies :
                              (c.kind = "store" /\ c.fill /\ c.k = k) => vis[k].h = MadeBy(GroupEpoch[c.g])
                         /\ \A t \in tasks :
                              (t.kind = "replicate" /\ cache[k] \in t.units) => vis[k].d = MadeBy(OwnerEpoch[t.o]))'
      BY DEF TrialVisibility
    <2>5. /\ (cache[k] # NONE /\ vis[k].d # NONE) =>
               /\ \A o \in Owners, p \in Positions :
                    table[o][p] = Shared(cache[k]) => OwnerEpoch[o] = vis[k].d
               /\ \A c \in copies : c.read = cache[k] => GroupEpoch[c.g] = vis[k].d
          /\ (hcache[k] # NONE /\ vis[k].h # NONE) =>
               /\ \A c \in copies : c.read = hcache[k] => GroupEpoch[c.g] = vis[k].h
               /\ \A t \in tasks : t.kind = "put" => t.h # hcache[k]
          /\ \A c \in copies :
               (c.kind = "load" /\ c.fill /\ c.k = k) => vis[k].d = MadeBy(GroupEpoch[c.g])
          /\ \A c \in copies :
               (c.kind = "store" /\ c.fill /\ c.k = k) => vis[k].h = MadeBy(GroupEpoch[c.g])
          /\ \A t \in tasks :
               (t.kind = "replicate" /\ cache[k] \in t.units) => vis[k].d = MadeBy(OwnerEpoch[t.o])
      BY DEF IndInv, TrialVisibility
    <2>6. CASE cache'[k] = cache[k]
      BY <2>5, <2>6, <2>2, <1>1 DEF MadeBy
    <2>7. CASE cache'[k] = NONE
      BY <2>5, <2>7, <2>2, <2>4, <1>1 DEF MadeBy
    <2> QED
      BY <2>3, <2>6, <2>7
  <1>20. CopiesPlaced' /\ CrossingsPlaced' /\ ImportsPlaced'
    <2>1. CopiesPlaced' /\ CrossingsPlaced'
      BY <1>1, <1>6, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
    <2>2. ImportsPlaced'
      <3> SUFFICES ASSUME NEW t \in tasks' PROVE ImportPlaced(t)'
        BY PlacedEach
      <3>1. CASE t = T
        <4>1. T.units \ T.writes = R
          BY <1>3, <1>5
        <4> QED
          BY <3>1, <4>1, <1>1, <1>3 DEF ImportPlaced
      <3>2. CASE t # T
        <4>1. t \in tasks /\ ImportPlaced(t)
          BY <3>2, <1>6, PlacedEach DEF IndInv
        <4> QED
          BY <1>1, <4>1 DEF ImportPlaced
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <2>1, <2>2
  <1> QED
    BY <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18, <1>19, <1>20
       DEF IndInv

-----------------------------------------------------------------------------
(* Borrow                                                                  *)

BorrowTask(o) ==
    [Task("borrow", NONE, NONE)
       EXCEPT !.o = o, !.units = {table[o][p].u : p \in Positions} \ {NONE}]

LEMMA BorrowTaskFields ==
  ASSUME NEW o
  PROVE  BorrowTask(o) =
           [kind |-> "borrow", k |-> NONE, h |-> NONE,
            expired |-> FALSE, p |-> NONE, o |-> o, to |-> NONE, held |-> TRUE,
            units |-> TableUnits(o), writes |-> {}, places |-> {}]
  BY DEF BorrowTask, Task, TableUnits

LEMMA Borrow_Effect ==
  ASSUME NEW o \in Owners, Borrow(o)
  PROVE  /\ \E t \in tasks : t.kind = "lend" /\ t.o = o
         /\ tasks' = tasks \cup {BorrowTask(o)}
         /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                        trial, member, valid, vis, dbt, launched, inflight, handoff, requests,
                        alloc, sharedKey>>
  BY DEF Borrow, BorrowTask, plane

LEMMA Borrow_Inv ==
  ASSUME IndInv, NEW o \in Owners, Borrow(o), UNCHANGED rare
  PROVE  IndInv'
  <1>0. /\ UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. /\ \E t \in tasks : t.kind = "lend" /\ t.o = o
        /\ tasks' = tasks \cup {BorrowTask(o)}
        /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                       trial, member, valid, vis, dbt, launched, inflight, handoff, requests,
                       alloc, sharedKey, remote, lost, bad>>
    BY Borrow_Effect DEF rare
  <1> DEFINE B == BorrowTask(o)
  <1>2. B = [kind |-> "borrow", k |-> NONE, h |-> NONE,
             expired |-> FALSE, p |-> NONE, o |-> o, to |-> NONE, held |-> TRUE,
             units |-> TableUnits(o), writes |-> {}, places |-> {}]
    BY BorrowTaskFields
  <1>3. /\ Live(o)
        /\ TableUnits(o) \subseteq Units
        /\ \A u \in TableUnits(o) :
             /\ \E p \in Positions : table[o][p].u = u
             /\ alloc[u] = "shared" \/ alloc[u] = o
             /\ \A b \in inflight : \A x \in b.writes : x[2] # u
             /\ \A c \in copies : ~c.fill => c.write # u
             /\ \A t \in tasks : t.kind = "import" => u \notin t.writes
    <2>1. Live(o)
      BY <1>1, TaskFacts DEF TaskOk
    <2>2. TableUnits(o) \subseteq Units
      BY TaskFacts
    <2>3. ASSUME NEW u \in TableUnits(o)
          PROVE  /\ \E p \in Positions : table[o][p].u = u
                 /\ alloc[u] = "shared" \/ alloc[u] = o
                 /\ \A b \in inflight : \A x \in b.writes : x[2] # u
                 /\ \A c \in copies : ~c.fill => c.write # u
                 /\ \A t \in tasks : t.kind = "import" => u \notin t.writes
      <3>1. PICK p \in Positions : table[o][p].u = u /\ u # NONE
        BY DEF TableUnits
      <3>2. table[o][p] # Empty
        BY <3>1 DEF Empty
      <3>3. table[o][p] = Private(u) \/ table[o][p] = Shared(u)
        BY <3>1, <3>2, OwnerTypes DEF OwnerTyped
      <3>4. alloc[u] = "shared" \/ alloc[u] = o
        BY <3>3 DEF IndInv, Tables, Private, Shared
      <3>5. \A b \in inflight : \A x \in b.writes : x[2] # u
        BY <1>1, <3>1, <3>2, LentUnwritten
      <3>6. \A c \in copies : ~c.fill => c.write # u
        BY <3>1 DEF IndInv, CopiesOk
      <3>7. \A t \in tasks : t.kind = "import" => u \notin t.writes
        BY <3>1, TaskFacts DEF TaskOk
      <3> QED
        BY <3>1, <3>4, <3>5, <3>6, <3>7
    <2> QED
      BY <2>1, <2>2, <2>3
  <1>4. /\ \A u \in AllUnits : Held(u)' = Held(u)
        /\ \A q : Lending(q)' = Lending(q)
        /\ \A q : Importing(q)' = Importing(q)
        /\ Crossings' = Crossings
        /\ \A q : (\E t \in tasks' : t.kind = "lend" /\ t.o = q)
                    <=> (\E t \in tasks : t.kind = "lend" /\ t.o = q)
        /\ \A t \in tasks' : t.kind # "borrow" => t \in tasks
        /\ \A t \in tasks' : t \in tasks \/ t = B
    <2>1. \A u \in AllUnits : Held(u)' = Held(u)
      <3> SUFFICES ASSUME NEW u \in AllUnits PROVE Held(u)' = Held(u)
        OBVIOUS
      <3>1. u # NONE /\ B.h = NONE /\ B.held /\ B.units = TableUnits(o)
        BY <1>2 DEF AllUnits
      <3>2. tasks' = tasks \cup {B}
        BY <1>1
      <3>3. u \in TableUnits(o) => \E p \in Positions : table[o][p].u = u
        BY <1>3
      <3>4. (\E t \in tasks' : t.held /\ (u \in t.units \/ u = t.h))
              <=> (\/ \E t \in tasks : t.held /\ (u \in t.units \/ u = t.h)
                   \/ u \in TableUnits(o))
        BY <3>1, <3>2
      <3>5. UNCHANGED <<table, oslot, cache, hcache, deferred, copies>>
        BY <1>1
      <3> QED
        BY <3>3, <3>4, <3>5 DEF Held, HeldIn
    <2> QED
      BY <1>1, <1>2, <2>1 DEF Lending, Importing, Crossings
  <1>5. TypeInv'
    <2>1. WorkTyped'
      <3>1. /\ B.units \subseteq AllUnits /\ B.writes \subseteq AllUnits /\ B.places = {}
        BY <1>2, <1>3 DEF AllUnits
      <3> QED
        BY <1>0, <1>1, <1>2, <3>1 DEF WorkTyped
    <2> QED
      BY <1>0, <1>1, <2>1, UnitTypedFrame, OwnerTypedFrame, PlaneTypedFrame, RequestTypedFrame,
         TypeInvNext
  <1>6. /\ FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
        /\ AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared' /\ Tables' /\ OwnersOk'
        /\ DeferredOk' /\ LendingUnmapped' /\ LostOk' /\ RequestsOk' /\ NoBad'
        /\ TablesMapped' /\ HandoffOk'
    <2>1. FreeClean' /\ NoLeak'
      BY <1>1, <1>4, UnitTypes DEF IndInv, FreeClean, NoLeak, InRegion, UnitTyped
    <2>2. DeferredOk'
      BY <1>1, <1>4 DEF IndInv, DeferredOk
    <2>3. LendingUnmapped'
      BY <1>1, <1>4 DEF IndInv, LendingUnmapped
    <2> QED
      BY <1>1, <1>4, <2>1, <2>2, <2>3, OutsideFrame, RegionsFrame, AllocOwnerFrame,
         SharedIffFrame, SharedCachedFrame, CacheSharedFrame, TablesFrame, OwnersOkFrame,
         LostOkFrame, RequestsOkFrame, NoBadFrame, TablesMappedFrame, HandoffOkFrame DEF IndInv
  <1>7. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY <1>1, CopiesOkEach
    <2>2. CopyOk(c)
      BY CopiesOkEach DEF IndInv
    <2>3. (c.kind \in {"clone", "load"} /\ ~c.fill) => c.write \notin TableUnits(o)
      BY <2>2, <1>3 DEF CopyOk
    <2> QED
      BY <1>1, <1>2, <1>4, <2>2, <2>3 DEF CopyOk, Live, Mem
  <1>8. TasksOk'
    <2>1. SUFFICES ASSUME NEW t \in tasks' PROVE TaskOk(t)'
      BY TasksOkEach
    <2>2. CASE t = B
      <3>1. Reads(B) = TableUnits(o)
        BY <1>2 DEF Reads
      <3>2. \A u \in Reads(B) :
              \/ alloc[u] = "shared"
              \/ alloc[u] \in Owners /\ (Live(alloc[u]) => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc[u])
        BY <1>1, <1>3, <1>4, <3>1
      <3>3. \A x \in tasks : x \in tasks'
        BY <1>1
      <3>4. /\ B.held /\ B.kind = "borrow" /\ B.h = NONE /\ B.p = NONE /\ B.to = NONE
            /\ B.o = o /\ B.units = TableUnits(o) /\ B.writes = {} /\ B.places = {}
        BY <1>2
      <3>5. ostate'[o] = "live" /\ o \in Owners /\ TableUnits(o)' = TableUnits(o) /\ TableUnits(o) \subseteq Units
        BY <1>1, <1>3 DEF Live, TableUnits
      <3>6. \E x \in tasks' : x.kind = "lend" /\ x.o = o
        BY <1>1, <3>3
      <3>7. \A u \in Reads(B) :
              \/ alloc'[u] = "shared"
              \/ alloc'[u] \in Owners /\ (ostate'[alloc'[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc'[u])
        BY <3>2, <1>1 DEF Live
      <3> QED
        BY <2>2, <3>4, <3>5, <3>6, <3>7, BorrowOk
    <2>3. CASE t # B
      <3>1. t \in tasks /\ TaskOk(t)
        BY <1>4, <2>3, TaskFacts
      <3>2. t.kind = "import" => \A u \in t.writes : u \notin B.units
        BY <1>2, <1>3, <3>1
      <3>3. \A x \in tasks : x \in tasks'
        BY <1>1
      <3>4. t.kind = "import" =>
              /\ \A u \in t.writes :
                   /\ \A q \in Owners, p \in Positions : table'[q][p].u # u
                   /\ \A k \in Keys : cache'[k] # u
                   /\ \A d \in deferred' : u \notin d.units
                   /\ \A c \in copies' : c.read # u /\ c.write # u
                   /\ \A x \in tasks' : x # t => u \notin x.units
              /\ \A x \in tasks' : (x.kind = "import" /\ x.o = t.o) => x = t
              /\ ~Lending(t.o)'
              /\ ostate'[t.o] = "live" =>
                   \A x \in t.places : table'[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])'
        <4>1. t.kind = "import" =>
                /\ \A u \in t.writes :
                     /\ \A q \in Owners, p \in Positions : table[q][p].u # u
                     /\ \A k \in Keys : cache[k] # u
                     /\ \A d \in deferred : u \notin d.units
                     /\ \A c \in copies : c.read # u /\ c.write # u
                     /\ \A x \in tasks : x # t => u \notin x.units
                /\ \A x \in tasks : (x.kind = "import" /\ x.o = t.o) => x = t
                /\ ~Lending(t.o)
                /\ ostate[t.o] = "live" =>
                     \A x \in t.places : table[t.o][x[1]] = Empty /\ ~CopyFor(t.o, x[1])
          BY <3>1 DEF TaskOk, Live
        <4>2. \A q, p : CopyFor(q, p)' = CopyFor(q, p)
          BY <1>1 DEF CopyFor
        <4> QED
          BY <4>1, <4>2, <3>2, <1>1, <1>2, <1>4
      <3>5. t.kind = "lend" =>
              /\ ostate'[t.o] = "live"
              /\ \A x \in tasks' : (x.kind = "lend" /\ x.o = t.o) => x = t
              /\ ~\E d \in deferred' : d.kind = "lend" /\ d.o = t.o
              /\ ~Importing(t.o)'
        <4>1. t.kind = "lend" =>
                /\ ostate[t.o] = "live"
                /\ \A x \in tasks : (x.kind = "lend" /\ x.o = t.o) => x = t
                /\ ~\E d \in deferred : d.kind = "lend" /\ d.o = t.o
                /\ ~Importing(t.o)
          BY <3>1 DEF TaskOk, Live
        <4> QED
          BY <4>1, <1>1, <1>2, <1>4
      <3>6. /\ (t.kind = "borrow" /\ ostate'[t.o] = "live") =>
                 /\ \E x \in tasks' : x.kind = "lend" /\ x.o = t.o
                 /\ t.units = TableUnits(t.o)'
            /\ \A u \in Reads(t) :
                 alloc[u] \in Owners =>
                   (ostate'[alloc[u]] = "live" => \E x \in tasks' : x.kind = "lend" /\ x.o = alloc[u])
            /\ t.kind = "prefetch" => \A x \in tasks' : (x.kind = "prefetch" /\ x.h = t.h) => x = t
        <4>1. /\ (t.kind = "borrow" /\ ostate[t.o] = "live") =>
                   /\ \E x \in tasks : x.kind = "lend" /\ x.o = t.o
                   /\ t.units = TableUnits(t.o)
              /\ \A u \in Reads(t) :
                   alloc[u] \in Owners =>
                     (ostate[alloc[u]] = "live" => \E x \in tasks : x.kind = "lend" /\ x.o = alloc[u])
              /\ t.kind = "prefetch" => \A x \in tasks : (x.kind = "prefetch" /\ x.h = t.h) => x = t
          BY <3>1 DEF TaskOk, Live
        <4> QED
          BY <4>1, <3>3, <1>1, <1>2, <1>4 DEF TableUnits
      <3> QED
        BY <3>1, <3>4, <3>5, <3>6, <1>1, TaskKeep
    <2> QED
      BY <2>2, <2>3
  <1>9. BatchesOk'
    <2>1. \A b, q, u : Covered(b, q, u)' = Covered(b, q, u)
      BY <1>1, <1>4 DEF Covered, Live, HoldsAt, ReadersOf
    <2> QED
      BY <1>1, <2>1 DEF IndInv, BatchesOk, Mem
  <1>10. CrossingsOk' /\ TrialVisibility'
    <2>1. \A x \in tasks' : x.kind \in {"put", "replicate"} => x \in tasks
      BY <1>4
    <2>2. CrossingsOk'
      BY <1>1, <1>4, <2>1 DEF IndInv, CrossingsOk, Lending, ReadersOf, Mem, MadeBy
    <2>3. TrialVisibility'
      BY <1>1, <2>1 DEF IndInv, TrialVisibility, MadeBy
    <2> QED
      BY <2>2, <2>3
  <1>11. CopiesPlaced' /\ CrossingsPlaced' /\ ImportsPlaced'
    <2>1. CopiesPlaced' /\ CrossingsPlaced'
      BY <1>1, <1>4, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
    <2>2. ImportsPlaced'
      <3> SUFFICES ASSUME NEW t \in tasks' PROVE ImportPlaced(t)'
        BY PlacedEach
      <3>1. CASE t = B
        BY <3>1, <1>1, <1>2, <1>3 DEF ImportPlaced
      <3>2. CASE t # B
        <4>1. t \in tasks /\ ImportPlaced(t)
          BY <3>2, <1>4, PlacedEach DEF IndInv
        <4> QED
          BY <1>1, <4>1 DEF ImportPlaced
      <3> QED
        BY <3>1, <3>2
    <2> QED
      BY <2>1, <2>2
  <1> QED
    BY <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11 DEF IndInv

-----------------------------------------------------------------------------
(* A borrow, or an import that fails or whose owner is gone, ends: the     *)
(* task leaves, and what nothing else holds of what it held is reclaimed.  *)

LEMMA TaskDrop ==
  ASSUME IndInv, NEW t \in tasks, t.kind \in {"borrow", "import"},
         tasks' = tasks \ {t},
         DropWith(t.units, table, oslot, cache, hcache, deferred, copies, tasks'),
         UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, valid,
                     vis, trial, member, dbt, launched, inflight, handoff, requests, remote, lost,
                     bad>>
  PROVE  IndInv'
  <1>0. /\ UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1> DEFINE G == ReclaimedWith(t.units, table, oslot, cache, hcache, deferred, copies, tasks')
  <1>1. /\ \A x : Held(x)' = HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks')
        /\ G = {x \in t.units : ~Held(x)'}
        /\ free' = free \cup G /\ alloc' = Reset(alloc, G) /\ sharedKey' = Reset(sharedKey, G)
    BY ReclaimedHeld DEF Held, DropWith, On, AllRules
  <1>2. t.units \subseteq AllUnits /\ \A x \in t.units : Held(x)
    <2>1. t.held /\ t.units \subseteq AllUnits
      BY TaskFacts DEF TaskOk
    <2> QED
      BY <2>1 DEF Held, HeldIn
  <1>3. /\ \A x \in AllUnits : Held(x)' => Held(x)
        /\ \A x \in AllUnits : Held(x) => Held(x)' \/ x \in G
        /\ \A x \in G : Held(x) /\ ~Held(x)'
        /\ G \subseteq AllUnits
    <2>1. \A x : HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks')
                   => HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
      BY DEF HeldIn
    <2>2. \A x : HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
                   => HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks') \/ x \in t.units
                      \/ x = t.h
      BY DEF HeldIn
    <2>3. t.h = NONE
      BY TaskFacts DEF TaskOk
    <2>4. \A x \in AllUnits : x # NONE
      BY DEF AllUnits
    <2> QED
      BY <1>1, <1>2, <2>1, <2>2, <2>3, <2>4 DEF Held
  <1>4. /\ \A v \in AllUnits : alloc'[v] = IF v \in G THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v \in G THEN NONE ELSE sharedKey[v]
        /\ \A v \in AllUnits : Held(v)' => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
    BY <1>1 DEF Reset
  <1>5. /\ \A o \in Owners, p \in Positions : \A x \in AllUnits : table[o][p].u = x => Held(x)'
        /\ \A o \in Owners : \A x \in AllUnits : oslot[o] = x => Held(x)'
        /\ \A k \in Keys : \A x \in AllUnits : (cache[k] = x \/ hcache[k] = x) => Held(x)'
        /\ \A e \in deferred : \A x \in e.units : Held(x)'
        /\ \A c \in copies : Held(c.read)' /\ Held(c.write)'
        /\ \A x \in tasks' : x.held => \A y \in x.units \cup {x.h} : Held(y)'
    BY <1>1 DEF HeldIn
  <1>6. /\ \A x \in tasks' : x \in tasks /\ x # t
        /\ \A q : Lending(q)' = Lending(q)
        /\ \A q : Importing(q)' => Importing(q)
        /\ Crossings' = Crossings
        /\ \A q : (\E x \in tasks' : x.kind = "lend" /\ x.o = q)
                    <=> (\E x \in tasks : x.kind = "lend" /\ x.o = q)
    BY DEF Lending, Importing, Crossings
  <1>7. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>3 DEF UnitTyped, Reset
    <2>2. WorkTyped'
      BY <1>0, <1>6 DEF WorkTyped
    <2> QED
      BY <2>1, <2>2, <1>0, OwnerTypedFrame, PlaneTypedFrame, RequestTypedFrame, TypeInvNext
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
      BY <1>0, <1>4, <1>5 DEF OwnerTyped, Empty, Private, Shared, AllUnits
    <2>2. \A o \in Owners : oslot[o] # NONE => alloc'[oslot[o]] = alloc[oslot[o]]
      BY <1>0, <1>4, <1>5 DEF OwnerTyped, AllUnits
    <2> QED
      BY <2>1, <2>2 DEF IndInv, Tables, OwnersOk, Live, Mem
  <1>11. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred PROVE DeferOk(e)'
      BY DeferredOkEach
    <2>2. DeferOk(e)
      BY DeferredOkEach DEF IndInv
    <2>3. e.kind = "commit" => \A u \in e.units : alloc'[u] = alloc[u]
      BY <1>0, <1>4, <1>5 DEF WorkTyped
    <2> QED
      BY <1>6, <2>2, <2>3 DEF DeferOk
  <1>12. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY CopiesOkEach
    <2>2. CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
      BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
      BY <1>4, <1>5, <2>2
    <2> QED
      BY <1>6, <2>2, <2>3 DEF CopyOk, Live, Mem
  <1>13. TasksOk'
    <2>1. SUFFICES ASSUME NEW x \in tasks' PROVE TaskOk(x)'
      BY TasksOkEach
    <2>2. x \in tasks /\ x # t /\ TaskOk(x)
      BY <1>6, TaskFacts
    <2>3. \A u \in x.units \cup {x.h} : u # NONE => alloc'[u] = alloc[u]
      BY <1>0, <1>4, <1>5, <2>2, TaskFacts DEF TaskOk, AllUnits
    <2>4. Reads(x) \subseteq x.units /\ x.units \subseteq AllUnits
      BY <2>2, TaskFacts
    <2>5. \A u \in x.units : alloc'[u] = alloc[u]
      BY <2>3, <2>4 DEF AllUnits
    <2>6. x.kind = "prefetch" => x.h # NONE
      BY <2>2 DEF TaskOk
    <2> QED
      BY <1>6, <2>2, <2>3, <2>4, <2>5, <2>6 DEF TaskOk, Live, TableUnits, CopyFor
  <1>14. /\ LendingUnmapped' /\ LostOk' /\ TablesMapped' /\ HandoffOk' /\ RequestsOk'
         /\ TrialVisibility' /\ NoBad'
    <2>1. LendingUnmapped'
      BY <1>6 DEF IndInv, LendingUnmapped
    <2>2. TablesMapped' /\ HandoffOk'
      BY <1>6, TablesMappedFrame, HandoffOkFrame DEF IndInv
    <2>3. TrialVisibility'
      BY <1>6 DEF IndInv, TrialVisibility, MadeBy
    <2> QED
      BY <2>1, <2>2, <2>3, LostOkFrame, RequestsOkFrame, NoBadFrame DEF IndInv
  <1>15. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. \A q, u : Covered(b, q, u)' = Covered(b, q, u)
      BY <1>6 DEF Covered, Live, HoldsAt, ReadersOf
    <2>4. ASSUME NEW q \in Owners, NEW u \in AllUnits, Covered(b, q, u)
          PROVE  Held(u)'
      <3>1. CASE \E p \in Rows : HoldsAt(q, p, u)
        BY <3>1, <1>5 DEF HoldsAt, Rows
      <3>2. CASE \E e \in deferred : e.kind = "release" /\ u \in e.units
        BY <3>2, <1>5
      <3> QED
        BY <2>4, <3>1, <3>2 DEF Covered
    <2>5. \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits /\ Covered(b, x[1], x[2])
      BY <1>0, <2>2 DEF BatchOk, RequestTyped
    <2>6. \A x \in b.reads : x[2] \notin free' /\ alloc'[x[2]] = alloc[x[2]]
      BY <1>1, <1>3, <1>4, <2>2, <2>4, <2>5 DEF BatchOk
    <2> QED
      BY <2>2, <2>3, <2>6 DEF BatchOk, Mem
  <1>16. CrossingsOk'
    <2>1. \A e \in deferred : CrossWaitOk(e)'
      <3> SUFFICES ASSUME NEW e \in deferred PROVE CrossWaitOk(e)'
        OBVIOUS
      <3>1. CrossWaitOk(e)
        BY CrossingsOkEach DEF IndInv
      <3>2. \A u \in e.units : alloc'[u] = alloc[u]
        BY <1>0, <1>4, <1>5 DEF WorkTyped
      <3> QED
        BY <1>6, <3>1, <3>2 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>2. \A x \in tasks' : CrossTaskOk(x)'
      <3> SUFFICES ASSUME NEW x \in tasks' PROVE CrossTaskOk(x)'
        OBVIOUS
      <3>1. x \in tasks /\ CrossTaskOk(x) /\ x.held
        BY <1>6, CrossingsOkEach, TaskFacts DEF IndInv, TaskOk
      <3>2. x.kind = "replicate" => \A u \in x.writes : alloc'[u] = alloc[u]
        <4>1. x.kind = "replicate" => x.writes \subseteq x.units /\ x.units \subseteq Units
          BY <3>1 DEF CrossTaskOk
        <4> QED
          BY <1>4, <1>5, <3>1, <4>1 DEF AllUnits
      <3> QED
        BY <1>6, <3>1, <3>2 DEF CrossTaskOk, ReadersOf, Mem
    <2> QED
      BY <1>1, <2>1, <2>2, CrossingsOkEach
  <1>17. CopiesPlaced' /\ CrossingsPlaced' /\ ImportsPlaced'
    <2>1. CopiesPlaced' /\ CrossingsPlaced'
      BY <1>6, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
    <2>2. ImportsPlaced'
      BY <1>6 DEF IndInv, ImportsPlaced
    <2> QED
      BY <2>1, <2>2
  <1> QED
    BY <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17 DEF IndInv

LEMMA ReleaseBorrow_Inv ==
  ASSUME IndInv, NEW t \in tasks, ReleaseBorrow(t), UNCHANGED rare
  PROVE  IndInv'
  <1>1. /\ t.kind = "borrow" /\ tasks' = tasks \ {t}
        /\ DropWith(t.units, table, oslot, cache, hcache, deferred, copies, tasks')
        /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, valid,
                       vis, trial, member, dbt, launched, inflight, handoff, requests>>
    BY DEF ReleaseBorrow, plane
  <1> QED
    BY <1>1, TaskDrop DEF rare

-----------------------------------------------------------------------------
(* GiveBack                                                                *)

(* The values GiveBack(t) computes for the lent owner o.                   *)
BackTable(o) == [table EXCEPT ![o] = [p \in Positions |-> Empty]]
BackSlot(o) == [oslot EXCEPT ![o] = NONE]
BackUnits(o) == ({table[o][p].u : p \in Positions} \cup {oslot[o]}) \ {NONE}

LEMMA GiveBack_Effect ==
  ASSUME NEW t \in tasks, GiveBack(t)
  PROVE  /\ t.kind = "lend"
         /\ tasks' = tasks \ {t}
         /\ table' = BackTable(t.o)
         /\ oslot' = BackSlot(t.o)
         /\ ostate' = [ostate EXCEPT ![t.o] = "gone"]
         /\ DropWith(BackUnits(t.o), BackTable(t.o), BackSlot(t.o), cache, hcache, deferred,
                     copies, tasks')
         /\ UNCHANGED <<ogroup, cache, hcache, deferred, copies, ncopy, valid, vis, trial, member,
                        dbt, launched, inflight, handoff, requests>>
  BY DEF GiveBack, BackTable, BackSlot, BackUnits, plane

LEMMA GiveBack_Inv ==
  ASSUME IndInv, NEW t \in tasks, GiveBack(t), UNCHANGED rare
  PROVE  IndInv'
  <1>0. /\ UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. /\ t.kind = "lend"
        /\ tasks' = tasks \ {t}
        /\ table' = BackTable(t.o)
        /\ oslot' = BackSlot(t.o)
        /\ ostate' = [ostate EXCEPT ![t.o] = "gone"]
        /\ DropWith(BackUnits(t.o), BackTable(t.o), BackSlot(t.o), cache, hcache, deferred,
                    copies, tasks')
        /\ UNCHANGED <<ogroup, cache, hcache, deferred, copies, ncopy, valid, vis, trial, member,
                       dbt, launched, inflight, handoff, requests, remote, lost, bad>>
    BY GiveBack_Effect DEF rare
  <1> DEFINE o == t.o
             U == BackUnits(o)
             G == ReclaimedWith(U, BackTable(o), BackSlot(o), cache, hcache, deferred, copies,
                                tasks')
  (* What IndInv says of the lent owner: it is attached, named in no       *)
  (* executor's table, written for by no copy, and has no import and no    *)
  (* crossing, and its lend is its only one.                               *)
  <1>2. /\ o \in Owners /\ Live(o) /\ t.units = {} /\ t.h = NONE
        /\ Lending(o)
        /\ \A g \in Groups, p \in Rows : dbt[g][o][p].u = NONE
        /\ \A h \in handoff : h.o # o
        /\ \A c \in copies : c.o # o
        /\ ~Importing(o)
        /\ \A x \in tasks : (x.kind = "lend" /\ x.o = o) => x = t
        /\ \A d \in deferred : d.kind = "replicate" => d.o # o
        /\ \A x \in tasks : x.kind = "replicate" => x.o # o
    <2>1. /\ o \in Owners /\ Live(o) /\ t.units = {} /\ t.h = NONE
          /\ ~Importing(o)
          /\ \A x \in tasks : (x.kind = "lend" /\ x.o = o) => x = t
      BY <1>1, TaskFacts DEF TaskOk
    <2>2. Lending(o)
      BY <1>1 DEF Lending
    <2>3. /\ \A g \in Groups, p \in Rows : dbt[g][o][p].u = NONE
          /\ \A h \in handoff : h.o # o
      BY <2>1, <2>2 DEF IndInv, LendingUnmapped
    <2>4. \A c \in copies : c.o # o
      BY <2>1, <2>2 DEF IndInv, CopiesOk
    <2>5. /\ \A d \in deferred : d.kind = "replicate" => d.o # o
          /\ \A x \in tasks : x.kind = "replicate" => x.o # o
      BY <2>2, CrossingsOkEach DEF IndInv, CrossWaitOk, CrossTaskOk
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5
  <1>3. /\ \A x : Held(x)' = HeldIn(x, BackTable(o), BackSlot(o), cache, hcache, deferred,
                                     copies, tasks')
        /\ G = {x \in U : ~Held(x)'}
        /\ free' = free \cup G /\ alloc' = Reset(alloc, G) /\ sharedKey' = Reset(sharedKey, G)
    BY <1>1, ReclaimedHeld DEF Held, DropWith, On, AllRules
  (* The owner's table and slot after the step, entry by entry.             *)
  <1>a. /\ \A q \in Owners, p \in Positions :
             BackTable(o)[q][p] = IF q = o THEN Empty ELSE table[q][p]
        /\ \A q \in Owners : BackSlot(o)[q] = IF q = o THEN NONE ELSE oslot[q]
        /\ DOMAIN BackTable(o) = Owners /\ \A q \in Owners : DOMAIN BackTable(o)[q] = Positions
    <2>1. DOMAIN table = Owners /\ \A q \in Owners : DOMAIN table[q] = Positions
      BY <1>0 DEF OwnerTyped
    <2>2. oslot \in [Owners -> Slots \cup {NONE}]
      BY <1>0 DEF OwnerTyped
    <2> QED
      BY <1>2, <2>1, <2>2 DEF BackTable, BackSlot
  <1>4. /\ U \subseteq AllUnits
        /\ \A x \in U : Held(x)
        /\ \A x \in AllUnits : x \in U <=> (\E p \in Positions : table[o][p].u = x) \/ oslot[o] = x
    <2>1. \A p \in Positions : table[o][p].u \in Units \cup {NONE}
      BY <1>0, <1>2 DEF OwnerTyped, Empty, Private, Shared
    <2>2. oslot[o] \in Slots \cup {NONE}
      BY <1>0, <1>2 DEF OwnerTyped
    <2>3. U \subseteq AllUnits
      BY <2>1, <2>2 DEF BackUnits, AllUnits
    <2>4. \A x \in U : Held(x)
      BY <1>2 DEF BackUnits, Held, HeldIn
    <2>5. \A x \in AllUnits : x \in U <=> (\E p \in Positions : table[o][p].u = x) \/ oslot[o] = x
      BY DEF BackUnits, AllUnits
    <2> QED
      BY <2>3, <2>4, <2>5
  <1>5. /\ \A x \in AllUnits : Held(x)' => Held(x)
        /\ \A x \in AllUnits : Held(x) => Held(x)' \/ x \in G
        /\ \A x \in G : Held(x) /\ ~Held(x)'
        /\ G \subseteq AllUnits
    <2>1. \A x \in AllUnits :
            HeldIn(x, BackTable(o), BackSlot(o), cache, hcache, deferred, copies, tasks')
              => HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
      <3> SUFFICES ASSUME NEW x \in AllUnits,
                          HeldIn(x, BackTable(o), BackSlot(o), cache, hcache, deferred, copies, tasks')
                   PROVE  HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
        OBVIOUS
      <3>1. x # NONE /\ Empty.u = NONE
        BY DEF AllUnits, Empty
      <3>2. (\E q \in Owners, p \in Positions : BackTable(o)[q][p].u = x)
              => \E q \in Owners, p \in Positions : table[q][p].u = x
        BY <1>a, <3>1
      <3>3. (\E q \in Owners : BackSlot(o)[q] = x) => \E q \in Owners : oslot[q] = x
        BY <1>a, <3>1
      <3>4. \A y \in tasks' : y \in tasks
        BY <1>1
      <3> QED
        BY <3>2, <3>3, <3>4 DEF HeldIn
    <2>2. \A x \in AllUnits :
            HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks)
              => HeldIn(x, BackTable(o), BackSlot(o), cache, hcache, deferred, copies, tasks') \/ x \in U
      <3> SUFFICES ASSUME NEW x \in AllUnits, HeldIn(x, table, oslot, cache, hcache, deferred, copies, tasks),
                          ~(x \in U)
                   PROVE  HeldIn(x, BackTable(o), BackSlot(o), cache, hcache, deferred, copies, tasks')
        OBVIOUS
      <3>1. /\ \A p \in Positions : table[o][p].u # x
            /\ oslot[o] # x
        BY <1>4
      <3>2. (\E q \in Owners, p \in Positions : table[q][p].u = x)
              => \E q \in Owners, p \in Positions : BackTable(o)[q][p].u = x
        BY <1>a, <3>1
      <3>3. (\E q \in Owners : oslot[q] = x) => \E q \in Owners : BackSlot(o)[q] = x
        BY <1>a, <3>1
      <3>4. (\E y \in tasks : y.held /\ (x \in y.units \/ x = y.h))
              => \E y \in tasks' : y.held /\ (x \in y.units \/ x = y.h)
        BY <1>1, <1>2 DEF AllUnits
      <3> QED
        BY <3>2, <3>3, <3>4 DEF HeldIn
    <2>3. \A x \in AllUnits : Held(x)' => Held(x)
      BY <1>3, <2>1 DEF Held
    <2>4. \A x \in AllUnits : Held(x) => Held(x)' \/ x \in G
      BY <1>3, <2>2 DEF Held
    <2>5. \A x \in G : Held(x) /\ ~Held(x)' /\ x \in AllUnits
      BY <1>3, <1>4
    <2> QED
      BY <2>3, <2>4, <2>5
  <1>6. /\ \A v \in AllUnits : alloc'[v] = IF v \in G THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v \in G THEN NONE ELSE sharedKey[v]
        /\ \A v \in AllUnits : Held(v)' => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
    BY <1>3 DEF Reset
  (* What stays held after the step: every holder but the owner.           *)
  <1>7. /\ \A q \in Owners, p \in Positions : q # o =>
             \A x \in AllUnits : table[q][p].u = x => Held(x)'
        /\ \A q \in Owners : q # o => \A x \in AllUnits : oslot[q] = x => Held(x)'
        /\ \A k \in Keys : \A x \in AllUnits : (cache[k] = x \/ hcache[k] = x) => Held(x)'
        /\ \A e \in deferred : \A x \in e.units : Held(x)'
        /\ \A c \in copies : Held(c.read)' /\ Held(c.write)'
        /\ \A x \in tasks' : x.held => \A y \in x.units \cup {x.h} : Held(y)'
    <2>1. \A q \in Owners, p \in Positions : q # o =>
             \A x \in AllUnits : table[q][p].u = x => Held(x)'
      BY <1>a, <1>3 DEF HeldIn
    <2>2. \A q \in Owners : q # o => \A x \in AllUnits : oslot[q] = x => Held(x)'
      BY <1>a, <1>3 DEF HeldIn
    <2> QED
      BY <2>1, <2>2, <1>1, <1>3 DEF HeldIn
  <1>8. /\ \A x \in tasks' : x \in tasks /\ x # t
        /\ \A q : Lending(q)' => Lending(q)
        /\ \A q : Importing(q)' = Importing(q)
        /\ Crossings' = Crossings
        /\ \A q : q # o => ((\E x \in tasks' : x.kind = "lend" /\ x.o = q)
                              <=> (\E x \in tasks : x.kind = "lend" /\ x.o = q))
        /\ \A q \in Owners : q # o => table'[q] = table[q] /\ oslot'[q] = oslot[q] /\ ostate'[q] = ostate[q]
        /\ table'[o] = [p \in Positions |-> Empty] /\ oslot'[o] = NONE /\ ostate'[o] = "gone"
    <2>1. \A x \in tasks' : x \in tasks /\ x # t
      BY <1>1
    <2>2. \A q : Lending(q)' => Lending(q)
      BY <1>1 DEF Lending
    <2>3. \A q : Importing(q)' = Importing(q)
      BY <1>1 DEF Importing
    <2>4. Crossings' = Crossings
      BY <1>1 DEF Crossings
    <2>5. \A q : q # o => ((\E x \in tasks' : x.kind = "lend" /\ x.o = q)
                              <=> (\E x \in tasks : x.kind = "lend" /\ x.o = q))
      BY <1>1
    <2>6. \A q \in Owners : q # o => table'[q] = table[q] /\ oslot'[q] = oslot[q] /\ ostate'[q] = ostate[q]
      <3>1. DOMAIN table = Owners /\ oslot \in [Owners -> Slots \cup {NONE}]
            /\ ostate \in [Owners -> {"new", "live", "gone"}]
        BY <1>0 DEF OwnerTyped
      <3> QED
        BY <3>1, <1>1, <1>2 DEF BackTable, BackSlot
    <2>7. table'[o] = [p \in Positions |-> Empty] /\ oslot'[o] = NONE /\ ostate'[o] = "gone"
      <3>1. DOMAIN table = Owners /\ oslot \in [Owners -> Slots \cup {NONE}]
            /\ ostate \in [Owners -> {"new", "live", "gone"}]
        BY <1>0 DEF OwnerTyped
      <3> QED
        BY <3>1, <1>1, <1>2 DEF BackTable, BackSlot
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7
  <1>9. TypeInv'
    <2>1. UnitTyped'
      BY <1>0, <1>1, <1>3, <1>5 DEF UnitTyped, Reset
    <2>2. OwnerTyped'
      <3>1. ostate' \in [Owners -> {"new", "live", "gone"}]
        BY <1>0, <1>1, <1>2 DEF OwnerTyped
      <3>2. oslot' \in [Owners -> Slots \cup {NONE}]
        BY <1>0, <1>1, <1>2 DEF OwnerTyped, BackSlot
      <3>3. /\ DOMAIN table' = Owners /\ \A q \in Owners : DOMAIN table'[q] = Positions
            /\ \A q \in Owners, p \in Positions :
                 \/ table'[q][p] = Empty
                 \/ table'[q][p] = Private(table'[q][p].u) /\ table'[q][p].u \in Units
                 \/ table'[q][p] = Shared(table'[q][p].u) /\ table'[q][p].u \in Units
        BY <1>0, <1>1, <1>a DEF OwnerTyped
      <3> QED
        BY <3>1, <3>2, <3>3, <1>0, <1>1 DEF OwnerTyped
    <2>3. WorkTyped'
      BY <1>0, <1>1, <1>8 DEF WorkTyped
    <2> QED
      BY <2>1, <2>2, <2>3, <1>0, <1>1, PlaneTypedFrame, RequestTypedFrame, TypeInvNext
  <1>10. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
    BY <1>0, <1>1, <1>3, <1>5, <1>6, FreeReclaim DEF IndInv
  <1>11. AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. AllocOwner'
      BY <1>6, <1>8, <1>2 DEF IndInv, AllocOwner, NONE
    <2>2. SharedIff'
      BY <1>6 DEF IndInv, SharedIff, NONE
    <2>3. \A v \in AllUnits : alloc[v] = "shared" /\ v \notin G =>
            cache[sharedKey[v]] = v \/ hcache[sharedKey[v]] = v
      BY DEF IndInv, SharedCached
    <2>4. SharedCached'
      BY <1>1, <1>6, <2>3 DEF SharedCached, NONE
    <2>5. CacheShared'
      BY <1>0, <1>1, <1>6, <1>7 DEF IndInv, CacheShared, UnitTyped, AllUnits
    <2> QED
      BY <2>1, <2>2, <2>4, <2>5
  <1>12. Tables' /\ OwnersOk'
    <2>1. \A q \in Owners, p \in Positions :
            (q # o /\ table[q][p].u # NONE) =>
              /\ table[q][p].u \in AllUnits
              /\ alloc'[table[q][p].u] = alloc[table[q][p].u]
      BY <1>0, <1>6, <1>7 DEF OwnerTyped, Empty, Private, Shared, AllUnits
    <2>2. \A q \in Owners : (q # o /\ oslot[q] # NONE) => alloc'[oslot[q]] = alloc[oslot[q]]
      BY <1>0, <1>6, <1>7 DEF OwnerTyped, AllUnits
    <2>3. Tables'
      BY <1>1, <1>8, <2>1 DEF IndInv, Tables, Live, Mem, Empty
    <2>4. OwnersOk'
      BY <1>1, <1>8, <2>2 DEF IndInv, OwnersOk, Live, Empty
    <2> QED
      BY <2>3, <2>4
  <1>13. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred PROVE DeferOk(e)'
      BY <1>1, DeferredOkEach
    <2>2. DeferOk(e)
      BY DeferredOkEach DEF IndInv
    <2>3. e.kind = "commit" => \A u \in e.units : alloc'[u] = alloc[u]
      BY <1>0, <1>6, <1>7 DEF WorkTyped
    <2>4. \A q \in Owners, p \in Positions : table'[q][p].u # NONE => table'[q][p] = table[q][p]
      BY <1>8 DEF Empty
    <2>5. \A q \in Owners : oslot'[q] # NONE => oslot'[q] = oslot[q]
      BY <1>8
    <2>6. ostate'[o] # "new"
      BY <1>8
    <2> QED
      BY <1>1, <1>8, <2>2, <2>3, <2>4, <2>5, <2>6 DEF DeferOk, Empty
  <1>14. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY <1>1, CopiesOkEach
    <2>2. CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits /\ c.o # o
      BY <1>0, <1>2, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
      BY <1>6, <1>7, <2>2
    <2>4. c.o \in Owners => ostate'[c.o] = ostate[c.o] /\ table'[c.o] = table[c.o]
      BY <1>8, <2>2
    <2>5. ~c.fill => \A q \in Owners, p \in Positions : table'[q][p].u # c.write
      <3>1. ~c.fill => \A q \in Owners, p \in Positions : table[q][p].u # c.write
        BY <2>2 DEF CopyOk
      <3>2. \A q \in Owners, p \in Positions : table'[q][p].u = table[q][p].u \/ table'[q][p].u = NONE
        BY <1>8 DEF Empty
      <3>3. c.write # NONE
        BY <2>2 DEF AllUnits
      <3> QED
        BY <3>1, <3>2, <3>3
    <2> QED
      BY <1>1, <1>8, <2>2, <2>3, <2>4, <2>5 DEF CopyOk, Live, Mem, Empty
  <1>15. TasksOk'
    <2>1. SUFFICES ASSUME NEW x \in tasks' PROVE TaskOk(x)'
      BY TasksOkEach
    <2>2. x \in tasks /\ x # t /\ TaskOk(x)
      BY <1>8, TaskFacts
    <2>3. \A u \in x.units \cup {x.h} : u # NONE => alloc'[u] = alloc[u]
      <3>1. x.held /\ x.units \subseteq AllUnits /\ x.h \in HostUnits \cup {NONE}
        BY <2>2, TaskFacts DEF TaskOk
      <3>2. \A u \in x.units \cup {x.h} : u # NONE => Held(u)' /\ u \in AllUnits
        BY <3>1, <1>7, <2>2 DEF AllUnits
      <3> QED
        BY <3>2, <1>6
    <2>4. x.kind = "prefetch" => alloc'[x.h] = alloc[x.h]
      BY <2>3, <2>2 DEF TaskOk, AllUnits
    <2>5. x.kind \in {"import", "lend"} => x.o # o
      BY <1>2, <2>2 DEF Importing
    <2>7. (x.kind = "borrow" /\ Live(x.o)') =>
            /\ \E y \in tasks' : y.kind = "lend" /\ y.o = x.o
            /\ x.units = TableUnits(x.o)'
      <3> SUFFICES ASSUME x.kind = "borrow", Live(x.o)'
                   PROVE  /\ \E y \in tasks' : y.kind = "lend" /\ y.o = x.o
                          /\ x.units = TableUnits(x.o)'
        OBVIOUS
      <3>1. x.o \in Owners /\ x.o # o
        BY <2>2, <1>8 DEF TaskOk, Live
      <3> QED
        BY <3>1, <1>8, <2>2 DEF TaskOk, Live, TableUnits
    <2>8. x.kind = "lend" =>
            /\ ostate'[x.o] = "live"
            /\ \A y \in tasks' : (y.kind = "lend" /\ y.o = x.o) => y = x
            /\ ~\E d \in deferred' : d.kind = "lend" /\ d.o = x.o
            /\ ~Importing(x.o)'
      <3>1. x.kind = "lend" =>
              /\ ostate[x.o] = "live" /\ x.o \in Owners
              /\ \A y \in tasks : (y.kind = "lend" /\ y.o = x.o) => y = x
              /\ ~\E d \in deferred : d.kind = "lend" /\ d.o = x.o
              /\ ~Importing(x.o)
        BY <2>2 DEF TaskOk, Live
      <3> QED
        BY <3>1, <2>5, <1>8, <1>1
    <2>9. x.kind = "import" =>
            /\ \A u \in x.writes :
                 /\ \A q \in Owners, p \in Positions : table'[q][p].u # u
                 /\ \A k \in Keys : cache'[k] # u
                 /\ \A d \in deferred' : u \notin d.units
                 /\ \A c \in copies' : c.read # u /\ c.write # u
                 /\ \A y \in tasks' : y # x => u \notin y.units
            /\ \A y \in tasks' : (y.kind = "import" /\ y.o = x.o) => y = x
            /\ ~Lending(x.o)'
            /\ ostate'[x.o] = "live" =>
                 \A y \in x.places : table'[x.o][y[1]] = Empty /\ ~CopyFor(x.o, y[1])'
      <3>1. x.kind = "import" =>
              /\ x.o \in Owners
              /\ \A u \in x.writes :
                   /\ \A q \in Owners, p \in Positions : table[q][p].u # u
                   /\ \A k \in Keys : cache[k] # u
                   /\ \A d \in deferred : u \notin d.units
                   /\ \A c \in copies : c.read # u /\ c.write # u
                   /\ \A y \in tasks : y # x => u \notin y.units
              /\ \A y \in tasks : (y.kind = "import" /\ y.o = x.o) => y = x
              /\ ~Lending(x.o)
              /\ ostate[x.o] = "live" =>
                   \A y \in x.places : table[x.o][y[1]] = Empty /\ ~CopyFor(x.o, y[1])
        BY <2>2 DEF TaskOk, Live
      <3>2. \A q \in Owners, p \in Positions : table'[q][p].u = table[q][p].u \/ table'[q][p].u = NONE
        BY <1>8 DEF Empty
      <3>3. x.kind = "import" => x.writes \subseteq Units
        BY <2>2 DEF TaskOk
      <3>4. \A q, p : CopyFor(q, p)' = CopyFor(q, p)
        BY <1>1 DEF CopyFor
      <3> QED
        BY <3>1, <3>2, <3>3, <3>4, <2>5, <1>8, <1>1, <1>2
    <2>10. /\ \A u \in Reads(x) :
                alloc[u] \in Owners =>
                  (ostate'[alloc[u]] = "live" => \E y \in tasks' : y.kind = "lend" /\ y.o = alloc[u])
           /\ x.kind = "prefetch" => \A y \in tasks' : (y.kind = "prefetch" /\ y.h = x.h) => y = x
           /\ x.kind = "put" => hcache'[x.k] = hcache[x.k] /\ vis'[x.k] = vis[x.k]
           /\ x.o \in Owners /\ ostate[x.o] # "new" => ostate'[x.o] # "new"
      <3>1. /\ \A u \in Reads(x) :
                 alloc[u] \in Owners =>
                   (ostate[alloc[u]] = "live" => \E y \in tasks : y.kind = "lend" /\ y.o = alloc[u])
            /\ x.kind = "prefetch" => \A y \in tasks : (y.kind = "prefetch" /\ y.h = x.h) => y = x
        BY <2>2 DEF TaskOk, Live
      <3>2. \A q \in Owners : ostate'[q] = "live" => ostate[q] = "live" /\ q # o
        BY <1>8
      <3>3. \A q \in Owners : q # o =>
              ((\E y \in tasks : y.kind = "lend" /\ y.o = q) => \E y \in tasks' : y.kind = "lend" /\ y.o = q)
        BY <1>8
      <3> QED
        BY <3>1, <3>2, <3>3, <1>8, <1>1
    <2> QED
      BY <2>2, <2>3, <2>7, <2>8, <2>9, <2>10, TaskKeep DEF Live
  <1>16. LendingUnmapped' /\ LostOk' /\ RequestsOk' /\ NoBad'
    <2>1. LendingUnmapped'
      BY <1>1, <1>8 DEF IndInv, LendingUnmapped
    <2> QED
      BY <1>1, <2>1, LostOkFrame, RequestsOkFrame, NoBadFrame DEF IndInv
  <1>17. TablesMapped' /\ HandoffOk'
    <2>1. TablesMapped'
      <3> SUFFICES ASSUME NEW g \in Groups, NEW q \in Owners, NEW p \in Rows, dbt[g][q][p].u # NONE
                   PROVE  (/\ Live(q)
                           /\ ogroup[q] = g \/ ReaderRow(g, q, p, dbt[g][q][p].u, dbt[g][q][p].w)
                           /\ HoldsAt(q, p, dbt[g][q][p].u)
                           /\ (p \in Positions /\ dbt[g][q][p].w) =>
                                /\ table[q][p].k = "private"
                                /\ ~\E d \in deferred :
                                      d.kind \in {"commit", "replicate"} /\ dbt[g][q][p].u \in d.units
                                /\ ~\E x \in tasks :
                                      x.kind = "replicate" /\ dbt[g][q][p].u \in x.units)'
        BY <1>1 DEF TablesMapped
      <3>1. q # o
        BY <1>2
      <3> QED
        BY <1>1, <1>8, <3>1 DEF IndInv, TablesMapped, Live, HoldsAt, ReaderRow, ReadersOf, Mem
    <2>2. HandoffOk'
      <3> SUFFICES ASSUME NEW h \in handoff
                   PROVE  (/\ h.o \in Owners /\ h.g \in Groups /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
                           /\ (~\E r \in requests : Names(r, h)) =>
                                /\ Live(h.o)
                                /\ ogroup[h.o] = h.g \/ ReaderRow(h.g, h.o, h.p, h.u, h.w)
                                /\ HoldsAt(h.o, h.p, h.u)
                                /\ (h.p \in Positions /\ h.w) =>
                                     /\ table[h.o][h.p].k = "private"
                                     /\ ~\E d \in deferred :
                                           d.kind \in {"commit", "replicate"} /\ h.u \in d.units
                                     /\ ~\E x \in tasks :
                                           x.kind = "replicate" /\ h.u \in x.units)'
        BY <1>1 DEF HandoffOk
      <3>1. h.o # o /\ h.o \in Owners
        BY <1>2, RequestTypes DEF RequestTyped
      <3> QED
        BY <1>1, <1>8, <3>1 DEF IndInv, HandoffOk, Live, HoldsAt, ReaderRow, ReadersOf, Mem
    <2> QED
      BY <2>1, <2>2
  <1>18. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. ASSUME NEW q \in Owners, NEW u \in AllUnits, Covered(b, q, u)
          PROVE  Covered(b, q, u)' /\ Held(u)'
      <3>1. CASE /\ Live(q) /\ (ogroup[q] = b.g \/ b.g \in ReadersOf(q))
                 /\ \E p \in Rows :
                      /\ HoldsAt(q, p, u)
                      /\ \A r \in requests :
                           TakesRow(r, q, p) => b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
                 /\ \A e \in deferred : (e.kind = "lend" /\ e.o = q) => e.after[b.g] >= b.seq
                 /\ ~\E x \in tasks : x.kind = "lend" /\ x.o = q
        <4>1. q # o
          BY <3>1, <1>1
        <4>2. Covered(b, q, u)'
          BY <1>1, <1>8, <3>1, <4>1 DEF Covered, Live, HoldsAt, ReadersOf
        <4>3. Held(u)'
          BY <1>7, <3>1, <4>1 DEF HoldsAt, Rows
        <4> QED
          BY <4>2, <4>3
      <3>2. CASE \E e \in deferred : e.kind = "release" /\ u \in e.units /\ e.after[b.g] >= b.seq
        BY <1>1, <1>7, <3>2 DEF Covered
      <3> QED
        BY <2>3, <3>1, <3>2 DEF Covered
    <2>4. \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits /\ x[2] \notin free
                             /\ Covered(b, x[1], x[2])
      BY <1>0, <2>2 DEF BatchOk, RequestTyped
    <2>5. \A x \in b.reads :
            /\ Covered(b, x[1], x[2])' /\ x[2] \notin free' /\ alloc'[x[2]] = alloc[x[2]]
      BY <1>3, <1>5, <1>6, <2>3, <2>4
    <2>6. \A x \in b.writes :
            \A r \in requests, p \in Positions :
              (FreezesRow(r, x[1], p) /\ table'[x[1]][p].u = x[2]) => table[x[1]][p].u = x[2]
      BY <1>8, <2>2, <2>4 DEF BatchOk, Empty, AllUnits
    <2> QED
      BY <1>1, <2>2, <2>5, <2>6 DEF BatchOk, Mem
  <1>19. CrossingsOk'
    <2>1. \A e \in deferred : CrossWaitOk(e)'
      <3> SUFFICES ASSUME NEW e \in deferred PROVE CrossWaitOk(e)'
        OBVIOUS
      <3>1. CrossWaitOk(e)
        BY CrossingsOkEach DEF IndInv
      <3>2. \A u \in e.units : alloc'[u] = alloc[u]
        BY <1>0, <1>6, <1>7 DEF WorkTyped
      <3>3. e.kind = "replicate" => e.o # o /\ ostate'[e.o] = ostate[e.o]
        BY <1>2, <1>8, <3>1 DEF CrossWaitOk
      <3> QED
        BY <1>1, <1>8, <3>1, <3>2, <3>3 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>2. \A x \in tasks' : CrossTaskOk(x)'
      <3> SUFFICES ASSUME NEW x \in tasks' PROVE CrossTaskOk(x)'
        OBVIOUS
      <3>1. x \in tasks /\ CrossTaskOk(x) /\ x.held
        BY <1>8, CrossingsOkEach, TaskFacts DEF IndInv, TaskOk
      <3>2. x.kind = "replicate" =>
              /\ \A u \in x.writes : alloc'[u] = alloc[u]
              /\ x.o # o /\ ostate'[x.o] = ostate[x.o]
        <4>1. x.kind = "replicate" => x.writes \subseteq x.units /\ x.units \subseteq Units /\ x.o \in Owners
          BY <3>1 DEF CrossTaskOk
        <4> QED
          BY <1>2, <1>6, <1>7, <1>8, <3>1, <4>1 DEF AllUnits
      <3> QED
        BY <1>1, <1>8, <3>1, <3>2 DEF CrossTaskOk, ReadersOf, Mem
    <2> QED
      BY <1>1, <2>1, <2>2, CrossingsOkEach
  <1>20. TrialVisibility'
    <2>1. \A q \in Owners, p \in Positions : table'[q][p] # Empty => table'[q][p] = table[q][p]
      BY <1>8
    <2> QED
      BY <1>1, <1>8, <2>1 DEF IndInv, TrialVisibility, MadeBy, Shared, Empty
  <1>21. CopiesPlaced' /\ CrossingsPlaced' /\ ImportsPlaced'
    <2>1. CopiesPlaced' /\ CrossingsPlaced'
      BY <1>1, <1>8, CopiesPlacedFrame, CrossingsPlacedFrame DEF IndInv
    <2>2. ImportsPlaced'
      BY <1>1, <1>8 DEF IndInv, ImportsPlaced
    <2> QED
      BY <2>1, <2>2
  <1> QED
    BY <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18, <1>19, <1>20, <1>21
       DEF IndInv

-----------------------------------------------------------------------------
(* ImportEnd                                                               *)

(* The values ImportEnd(t, TRUE) computes for an owner that is there: the  *)
(* position's block, the owner's table, where the new blocks are valid,    *)
(* and the entries its group's executor is handed.                          *)
PlaceOf(t, p) == CHOOSE x \in t.places : x[1] = p
Placed(t) == {x[1] : x \in t.places}
ImportTable(t) ==
    [table EXCEPT ![t.o] =
       [p \in Positions |->
          IF \E x \in t.places : x[1] = p THEN Private(PlaceOf(t, p)[2]) ELSE table[t.o][p]]]
ImportValid(t) ==
    [u \in DOMAIN valid |-> IF u \in t.writes THEN {Mem(ogroup[t.o])} ELSE valid[u]]
Entry(t, p) == [g |-> ogroup[t.o], o |-> t.o, p |-> p, u |-> PlaceOf(t, p)[2], w |-> TRUE]
ImportEntries(t) == {[g |-> ogroup[t.o], o |-> t.o, p |-> x[1], u |-> x[2], w |-> TRUE] : x \in t.places}

LEMMA ImportEnd_Effect ==
  ASSUME NEW t \in tasks, NEW ok \in BOOLEAN, ImportEnd(t, ok)
  PROVE  /\ t.kind = "import"
         /\ tasks' = tasks \ {t}
         /\ \/ /\ ok /\ ostate[t.o] = "live"
               /\ table' = ImportTable(t)
               /\ valid' = ImportValid(t)
               /\ DropWith(t.units, ImportTable(t), oslot, cache, hcache, deferred, copies, tasks')
               /\ GiveAll(ImportEntries(t))
            \/ /\ ~(ok /\ ostate[t.o] = "live")
               /\ DropWith(t.units, table, oslot, cache, hcache, deferred, copies, tasks')
               /\ UNCHANGED <<table, valid, dbt, handoff>>
         /\ UNCHANGED <<ostate, ogroup, oslot, cache, hcache, deferred, copies, ncopy, vis, trial,
                        member, launched, inflight, requests>>
  BY DEF ImportEnd, ImportTable, ImportValid, ImportEntries, PlaceOf

LEMMA ImportEnd_Fail ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, ImportEnd(t, ok), UNCHANGED rare,
         ~(ok /\ ostate[t.o] = "live")
  PROVE  IndInv'
  BY ImportEnd_Effect, TaskDrop DEF rare

(* What IndInv says of an import that ends: its places, one block per      *)
(* position and one position per block, and its owner's free positions.    *)
LEMMA ImportEnd_Facts ==
  ASSUME IndInv, NEW t \in tasks, t.kind = "import", Live(t.o)
  PROVE  /\ t.o \in Owners /\ ogroup[t.o] \in Groups /\ GroupEpoch[ogroup[t.o]] = OwnerEpoch[t.o]
         /\ OwnerEpoch[t.o] \notin lost
         /\ t.places \subseteq Positions \X Units
         /\ t.writes = {x[2] : x \in t.places} /\ t.writes \subseteq t.units
         /\ t.units \subseteq Units /\ t.h = NONE
         /\ Placed(t) \subseteq Positions
         /\ \A p \in Placed(t) : PlaceOf(t, p) \in t.places /\ PlaceOf(t, p)[1] = p
                                 /\ PlaceOf(t, p)[2] \in t.writes /\ PlaceOf(t, p)[2] \in Units
         /\ \A x \in t.places : PlaceOf(t, x[1]) = x
         /\ \A p, q \in Placed(t) : PlaceOf(t, p)[2] = PlaceOf(t, q)[2] => p = q
         /\ \A u \in t.writes : \E p \in Placed(t) : PlaceOf(t, p)[2] = u
         /\ \A u \in t.writes :
              /\ alloc[u] = t.o
              /\ \A o \in Owners, p \in Positions : table[o][p].u # u
              /\ \A k \in Keys : cache[k] # u /\ hcache[k] # u
              /\ \A d \in deferred : u \notin d.units
              /\ \A c \in copies : c.read # u /\ c.write # u
              /\ \A x \in tasks : x # t => u \notin x.units
              /\ \A o \in Owners : oslot[o] # u
              /\ \A b \in inflight : \A x \in b.reads : x[2] # u
         /\ \A p \in Placed(t) : table[t.o][p] = Empty /\ ~CopyFor(t.o, p)
         /\ ~Lending(t.o)
         /\ \A g \in Groups, r \in Rows : r \in Placed(t) => dbt[g][t.o][r].u = NONE
         /\ \A h \in handoff : (h.o = t.o /\ h.p \in Placed(t)) => \E r \in requests : Names(r, h)
  <1>1. TaskOk(t)
    BY TaskFacts
  <1>2. /\ t.o \in Owners /\ ogroup[t.o] \in Groups /\ GroupEpoch[ogroup[t.o]] = OwnerEpoch[t.o]
        /\ OwnerEpoch[t.o] \notin lost
    BY <1>1 DEF TaskOk, IndInv, OwnersOk, Live
  <1>3. t.places \subseteq Positions \X Units /\ t.units \subseteq Units /\ t.h = NONE
    BY <1>1, WorkTypes DEF WorkTyped, TaskOk
  <1>4. \A x, y \in t.places : (x[1] = y[1] \/ x[2] = y[2]) => x = y
    BY <1>1 DEF TaskOk
  <1>5. \A p \in Placed(t) : PlaceOf(t, p) \in t.places /\ PlaceOf(t, p)[1] = p
    BY DEF Placed, PlaceOf
  <1>6. \A x \in t.places : PlaceOf(t, x[1]) = x
    BY <1>4, <1>5 DEF Placed
  <1>7. \A u \in t.writes : u \in Units /\ u # NONE
    BY <1>1, <1>3 DEF TaskOk
  <1>8. \A u \in t.writes : \A o \in Owners : oslot[o] # u
    <2>1. oslot \in [Owners -> Slots \cup {NONE}]
      BY DEF IndInv, TypeInv, TypeOK
    <2> QED
      BY <1>7, <2>1
  <1>9. \A u \in t.writes : \A k \in Keys : hcache[k] # u
    <2>1. hcache \in [Keys -> HostUnits \cup {NONE}]
      BY DEF IndInv, TypeInv, TypeOK
    <2> QED
      BY <1>7, <2>1
  <1>10. \A u \in t.writes : \A b \in inflight : \A x \in b.reads : x[2] # u
    <2> SUFFICES ASSUME NEW u \in t.writes, NEW b \in inflight, NEW x \in b.reads, x[2] = u
                 PROVE  FALSE
      OBVIOUS
    <2>1. Covered(b, x[1], u) /\ x[1] \in Owners
      BY DEF IndInv, BatchesOk
    <2>2. CASE \E p \in Rows : HoldsAt(x[1], p, u)
      <3>1. PICK p \in Rows : HoldsAt(x[1], p, u)
        BY <2>2
      <3>2. \A q \in Owners, r \in Positions : table[q][r].u # u
        BY <1>1 DEF TaskOk
      <3>3. \A q \in Owners : oslot[q] # u
        BY <1>8
      <3> QED
        BY <2>1, <3>1, <3>2, <3>3 DEF HoldsAt, Rows
    <2>3. CASE \E d \in deferred : d.kind = "release" /\ u \in d.units
      BY <2>3, <1>1 DEF TaskOk
    <2> QED
      BY <2>1, <2>2, <2>3 DEF Covered
  <1>11. \A g \in Groups, r \in Rows : r \in Placed(t) => dbt[g][t.o][r].u = NONE
    <2> SUFFICES ASSUME NEW g \in Groups, NEW r \in Rows, r \in Placed(t), dbt[g][t.o][r].u # NONE
                 PROVE  FALSE
      OBVIOUS
    <2>1. HoldsAt(t.o, r, dbt[g][t.o][r].u)
      BY <1>2 DEF IndInv, TablesMapped
    <2>2. r \in Positions /\ table[t.o][r] = Empty
      BY <1>1, <1>3, <1>5 DEF TaskOk, Placed
    <2> QED
      BY <2>1, <2>2, SLOT \notin Positions DEF HoldsAt, Empty
  <1>12. \A h \in handoff : (h.o = t.o /\ h.p \in Placed(t)) => \E r \in requests : Names(r, h)
    <2> SUFFICES ASSUME NEW h \in handoff, h.o = t.o, h.p \in Placed(t),
                        ~\E r \in requests : Names(r, h)
                 PROVE  FALSE
      OBVIOUS
    <2>1. HoldsAt(h.o, h.p, h.u) /\ h.u \in AllUnits
      BY RequestTypes DEF IndInv, HandoffOk, RequestTyped
    <2>2. h.p \in Positions /\ table[t.o][h.p] = Empty
      BY <1>1, <1>3, <1>5 DEF TaskOk, Placed
    <2> QED
      BY <2>1, <2>2, SLOT \notin Positions DEF HoldsAt, Empty, AllUnits
  <1>13. \A u \in t.writes : \E p \in Placed(t) : PlaceOf(t, p)[2] = u
    BY <1>1, <1>6 DEF TaskOk, Placed
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13
       DEF TaskOk, Placed

(* The rows GiveAll writes: the entry of each placed position that no      *)
(* request under way names, when completions are handed on at once.        *)
LEMMA ImportEnd_Rows ==
  ASSUME IndInv, NEW t \in tasks, t.kind = "import", Live(t.o), GiveAll(ImportEntries(t))
  PROVE  /\ handoff' = IF Eager THEN handoff ELSE handoff \cup ImportEntries(t)
         /\ DOMAIN dbt' = Groups
         /\ \A g \in Groups : DOMAIN dbt'[g] = Owners
         /\ \A g \in Groups, q \in Owners : DOMAIN dbt'[g][q] = Rows
         /\ \A g \in Groups, q \in Owners, r \in Rows :
              dbt'[g][q][r] =
                IF Eager /\ g = ogroup[t.o] /\ q = t.o /\ r \in Placed(t) /\ Unnamed(Entry(t, r))
                THEN [u |-> PlaceOf(t, r)[2], w |-> TRUE]
                ELSE dbt[g][q][r]
         /\ ImportEntries(t) = {Entry(t, p) : p \in Placed(t)}
  <1>1. \A x \in t.places : PlaceOf(t, x[1]) = x
    BY ImportEnd_Facts
  <1>2. ImportEntries(t) = {Entry(t, p) : p \in Placed(t)}
    BY <1>1 DEF ImportEntries, Entry, Placed
  <1>3. \A h \in ImportEntries(t) : h = Entry(t, h.p) /\ h.p \in Placed(t)
    BY <1>2 DEF Entry
  <1>4. CASE Eager
    <2>1. dbt' = [g \in Groups |-> [q \in Owners |-> [r \in Rows |->
                   IF \E h \in ImportEntries(t) : /\ h.g = g /\ h.o = q /\ h.p = r
                                                 /\ ~\E x \in requests : Names(x, h)
                   THEN LET h == CHOOSE x \in ImportEntries(t) : x.g = g /\ x.o = q /\ x.p = r
                        IN [u |-> h.u, w |-> h.w]
                   ELSE dbt[g][q][r]]]]
          /\ handoff' = handoff
      BY <1>4 DEF GiveAll
    <2>2. \A g \in Groups, q \in Owners, r \in Rows :
            (\E h \in ImportEntries(t) : h.g = g /\ h.o = q /\ h.p = r)
              <=> (g = ogroup[t.o] /\ q = t.o /\ r \in Placed(t))
      BY <1>2, <1>3 DEF Entry
    <2>3. \A g \in Groups, q \in Owners, r \in Rows :
            (g = ogroup[t.o] /\ q = t.o /\ r \in Placed(t)) =>
              (CHOOSE x \in ImportEntries(t) : x.g = g /\ x.o = q /\ x.p = r) = Entry(t, r)
      BY <1>2, <1>3 DEF Entry
    <2> QED
      BY <1>2, <1>4, <2>1, <2>2, <2>3 DEF Unnamed, Entry
  <1>5. CASE ~Eager
    <2>1. dbt' = dbt /\ handoff' = handoff \cup ImportEntries(t)
      BY <1>5 DEF GiveAll
    <2> QED
      BY <1>2, <1>5, <2>1, PlaneTypes DEF PlaneTyped
  <1> QED
    BY <1>4, <1>5

LEMMA ImportEnd_Done ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, ImportEnd(t, ok), UNCHANGED rare,
         ok, ostate[t.o] = "live"
  PROVE  IndInv'
  <1>0. /\ UnitTyped /\ OwnerTyped /\ PlaneTyped /\ WorkTyped /\ RequestTyped
    BY UnitTypes, OwnerTypes, PlaneTypes, WorkTypes, RequestTypes
  <1>1. /\ t.kind = "import"
        /\ tasks' = tasks \ {t}
        /\ table' = ImportTable(t)
        /\ valid' = ImportValid(t)
        /\ DropWith(t.units, ImportTable(t), oslot, cache, hcache, deferred, copies, tasks')
        /\ GiveAll(ImportEntries(t))
        /\ UNCHANGED <<ostate, ogroup, oslot, cache, hcache, deferred, copies, ncopy, vis, trial,
                       member, launched, inflight, requests, remote, lost, bad>>
    BY ImportEnd_Effect DEF rare
  <1> DEFINE o == t.o
             W == t.writes
             P == Placed(t)
             G == ReclaimedWith(t.units, ImportTable(t), oslot, cache, hcache, deferred, copies,
                                tasks')
  <1>2. /\ o \in Owners /\ Live(o) /\ ogroup[o] \in Groups /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o]
        /\ OwnerEpoch[o] \notin lost
        /\ t.places \subseteq Positions \X Units
        /\ W = {x[2] : x \in t.places} /\ W \subseteq t.units
        /\ t.units \subseteq Units /\ t.h = NONE
        /\ P \subseteq Positions
        /\ \A p \in P : PlaceOf(t, p) \in t.places /\ PlaceOf(t, p)[1] = p
                        /\ PlaceOf(t, p)[2] \in W /\ PlaceOf(t, p)[2] \in Units
        /\ \A x \in t.places : PlaceOf(t, x[1]) = x
        /\ \A p, q \in P : PlaceOf(t, p)[2] = PlaceOf(t, q)[2] => p = q
        /\ \A u \in W : \E p \in P : PlaceOf(t, p)[2] = u
        /\ \A u \in W :
             /\ alloc[u] = o
             /\ \A q \in Owners, p \in Positions : table[q][p].u # u
             /\ \A k \in Keys : cache[k] # u /\ hcache[k] # u
             /\ \A d \in deferred : u \notin d.units
             /\ \A c \in copies : c.read # u /\ c.write # u
             /\ \A x \in tasks : x # t => u \notin x.units
             /\ \A q \in Owners : oslot[q] # u
             /\ \A b \in inflight : \A x \in b.reads : x[2] # u
        /\ \A p \in P : table[o][p] = Empty /\ ~CopyFor(o, p)
        /\ ~Lending(o)
        /\ \A g \in Groups, r \in Rows : r \in P => dbt[g][o][r].u = NONE
        /\ \A h \in handoff : (h.o = o /\ h.p \in P) => \E r \in requests : Names(r, h)
    BY <1>1, ImportEnd_Facts DEF Live
  <1>3. /\ handoff' = IF Eager THEN handoff ELSE handoff \cup ImportEntries(t)
        /\ DOMAIN dbt' = Groups
        /\ \A g \in Groups : DOMAIN dbt'[g] = Owners
        /\ \A g \in Groups, q \in Owners : DOMAIN dbt'[g][q] = Rows
        /\ \A g \in Groups, q \in Owners, r \in Rows :
             dbt'[g][q][r] =
               IF Eager /\ g = ogroup[o] /\ q = o /\ r \in P /\ Unnamed(Entry(t, r))
               THEN [u |-> PlaceOf(t, r)[2], w |-> TRUE]
               ELSE dbt[g][q][r]
        /\ ImportEntries(t) = {Entry(t, p) : p \in P}
    BY <1>1, <1>2, ImportEnd_Rows DEF Live
  (* The owners' tables after the step: o's placed positions hold the new  *)
  (* blocks, and every other entry is as it was.                           *)
  <1>4. /\ \A p \in P : table'[o][p] = Private(PlaceOf(t, p)[2])
        /\ \A q \in Owners, p \in Positions : (q # o \/ p \notin P) => table'[q][p] = table[q][p]
        /\ DOMAIN table' = Owners /\ \A q \in Owners : DOMAIN table'[q] = Positions
        /\ \A q \in Owners : q # o => table'[q] = table[q]
    <2>1. \A p \in Positions : (\E x \in t.places : x[1] = p) <=> p \in P
      BY DEF Placed
    <2>2. DOMAIN table = Owners /\ \A q \in Owners : DOMAIN table[q] = Positions
      BY <1>0 DEF OwnerTyped
    <2>3. table' = [table EXCEPT ![o] =
                      [p \in Positions |->
                         IF \E x \in t.places : x[1] = p THEN Private(PlaceOf(t, p)[2]) ELSE table[o][p]]]
      BY <1>1 DEF ImportTable
    <2> QED
      BY <1>2, <2>1, <2>2, <2>3
  <1>5. /\ \A u \in W : valid'[u] = {Mem(ogroup[o])}
        /\ \A u \in Units \cup HostUnits : u \notin W => valid'[u] = valid[u]
        /\ DOMAIN valid' = DOMAIN valid
    <2>1. valid' = [u \in DOMAIN valid |-> IF u \in W THEN {Mem(ogroup[o])} ELSE valid[u]]
      BY <1>1 DEF ImportValid
    <2>2. DOMAIN valid = Units \cup HostUnits /\ W \subseteq Units
      BY <1>0, <1>2 DEF UnitTyped
    <2> QED
      BY <2>1, <2>2
  (* The holders: the task's units leave, the new blocks are held by the   *)
  (* owner's table.                                                        *)
  <1>6. /\ \A x : Held(x)' = HeldIn(x, ImportTable(t), oslot, cache, hcache, deferred, copies,
                                     tasks')
        /\ G = {x \in t.units : ~Held(x)'}
        /\ free' = free \cup G /\ alloc' = Reset(alloc, G) /\ sharedKey' = Reset(sharedKey, G)
    BY <1>1, ReclaimedHeld DEF Held, DropWith, On, AllRules
  <1>7. /\ \A x \in W : Held(x)'
        /\ \A q \in Owners, p \in Positions : \A x \in AllUnits : table[q][p].u = x => Held(x)'
        /\ \A q \in Owners : \A x \in AllUnits : oslot[q] = x => Held(x)'
        /\ \A k \in Keys : \A x \in AllUnits : (cache[k] = x \/ hcache[k] = x) => Held(x)'
        /\ \A e \in deferred : \A x \in e.units : Held(x)'
        /\ \A c \in copies : Held(c.read)' /\ Held(c.write)'
        /\ \A x \in tasks' : x.held => \A y \in x.units \cup {x.h} : Held(y)'
    <2>1. \A x \in W : Held(x)'
      <3> SUFFICES ASSUME NEW x \in W PROVE Held(x)'
        OBVIOUS
      <3>1. PICK p \in P : PlaceOf(t, p)[2] = x
        BY <1>2
      <3>2. p \in Positions /\ o \in Owners /\ table'[o][p].u = x
        BY <1>2, <1>4, <3>1 DEF Private
      <3> QED
        BY <1>1, <1>6, <3>2 DEF HeldIn
    <2>2. \A q \in Owners, p \in Positions : \A x \in AllUnits : table[q][p].u = x => Held(x)'
      <3> SUFFICES ASSUME NEW q \in Owners, NEW p \in Positions, NEW x \in AllUnits, table[q][p].u = x
                   PROVE  Held(x)'
        OBVIOUS
      <3>1. ~(q = o /\ p \in P)
        BY <1>2 DEF Empty, AllUnits
      <3>2. table'[q][p].u = x
        BY <3>1, <1>4
      <3> QED
        BY <1>1, <1>6, <3>2 DEF HeldIn
    <2> QED
      BY <1>6, <2>1, <2>2 DEF HeldIn
  <1>8. /\ t.units \subseteq AllUnits /\ \A x \in t.units : Held(x)
        /\ \A x \in AllUnits : Held(x)' => Held(x)
        /\ \A x \in AllUnits : Held(x) => Held(x)' \/ x \in G
        /\ \A x \in G : Held(x) /\ ~Held(x)'
        /\ G \subseteq AllUnits
        /\ \A x \in G : x \notin W
    <2>1. t.units \subseteq AllUnits /\ \A x \in t.units : Held(x)
      BY <1>2, TasksHeld DEF Held, HeldIn, AllUnits
    <2>2. \A x \in AllUnits : Held(x)' => Held(x)
      <3> SUFFICES ASSUME NEW x \in AllUnits, Held(x)' PROVE Held(x)
        OBVIOUS
      <3>1. CASE \E q \in Owners, p \in Positions : ImportTable(t)[q][p].u = x
        <4>1. PICK q \in Owners, p \in Positions : ImportTable(t)[q][p].u = x
          BY <3>1
        <4>2. CASE q = o /\ p \in P
          <5>1. x \in W
            BY <4>1, <4>2, <1>1, <1>2, <1>4 DEF Private
          <5> QED
            BY <5>1, <2>1, <1>2
        <4>3. CASE ~(q = o /\ p \in P)
          BY <4>1, <4>3, <1>1, <1>4 DEF Held, HeldIn
        <4> QED
          BY <4>2, <4>3
      <3>2. CASE ~\E q \in Owners, p \in Positions : ImportTable(t)[q][p].u = x
        BY <3>2, <1>1, <1>6 DEF Held, HeldIn
      <3> QED
        BY <3>1, <3>2
    <2>3. \A x \in AllUnits : Held(x) => Held(x)' \/ x \in t.units
      BY <1>1, <1>2, <1>6, <1>7 DEF Held, HeldIn, AllUnits
    <2>4. \A x \in G : x \notin W
      BY <1>6, <1>7
    <2> QED
      BY <1>6, <2>1, <2>2, <2>3, <2>4
  <1>9. /\ \A v \in AllUnits : alloc'[v] = IF v \in G THEN NONE ELSE alloc[v]
        /\ \A v \in AllUnits : sharedKey'[v] = IF v \in G THEN NONE ELSE sharedKey[v]
        /\ \A v \in AllUnits : Held(v)' => alloc'[v] = alloc[v] /\ sharedKey'[v] = sharedKey[v]
    BY <1>6 DEF Reset
  <1>10. /\ \A x \in tasks' : x \in tasks /\ x # t
         /\ \A q : Lending(q)' = Lending(q)
         /\ \A q : Importing(q)' => Importing(q)
         /\ Crossings' = Crossings
         /\ \A q : (\E x \in tasks' : x.kind = "lend" /\ x.o = q)
                     <=> (\E x \in tasks : x.kind = "lend" /\ x.o = q)
    BY <1>1 DEF Lending, Importing, Crossings
  <1>11. TypeInv'
    <2>1. UnitTyped'
      <3>1. ValidTyped'
        <4>1. Mem(ogroup[o]) \in Memories
          BY <1>2 DEF Mem, Memories
        <4>2. valid' = [u \in Units \cup HostUnits |-> IF u \in W THEN {Mem(ogroup[o])} ELSE valid[u]]
          BY <1>0, <1>1 DEF ImportValid, UnitTyped
        <4>3. W \subseteq Units
          BY <1>2
        <4> QED
          BY <1>0, <4>1, <4>2, <4>3 DEF ValidTyped, UnitTyped
      <3>2. free' \subseteq AllUnits /\ member' \subseteq AllUnits
        BY <1>0, <1>1, <1>6, <1>8 DEF UnitTyped
      <3>3. alloc' \in [AllUnits -> Owners \cup {NONE, "shared", "copy", "task"}]
            /\ sharedKey' \in [AllUnits -> Keys \cup {NONE}]
        BY <1>0, <1>6 DEF UnitTyped, Reset
      <3>4. cache' \in [Keys -> Units \cup {NONE}] /\ hcache' \in [Keys -> HostUnits \cup {NONE}]
        BY <1>0, <1>1 DEF UnitTyped
      <3> QED
        BY <3>1, <3>2, <3>3, <3>4 DEF UnitTyped, ValidTyped
    <2>2. OwnerTyped'
      BY <1>0, <1>1, <1>2, <1>4 DEF OwnerTyped, Private
    <2>3. PlaneTyped'
      BY <1>0, <1>1, <1>2, <1>3 DEF PlaneTyped, AllUnits
    <2>4. WorkTyped'
      BY <1>0, <1>1, <1>10 DEF WorkTyped
    <2>5. RequestTyped'
      <3>1. \A h \in ImportEntries(t) :
              h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows /\ h.u \in AllUnits
        BY <1>2, <1>3 DEF Entry, Rows, AllUnits
      <3> QED
        BY <1>0, <1>1, <1>3, <3>1 DEF RequestTyped
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4, <2>5, TypeInvNext
  <1>12. FreeClean' /\ NoLeak' /\ Outside' /\ Regions'
    BY <1>0, <1>1, <1>6, <1>8, <1>9, FreeReclaim DEF IndInv
  <1>13. AllocOwner' /\ SharedIff' /\ SharedCached' /\ CacheShared'
    <2>1. AllocOwner' /\ SharedIff'
      BY <1>1, <1>9 DEF IndInv, AllocOwner, SharedIff, NONE
    <2>2. \A v \in AllUnits : alloc[v] = "shared" /\ v \notin G =>
            cache[sharedKey[v]] = v \/ hcache[sharedKey[v]] = v
      BY DEF IndInv, SharedCached
    <2>3. SharedCached'
      BY <1>1, <1>9, <2>2 DEF SharedCached, NONE
    <2>4. CacheShared'
      BY <1>0, <1>1, <1>9, <1>7 DEF IndInv, CacheShared, UnitTyped, AllUnits
    <2> QED
      BY <2>1, <2>3, <2>4
  <1>14. Tables' /\ OwnersOk'
    <2>1. Tables'
      <3> SUFFICES ASSUME NEW q \in Owners, NEW p \in Positions
                   PROVE  /\ table'[q][p].k = "none" => table'[q][p].u = NONE
                          /\ table'[q][p].k = "private" =>
                               table'[q][p].u \in Units /\ alloc'[table'[q][p].u] = q
                          /\ table'[q][p].k = "shared" =>
                               table'[q][p].u \in Units /\ alloc'[table'[q][p].u] = "shared"
                          /\ table'[q][p].k # "none" => ostate'[q] = "live"
                          /\ \A r \in Positions :
                               (table'[q][p].k = "private" /\ table'[q][r].u = table'[q][p].u) => p = r
                          /\ table'[q][p].u # NONE => Mem(ogroup'[q]) \in valid'[table'[q][p].u]
        BY DEF Tables, Live
      <3>1. CASE q = o /\ p \in P
        <4>1. table'[q][p] = Private(PlaceOf(t, p)[2]) /\ PlaceOf(t, p)[2] \in W
              /\ PlaceOf(t, p)[2] \in Units
          BY <3>1, <1>2, <1>4
        <4>2. alloc'[PlaceOf(t, p)[2]] = o
          BY <4>1, <1>2, <1>7, <1>9 DEF AllUnits
        <4>3. \A r \in Positions : table'[q][r].u = PlaceOf(t, p)[2] => r = p
          <5> SUFFICES ASSUME NEW r \in Positions, table'[q][r].u = PlaceOf(t, p)[2] PROVE r = p
            OBVIOUS
          <5>1. CASE r \in P
            BY <5>1, <3>1, <1>2, <1>4 DEF Private
          <5>2. CASE r \notin P
            BY <5>2, <3>1, <1>2, <1>4, <4>1
          <5> QED
            BY <5>1, <5>2
        <4>4. valid'[PlaceOf(t, p)[2]] = {Mem(ogroup[o])}
          BY <4>1, <1>5
        <4> QED
          BY <3>1, <4>1, <4>2, <4>3, <4>4, <1>1, <1>2 DEF Private, Live
      <3>2. CASE ~(q = o /\ p \in P)
        <4>1. table'[q][p] = table[q][p]
          BY <3>2, <1>4
        <4>2. table[q][p].u # NONE =>
                /\ table[q][p].u \in AllUnits /\ alloc'[table[q][p].u] = alloc[table[q][p].u]
                /\ table[q][p].u \notin W /\ valid'[table[q][p].u] = valid[table[q][p].u]
          BY <1>0, <1>2, <1>5, <1>7, <1>9 DEF OwnerTyped, Empty, Private, Shared, AllUnits
        <4>3. \A r \in Positions : (table[q][p].k = "private" /\ table'[q][r].u = table[q][p].u) =>
                                     table'[q][r] = table[q][r]
          BY <1>2, <1>4, <4>2, <3>2 DEF Private
        <4> QED
          BY <1>1, <4>1, <4>2, <4>3 DEF IndInv, Tables, Live, Mem
      <3> QED
        BY <3>1, <3>2
    <2>2. OwnersOk'
      <3>1. \A q \in Owners : oslot[q] # NONE => alloc'[oslot[q]] = alloc[oslot[q]]
        BY <1>0, <1>7, <1>9 DEF OwnerTyped, AllUnits
      <3>2. \A q \in Owners : ~Live(q) => table'[q] = table[q]
        BY <1>2, <1>4
      <3> QED
        BY <1>1, <3>1, <3>2 DEF IndInv, OwnersOk, Live
    <2> QED
      BY <2>1, <2>2
  <1>15. DeferredOk'
    <2>1. SUFFICES ASSUME NEW e \in deferred PROVE DeferOk(e)'
      BY <1>1, DeferredOkEach
    <2>2. DeferOk(e)
      BY DeferredOkEach DEF IndInv
    <2>3. e.kind = "commit" => \A u \in e.units : alloc'[u] = alloc[u]
      BY <1>0, <1>9, <1>7 DEF WorkTyped
    <2>4. \A q \in Owners, p \in Positions : table'[q][p].u \in e.units => table'[q][p] = table[q][p]
      BY <1>2, <1>4 DEF Private
    <2> QED
      BY <1>1, <1>10, <2>2, <2>3, <2>4 DEF DeferOk
  <1>16. CopiesOk'
    <2>1. SUFFICES ASSUME NEW c \in copies PROVE CopyOk(c)'
      BY <1>1, CopiesOkEach
    <2>2. CopyOk(c) /\ c.read \in AllUnits /\ c.write \in AllUnits
      BY <1>0, CopiesOkEach DEF IndInv, WorkTyped
    <2>3. alloc'[c.read] = alloc[c.read] /\ alloc'[c.write] = alloc[c.write]
      BY <1>7, <1>9, <2>2
    <2>4. \A q \in Owners, p \in Positions : table'[q][p].u = c.write => table'[q][p] = table[q][p]
      BY <1>2, <1>4 DEF Private
    <2>5. c.kind \in {"clone", "load"} => table'[c.o][c.p] = table[c.o][c.p]
      <3> SUFFICES ASSUME c.kind \in {"clone", "load"} PROVE table'[c.o][c.p] = table[c.o][c.p]
        OBVIOUS
      <3>1. c.o \in Owners /\ c.p \in Positions
        BY <2>2 DEF CopyOk
      <3>2. ~(c.o = o /\ c.p \in P)
        BY <1>2 DEF CopyFor
      <3> QED
        BY <3>1, <3>2, <1>4
    <2>6. c.write \in Units => valid'[c.write] = valid[c.write]
      BY <1>2, <1>5
    <2>7. ~c.fill =>
            /\ c.kind \in {"clone", "load", "store", "save"}
            /\ c.kind \in {"clone", "load"} => c.write \in Units /\ alloc'[c.write] = c.o
            /\ c.kind \in {"clone", "load"} => \A x \in tasks' : c.write \notin x.units
            /\ c.kind \in {"store", "save"} => c.write \in HostUnits /\ alloc'[c.write] = "copy"
            /\ \A q \in Owners, p \in Positions : table'[q][p].u # c.write
            /\ \A x \in copies' : x.write = c.write => x = c
      BY <1>1, <1>10, <2>2, <2>3, <2>4 DEF CopyOk
    <2>8. c.fill =>
            \/ /\ c.kind = "load"
               /\ cache'[c.k] = c.write
               /\ Mem(c.g) \notin valid'[c.write]
               /\ \A x \in copies' : (x.write = c.write /\ Mem(x.g) = Mem(c.g)) => x = c
            \/ c.kind = "restore" /\ c.write \in Slots /\ alloc'[c.write] = c.o
            \/ /\ c.kind = "store"
               /\ c.write \in HostUnits /\ hcache'[c.k] = c.write
               /\ \A x \in copies' : x.write = c.write => x = c
      <3>1. (c.fill /\ c.kind = "load") => c.write \in Units
        <4>1. c.k \in Keys /\ cache \in [Keys -> Units \cup {NONE}]
          BY <1>0 DEF UnitTyped, WorkTyped
        <4> QED
          BY <4>1, <2>2 DEF CopyOk, AllUnits
      <3> QED
        BY <1>1, <2>2, <2>3, <2>6, <3>1 DEF CopyOk, Mem
    <2>9. (c.o \in Owners /\ ostate'[c.o] = "live") => ogroup'[c.o] = c.g /\ ~Lending(c.o)'
      BY <1>1, <1>10, <2>2 DEF CopyOk, Live
    <2>10. /\ c.id \in 1 .. ncopy'
           /\ \A x \in copies' : x.id = c.id => x = c
           /\ c.g \in Groups /\ GroupEpoch[c.g] \notin lost'
           /\ c.seq <= launched'[c.g]
           /\ c.done => c.seq > 0
           /\ c.kind \in {"clone", "load", "store", "save", "restore"}
           /\ IF c.kind = "save" THEN c.read \in Slots /\ alloc'[c.read] = c.o
                                 ELSE alloc'[c.read] = "shared"
           /\ c.o \in Owners => OwnerEpoch[c.o] = GroupEpoch[c.g]
           /\ c.kind \in {"clone", "load"} =>
                /\ c.o \in Owners /\ c.p \in Positions
                /\ table'[c.o][c.p] = Empty
                /\ \A x \in copies' : (x.o = c.o /\ x.p = c.p) => x = c
           /\ c.kind \in {"save", "restore"} => c.o \in Owners
           /\ c.o \in Owners => ostate'[c.o] # "new"
      BY <1>1, <2>2, <2>3, <2>5 DEF CopyOk
    <2> QED
      BY <2>7, <2>8, <2>9, <2>10 DEF CopyOk, Live
  <1>17. TasksOk'
    <2>1. SUFFICES ASSUME NEW x \in tasks' PROVE TaskOk(x)'
      BY TasksOkEach
    <2>2. x \in tasks /\ x # t /\ TaskOk(x)
      BY <1>10, TaskFacts
    <2>3. \A u \in x.units : alloc'[u] = alloc[u] /\ u \notin W
      <3>1. x.held /\ x.units \subseteq AllUnits /\ x \in tasks'
        BY <2>2, <1>1, TaskFacts DEF TaskOk
      <3>2. \A u \in x.units : Held(u)' /\ u \notin W
        BY <3>1, <1>7, <1>2, <2>2
      <3> QED
        BY <3>1, <3>2, <1>9
    <2>4. x.kind = "prefetch" => alloc'[x.h] = alloc[x.h]
      BY <1>7, <1>9, <2>2 DEF TaskOk, AllUnits
    <2>5. x.kind = "import" =>
            /\ x.o # o
            /\ \A u \in x.writes : \A q \in Owners, p \in Positions : table'[q][p].u # u
      <3> SUFFICES ASSUME x.kind = "import"
                   PROVE  /\ x.o # o
                          /\ \A u \in x.writes : \A q \in Owners, p \in Positions : table'[q][p].u # u
        OBVIOUS
      <3>1. x.o # o
        BY <1>1, <2>2 DEF TaskOk
      <3>2. \A u \in x.writes : u \notin W /\ \A q \in Owners, p \in Positions : table[q][p].u # u
        BY <2>2, <2>3 DEF TaskOk
      <3> QED
        BY <3>1, <3>2, <1>2, <1>4 DEF Private
    <2>6. (x.kind = "borrow" /\ Live(x.o)') => x.o # o
      BY <1>2, <2>2 DEF TaskOk, Lending, Live
    <2>7. x.kind = "lend" => x.o # o
      BY <1>2, <2>2 DEF Lending
    <2>8. \A q \in Owners : q # o => table'[q] = table[q]
      BY <1>4
    <2> QED
      BY <1>1, <1>10, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8
         DEF TaskOk, Live, TableUnits, CopyFor, Reads
  <1>18. LendingUnmapped' /\ LostOk'
    <2>1. LendingUnmapped'
      <3> SUFFICES ASSUME NEW q \in Owners, Lending(q)'
                   PROVE  /\ \A g \in Groups, r \in Rows : dbt'[g][q][r].u = NONE
                          /\ \A h \in handoff' : h.o # q
                          /\ \A d, e \in deferred :
                               (d.kind = "lend" /\ e.kind = "lend" /\ d.o = q /\ e.o = q) => d = e
        BY <1>1 DEF LendingUnmapped
      <3>1. Lending(q) /\ q # o
        BY <1>10, <1>2
      <3>2. \A h \in ImportEntries(t) : h.o = o
        BY <1>3 DEF Entry
      <3> QED
        BY <1>3, <3>1, <3>2 DEF IndInv, LendingUnmapped
    <2>2. LostOk'
      <3>1. \A h \in ImportEntries(t) : h.g = ogroup[o] /\ GroupEpoch[h.g] \notin lost
        BY <1>2, <1>3 DEF Entry
      <3>2. \A g \in Groups, q \in Owners, r \in Rows :
              dbt'[g][q][r].u # NONE => dbt[g][q][r].u # NONE \/ g = ogroup[o]
        BY <1>3
      <3> QED
        BY <1>1, <1>2, <1>3, <3>1, <3>2 DEF IndInv, LostOk
    <2> QED
      BY <2>1, <2>2
  (* The executors' tables: a new entry is the owner's new private block,   *)
  (* writable; every other entry is as it was.                              *)
  <1>19. TablesMapped'
    <2> SUFFICES ASSUME NEW g \in Groups, NEW q \in Owners, NEW r \in Rows, dbt'[g][q][r].u # NONE
                 PROVE  /\ ostate'[q] = "live"
                        /\ \/ ogroup'[q] = g
                           \/ ReaderRow(g, q, r, dbt[g][q][r].u, dbt[g][q][r].w)'
                        /\ IF r = SLOT THEN oslot'[q] = dbt'[g][q][r].u
                                       ELSE table'[q][r].u = dbt'[g][q][r].u
                        /\ (r \in Positions /\ dbt'[g][q][r].w) =>
                             /\ table'[q][r].k = "private"
                             /\ ~\E d \in deferred' :
                                   d.kind \in {"commit", "replicate"} /\ dbt'[g][q][r].u \in d.units
                             /\ ~\E x \in tasks' : x.kind = "replicate" /\ dbt'[g][q][r].u \in x.units
      BY DEF TablesMapped, Live, HoldsAt
    <2>1. CASE Eager /\ g = ogroup[o] /\ q = o /\ r \in P /\ Unnamed(Entry(t, r))
      <3>1. dbt'[g][q][r] = [u |-> PlaceOf(t, r)[2], w |-> TRUE]
        BY <2>1, <1>3
      <3>2. r \in Positions /\ r # SLOT /\ table'[o][r] = Private(PlaceOf(t, r)[2])
            /\ PlaceOf(t, r)[2] \in W
        BY <2>1, <1>2, <1>4, SLOT \notin Positions
      <3>3. /\ ~\E d \in deferred : d.kind \in {"commit", "replicate"} /\ PlaceOf(t, r)[2] \in d.units
            /\ ~\E x \in tasks' : x.kind = "replicate" /\ PlaceOf(t, r)[2] \in x.units
        BY <1>2, <1>10, <3>2
      <3> QED
        BY <1>1, <1>2, <2>1, <3>1, <3>2, <3>3 DEF Private
    <2>2. CASE ~(Eager /\ g = ogroup[o] /\ q = o /\ r \in P /\ Unnamed(Entry(t, r)))
      <3>1. dbt'[g][q][r] = dbt[g][q][r]
        BY <2>2, <1>3
      <3>2. /\ Live(q)
            /\ ogroup[q] = g \/ ReaderRow(g, q, r, dbt[g][q][r].u, dbt[g][q][r].w)
            /\ HoldsAt(q, r, dbt[g][q][r].u)
            /\ (r \in Positions /\ dbt[g][q][r].w) =>
                 /\ table[q][r].k = "private"
                 /\ ~\E d \in deferred :
                       d.kind \in {"commit", "replicate"} /\ dbt[g][q][r].u \in d.units
                 /\ ~\E x \in tasks : x.kind = "replicate" /\ dbt[g][q][r].u \in x.units
        BY <3>1 DEF IndInv, TablesMapped
      <3>3. ~(q = o /\ r \in P)
        BY <3>1, <1>2
      <3>4. r \in Positions => table'[q][r] = table[q][r]
        BY <3>3, <1>4
      <3>5. r \in Positions => dbt[g][q][r].u \in Units /\ dbt[g][q][r].u \notin W
        BY <3>1, <3>2, <1>0, <1>2 DEF HoldsAt, OwnerTyped, Empty, Private, Shared
      <3>6. ReaderRow(g, q, r, dbt[g][q][r].u, dbt[g][q][r].w) =>
              ReaderRow(g, q, r, dbt[g][q][r].u, dbt[g][q][r].w)'
        <4> SUFFICES ASSUME ReaderRow(g, q, r, dbt[g][q][r].u, dbt[g][q][r].w)
                     PROVE  ReaderRow(g, q, r, dbt[g][q][r].u, dbt[g][q][r].w)'
          OBVIOUS
        <4>1. r \in Positions /\ g \in ReadersOf(q) /\ ~dbt[g][q][r].w
              /\ Mem(g) \in valid[dbt[g][q][r].u]
          BY DEF ReaderRow
        <4>2. valid'[dbt[g][q][r].u] = valid[dbt[g][q][r].u]
          BY <4>1, <3>5, <1>5
        <4>3. ReadersOf(q)' = ReadersOf(q)
          BY <1>1 DEF ReadersOf
        <4> QED
          BY <4>1, <4>2, <4>3, <3>1 DEF ReaderRow
      <3>7. IF r = SLOT THEN oslot'[q] = dbt'[g][q][r].u ELSE table'[q][r].u = dbt'[g][q][r].u
        BY <3>1, <3>2, <3>4, <1>1 DEF HoldsAt, Rows
      <3>8. (r \in Positions /\ dbt'[g][q][r].w) =>
              /\ table'[q][r].k = "private"
              /\ ~\E d \in deferred' :
                    d.kind \in {"commit", "replicate"} /\ dbt'[g][q][r].u \in d.units
              /\ ~\E x \in tasks' : x.kind = "replicate" /\ dbt'[g][q][r].u \in x.units
        BY <3>1, <3>2, <3>4, <1>1, <1>10 DEF Crossings
      <3> QED
        BY <1>1, <3>1, <3>2, <3>6, <3>7, <3>8 DEF Live
    <2> QED
      BY <2>1, <2>2
  <1>20. HandoffOk'
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
                                  /\ ~\E x \in tasks' : x.kind = "replicate" /\ y.u \in x.units
      BY <1>1 DEF HandoffOk, Live, Unnamed, HoldsAt
    <2>1. CASE y \in ImportEntries(t)
      <3>1. PICK p \in P : y = Entry(t, p)
        BY <2>1, <1>3
      <3>2. /\ y.o = o /\ y.g = ogroup[o] /\ y.p = p /\ y.u = PlaceOf(t, p)[2] /\ y.w
            /\ p \in Positions /\ p # SLOT
        BY <3>1, <1>2, SLOT \notin Positions DEF Entry
      <3>3. /\ table'[o][p] = Private(PlaceOf(t, p)[2]) /\ PlaceOf(t, p)[2] \in W
        BY <1>2, <1>4
      <3>4. /\ ~\E d \in deferred : d.kind \in {"commit", "replicate"} /\ PlaceOf(t, p)[2] \in d.units
            /\ ~\E x \in tasks' : x.kind = "replicate" /\ PlaceOf(t, p)[2] \in x.units
        BY <1>2, <1>10, <3>3
      <3> QED
        BY <1>1, <1>2, <3>2, <3>3, <3>4 DEF Private
    <2>2. CASE y \notin ImportEntries(t)
      <3>1. y \in handoff
        BY <2>2, <1>3
      <3>2. /\ y.o \in Owners /\ y.g \in Groups /\ GroupEpoch[y.g] = OwnerEpoch[y.o]
            /\ Unnamed(y) =>
                 /\ ostate[y.o] = "live" /\ (ogroup[y.o] = y.g \/ ReaderRow(y.g, y.o, y.p, y.u, y.w))
                 /\ HoldsAt(y.o, y.p, y.u)
                 /\ (y.p \in Positions /\ y.w) =>
                      /\ table[y.o][y.p].k = "private"
                      /\ ~\E d \in deferred : d.kind \in {"commit", "replicate"} /\ y.u \in d.units
                      /\ ~\E x \in tasks : x.kind = "replicate" /\ y.u \in x.units
        BY <3>1 DEF IndInv, HandoffOk, Live, Unnamed
      <3>3. y.p \in Rows /\ y.u \in AllUnits
        BY <3>1, RequestTypes DEF RequestTyped
      <3>4. Unnamed(y) => ~(y.o = o /\ y.p \in P)
        BY <3>1, <1>2 DEF Unnamed
      <3>5. Unnamed(y) => (y.p \in Positions => table'[y.o][y.p] = table[y.o][y.p])
        BY <3>2, <3>4, <1>4
      <3>6. (Unnamed(y) /\ y.p \in Positions) => y.u \in Units /\ y.u \notin W
        BY <3>2, <3>3, <1>0, <1>2 DEF HoldsAt, OwnerTyped, Empty, Private, Shared, AllUnits
      <3>7. (Unnamed(y) /\ ReaderRow(y.g, y.o, y.p, y.u, y.w)) => ReaderRow(y.g, y.o, y.p, y.u, y.w)'
        BY <3>6, <1>5, <1>1 DEF ReaderRow, ReadersOf, Mem
      <3> QED
        BY <1>1, <1>10, <3>2, <3>3, <3>5, <3>7 DEF HoldsAt, Unnamed, Rows
    <2> QED
      BY <2>1, <2>2
  <1>21. RequestsOk'
    <2>1. \A r \in requests :
            r.kind # "finish" =>
              \A g \in Groups, q \in Rows :
                (dbt'[g][r.o][q].u # NONE /\ (TakesRow(r, r.o, q) \/ FreezesRow(r, r.o, q))) =>
                  /\ g \in r.sg
                  /\ g \in r.by => FreezesRow(r, r.o, q) /\ ~dbt'[g][r.o][q].w
      <3> SUFFICES ASSUME NEW r \in requests, r.kind # "finish", NEW g \in Groups, NEW q \in Rows,
                          dbt'[g][r.o][q].u # NONE, TakesRow(r, r.o, q) \/ FreezesRow(r, r.o, q)
                   PROVE  /\ g \in r.sg
                          /\ g \in r.by => FreezesRow(r, r.o, q) /\ ~dbt'[g][r.o][q].w
        OBVIOUS
      <3>1. r.o \in Owners
        BY DEF IndInv, TypeInv
      <3>2. CASE Eager /\ g = ogroup[o] /\ r.o = o /\ q \in P /\ Unnamed(Entry(t, q))
        <4>1. Names(r, Entry(t, q))
          BY <3>2, NamesRow DEF Entry
        <4> QED
          BY <3>2, <4>1 DEF Unnamed
      <3>3. CASE ~(Eager /\ g = ogroup[o] /\ r.o = o /\ q \in P /\ Unnamed(Entry(t, q)))
        <4>1. dbt'[g][r.o][q] = dbt[g][r.o][q]
          BY <3>1, <3>3, <1>3
        <4> QED
          BY <3>1, <4>1 DEF IndInv, RequestsOk
      <3> QED
        BY <3>2, <3>3
    <2> QED
      BY <1>1, <2>1 DEF IndInv, RequestsOk
  <1>22. BatchesOk'
    <2>1. SUFFICES ASSUME NEW b \in inflight PROVE BatchOk(b)'
      BY <1>1, BatchesOkEach
    <2>2. BatchOk(b)
      BY BatchesOkEach DEF IndInv
    <2>3. ASSUME NEW q \in Owners, NEW u \in AllUnits, Covered(b, q, u)
          PROVE  Covered(b, q, u)' /\ Held(u)'
      <3>1. \A r \in Rows : HoldsAt(q, r, u) => HoldsAt(q, r, u)'
        <4> SUFFICES ASSUME NEW r \in Rows, HoldsAt(q, r, u) PROVE HoldsAt(q, r, u)'
          OBVIOUS
        <4>1. CASE r = SLOT
          BY <4>1, <1>1 DEF HoldsAt
        <4>2. CASE r # SLOT
          <5>1. r \in Positions /\ table[q][r].u = u
            BY <4>2 DEF HoldsAt, Rows
          <5>2. ~(q = o /\ r \in P)
            BY <5>1, <1>2 DEF Empty, AllUnits
          <5> QED
            BY <5>1, <5>2, <1>4, <4>2 DEF HoldsAt
        <4> QED
          BY <4>1, <4>2
      <3>2. Covered(b, q, u)'
        BY <2>3, <3>1, <1>1, <1>10, CoverStays
      <3>3. Held(u)'
        <4>1. CASE \E p \in Rows : HoldsAt(q, p, u)
          BY <4>1, <1>7 DEF HoldsAt, Rows
        <4>2. CASE \E e \in deferred : e.kind = "release" /\ u \in e.units
          BY <4>2, <1>7
        <4> QED
          BY <2>3, <4>1, <4>2 DEF Covered
      <3> QED
        BY <3>2, <3>3
    <2>4. \A x \in b.reads : x[1] \in Owners /\ x[2] \in AllUnits /\ x[2] \notin free
                             /\ Covered(b, x[1], x[2]) /\ x[2] \notin W
      BY <1>0, <1>2, <2>2 DEF BatchOk, RequestTyped
    <2>5. \A x \in b.reads :
            /\ Covered(b, x[1], x[2])' /\ x[2] \notin free' /\ alloc'[x[2]] = alloc[x[2]]
            /\ x[2] \in Units => valid'[x[2]] = valid[x[2]]
      BY <1>6, <1>8, <1>9, <1>5, <2>3, <2>4
    <2>6. \A x \in b.writes :
            \A r \in requests, p \in Positions :
              (FreezesRow(r, x[1], p) /\ table'[x[1]][p].u = x[2]) => table[x[1]][p].u = x[2]
      <3> SUFFICES ASSUME NEW x \in b.writes, NEW r \in requests, NEW p \in Positions,
                          table'[x[1]][p].u = x[2]
                   PROVE  table[x[1]][p].u = x[2]
        OBVIOUS
      <3>1. x \in b.reads /\ x[2] \notin W /\ x[1] \in Owners
        BY <2>2, <2>4 DEF BatchOk
      <3> QED
        BY <3>1, <1>2, <1>4 DEF Private
    <2> QED
      BY <1>1, <2>2, <2>5, <2>6 DEF BatchOk, Mem
  <1>23. CrossingsOk'
    <2>1. \A e \in deferred : CrossWaitOk(e)'
      <3> SUFFICES ASSUME NEW e \in deferred PROVE CrossWaitOk(e)'
        OBVIOUS
      <3>1. CrossWaitOk(e)
        BY CrossingsOkEach DEF IndInv
      <3>2. \A u \in e.units : alloc'[u] = alloc[u]
        BY <1>0, <1>9, <1>7 DEF WorkTyped
      <3> QED
        BY <1>1, <1>10, <3>1, <3>2 DEF CrossWaitOk, ReadersOf, MadeBy
    <2>2. \A x \in tasks' : CrossTaskOk(x)'
      <3> SUFFICES ASSUME NEW x \in tasks' PROVE CrossTaskOk(x)'
        OBVIOUS
      <3>1. x \in tasks /\ x # t /\ CrossTaskOk(x) /\ x.held
        BY <1>10, CrossingsOkEach, TaskFacts DEF IndInv, TaskOk
      <3>2. x.kind = "replicate" =>
              \A u \in x.writes : alloc'[u] = alloc[u] /\ valid'[u] = valid[u]
        <4>1. x.kind = "replicate" => x.writes \subseteq x.units /\ x.units \subseteq Units
          BY <3>1 DEF CrossTaskOk
        <4>2. \A u \in x.units : u \notin W
          BY <3>1, <1>2
        <4> QED
          BY <1>5, <1>7, <1>9, <3>1, <4>1, <4>2 DEF AllUnits
      <3> QED
        BY <1>1, <1>10, <3>1, <3>2 DEF CrossTaskOk, ReadersOf, Mem
    <2> QED
      BY <1>1, <2>1, <2>2, CrossingsOkEach
  <1>24. TrialVisibility'
    <2>1. \A q \in Owners, p \in Positions : table'[q][p] # table[q][p] => table'[q][p].k = "private"
      BY <1>2, <1>4 DEF Private
    <2>2. \A q \in Owners, p \in Positions : \A u : table'[q][p] = Shared(u) => table[q][p] = Shared(u)
      BY <2>1 DEF Shared
    <2> QED
      BY <1>1, <1>10, <2>2 DEF IndInv, TrialVisibility, MadeBy
  <1>25. CopiesPlaced' /\ CrossingsPlaced' /\ ImportsPlaced' /\ NoBad'
    <2>1. CopiesPlaced'
      <3> SUFFICES ASSUME NEW c \in copies PROVE CopyPlaced(c)'
        BY <1>1, PlacedEach
      <3>1. CopyPlaced(c) /\ c.read \notin W /\ c.write \notin W
        BY <1>2, PlacedEach DEF IndInv
      <3> QED
        BY <1>1, <1>5, <3>1 DEF CopyPlaced, Mem
    <2>2. CrossingsPlaced'
      <3> SUFFICES ASSUME NEW x \in tasks' PROVE CrossingPlaced(x)'
        BY PlacedEach
      <3>1. x \in tasks /\ x # t /\ CrossingPlaced(x)
        BY <1>10, PlacedEach DEF IndInv
      <3>2. x.kind = "replicate" => \A u \in x.units : u \in Units /\ u \notin W
        BY <3>1, <1>2, CrossingsOkEach DEF IndInv, CrossTaskOk
      <3> QED
        BY <1>1, <1>5, <3>1, <3>2 DEF CrossingPlaced, Mem
    <2>3. ImportsPlaced'
      BY <1>1, <1>10 DEF IndInv, ImportsPlaced
    <2>4. NoBad'
      BY DEF IndInv, NoBad, rare
    <2> QED
      BY <2>1, <2>2, <2>3, <2>4
  <1> QED
    BY <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18, <1>19, <1>20, <1>21, <1>22, <1>23,
       <1>24, <1>25 DEF IndInv

LEMMA ImportEnd_Inv ==
  ASSUME IndInv, NEW t \in tasks, NEW ok \in BOOLEAN, ImportEnd(t, ok), UNCHANGED rare
  PROVE  IndInv'
  <1>1. CASE ok /\ ostate[t.o] = "live"
    BY <1>1, ImportEnd_Done
  <1>2. CASE ~(ok /\ ostate[t.o] = "live")
    BY <1>2, ImportEnd_Fail
  <1> QED
    BY <1>1, <1>2

-----------------------------------------------------------------------------

THEOREM Step8 ==
    IndInv /\ (/\ \/ \E d \in deferred : LendDue(d)
                  \/ \E o \in Owners, ps \in SUBSET Positions, l \in Owners \cup {NONE} :
                       Import(o, ps, l)
                  \/ \E o \in Owners : Borrow(o)
                  \/ \E t \in tasks : GiveBack(t) \/ ReleaseBorrow(t)
                  \/ \E t \in tasks, ok \in BOOLEAN : ImportEnd(t, ok)
               /\ UNCHANGED rare)
    => IndInv'
  BY LendDue_Inv, Import_Inv, Borrow_Inv, GiveBack_Inv, ReleaseBorrow_Inv, ImportEnd_Inv

=============================================================================
