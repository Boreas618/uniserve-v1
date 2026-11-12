------------------------------- MODULE KvCore -------------------------------
(***************************************************************************)
(* The design model of the buffer pool of specs/buffer-pool.md: who holds  *)
(* each block, with the buffer pool's request handlers, the work it defers *)
(* until a worker group's `completed` word reaches a stamp, and the        *)
(* neighbours it relies on: an executor that meets the assumptions A1 to   *)
(* A4 with batches of unbounded delay, and a control plane that submits    *)
(* any request in any order.                                               *)
(*                                                                         *)
(* A block is one unit; a slot row is a unit of its group's slot region.   *)
(* Holders are sets: a unit is free, or it is held by owners' table        *)
(* entries, the cache, deferred work and copies. Each handler is one atomic       *)
(* action, because the buffer pool is one value changed through one        *)
(* thread; what is not atomic is three actions: taken, due, finished.      *)
(*                                                                         *)
(* Each rule R1 to R10 of §9 is one guard, in force when its name is in    *)
(* `Rules`; a configuration that leaves one out is that rule's mutation.   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Units,       \* the units of the device KV region
    Slots,       \* the slot rows of every worker group
    HostUnits,   \* the units of the host pool
    Keys,        \* the keys a block may be cached under
    Owners,      \* owner identifiers; the buffer pool takes each once
    Groups,      \* worker groups, each of one engine epoch
    Positions,   \* the positions of an owner's table
    Epochs,      \* engine epochs: the clients, one session each
    SlotGroup,   \* [Slots -> Groups]: the group a slot row belongs to
    GroupEpoch,  \* [Groups -> Epochs]: the epoch a group belongs to
    OwnerEpoch,  \* [Owners -> Epochs]: the client an owner identifier names
    MaxBatch,    \* the batches a group launches (a bound of the model)
    MaxRequests, \* the stamped requests the control plane has under way (a bound)
    MaxCopies,   \* the copies the buffer pool is asked for (a bound)
    Eager,       \* whether a completion is handed to its executor when it is produced
    Rules        \* the rules in force, a subset of {"R1", ..., "R10"}

VARIABLES
    \* The buffer pool.
    free,        \* the units no block holds
    ostate,      \* [Owners -> {"new", "live", "gone"}]
    ogroup,      \* [Owners -> Groups \cup {NONE}]: the group a live owner is attached to
    oslot,       \* [Owners -> Slots \cup {NONE}]: its slot row
    table,       \* [Owners -> [Positions -> entry]]: its table of the KV class
    cache,       \* [Keys -> Units \cup {NONE}]: the device block cached under a key
    hcache,      \* [Keys -> HostUnits \cup {NONE}]: the host block cached under a key
    deferred,    \* work that waits for stamps: releases, commits and the ends of copies
    copies,      \* the copies in flight, each holding the block it reads and the one it writes
    ncopy,       \* the copies ever asked for
    \* The control plane: the executors and the requests under way.
    dbt,         \* [Groups -> [Owners -> [Rows -> entry]]]: the device block tables
    launched,    \* [Groups -> 0 .. MaxBatch]: the sequence of the last batch launched
    inflight,    \* the batches launched and not ended
    handoff,     \* completions with runs that no executor was handed yet
    requests,    \* stamped requests: being stamped, or submitted and not taken
    \* History, assigned by actions and read by no guard.
    alloc,       \* [AllUnits -> Owners \cup {NONE, "shared"}]: whom a held unit was allocated to
    sharedKey    \* [AllUnits -> Keys \cup {NONE}]: the key a unit became shared under

pool == <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy>>
plane == <<dbt, launched, inflight, handoff, requests>>
vars == <<pool, plane, alloc, sharedKey>>

NONE == "none"
SLOT == "slot"                       \* the row of an owner's slot in an executor's tables
AllUnits == Units \cup Slots \cup HostUnits
Rows == Positions \cup {SLOT}
On(rule) == rule \in Rules

Empty == [k |-> "none", u |-> NONE]
Private(u) == [k |-> "private", u |-> u]
Shared(u) == [k |-> "shared", u |-> u]
NoEntry == [u |-> NONE, w |-> FALSE]
NoStamp == [g \in Groups |-> 0]

Min(S) == CHOOSE x \in S : \A y \in S : x <= y

(* A3: `completed` of a group is the last batch whose end its executor has *)
(* seen with every earlier batch ended.                                    *)
Completed(g) ==
    LET mine == {b.seq : b \in {c \in inflight : c.g = g}}
    IN IF mine = {} THEN launched[g] ELSE Min(mine) - 1

Reached(after) == \A g \in Groups : Completed(g) >= after[g]

-----------------------------------------------------------------------------
(* Holders. *)

HeldIn(u, tbl, slt, cch, hch, dfr, cps) ==
    \/ \E o \in Owners, p \in Positions : tbl[o][p].u = u
    \/ \E o \in Owners : slt[o] = u
    \/ \E k \in Keys : cch[k] = u \/ hch[k] = u
    \/ \E d \in dfr : u \in d.units
    \/ \E c \in cps : u = c.read \/ u = c.write

(* R2: runs return to a region only when the last reference is dropped.    *)
Reclaimed(dropped, tbl, slt, cch, hch, dfr, cps) ==
    IF On("R2") THEN {u \in dropped : ~HeldIn(u, tbl, slt, cch, hch, dfr, cps)} ELSE dropped

Reset(f, gone) == [u \in AllUnits |-> IF u \in gone THEN NONE ELSE f[u]]

(* Drop references to `units` in a state whose tables, slots, caches,      *)
(* deferred work and copies are given: what nothing holds any more is      *)
(* reclaimed.                                                              *)
Drop(units, tbl, slt, cch, hch, dfr, cps) ==
    LET gone == Reclaimed(units, tbl, slt, cch, hch, dfr, cps)
    IN /\ free' = free \cup gone
       /\ alloc' = Reset(alloc, gone)
       /\ sharedKey' = Reset(sharedKey, gone)

Pinned(u) == \E o \in Owners, p \in Positions : table[o][p] = Shared(u)

(* R7: eviction takes a shared block without a pin; R2: and with no other  *)
(* holder.                                                                 *)
Evictable(u) ==
    /\ \E k \in Keys : cache[k] = u
    /\ On("R7") => ~Pinned(u)
    /\ On("R2") => ~\E d \in deferred : u \in d.units
    /\ On("R2") => ~\E c \in copies : u = c.read \/ u = c.write

(* A device unit for a new private block: a free one, or when the region   *)
(* has none an evictable one, which leaves the cache.                      *)
Obtain(u) ==
    \/ u \in free /\ cache' = cache
    \/ /\ free \cap Units = {}
       /\ Evictable(u)
       /\ cache' = [k \in Keys |-> IF cache[k] = u THEN NONE ELSE cache[k]]

(* Whether a copy writes for position p of owner o.                        *)
CopyFor(o, p) == \E c \in copies : c.o = o /\ c.p = p

(* Whether stamped request r takes away the entry completion h fills.      *)
Names(r, h) ==
    /\ r.e = OwnerEpoch[h.o]
    /\ r.o = h.o
    /\ CASE r.kind = "release" -> h.p \in r.ps
         [] r.kind = "detach" -> TRUE
         [] r.kind = "commit" -> h.p = r.p
         [] OTHER -> FALSE

(* A completion with runs leaves the buffer pool. The control plane hands  *)
(* it to its executor now, or later, or drops it; it drops it when a       *)
(* stamped request under way names its entry (A1). With `Eager` it is      *)
(* handed on at once, which is the case that addresses the most units:     *)
(* an executor's table then holds at every moment at least what it holds   *)
(* when completions are handed on later or dropped.                        *)
Give(h) ==
    IF Eager
    THEN /\ handoff' = handoff
         /\ dbt' = IF \E r \in requests : Names(r, h)
                   THEN dbt
                   ELSE [dbt EXCEPT ![h.g][h.o][h.p] = [u |-> h.u, w |-> h.w]]
    ELSE /\ handoff' = handoff \cup {h}
         /\ dbt' = dbt

-----------------------------------------------------------------------------
(* The buffer pool's handlers of requests that carry no stamp.             *)

Attach(o, g) ==
    /\ On("R1") => ostate[o] = "new" /\ GroupEpoch[g] = OwnerEpoch[o]
    /\ \E s \in free \cap Slots :
         /\ SlotGroup[s] = g
         /\ free' = free \ {s}
         /\ oslot' = [oslot EXCEPT ![o] = s]
         /\ alloc' = [alloc EXCEPT ![s] = o]
         /\ Give([g |-> g, o |-> o, p |-> SLOT, u |-> s, w |-> TRUE])
    /\ ostate' = [ostate EXCEPT ![o] = "live"]
    /\ ogroup' = [ogroup EXCEPT ![o] = g]
    /\ table' = [table EXCEPT ![o] = [p \in Positions |-> Empty]]
    /\ UNCHANGED <<cache, hcache, deferred, copies, ncopy, launched, inflight, requests, sharedKey>>

Allocate(o, p) ==
    /\ ostate[o] = "live"
    /\ On("R1") => table[o][p] = Empty /\ ~CopyFor(o, p)
    /\ \E u \in Units :
         /\ Obtain(u)
         /\ free' = free \ {u}
         /\ table' = [table EXCEPT ![o][p] = Private(u)]
         /\ alloc' = [alloc EXCEPT ![u] = o]
         /\ sharedKey' = [sharedKey EXCEPT ![u] = NONE]
         /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> u, w |-> TRUE])
    /\ UNCHANGED <<ostate, ogroup, oslot, hcache, deferred, copies, ncopy, launched, inflight,
                   requests>>

Pin(o, p, k) ==
    /\ ostate[o] = "live"
    /\ cache[k] # NONE
    /\ On("R1") => table[o][p] = Empty /\ ~CopyFor(o, p)
    /\ table' = [table EXCEPT ![o][p] = Shared(cache[k])]
    /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> cache[k], w |-> FALSE])
    /\ UNCHANGED <<free, ostate, ogroup, oslot, cache, hcache, deferred, copies, ncopy, launched,
                   inflight, requests, alloc, sharedKey>>

Evict(k) ==
    /\ cache[k] # NONE
    /\ Evictable(cache[k])
    /\ cache' = [cache EXCEPT ![k] = NONE]
    /\ Drop({cache[k]}, table, oslot, cache', hcache, deferred, copies)
    /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, copies, ncopy, plane>>

(* A host copy leaves the host tier when nothing but the cache holds it.   *)
EvictHost(k) ==
    /\ hcache[k] # NONE
    /\ On("R2") => ~\E c \in copies : hcache[k] = c.read \/ hcache[k] = c.write
    /\ hcache' = [hcache EXCEPT ![k] = NONE]
    /\ Drop({hcache[k]}, table, oslot, cache, hcache', deferred, copies)
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, deferred, copies, ncopy, plane>>

-----------------------------------------------------------------------------
(* Copies the executor runs. The handler allocates the destination and the *)
(* copy holds the block it reads and the block it writes until its         *)
(* `Finish` takes effect.                                                  *)

Copy(kind, o, p, k, g, read, write) ==
    [id |-> ncopy + 1, kind |-> kind, o |-> o, p |-> p, k |-> k, g |-> g,
     read |-> read, write |-> write,
     seq |-> 0,           \* the batch it ran with; 0 before it starts
     done |-> FALSE,      \* whether its executor returned its `Finish`
     stopped |-> FALSE]   \* whether the scheduler gave it up

(* R5: a clone is a new private block that the executor fills from the     *)
(* cached one. Without the rule the cached block itself becomes private.   *)
Clone(o, p, k) ==
    /\ ncopy < MaxCopies
    /\ ostate[o] = "live"
    /\ cache[k] # NONE
    /\ On("R1") => table[o][p] = Empty /\ ~CopyFor(o, p)
    /\ IF On("R5")
       THEN /\ \E w \in Units \ {cache[k]} :
                 /\ Obtain(w)
                 /\ free' = free \ {w}
                 /\ alloc' = [alloc EXCEPT ![w] = o]
                 /\ sharedKey' = [sharedKey EXCEPT ![w] = NONE]
                 /\ copies' = copies \cup {Copy("clone", o, p, k, ogroup[o], cache[k], w)}
            /\ ncopy' = ncopy + 1
            /\ UNCHANGED <<table, dbt, handoff>>
       ELSE /\ table' = [table EXCEPT ![o][p] = Private(cache[k])]
            /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> cache[k], w |-> TRUE])
            /\ UNCHANGED <<free, cache, copies, ncopy, alloc, sharedKey>>
    /\ UNCHANGED <<ostate, ogroup, oslot, hcache, deferred, launched, inflight, requests>>

Load(o, p, k) ==
    /\ ncopy < MaxCopies
    /\ ostate[o] = "live"
    /\ hcache[k] # NONE
    /\ cache[k] = NONE
    /\ On("R1") => table[o][p] = Empty /\ ~CopyFor(o, p)
    /\ \E w \in Units :
         /\ Obtain(w)
         /\ free' = free \ {w}
         /\ alloc' = [alloc EXCEPT ![w] = o]
         /\ sharedKey' = [sharedKey EXCEPT ![w] = NONE]
         /\ copies' = copies \cup {Copy("load", o, p, k, ogroup[o], hcache[k], w)}
    /\ ncopy' = ncopy + 1
    /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, plane>>

Store(g, k) ==
    /\ ncopy < MaxCopies
    /\ cache[k] # NONE
    /\ hcache[k] = NONE
    /\ \E h \in free \cap HostUnits :
         /\ free' = free \ {h}
         /\ alloc' = [alloc EXCEPT ![h] = "copy"]
         /\ copies' = copies \cup {Copy("store", NONE, NONE, k, g, cache[k], h)}
    /\ ncopy' = ncopy + 1
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, plane, sharedKey>>

-----------------------------------------------------------------------------
(* The control plane issues a stamped request; the executor of every group *)
(* whose device block tables may hold the entries concerned applies and    *)
(* stamps it (A2).                                                         *)

Request(kind, e, o, ps, p, k, c, ok) ==
    [kind |-> kind, e |-> e, o |-> o, ps |-> ps, p |-> p, k |-> k, c |-> c, ok |-> ok,
     after |-> NoStamp, by |-> {}]

Issue ==
    /\ Cardinality(requests) < MaxRequests
    /\ \E e \in Epochs :
         \/ \E o \in Owners :
              \/ \E ps \in (SUBSET Positions) \ {{}} :
                   requests' = requests \cup {Request("release", e, o, ps, NONE, NONE, 0, FALSE)}
              \/ requests' = requests \cup {Request("detach", e, o, {}, NONE, NONE, 0, FALSE)}
              \/ \E p \in Positions, k \in Keys :
                   requests' = requests \cup {Request("commit", e, o, {}, p, k, 0, FALSE)}
         \* The scheduler gives a copy up.
         \/ \E c \in 1 .. ncopy :
              requests' = requests \cup {Request("finish", e, NONE, {}, NONE, NONE, c, FALSE)}
    /\ UNCHANGED <<pool, dbt, launched, inflight, handoff, alloc, sharedKey>>

(* A2: the groups that stamp r: every group of r's session whose device     *)
(* block tables may hold what r concerns, which is the group its owner is  *)
(* attached to, or its copy's group. Each stamps whether or not its table  *)
(* still holds the entries.                                                *)
StampGroups(r) ==
    {g \in Groups :
        /\ GroupEpoch[g] = r.e
        /\ IF r.kind = "finish"
           THEN \E c \in copies : c.id = r.c /\ c.g = g
           ELSE ogroup[r.o] = g}

(* The executor applies r: it removes the entries of a release or a detach, *)
(* marks the entry of a commit read-only, and attaches a copy that is      *)
(* given up to no further batch. Then it stamps r with the last batch it   *)
(* launched. A give-up for a copy whose own `Finish` it has returned is    *)
(* discarded.                                                              *)
ExecStamp(g, r) ==
    /\ r \in requests
    /\ g \in StampGroups(r) \ r.by
    /\ IF r.kind = "finish"
       THEN /\ dbt' = dbt
            /\ IF \E c \in copies : c.id = r.c /\ c.done
               THEN /\ requests' = requests \ {r}
                    /\ copies' = copies
               ELSE /\ copies' = {IF c.id = r.c THEN [c EXCEPT !.stopped = TRUE] ELSE c : c \in copies}
                    /\ requests' = (requests \ {r}) \cup
                                     {[r EXCEPT !.after[g] = launched[g], !.by = @ \cup {g}]}
       ELSE /\ dbt' = [dbt EXCEPT ![g][r.o] =
                        [p \in Rows |->
                           CASE r.kind = "release" /\ p \in r.ps -> NoEntry
                             [] r.kind = "detach" -> NoEntry
                             [] r.kind = "commit" /\ p = r.p -> [u |-> dbt[g][r.o][p].u, w |-> FALSE]
                             [] OTHER -> dbt[g][r.o][p]]]
            /\ requests' = (requests \ {r}) \cup
                             {[r EXCEPT !.after[g] = launched[g], !.by = @ \cup {g}]}
            /\ copies' = copies
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, launched,
                   inflight, handoff, alloc, sharedKey>>

(* A2: a request is submitted once every such group has stamped it.        *)
Submittable(r) == StampGroups(r) \subseteq r.by

(* The completions the control plane still hands on once r is taken: those *)
(* of requests submitted before r that fill what r takes away are dropped. *)
Kept(r) == {h \in handoff : ~Names(r, h)}

(* A1: an executor writes only the runs a completion gave it, at the        *)
(* position the request named, and is handed a completion only before any  *)
(* stamped request that names its entry is handed to an executor.          *)
ExecMap(h) ==
    /\ h \in handoff
    /\ ~\E r \in requests : Names(r, h)
    /\ handoff' = handoff \ {h}
    /\ dbt' = [dbt EXCEPT ![h.g][h.o][h.p] = [u |-> h.u, w |-> h.w]]
    /\ UNCHANGED <<pool, launched, inflight, requests, alloc, sharedKey>>

(* The scheduler drops a completion instead of handing it on.              *)
DropCompletion(h) ==
    /\ h \in handoff
    /\ handoff' = handoff \ {h}
    /\ UNCHANGED <<pool, dbt, launched, inflight, requests, alloc, sharedKey>>

(* A4: a batch reads the entries of its group's table and writes the        *)
(* writable ones; the batch that addresses every entry stands for all. The *)
(* copies of the group that have not started run with it, each reading its *)
(* source and writing its destination, which nothing else writes.          *)
Launch(g) ==
    /\ launched[g] < MaxBatch
    /\ LET rows == {r \in Owners \X Rows : dbt[g][r[1]][r[2]].u # NONE}
           start == {c \in copies : c.g = g /\ c.seq = 0 /\ ~c.stopped}
       IN /\ rows # {} \/ start # {}
          /\ inflight' = inflight \cup
               {[g |-> g, seq |-> launched[g] + 1,
                 reads |-> {<<r[1], dbt[g][r[1]][r[2]].u>> : r \in rows},
                 writes |-> {<<r[1], dbt[g][r[1]][r[2]].u>> :
                               r \in {x \in rows : dbt[g][x[1]][x[2]].w}},
                 cr |-> {c.read : c \in start},
                 cw |-> {c.write : c \in start}]}
          /\ copies' = {IF c \in start THEN [c EXCEPT !.seq = launched[g] + 1] ELSE c : c \in copies}
    /\ launched' = [launched EXCEPT ![g] = @ + 1]
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, dbt,
                   handoff, requests, alloc, sharedKey>>

(* A batch ends after any delay.                                           *)
EndBatch(b) ==
    /\ b \in inflight
    /\ inflight' = inflight \ {b}
    /\ UNCHANGED <<pool, dbt, launched, handoff, requests, alloc, sharedKey>>

(* The executor returns the `Finish` of a copy it started, with its stamp. *)
ExecFinish(c, ok) ==
    /\ c \in copies
    /\ c.seq > 0 /\ ~c.done /\ ~c.stopped
    /\ copies' = (copies \ {c}) \cup {[c EXCEPT !.done = TRUE]}
    /\ requests' = requests \cup
         {[Request("finish", GroupEpoch[c.g], NONE, {}, NONE, NONE, c.id, ok)
             EXCEPT !.after[c.g] = launched[c.g], !.by = {c.g}]}
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, dbt,
                   launched, inflight, handoff, alloc, sharedKey>>

-----------------------------------------------------------------------------
(* The buffer pool takes a stamped request.                                *)

(* R3: what a stamped request takes away is released when `completed` of   *)
(* every stamping group has reached its stamp.                             *)
ReleaseUnits(units, after, tbl, slt) ==
    /\ table' = tbl
    /\ oslot' = slt
    /\ IF On("R3") /\ ~Reached(after) /\ units # {}
       THEN /\ deferred' = deferred \cup
                 {[kind |-> "release", units |-> units, after |-> after,
                   o |-> NONE, p |-> NONE, k |-> NONE, c |-> 0, ok |-> FALSE]}
            /\ UNCHANGED <<free, alloc, sharedKey>>
       ELSE /\ Drop(units, tbl, slt, cache, hcache, deferred, copies)
            /\ UNCHANGED deferred

TakeRelease(r) ==
    /\ r \in requests /\ r.kind = "release" /\ Submittable(r)
    /\ requests' = requests \ {r}
    /\ IF On("R1") /\ ~(/\ ostate[r.o] = "live"
                        /\ OwnerEpoch[r.o] = r.e
                        /\ \A p \in r.ps : table[r.o][p] # Empty)
       THEN UNCHANGED <<pool, alloc, sharedKey>>
       ELSE /\ ReleaseUnits({table[r.o][p].u : p \in r.ps} \ {NONE}, r.after,
                            [table EXCEPT ![r.o] =
                               [p \in Positions |-> IF p \in r.ps THEN Empty ELSE table[r.o][p]]],
                            oslot)
            /\ UNCHANGED <<ostate, ogroup, cache, hcache, copies, ncopy>>
    /\ handoff' = Kept(r)
    /\ UNCHANGED <<dbt, launched, inflight>>

TakeDetach(r) ==
    /\ r \in requests /\ r.kind = "detach" /\ Submittable(r)
    /\ requests' = requests \ {r}
    /\ IF On("R1") /\ ~(ostate[r.o] = "live" /\ OwnerEpoch[r.o] = r.e)
       THEN UNCHANGED <<pool, alloc, sharedKey>>
       ELSE /\ ReleaseUnits(({table[r.o][p].u : p \in Positions} \cup {oslot[r.o]}) \ {NONE},
                            r.after,
                            [table EXCEPT ![r.o] = [p \in Positions |-> Empty]],
                            [oslot EXCEPT ![r.o] = NONE])
            /\ ostate' = [ostate EXCEPT ![r.o] = IF @ = "live" THEN "gone" ELSE @]
            /\ UNCHANGED <<ogroup, cache, hcache, copies, ncopy>>
    /\ handoff' = Kept(r)
    /\ UNCHANGED <<dbt, launched, inflight>>

(* R6: a commit waits for its stamps holding the block it was taken for.   *)
(* Without the rule it holds nothing and names the position alone.         *)
TakeCommit(r) ==
    /\ r \in requests /\ r.kind = "commit" /\ Submittable(r)
    /\ requests' = requests \ {r}
    /\ IF ~(/\ ostate[r.o] = "live"
            /\ table[r.o][r.p].k = "private"
            /\ On("R1") => /\ OwnerEpoch[r.o] = r.e
                           /\ ~\E d \in deferred : d.kind = "commit" /\ d.o = r.o /\ d.p = r.p)
       THEN UNCHANGED deferred
       ELSE deferred' = deferred \cup
              {[kind |-> "commit",
                units |-> IF On("R6") THEN {table[r.o][r.p].u} ELSE {},
                after |-> r.after,
                o |-> r.o, p |-> r.p, k |-> r.k, c |-> 0, ok |-> FALSE]}
    /\ handoff' = Kept(r)
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, dbt, launched,
                   inflight, alloc, sharedKey>>

(* One `Finish` ends a copy: R1 refuses one that names no copy in flight,  *)
(* a copy of another session, or a copy whose `Finish` already waits.      *)
TakeFinish(r) ==
    /\ r \in requests /\ r.kind = "finish" /\ Submittable(r)
    /\ requests' = requests \ {r}
    /\ IF ~(/\ \E c \in copies : c.id = r.c /\ (On("R1") => GroupEpoch[c.g] = r.e)
            /\ On("R1") => ~\E d \in deferred : d.kind = "finish" /\ d.c = r.c)
       THEN UNCHANGED deferred
       ELSE deferred' = deferred \cup
              {[kind |-> "finish", units |-> {}, after |-> r.after,
                o |-> NONE, p |-> NONE, k |-> NONE, c |-> r.c, ok |-> r.ok]}
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, dbt, launched,
                   inflight, handoff, alloc, sharedKey>>

-----------------------------------------------------------------------------
(* Deferred work comes due.                                                *)

ReleaseDue(d) ==
    /\ d \in deferred /\ d.kind = "release"
    /\ Reached(d.after)
    /\ deferred' = deferred \ {d}
    /\ Drop(d.units, table, oslot, cache, hcache, deferred', copies)
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, plane>>

CommitDue(d) ==
    /\ d \in deferred /\ d.kind = "commit"
    /\ On("R3") => Reached(d.after)
    /\ deferred' = deferred \ {d}
    /\ LET \* The block the commit shares: the one it holds, or without R6
           \* whatever private block its position holds when it comes due.
           u == IF On("R6") THEN CHOOSE x \in d.units : TRUE
                ELSE IF table[d.o][d.p].k = "private" THEN table[d.o][d.p].u ELSE NONE
       IN IF u = NONE
          THEN UNCHANGED <<free, table, cache, alloc, sharedKey>>
          ELSE IF On("R4") /\ cache[d.k] # NONE
          \* R4: a key that is cached keeps its block; this one stays private.
          THEN /\ Drop(d.units, table, oslot, cache, hcache, deferred', copies)
               /\ UNCHANGED <<table, cache>>
          ELSE /\ cache' = [cache EXCEPT ![d.k] = u]
               /\ table' = [table EXCEPT ![d.o][d.p] = IF @ = Private(u) THEN Shared(u) ELSE @]
               /\ free' = free
               /\ alloc' = [alloc EXCEPT ![u] = "shared"]
               /\ sharedKey' = [sharedKey EXCEPT ![u] = IF @ = NONE THEN d.k ELSE @]
    /\ UNCHANGED <<ostate, ogroup, oslot, hcache, copies, ncopy, plane>>

(* The end of a copy takes effect (R3: once `completed` of its group has   *)
(* reached the stamp of its `Finish`, so the copy has ended).              *)
FinishDue(d) ==
    /\ d \in deferred /\ d.kind = "finish"
    /\ On("R3") => Reached(d.after)
    /\ deferred' = deferred \ {d}
    /\ \E c \in copies :
         /\ c.id = d.c
         /\ copies' = copies \ {c}
         /\ LET target == c.o # NONE /\ ostate[c.o] = "live" /\ table[c.o][c.p] = Empty
            IN CASE ~d.ok ->
                      \* Nothing was copied: the destination retires.
                      /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                      /\ UNCHANGED <<table, cache, hcache, dbt, handoff>>
                 [] d.ok /\ c.kind = "clone" ->
                      \* The owner holds a private block with the cached content.
                      IF target
                      THEN /\ table' = [table EXCEPT ![c.o][c.p] = Private(c.write)]
                           /\ Give([g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write, w |-> TRUE])
                           /\ Drop({c.read}, table', oslot, cache, hcache, deferred', copies')
                           /\ UNCHANGED <<cache, hcache>>
                      ELSE /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                           /\ UNCHANGED <<table, cache, hcache, dbt, handoff>>
                 [] d.ok /\ c.kind = "load" ->
                      IF On("R4") /\ cache[c.k] # NONE
                      \* R4: the key gained a device copy meanwhile; this block
                      \* stays private to the owner, which only reads it.
                      THEN IF target
                           THEN /\ table' = [table EXCEPT ![c.o][c.p] = Private(c.write)]
                                /\ Give([g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write, w |-> FALSE])
                                /\ Drop({c.read}, table', oslot, cache, hcache, deferred', copies')
                                /\ UNCHANGED <<cache, hcache>>
                           ELSE /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred',
                                        copies')
                                /\ UNCHANGED <<table, cache, hcache, dbt, handoff>>
                      \* The block is shared on the device and pinned for the owner.
                      ELSE /\ cache' = [cache EXCEPT ![c.k] = c.write]
                           /\ IF target
                              THEN /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
                                   /\ Give([g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write,
                                            w |-> FALSE])
                              ELSE UNCHANGED <<table, dbt, handoff>>
                           /\ free' = free
                           /\ alloc' = [alloc EXCEPT ![c.write] = "shared"]
                           /\ sharedKey' = [sharedKey EXCEPT ![c.write] = c.k]
                           /\ UNCHANGED hcache
                 [] d.ok /\ c.kind = "store" ->
                      IF On("R4") /\ hcache[c.k] # NONE
                      THEN /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                           /\ UNCHANGED <<table, cache, hcache, dbt, handoff>>
                      ELSE /\ hcache' = [hcache EXCEPT ![c.k] = c.write]
                           /\ free' = free
                           /\ alloc' = [alloc EXCEPT ![c.write] = "shared"]
                           /\ sharedKey' = [sharedKey EXCEPT ![c.write] = c.k]
                           /\ UNCHANGED <<table, cache, dbt, handoff>>
    /\ UNCHANGED <<ostate, ogroup, oslot, ncopy, launched, inflight, requests>>

-----------------------------------------------------------------------------

Init ==
    /\ free = AllUnits
    /\ ostate = [o \in Owners |-> "new"]
    /\ ogroup = [o \in Owners |-> NONE]
    /\ oslot = [o \in Owners |-> NONE]
    /\ table = [o \in Owners |-> [p \in Positions |-> Empty]]
    /\ cache = [k \in Keys |-> NONE]
    /\ hcache = [k \in Keys |-> NONE]
    /\ deferred = {}
    /\ copies = {}
    /\ ncopy = 0
    /\ dbt = [g \in Groups |-> [o \in Owners |-> [p \in Rows |-> NoEntry]]]
    /\ launched = [g \in Groups |-> 0]
    /\ inflight = {}
    /\ handoff = {}
    /\ requests = {}
    /\ alloc = [u \in AllUnits |-> NONE]
    /\ sharedKey = [u \in AllUnits |-> NONE]

Next ==
    \/ \E o \in Owners, g \in Groups : Attach(o, g)
    \/ \E o \in Owners, p \in Positions : Allocate(o, p)
    \/ \E o \in Owners, p \in Positions, k \in Keys : Pin(o, p, k) \/ Clone(o, p, k) \/ Load(o, p, k)
    \/ \E g \in Groups, k \in Keys : Store(g, k)
    \/ \E k \in Keys : Evict(k) \/ EvictHost(k)
    \/ Issue
    \/ \E g \in Groups, r \in requests : ExecStamp(g, r)
    \/ \E h \in handoff : ExecMap(h) \/ DropCompletion(h)
    \/ \E g \in Groups : Launch(g)
    \/ \E b \in inflight : EndBatch(b)
    \/ \E c \in copies, ok \in BOOLEAN : ExecFinish(c, ok)
    \/ \E r \in requests : TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r)
    \/ \E d \in deferred : ReleaseDue(d) \/ CommitDue(d) \/ FinishDue(d)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* The properties. Their statements are those of specs/buffer-pool.md §17. *)

Held(u) == HeldIn(u, table, oslot, cache, hcache, deferred, copies)
PrivateHolders(u) ==
    {x \in Owners \X Positions : table[x[1]][x[2]] = Private(u)}
        \cup {x \in Owners \X {SLOT} : oslot[x[1]] = u}
        \cup {<<c.id, "copy">> : c \in {x \in copies : x.write = u}}
CacheKeys(u) == {k \in Keys : cache[k] = u \/ hcache[k] = u}

(* P1 Isolation, the buffer pool's part: a free unit has no holder, and a  *)
(* private block has one owner or one copy that writes it and is neither   *)
(* pinned nor cached.                                                      *)
FreeUnheld == \A u \in free : ~Held(u)
Exclusive ==
    \A u \in AllUnits :
        /\ Cardinality(PrivateHolders(u)) <= 1
        /\ PrivateHolders(u) # {} => ~Pinned(u) /\ CacheKeys(u) = {}

(* P1 Isolation, with A1 to A4: a write lands only on a block allocated to *)
(* the writing request, nothing a batch in flight addresses was reclaimed  *)
(* or given to another owner, and a copy in flight still holds what it     *)
(* reads and writes.                                                       *)
WriteIsolation ==
    \A b \in inflight : \A w \in b.writes : w[2] \notin free /\ alloc[w[2]] = w[1]
ReadIsolation ==
    \A b \in inflight : \A r \in b.reads :
        r[2] \notin free /\ alloc[r[2]] \in {r[1], "shared"}
CopyIsolation ==
    \A b \in inflight :
        /\ \A u \in b.cw : \E c \in copies : c.write = u /\ c.g = b.g /\ c.seq = b.seq
        /\ \A u \in b.cr : \E c \in copies : c.read = u /\ c.g = b.g /\ c.seq = b.seq

P1 == FreeUnheld /\ Exclusive /\ WriteIsolation /\ ReadIsolation /\ CopyIsolation

(* P2 Integrity: a shared block is never private again, keeps its key, and *)
(* is written neither by a batch nor by a copy; a copy reads a shared      *)
(* block.                                                                  *)
KeyStable ==
    \A u \in AllUnits :
        sharedKey[u] # NONE =>
            /\ u \notin free
            /\ cache[sharedKey[u]] = u \/ hcache[sharedKey[u]] = u
            /\ PrivateHolders(u) = {}
PinsCached ==
    \A o \in Owners, p \in Positions :
        table[o][p].k = "shared" => sharedKey[table[o][p].u] # NONE
OneKey == \A u \in AllUnits : Cardinality(CacheKeys(u)) <= 1
(* A block is committed only after `completed` reached the stamp of its    *)
(* commit, and the destination of a copy becomes shared only after the     *)
(* stamp of its `Finish`: nothing in flight writes a shared block.         *)
SharedUnwritten ==
    \A b \in inflight :
        /\ \A w \in b.writes : alloc[w[2]] # "shared"
        /\ \A u \in b.cw : alloc[u] # "shared"
CopiesReadShared == \A b \in inflight : \A u \in b.cr : alloc[u] = "shared"

P2 == KeyStable /\ PinsCached /\ OneKey /\ SharedUnwritten /\ CopiesReadShared

(* P3 Conservation: every unit is free or held, and when no owner exists,  *)
(* nothing is deferred and no copy is in flight every block is free or     *)
(* shared without a pin.                                                   *)
NoLeak == \A u \in AllUnits : u \in free \/ Held(u)
Quiescent ==
    (/\ \A o \in Owners : ostate[o] # "live"
     /\ deferred = {}
     /\ copies = {})
        => \A u \in AllUnits : u \in free \/ (CacheKeys(u) # {} /\ ~Pinned(u))

P3 == NoLeak /\ Quiescent

TypeOK ==
    /\ free \subseteq AllUnits
    /\ ostate \in [Owners -> {"new", "live", "gone"}]
    /\ ogroup \in [Owners -> Groups \cup {NONE}]
    /\ oslot \in [Owners -> Slots \cup {NONE}]
    /\ \A o \in Owners, p \in Positions :
         /\ table[o][p].k \in {"none", "private", "shared"}
         /\ table[o][p].u \in Units \cup {NONE}
    /\ cache \in [Keys -> Units \cup {NONE}]
    /\ hcache \in [Keys -> HostUnits \cup {NONE}]
    /\ ncopy \in 0 .. MaxCopies
    /\ launched \in [Groups -> 0 .. MaxBatch]
    /\ alloc \in [AllUnits -> Owners \cup {NONE, "shared", "copy"}]
    /\ sharedKey \in [AllUnits -> Keys \cup {NONE}]

Safety == TypeOK /\ P1 /\ P2 /\ P3
=============================================================================
