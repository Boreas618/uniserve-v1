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
(* entries, the cache, deferred work, copies and tasks. Each handler is one atomic       *)
(* action, because the buffer pool is one value changed through one        *)
(* thread; what is not atomic is three actions: taken, due, finished.      *)
(*                                                                         *)
(* Each rule R1 to R10 of §9 is one guard, in force when its name is in    *)
(* `Rules`; a configuration that leaves one out is that rule's mutation.   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    \* @type: Set(Str);
    Units,       \* the units of the device KV region
    \* @type: Set(Str);
    Slots,       \* the slot rows of every worker group
    \* @type: Set(Str);
    HostUnits,   \* the units of the host pool
    \* @type: Set(Str);
    Keys,        \* the keys a block may be cached under
    \* @type: Set(Str);
    Owners,      \* owner identifiers; the buffer pool takes each once
    \* @type: Set(Str);
    Groups,      \* worker groups, each of one engine epoch
    \* @type: Set(Str);
    Positions,   \* the positions of an owner's table
    \* @type: Set(Str);
    Epochs,      \* engine epochs: the clients, one session each
    \* @type: Str -> Str;
    SlotGroup,   \* [Slots -> Groups]: the group a slot row belongs to
    \* @type: Str -> Str;
    GroupEpoch,  \* [Groups -> Epochs]: the epoch a group belongs to
    \* @type: Str -> Str;
    OwnerEpoch,  \* [Owners -> Epochs]: the client an owner identifier names
    \* @type: Str -> Str;
    GroupMemory, \* [Groups -> memories]: the device memory a group's ranks address; the
                 \* groups of two epochs of one layout that name the same ranks map the same
    \* @type: Set(Str);
    TrialEpochs, \* the epochs on trial at the start
    \* @type: Str -> Str;
    UnitRegion,  \* [Units -> regions]: the region a device unit belongs to
    \* @type: Str -> Str;
    UnitChunk,   \* [Units -> chunks]: the device memory a unit occupies; units of the regions
                 \* of two layouts may occupy the same
    \* @type: Str -> Str;
    GroupRegion, \* [Groups -> regions]: the KV region of a group's scheduler
    \* @type: Set(Str);
    InitialUnits, \* the device units that are in their regions at the start
    \* @type: Set(Str);
    Movable,     \* the device units `Resize` may take out of their region or put into it
    \* @type: Int;
    MaxBatch,    \* the batches a group launches (a bound of the model)
    \* @type: Int;
    MaxRequests, \* the stamped requests the control plane has under way (a bound)
    \* @type: Int;
    MaxCopies,   \* the copies the buffer pool is asked for (a bound)
    \* @type: Int;
    MaxTasks,    \* the prefetches and remote stores in flight at once (a bound)
    \* @type: Set(Str);
    Losable,     \* the epochs whose engine may be lost
    \* @type: Set(Str);
    Hazards,     \* the hazards of the system as built that are switched on
    \* @type: Bool;
    Eager,       \* whether a completion is handed to its executor when it is produced
    \* @type: Set(Str);
    Rules        \* the rules in force, a subset of {"R1", ..., "R10"}

VARIABLES
    \* The buffer pool.
    \* @type: Set(Str);
    free,        \* the units no block holds
    \* @type: Str -> Str;
    ostate,      \* [Owners -> {"new", "live", "gone"}]
    \* @type: Str -> Str;
    ogroup,      \* [Owners -> Groups \cup {NONE}]: the group a live owner is attached to
    \* @type: Str -> Str;
    oslot,       \* [Owners -> Slots \cup {NONE}]: its slot row
    \* @type: Str -> (Str -> {k: Str, u: Str});
    table,       \* [Owners -> [Positions -> entry]]: its table of the KV class
    \* @type: Str -> Str;
    cache,       \* [Keys -> Units \cup {NONE}]: the device block cached under a key
    \* @type: Str -> Str;
    hcache,      \* [Keys -> HostUnits \cup {NONE}]: the host block cached under a key
    \* @type: Set({kind: Str, units: Set(Str), after: Str -> Int, o: Str, p: Str, k: Str, c: Int, ok: Bool, g: Str, to: Str});
    deferred,    \* work that waits for stamps: releases, commits and the ends of copies
    \* @type: Set({id: Int, kind: Str, o: Str, p: Str, k: Str, g: Str, read: Str, write: Str, fill: Bool, seq: Int, done: Bool, stopped: Bool});
    copies,      \* the copies in flight, each holding the block it reads and the one it writes
    \* @type: Int;
    ncopy,       \* the copies ever asked for
    \* @type: Set({kind: Str, k: Str, h: Str, src: Str, ok: Bool, expired: Bool, o: Str, to: Str, slot: Str, held: Bool, units: Set(Str), writes: Set(Str)});
    tasks,       \* the transfers in flight, each holding the blocks it copies
    \* @type: Str -> Set(Str);
    valid,       \* [Units -> SUBSET memories]: the memories that hold a block's content
    \* @type: Str -> {d: Str, h: Str};
    vis,         \* [Keys -> [d, h]]: per key, the epoch on trial its device copy and its host
                 \* copy are visible to alone, or NONE for a copy every epoch may use
    \* @type: Set(Str);
    member,      \* the device units that are in their region; the others' memory is the
                 \* deployment's free quota on the device, or another region's
    \* @type: Str -> Str;
    trial,       \* [Epochs -> {"no", "yes", "rolled"}]: whether an epoch is on trial, or was
                 \* and its update was rolled back
    \* The control plane: the executors and the requests under way.
    \* @type: Str -> (Str -> (Str -> {u: Str, w: Bool}));
    dbt,         \* [Groups -> [Owners -> [Rows -> entry]]]: the device block tables
    \* @type: Str -> Int;
    launched,    \* [Groups -> 0 .. MaxBatch]: the sequence of the last batch launched
    \* @type: Set({g: Str, seq: Int, reads: Set(<<Str, Str>>), writes: Set(<<Str, Str>>), cr: Set(Str), cw: Set(Str)});
    inflight,    \* the batches launched and not ended
    \* @type: Set({g: Str, o: Str, p: Str, u: Str, w: Bool});
    handoff,     \* completions with runs that no executor was handed yet
    \* @type: Set({kind: Str, e: Str, o: Str, ps: Set(Str), p: Str, k: Str, c: Int, ok: Bool, to: Str, after: Str -> Int, by: Set(Str), sg: Set(Str)});
    requests,    \* stamped requests: being stamped, or submitted and not taken
    \* History, assigned by actions and read by no guard.
    \* @type: Str -> Str;
    alloc,       \* [AllUnits -> Owners \cup {NONE, "shared"}]: whom a held unit was allocated to
    \* @type: Str -> Str;
    sharedKey,   \* [AllUnits -> Keys \cup {NONE}]: the key a unit became shared under
    \* Assigned only by the actions of tasks and of an engine's loss.
    \* @type: Set(Str);
    remote,      \* the keys the remote store acknowledged a copy of
    \* @type: Set(Str);
    lost,        \* the epochs whose engine's processes have exited
    \* @type: Set(<<Str, Str>>);
    bad          \* history: the copies cached from a source not held throughout, <<key, tier>>

pool == <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, tasks, valid,
          vis, trial, member>>
plane == <<dbt, launched, inflight, handoff, requests>>
rare == <<remote, lost, bad>>
vars == <<pool, plane, alloc, sharedKey, rare>>

\* @typeAlias: req = {kind: Str, e: Str, o: Str, ps: Set(Str), p: Str, k: Str, c: Int, ok: Bool, to: Str, after: Str -> Int, by: Set(Str), sg: Set(Str)};
\* @typeAlias: def = {kind: Str, units: Set(Str), after: Str -> Int, o: Str, p: Str, k: Str, c: Int, ok: Bool, g: Str, to: Str};
\* @typeAlias: copy = {id: Int, kind: Str, o: Str, p: Str, k: Str, g: Str, read: Str, write: Str, fill: Bool, seq: Int, done: Bool, stopped: Bool};
\* @typeAlias: task = {kind: Str, k: Str, h: Str, src: Str, ok: Bool, expired: Bool, o: Str, to: Str, slot: Str, held: Bool, units: Set(Str), writes: Set(Str)};
\* @typeAlias: batch = {g: Str, seq: Int, reads: Set(<<Str, Str>>), writes: Set(<<Str, Str>>), cr: Set(Str), cw: Set(Str)};
\* @typeAlias: hand = {g: Str, o: Str, p: Str, u: Str, w: Bool};
KvAliases == TRUE
NONE == "none"
SLOT == "slot"                       \* the row of an owner's slot in an executor's tables
AllUnits == Units \cup Slots \cup HostUnits
Rows == Positions \cup {SLOT}
On(rule) == rule \in Rules
Mem(g) == GroupMemory[g]
(* R1: a request acts only on owners of its own session. Hazard H7: the    *)
(* Storage Server does not check whose a slot is.                          *)
OwnSession(o, e) == "H7" \in Hazards \/ OwnerEpoch[o] = e
Memories == {GroupMemory[g] : g \in Groups}

Empty == [k |-> "none", u |-> NONE]
Private(u) == [k |-> "private", u |-> u]
Shared(u) == [k |-> "shared", u |-> u]
NoEntry == [u |-> NONE, w |-> FALSE]
NoStamp == [g \in Groups |-> 0]

(* A3: `completed` of a group is the last batch whose end its executor has *)
(* seen with every earlier batch ended: one less than the earliest batch   *)
(* in flight, or the last batch launched when none is. A stamp is reached  *)
(* when `completed` of every group is at or past the group's stamp.        *)
\* @type: (Str -> Int) => Bool;
Reached(after) ==
    \A g \in Groups :
        /\ \A b \in inflight : b.g = g => after[g] < b.seq
        /\ (~\E b \in inflight : b.g = g) => after[g] <= launched[g]

-----------------------------------------------------------------------------
(* Holders. *)

\* @type: (Str, Str -> (Str -> {k: Str, u: Str}), Str -> Str, Str -> Str, Str -> Str, Set($def), Set($copy), Set($task)) => Bool;
HeldIn(u, tbl, slt, cch, hch, dfr, cps, tsk) ==
    \/ \E o \in Owners, p \in Positions : tbl[o][p].u = u
    \/ \E o \in Owners : slt[o] = u
    \/ \E k \in Keys : cch[k] = u \/ hch[k] = u
    \/ \E d \in dfr : u \in d.units
    \/ \E c \in cps : u = c.read \/ u = c.write
    \/ \E t \in tsk : t.held /\ (u \in t.units \/ u = t.slot \/ u = t.h)

(* R2: runs return to a region only when the last reference is dropped.    *)
\* @type: (Set(Str), Str -> (Str -> {k: Str, u: Str}), Str -> Str, Str -> Str, Str -> Str, Set($def), Set($copy), Set($task)) => Set(Str);
ReclaimedWith(dropped, tbl, slt, cch, hch, dfr, cps, tsk) ==
    IF On("R2") THEN {u \in dropped : ~HeldIn(u, tbl, slt, cch, hch, dfr, cps, tsk)} ELSE dropped

\* @type: (Str -> Str, Set(Str)) => (Str -> Str);
Reset(f, gone) == [u \in AllUnits |-> IF u \in gone THEN NONE ELSE f[u]]

(* Drop references to `units` in a state whose tables, slots, caches,      *)
(* deferred work, copies and tasks are given: what nothing holds any more  *)
(* is reclaimed.                                                           *)
\* @type: (Set(Str), Str -> (Str -> {k: Str, u: Str}), Str -> Str, Str -> Str, Str -> Str, Set($def), Set($copy), Set($task)) => Bool;
DropWith(units, tbl, slt, cch, hch, dfr, cps, tsk) ==
    LET gone == ReclaimedWith(units, tbl, slt, cch, hch, dfr, cps, tsk)
    IN /\ free' = free \cup gone
       /\ alloc' = Reset(alloc, gone)
       /\ sharedKey' = Reset(sharedKey, gone)

\* @type: (Set(Str), Str -> (Str -> {k: Str, u: Str}), Str -> Str, Str -> Str, Str -> Str, Set($def), Set($copy)) => Bool;
Drop(units, tbl, slt, cch, hch, dfr, cps) ==
    DropWith(units, tbl, slt, cch, hch, dfr, cps, tasks)

Pinned(u) == \E o \in Owners, p \in Positions : table[o][p] = Shared(u)
(* Hazard H9: an epoch that adopted another's cache sees its own pins only. *)
PinnedFor(u, e) ==
    \E o \in Owners, p \in Positions :
        table[o][p] = Shared(u) /\ ("H9" \notin Hazards \/ OwnerEpoch[o] = e)

(* R7: eviction takes a shared block without a pin; R2: and with no other  *)
(* holder.                                                                 *)
EvictableFor(u, e) ==
    /\ \E k \in Keys : cache[k] = u
    /\ On("R7") => ~PinnedFor(u, e)
    /\ On("R2") => ~\E d \in deferred : u \in d.units
    /\ On("R2") => ~\E c \in copies : u = c.read \/ u = c.write
    /\ On("R2") => ~\E t \in tasks : t.held /\ u \in t.units

Evictable(u) == EvictableFor(u, NONE) /\ (On("R7") => ~Pinned(u))

(* The units of the KV region of owner o's scheduler.                      *)
RegionUnits(o) == {u \in Units : UnitRegion[u] = GroupRegion[ogroup[o]]}

(* A device unit of region R for a new private block: a free one, or when  *)
(* the region has none an evictable one, which leaves the cache.           *)
Obtain(u, R, e) ==
    \/ u \in free /\ cache' = cache
    \/ /\ free \cap R = {}
       /\ EvictableFor(u, e)
       /\ cache' = [k \in Keys |-> IF cache[k] = u THEN NONE ELSE cache[k]]

(* R8: a copy made by an epoch on trial is pinned, cloned, loaded, restored *)
(* and stored only for that epoch until the update is committed, and an    *)
(* epoch on trial fills no memory of a copy that others may use.           *)
VisibleDevice(k, o) == On("R8") => vis[k].d \in {NONE, OwnerEpoch[o]}
VisibleHost(k, o) == On("R8") => vis[k].h \in {NONE, OwnerEpoch[o]}
(* The visibility of a copy that epoch e makes.                            *)
MadeBy(e) == IF trial[e] = "no" THEN NONE ELSE e

(* Whether a copy writes for position p of owner o.                        *)
CopyFor(o, p) == \E c \in copies : c.o = o /\ c.p = p

(* Whether owner o's transfer waits or runs. R1 refuses every request that *)
(* names such an owner but its `Detach`.                                   *)
Moving(o) ==
    \/ \E t \in tasks : t.kind = "transfer" /\ t.o = o
    \/ \E d \in deferred : d.kind = "transfer" /\ d.o = o

(* Whether stamped request r takes away the entry completion h fills.      *)
\* @type: ($req, $hand) => Bool;
Names(r, h) ==
    /\ r.e = OwnerEpoch[h.o]
    /\ r.o = h.o
    /\ CASE r.kind = "release" -> h.p \in r.ps
         [] r.kind \in {"detach", "transfer"} -> TRUE
         [] r.kind = "commit" -> h.p = r.p
         [] OTHER -> FALSE

(* A completion with runs leaves the buffer pool. The control plane hands  *)
(* it to its executor now, or later, or drops it; it drops it when a       *)
(* stamped request under way names its entry (A1). With `Eager` it is      *)
(* handed on at once, which is the case that addresses the most units:     *)
(* an executor's table then holds at every moment at least what it holds   *)
(* when completions are handed on later or dropped.                        *)
\* @type: ($hand) => Bool;
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
    /\ On("R1") => /\ ostate[o] = "new"
                    /\ GroupEpoch[g] = OwnerEpoch[o]
                    /\ trial[OwnerEpoch[o]] # "rolled"
                    /\ OwnerEpoch[o] \notin lost
    /\ \E s \in free \cap Slots :
         /\ SlotGroup[s] = g
         /\ free' = free \ {s}
         /\ oslot' = [oslot EXCEPT ![o] = s]
         /\ alloc' = [alloc EXCEPT ![s] = o]
         /\ Give([g |-> g, o |-> o, p |-> SLOT, u |-> s, w |-> TRUE])
    /\ ostate' = [ostate EXCEPT ![o] = "live"]
    /\ ogroup' = [ogroup EXCEPT ![o] = g]
    /\ table' = [table EXCEPT ![o] = [p \in Positions |-> Empty]]
    /\ UNCHANGED <<cache, hcache, deferred, copies, ncopy, tasks, trial, member, valid, vis, launched, inflight, requests, sharedKey>>

Allocate(o, p) ==
    /\ ostate[o] = "live"
    /\ On("R1") => table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
    /\ \E u \in RegionUnits(o) :
         /\ Obtain(u, RegionUnits(o), OwnerEpoch[o])
         /\ free' = free \ {u}
         /\ table' = [table EXCEPT ![o][p] = Private(u)]
         /\ alloc' = [alloc EXCEPT ![u] = o]
         /\ sharedKey' = [sharedKey EXCEPT ![u] = NONE]
         /\ valid' = [valid EXCEPT ![u] = {Mem(ogroup[o])}]
         /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> u, w |-> TRUE])
    /\ UNCHANGED <<ostate, ogroup, oslot, hcache, deferred, copies, ncopy, tasks, trial, member, vis,
                   launched, inflight, requests>>

Pin(o, p, k) ==
    /\ ostate[o] = "live"
    /\ cache[k] # NONE
    /\ Mem(ogroup[o]) \in valid[cache[k]]
    /\ VisibleDevice(k, o)
    /\ On("R1") => table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
    /\ table' = [table EXCEPT ![o][p] = Shared(cache[k])]
    /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> cache[k], w |-> FALSE])
    /\ UNCHANGED <<free, ostate, ogroup, oslot, cache, hcache, deferred, copies, ncopy, tasks, trial, member, valid, vis, launched,
                   inflight, requests, alloc, sharedKey>>

Evict(k) ==
    /\ cache[k] # NONE
    /\ Evictable(cache[k])
    /\ cache' = [cache EXCEPT ![k] = NONE]
    \* Hazard H12: state kept twice; here the free list misses the block.
    /\ IF "H12" \in Hazards
       THEN /\ free' = free
            /\ alloc' = Reset(alloc, {cache[k]})
            /\ sharedKey' = Reset(sharedKey, {cache[k]})
       ELSE Drop({cache[k]}, table, oslot, cache', hcache, deferred, copies)
    /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, copies, ncopy, tasks, trial, member, valid, vis, plane>>

(* A host copy leaves the host tier when nothing but the cache holds it.   *)
EvictHost(k) ==
    /\ hcache[k] # NONE
    /\ On("R2") => ~\E c \in copies : hcache[k] = c.read \/ hcache[k] = c.write
    /\ On("R2") => ~\E t \in tasks : t.held /\ hcache[k] = t.h
    /\ hcache' = [hcache EXCEPT ![k] = NONE]
    /\ Drop({hcache[k]}, table, oslot, cache, hcache', deferred, copies)
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, deferred, copies, ncopy, tasks, trial, member, valid, vis, plane>>

-----------------------------------------------------------------------------
(* Copies the executor runs. The handler allocates the destination and the *)
(* copy holds the block it reads and the block it writes until its         *)
(* `Finish` takes effect.                                                  *)

\* @type: (Str, Str, Str, Str, Str, Str, Str) => $copy;
Copy(kind, o, p, k, g, read, write) ==
    [id |-> ncopy + 1, kind |-> kind, o |-> o, p |-> p, k |-> k, g |-> g,
     read |-> read, write |-> write,
     fill |-> FALSE,      \* whether it fills a group's memory of a shared block
     seq |-> 0,           \* the batch it ran with; 0 before it starts
     done |-> FALSE,      \* whether its executor returned its `Finish`
     stopped |-> FALSE]   \* whether the scheduler gave it up

(* R5: a clone is a new private block that the executor fills from the     *)
(* cached one. Without the rule the cached block itself becomes private.   *)
Clone(o, p, k) ==
    /\ ncopy < MaxCopies
    /\ ostate[o] = "live"
    /\ cache[k] # NONE
    /\ Mem(ogroup[o]) \in valid[cache[k]]
    /\ VisibleDevice(k, o)
    /\ On("R1") => table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
    /\ IF On("R5")
       THEN /\ \E w \in RegionUnits(o) \ {cache[k]} :
                 /\ Obtain(w, RegionUnits(o), OwnerEpoch[o])
                 /\ free' = free \ {w}
                 /\ alloc' = [alloc EXCEPT ![w] = o]
                 /\ sharedKey' = [sharedKey EXCEPT ![w] = NONE]
                 /\ copies' = copies \cup {Copy("clone", o, p, k, ogroup[o], cache[k], w)}
            /\ ncopy' = ncopy + 1
            /\ UNCHANGED <<table, dbt, handoff>>
       ELSE /\ table' = [table EXCEPT ![o][p] = Private(cache[k])]
            /\ Give([g |-> ogroup[o], o |-> o, p |-> p, u |-> cache[k], w |-> TRUE])
            /\ UNCHANGED <<free, cache, copies, ncopy, alloc, sharedKey>>
    /\ UNCHANGED <<ostate, ogroup, oslot, hcache, deferred, tasks, trial, member, valid, vis, launched, inflight,
                   requests>>

(* A load brings a key from the host to the owner's group: into a new      *)
(* block when the key has no device block, into the group's memory of its  *)
(* block when that block is valid in other groups only.                    *)
Load(o, p, k) ==
    /\ ncopy < MaxCopies
    /\ ostate[o] = "live"
    /\ hcache[k] # NONE
    /\ VisibleHost(k, o)
    /\ On("R1") => table[o][p] = Empty /\ ~CopyFor(o, p) /\ ~Moving(o)
    /\ LET here == Mem(ogroup[o])
           \* The key's device block can be filled for this owner: it is not
           \* valid in the owner's memory, and its visibility is the one a
           \* copy this owner's epoch makes would have.
           fills == /\ cache[k] # NONE
                    /\ here \notin valid[cache[k]]
                    /\ On("R8") => vis[k].d = MadeBy(OwnerEpoch[o])
           \* The owner can pin the key's device block instead.
           pins == cache[k] # NONE /\ here \in valid[cache[k]] /\ VisibleDevice(k, o)
       IN IF fills
          THEN \* Nothing else fills that memory: no other load, no transfer.
               /\ ~\E c \in copies : c.write = cache[k] /\ Mem(c.g) = here
               /\ ~\E t \in tasks :
                    t.kind = "transfer" /\ Mem(t.to) = here /\ cache[k] \in t.writes
               /\ copies' = copies \cup
                    {[Copy("load", o, p, k, ogroup[o], hcache[k], cache[k]) EXCEPT !.fill = TRUE]}
               /\ UNCHANGED <<free, cache, alloc, sharedKey>>
          ELSE \* A new block: shared when its `Finish` takes effect if the key
               \* has no device block then, private to the owner otherwise.
               /\ ~pins
               /\ \E w \in RegionUnits(o) \ {cache[k]} :
                    /\ Obtain(w, RegionUnits(o), OwnerEpoch[o])
                    /\ free' = free \ {w}
                    /\ alloc' = [alloc EXCEPT ![w] = o]
                    /\ sharedKey' = [sharedKey EXCEPT ![w] = NONE]
                    /\ copies' = copies \cup {Copy("load", o, p, k, ogroup[o], hcache[k], w)}
    /\ ncopy' = ncopy + 1
    /\ UNCHANGED <<ostate, ogroup, oslot, table, hcache, deferred, tasks, trial, member, valid, vis,
                   plane>>

Store(g, k) ==
    /\ ncopy < MaxCopies
    \* R1: the group is of a session that is open.
    /\ On("R1") => GroupEpoch[g] \notin lost
    /\ cache[k] # NONE
    /\ Mem(g) \in valid[cache[k]]
    /\ On("R8") => vis[k].d \in {NONE, GroupEpoch[g]}
    /\ hcache[k] = NONE
    /\ \E h \in free \cap HostUnits :
         /\ free' = free \ {h}
         /\ alloc' = [alloc EXCEPT ![h] = "copy"]
         /\ copies' = copies \cup {Copy("store", NONE, NONE, k, g, cache[k], h)}
    /\ ncopy' = ncopy + 1
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, tasks, trial, member, valid, vis, plane,
                   sharedKey>>

(* A media request's trajectory: `Save` copies the owner's slot into a new *)
(* host block cached under a key; `Restore` copies such a block back into  *)
(* the owner's slot.                                                       *)
Save(o, k) ==
    /\ ncopy < MaxCopies
    /\ ostate[o] = "live"
    /\ hcache[k] = NONE
    /\ On("R1") => ~Moving(o) /\ ~\E c \in copies : c.o = o /\ c.kind \in {"save", "restore"}
    /\ \E h \in free \cap HostUnits :
         /\ free' = free \ {h}
         /\ alloc' = [alloc EXCEPT ![h] = "copy"]
         /\ copies' = copies \cup {Copy("save", o, NONE, k, ogroup[o], oslot[o], h)}
    /\ ncopy' = ncopy + 1
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, tasks, trial, member,
                   valid, vis, plane, sharedKey>>

Restore(o, k) ==
    /\ ncopy < MaxCopies
    /\ ostate[o] = "live"
    /\ hcache[k] # NONE
    /\ VisibleHost(k, o)
    /\ On("R1") => ~Moving(o) /\ ~\E c \in copies : c.o = o /\ c.kind \in {"save", "restore"}
    /\ copies' = copies \cup
         {[Copy("restore", o, NONE, k, ogroup[o], hcache[k], oslot[o]) EXCEPT !.fill = TRUE]}
    /\ ncopy' = ncopy + 1
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, tasks, trial,
                   member, valid, vis, plane, alloc, sharedKey>>

-----------------------------------------------------------------------------
(* The control plane issues a stamped request; the executor of every group *)
(* whose device block tables may hold the entries concerned applies and    *)
(* stamps it (A2).                                                         *)

\* @type: (Str, Str, Str, Set(Str), Str, Str, Int, Bool) => $req;
Request(kind, e, o, ps, p, k, c, ok) ==
    [kind |-> kind, e |-> e, o |-> o, ps |-> ps, p |-> p, k |-> k, c |-> c, ok |-> ok,
     to |-> NONE, after |-> NoStamp, by |-> {}, sg |-> {}]

(* A2: the groups that stamp r, fixed when the scheduler issues it: every  *)
(* group of r's session whose device block tables may hold what r          *)
(* concerns, which is the group its owner is attached to, or its copy's    *)
(* group. Each stamps whether or not its table still holds the entries.    *)
\* @type: ($req) => $req;
Issued(r) ==
    [r EXCEPT !.sg = {g \in Groups :
                        /\ GroupEpoch[g] = r.e
                        /\ IF r.kind = "finish"
                           THEN \E c \in copies : c.id = r.c /\ c.g = g
                           ELSE ogroup[r.o] = g}]

Issue ==
    /\ Cardinality(requests) < MaxRequests
    /\ \E e \in Epochs \ lost :
         \/ \E o \in Owners :
              \/ \E ps \in (SUBSET Positions) \ {{}} :
                   requests' = requests \cup {Issued(Request("release", e, o, ps, NONE, NONE, 0, FALSE))}
              \/ requests' = requests \cup {Issued(Request("detach", e, o, {}, NONE, NONE, 0, FALSE))}
              \/ \E p \in Positions, k \in Keys :
                   requests' = requests \cup {Issued(Request("commit", e, o, {}, p, k, 0, FALSE))}
              \/ \E g \in Groups :
                   requests' = requests \cup
                     {Issued([Request("transfer", e, o, {}, NONE, NONE, 0, FALSE)
                                EXCEPT !.to = g])}
         \* The scheduler gives a copy up.
         \* Hazard H13: nothing bounds how long a copy holds its blocks.
         \/ \E c \in 1 .. ncopy :
              /\ "H13" \notin Hazards
              /\ requests' = requests \cup
                   {Issued(Request("finish", e, NONE, {}, NONE, NONE, c, FALSE))}
    /\ UNCHANGED <<pool, dbt, launched, inflight, handoff, alloc, sharedKey>>

\* @type: ($req) => Set(Str);
StampGroups(r) == r.sg

(* The executor applies r: it removes the entries of a release, a detach   *)
(* or a transfer, marks the entry of a commit read-only, and attaches a    *)
(* copy that is                                                            *)
(* given up to no further batch. Then it stamps r with the last batch it   *)
(* launched. A give-up for a copy whose own `Finish` it has returned is    *)
(* discarded.                                                              *)
\* @type: (Str, $req) => Bool;
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
       ELSE \* Hazard H8: the executor's rows are never cleared.
            /\ dbt' = IF "H8" \in Hazards THEN dbt ELSE [dbt EXCEPT ![g][r.o] =
                        [p \in Rows |->
                           CASE r.kind = "release" /\ p \in r.ps -> NoEntry
                             [] r.kind \in {"detach", "transfer"} -> NoEntry
                             [] r.kind = "commit" /\ p = r.p -> [u |-> dbt[g][r.o][p].u, w |-> FALSE]
                             [] OTHER -> dbt[g][r.o][p]]]
            /\ requests' = (requests \ {r}) \cup
                             {[r EXCEPT !.after[g] = launched[g], !.by = @ \cup {g}]}
            /\ copies' = copies
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks, trial, member, valid, vis, launched,
                   inflight, handoff, alloc, sharedKey>>

(* A2: a request is submitted once every such group has stamped it.        *)
\* @type: ($req) => Bool;
Submittable(r) == StampGroups(r) \subseteq r.by

(* The completions the control plane still hands on once r is taken: those *)
(* of requests submitted before r that fill what r takes away are dropped. *)
\* @type: ($req) => Set($hand);
Kept(r) == {h \in handoff : ~Names(r, h)}

(* A1: an executor writes only the runs a completion gave it, at the        *)
(* position the request named, and is handed a completion only before any  *)
(* stamped request that names its entry is handed to an executor.          *)
\* @type: ($hand) => Bool;
ExecMap(h) ==
    /\ h \in handoff
    /\ ~\E r \in requests : Names(r, h)
    /\ handoff' = handoff \ {h}
    /\ dbt' = [dbt EXCEPT ![h.g][h.o][h.p] = [u |-> h.u, w |-> h.w]]
    /\ UNCHANGED <<pool, launched, inflight, requests, alloc, sharedKey>>

(* The scheduler drops a completion instead of handing it on.              *)
\* @type: ($hand) => Bool;
DropCompletion(h) ==
    /\ h \in handoff
    /\ handoff' = handoff \ {h}
    /\ UNCHANGED <<pool, dbt, launched, inflight, requests, alloc, sharedKey>>

(* A4: a batch reads the entries of its group's table and writes the        *)
(* writable ones; the batch that addresses every entry stands for all. The *)
(* copies of the group that have not started run with it, each reading its *)
(* source and writing its destination, which nothing else writes.          *)
Launch(g) ==
    /\ GroupEpoch[g] \notin lost
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
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks, trial, member, valid, vis, dbt,
                   handoff, requests, alloc, sharedKey>>

(* A batch ends after any delay.                                           *)
\* @type: ($batch) => Bool;
EndBatch(b) ==
    /\ b \in inflight
    /\ inflight' = inflight \ {b}
    /\ UNCHANGED <<pool, dbt, launched, handoff, requests, alloc, sharedKey>>

(* The executor returns the `Finish` of a copy it started, with its stamp. *)
\* @type: ($copy, Bool) => Bool;
ExecFinish(c, ok) ==
    /\ c \in copies
    /\ c.seq > 0 /\ ~c.done /\ ~c.stopped
    /\ copies' = (copies \ {c}) \cup {[c EXCEPT !.done = TRUE]}
    /\ requests' = requests \cup
         {[Request("finish", GroupEpoch[c.g], NONE, {}, NONE, NONE, c.id, ok)
             EXCEPT !.after[c.g] = launched[c.g], !.by = {c.g}, !.sg = {c.g}]}
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, ncopy, tasks, trial, member, valid, vis, dbt,
                   launched, inflight, handoff, alloc, sharedKey>>

-----------------------------------------------------------------------------
(* The buffer pool takes a stamped request.                                *)

(* R3: what a stamped request takes away is released when `completed` of   *)
(* every stamping group has reached its stamp.                             *)
\* @type: (Set(Str), Str -> Int, Str -> (Str -> {k: Str, u: Str}), Str -> Str, Set($task), Set($copy)) => Bool;
ReleaseUnits(units, after, tbl, slt, tsk, cps) ==
    /\ table' = tbl
    /\ oslot' = slt
    /\ IF On("R3") /\ ~Reached(after) /\ units # {}
       THEN /\ deferred' = deferred \cup
                 {[kind |-> "release", units |-> units, after |-> after,
                   o |-> NONE, p |-> NONE, k |-> NONE, c |-> 0, ok |-> FALSE,
                   g |-> NONE, to |-> NONE]}
            /\ UNCHANGED <<free, alloc, sharedKey>>
       ELSE /\ DropWith(units, tbl, slt, cache, hcache, deferred, cps, tsk)
            /\ UNCHANGED deferred

\* @type: ($req) => Bool;
TakeRelease(r) ==
    /\ r \in requests /\ r.kind = "release" /\ Submittable(r)
    /\ requests' = requests \ {r}
    /\ IF On("R1") /\ ~(/\ ostate[r.o] = "live"
                        /\ OwnSession(r.o, r.e)
                        /\ ~Moving(r.o)
                        /\ \A p \in r.ps : table[r.o][p] # Empty)
       THEN UNCHANGED <<pool, alloc, sharedKey>>
       ELSE /\ ReleaseUnits({table[r.o][p].u : p \in r.ps} \ {NONE}, r.after,
                            [table EXCEPT ![r.o] =
                               [p \in Positions |-> IF p \in r.ps THEN Empty ELSE table[r.o][p]]],
                            oslot, tasks, copies)
            /\ UNCHANGED <<ostate, ogroup, cache, hcache, copies, ncopy, tasks, trial, member, valid, vis>>
    /\ handoff' = Kept(r)
    /\ UNCHANGED <<dbt, launched, inflight>>

\* @type: ($req) => Bool;
TakeDetach(r) ==
    /\ r \in requests /\ r.kind = "detach" /\ Submittable(r)
    /\ requests' = requests \ {r}
    /\ IF On("R1") /\ ~(ostate[r.o] = "live" /\ OwnSession(r.o, r.e))
       THEN UNCHANGED <<pool, alloc, sharedKey>>
       ELSE \* R9: a block a task holds is released only on the task's
            \* completion. Without the rule the detach releases what the
            \* owner's task holds, the task's slot row included, which is the
            \* hazard H1 of the system as built. Hazard H5: the detach also
            \* lets go of what the owner's copies hold while they run.
            LET mine == {t \in tasks : t.kind = "transfer" /\ t.o = r.o /\ t.held}
                keeps == On("R9") /\ "H1" \notin Hazards
                tsk == IF keeps THEN tasks
                       ELSE {IF t \in mine THEN [t EXCEPT !.held = FALSE] ELSE t : t \in tasks}
                extra == IF keeps THEN {} ELSE {t.slot : t \in mine}
                cps == IF "H5" \in Hazards THEN {c \in copies : c.o # r.o} ELSE copies
                let == UNION {{c.read, c.write} : c \in copies \ cps}
            IN /\ tasks' = tsk
               /\ copies' = cps
               /\ ReleaseUnits(({table[r.o][p].u : p \in Positions} \cup {oslot[r.o]} \cup extra
                                   \cup let) \ {NONE},
                               r.after,
                               [table EXCEPT ![r.o] = [p \in Positions |-> Empty]],
                               [oslot EXCEPT ![r.o] = NONE],
                               tsk, cps)
               /\ ostate' = [ostate EXCEPT ![r.o] = IF @ = "live" THEN "gone" ELSE @]
               /\ UNCHANGED <<ogroup, cache, hcache, ncopy, valid, vis, trial, member>>
    /\ handoff' = Kept(r)
    /\ UNCHANGED <<dbt, launched, inflight>>

(* R6: a commit waits for its stamps holding the block it was taken for.   *)
(* Without the rule it holds nothing and names the position alone.         *)
\* @type: ($req) => Bool;
TakeCommit(r) ==
    /\ r \in requests /\ r.kind = "commit" /\ Submittable(r)
    /\ requests' = requests \ {r}
    /\ IF ~(/\ ostate[r.o] = "live"
            /\ table[r.o][r.p].k = "private"
            /\ On("R1") => /\ OwnSession(r.o, r.e)
                           /\ ~Moving(r.o)
                           /\ ~\E d \in deferred : d.kind = "commit" /\ d.o = r.o /\ d.p = r.p)
       THEN UNCHANGED deferred
       ELSE deferred' = deferred \cup
              {[kind |-> "commit",
                units |-> IF On("R6") THEN {table[r.o][r.p].u} ELSE {},
                after |-> r.after,
                o |-> r.o, p |-> r.p, k |-> r.k, c |-> 0, ok |-> FALSE,
                \* The group whose executor wrote the block: where it is valid.
                g |-> ogroup[r.o], to |-> NONE]}
    /\ handoff' = Kept(r)
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks, trial, member, valid, vis, dbt, launched,
                   inflight, alloc, sharedKey>>

(* One `Finish` ends a copy: R1 refuses one that names no copy in flight,  *)
(* a copy of another session, or a copy whose `Finish` already waits.      *)
\* @type: ($req) => Bool;
TakeFinish(r) ==
    /\ r \in requests /\ r.kind = "finish" /\ Submittable(r)
    /\ requests' = requests \ {r}
    /\ IF ~(/\ \E c \in copies : c.id = r.c /\ (On("R1") => GroupEpoch[c.g] = r.e)
            /\ On("R1") => ~\E d \in deferred : d.kind = "finish" /\ d.c = r.c)
       THEN UNCHANGED deferred
       ELSE deferred' = deferred \cup
              {[kind |-> "finish", units |-> {}, after |-> r.after,
                o |-> NONE, p |-> NONE, k |-> NONE, c |-> r.c, ok |-> r.ok,
                g |-> NONE, to |-> NONE]}
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks, trial, member, valid, vis, dbt, launched,
                   inflight, handoff, alloc, sharedKey>>

(* A transfer waits for its stamps: until then the source group's batches  *)
(* may still write the owner's blocks.                                     *)
\* @type: ($req) => Bool;
TakeTransfer(r) ==
    /\ r \in requests /\ r.kind = "transfer" /\ Submittable(r)
    /\ requests' = requests \ {r}
    /\ IF ~(/\ ostate[r.o] = "live"
            /\ On("R1") => /\ OwnSession(r.o, r.e)
                           /\ GroupEpoch[r.to] = r.e
                           /\ r.to # ogroup[r.o]
                           /\ ~Moving(r.o)
                           /\ ~\E c \in copies : c.o = r.o)
       THEN UNCHANGED deferred
       ELSE deferred' = deferred \cup
              {[kind |-> "transfer", units |-> {}, after |-> r.after,
                o |-> r.o, p |-> NONE, k |-> NONE, c |-> 0, ok |-> FALSE,
                g |-> NONE, to |-> r.to]}
    /\ handoff' = Kept(r)
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks, trial, member, valid, vis,
                   dbt, launched, inflight, alloc, sharedKey>>

-----------------------------------------------------------------------------
(* Deferred work comes due.                                                *)

\* @type: ($def) => Bool;
ReleaseDue(d) ==
    /\ d \in deferred /\ d.kind = "release"
    /\ Reached(d.after)
    /\ deferred' = deferred \ {d}
    /\ Drop(d.units, table, oslot, cache, hcache, deferred', copies)
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, tasks, trial, member, valid, vis, plane>>

\* @type: ($def) => Bool;
CommitDue(d) ==
    /\ d \in deferred /\ d.kind = "commit"
    \* Hazard H10: a block is read as shared while its writer still appends.
    /\ (On("R3") /\ "H10" \notin Hazards) => Reached(d.after)
    /\ deferred' = deferred \ {d}
    /\ LET \* The block the commit shares: the one it holds, or without R6
           \* whatever private block its position holds when it comes due.
           u == IF On("R6") THEN CHOOSE x \in d.units : TRUE
                ELSE IF table[d.o][d.p].k = "private" THEN table[d.o][d.p].u ELSE NONE
       IN IF u = NONE
          THEN UNCHANGED <<free, table, cache, valid, vis, alloc, sharedKey>>
          ELSE IF On("R4") /\ cache[d.k] # NONE
          \* R4: a key that is cached keeps its block; this one stays private.
          THEN /\ Drop(d.units, table, oslot, cache, hcache, deferred', copies)
               /\ UNCHANGED <<table, cache, valid, vis>>
          ELSE /\ cache' = [cache EXCEPT ![d.k] = u]
               /\ valid' = valid
               \* A block an epoch on trial commits is visible to that epoch alone.
               /\ vis' = [vis EXCEPT ![d.k].d = MadeBy(OwnerEpoch[d.o])]
               /\ table' = [table EXCEPT ![d.o][d.p] = IF @ = Private(u) THEN Shared(u) ELSE @]
               /\ free' = free
               /\ alloc' = [alloc EXCEPT ![u] = "shared"]
               /\ sharedKey' = [sharedKey EXCEPT ![u] = IF @ = NONE THEN d.k ELSE @]
    /\ UNCHANGED <<ostate, ogroup, oslot, hcache, copies, ncopy, tasks, trial, member, plane>>

(* The end of a copy takes effect (R3: once `completed` of its group has   *)
(* reached the stamp of its `Finish`, so the copy has ended).              *)
\* @type: ($def) => Bool;
FinishDue(d) ==
    /\ d \in deferred /\ d.kind = "finish"
    \* Hazard H3: the end of a failed copy takes effect without waiting.
    /\ (On("R3") /\ ~("H3" \in Hazards /\ ~d.ok)) => Reached(d.after)
    /\ deferred' = deferred \ {d}
    /\ \E c \in copies :
         /\ c.id = d.c
         /\ copies' = copies \ {c}
         \* A copy an epoch on trial made is visible to that epoch alone.
         /\ vis' = IF d.ok /\ c.kind = "load" /\ ~c.fill /\ ~(On("R4") /\ cache[c.k] # NONE)
                   THEN [vis EXCEPT ![c.k].d = MadeBy(GroupEpoch[c.g])]
                   ELSE IF /\ d.ok /\ c.kind \in {"store", "save"}
                           /\ ~(On("R4") /\ hcache[c.k] # NONE)
                   THEN [vis EXCEPT ![c.k].h = MadeBy(GroupEpoch[c.g])]
                   ELSE vis
         /\ LET target == c.o # NONE /\ ostate[c.o] = "live" /\ table[c.o][c.p] = Empty
            IN CASE ~d.ok ->
                      \* Nothing was copied: the destination retires.
                      /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                      /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff>>
                 [] d.ok /\ c.kind = "clone" ->
                      \* The owner holds a private block with the cached content.
                      IF target
                      THEN /\ table' = [table EXCEPT ![c.o][c.p] = Private(c.write)]
                           /\ valid' = [valid EXCEPT ![c.write] = {Mem(c.g)}]
                           /\ Give([g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write, w |-> TRUE])
                           /\ Drop({c.read}, table', oslot, cache, hcache, deferred', copies')
                           /\ UNCHANGED <<cache, hcache>>
                      ELSE /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                           /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff>>
                 [] d.ok /\ c.kind = "load" /\ c.fill ->
                      \* The group's memory of the shared block holds its content.
                      /\ valid' = [valid EXCEPT ![c.write] = @ \cup {Mem(c.g)}]
                      /\ IF target
                         THEN /\ table' = [table EXCEPT ![c.o][c.p] = Shared(c.write)]
                              /\ Give([g |-> c.g, o |-> c.o, p |-> c.p, u |-> c.write, w |-> FALSE])
                         ELSE UNCHANGED <<table, dbt, handoff>>
                      /\ UNCHANGED <<free, cache, hcache, alloc, sharedKey>>
                 [] d.ok /\ c.kind = "load" /\ ~c.fill ->
                      IF On("R4") /\ cache[c.k] # NONE
                      \* R4: the key gained a device block meanwhile; this block
                      \* stays private to the owner, which only reads it.
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
                      \* The block is shared on the device and pinned for the owner.
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
                      \* The slot holds the stored trajectory.
                      /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                      /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff>>
                 [] d.ok /\ c.kind \in {"store", "save"} ->
                      IF On("R4") /\ hcache[c.k] # NONE
                      THEN /\ Drop({c.read, c.write}, table, oslot, cache, hcache, deferred', copies')
                           /\ UNCHANGED <<table, cache, hcache, valid, dbt, handoff>>
                      ELSE /\ hcache' = [hcache EXCEPT ![c.k] = c.write]
                           \* The copy lets go of what it read: the cached device
                           \* block, or the slot of an owner that may be gone.
                           /\ LET gone == ReclaimedWith({c.read}, table, oslot, cache, hcache',
                                                        deferred', copies', tasks)
                              IN /\ free' = free \cup gone
                                 /\ alloc' = [Reset(alloc, gone) EXCEPT ![c.write] = "shared"]
                                 /\ sharedKey' = [Reset(sharedKey, gone) EXCEPT ![c.write] = c.k]
                           /\ UNCHANGED <<table, cache, valid, dbt, handoff>>
    /\ UNCHANGED <<ostate, ogroup, oslot, ncopy, tasks, trial, member, launched, inflight, requests>>

(* A transfer comes due: a task starts that copies into the target group's *)
(* memory the owner's blocks that are not valid there. It holds every      *)
(* block of the owner and a slot row of the target group.                  *)
(* Without a free row there the transfer fails and nothing changes.        *)
\* @type: ($def) => Bool;
TransferDue(d) ==
    /\ d \in deferred /\ d.kind = "transfer"
    /\ On("R3") => Reached(d.after)
    /\ LET units == {table[d.o][p].u : p \in Positions} \ {NONE}
           rows == {s \in free \cap Slots : SlotGroup[s] = d.to}
       IN \* It waits while a load or another owner's transfer fills the target
          \* group's memory of a block it holds: when that has ended the block
          \* is valid there, or is copied by this transfer alone.
          /\ ~\E c \in copies : c.fill /\ Mem(c.g) = Mem(d.to) /\ c.write \in units
          /\ ~\E t \in tasks :
               t.kind = "transfer" /\ Mem(t.to) = Mem(d.to) /\ t.writes \cap units # {}
          /\ deferred' = deferred \ {d}
          /\ IF ostate[d.o] # "live" \/ rows = {}
             THEN UNCHANGED <<free, tasks, alloc>>
             ELSE \E s \in rows :
                    /\ free' = free \ {s}
                    /\ alloc' = [alloc EXCEPT ![s] = d.o]
                    /\ tasks' = tasks \cup
                         {[kind |-> "transfer", k |-> NONE, h |-> NONE, src |-> NONE,
                           ok |-> TRUE, expired |-> FALSE,
                           o |-> d.o, to |-> d.to, slot |-> s, held |-> TRUE,
                           units |-> units \cup {oslot[d.o]},
                           writes |-> {u \in units : Mem(d.to) \notin valid[u]}]}
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, copies, ncopy, valid, vis, trial, member,
                   plane, sharedKey>>

(* A5: a task's completion is produced only after it stopped touching its  *)
(* blocks. It may come at any time, as a success or a failure.             *)
\* @type: ($task, Bool) => Bool;
TransferEnd(t, ok) ==
    /\ t \in tasks /\ t.kind = "transfer"
    /\ tasks' = tasks \ {t}
    /\ IF ok /\ ostate[t.o] = "live"
       THEN \* The owner is attached to the target group with its new slot row;
            \* the executor of that group is handed every entry of the owner.
            /\ ogroup' = [ogroup EXCEPT ![t.o] = t.to]
            /\ oslot' = [oslot EXCEPT ![t.o] = t.slot]
            \* The target group's memory holds what the task copied. A private
            \* block that no commit froze is written there from now on, so the
            \* source group's memory of it is stale.
            /\ valid' = [u \in Units |->
                           IF u \notin t.writes THEN valid[u]
                           ELSE IF alloc[u] = "shared"
                                   \/ \E d \in deferred : d.kind = "commit" /\ u \in d.units
                                THEN valid[u] \cup {Mem(t.to)}
                                ELSE {Mem(t.to)}]
            /\ DropWith(t.units, table, oslot', cache, hcache, deferred, copies, tasks')
            \* A1: an entry a stamped request under way names is not handed on.
            /\ LET entries ==
                     {[g |-> t.to, o |-> t.o, p |-> p, u |-> table[t.o][p].u,
                       w |-> /\ table[t.o][p].k = "private"
                             /\ ~\E d \in deferred :
                                   d.kind = "commit" /\ d.o = t.o /\ d.p = p] :
                        p \in {q \in Positions : table[t.o][q] # Empty}}
                     \cup {[g |-> t.to, o |-> t.o, p |-> SLOT, u |-> t.slot, w |-> TRUE]}
                   given == {h \in entries : ~\E r \in requests : Names(r, h)}
               IN IF Eager
                  THEN /\ handoff' = handoff
                       /\ dbt' = [dbt EXCEPT ![t.to][t.o] =
                                    [p \in Rows |->
                                       IF \E h \in given : h.p = p
                                       THEN LET h == CHOOSE x \in given : x.p = p
                                            IN [u |-> h.u, w |-> h.w]
                                       ELSE NoEntry]]
                  ELSE /\ handoff' = handoff \cup given
                       /\ dbt' = dbt
       ELSE \* Failed, or the owner was detached meanwhile: what the task held retires.
            /\ DropWith(t.units \cup {t.slot}, table, oslot, cache, hcache, deferred, copies, tasks')
            /\ UNCHANGED <<ogroup, oslot, valid, dbt, handoff>>
    /\ UNCHANGED <<ostate, table, cache, hcache, deferred, copies, ncopy, vis, trial, member, launched,
                   inflight, requests>>

(* The prefetches and remote stores in flight.                             *)
TierTasks == {t \in tasks : t.kind # "transfer"}

\* @type: (Str, Str, Str, Str) => $task;
Task(kind, k, h, src) ==
    [kind |-> kind, k |-> k, h |-> h, src |-> src, ok |-> TRUE, expired |-> FALSE,
     o |-> NONE, to |-> NONE, slot |-> NONE, held |-> TRUE, units |-> {}, writes |-> {}]

(* `Prefetch`: a task brings a key into a new host block from the remote   *)
(* store or from a peer, where the reader pins the block it reads.         *)
Prefetch(k, src) ==
    /\ Cardinality(TierTasks) < MaxTasks
    /\ hcache[k] = NONE
    /\ ~\E t \in tasks : t.kind = "prefetch" /\ t.k = k
    /\ \E h \in free \cap HostUnits :
         /\ free' = free \ {h}
         /\ alloc' = [alloc EXCEPT ![h] = "task"]
         /\ tasks' = tasks \cup {Task("prefetch", k, h, src)}
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, trial,
                   member, valid, vis, plane, sharedKey, rare>>

(* `Store` to the remote store: the connector's task reads the host copy.  *)
StoreRemote(k) ==
    /\ Cardinality(TierTasks) < MaxTasks
    /\ hcache[k] # NONE
    /\ On("R8") => vis[k].h = NONE
    /\ k \notin remote
    /\ ~\E t \in tasks : t.kind = "put" /\ t.k = k
    /\ tasks' = tasks \cup {Task("put", k, hcache[k], "remote")}
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                   trial, member, valid, vis, plane, alloc, sharedKey, rare>>

(* The peer a prefetch reads from lets go of its pin: the peer's           *)
(* connection closed, or it answered `Gone`.                               *)
\* @type: ($task) => Bool;
PeerLost(t) ==
    /\ t \in tasks /\ t.kind = "prefetch" /\ t.src = "peer" /\ t.ok
    /\ tasks' = (tasks \ {t}) \cup {[t EXCEPT !.ok = FALSE]}
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                   trial, member, valid, vis, plane, alloc, sharedKey, rare>>

(* A task passes its deadline, or its remote put is given up: the request  *)
(* is answered with what was found. R9: the block the task holds is        *)
(* released only on the task's completion; without the rule, now.          *)
\* @type: ($task) => Bool;
Deadline(t) ==
    \* Hazard H6: a stage of the tier walk has no deadline.
    /\ "H6" \notin Hazards
    /\ t \in TierTasks /\ ~t.expired
    /\ LET after == [t EXCEPT !.expired = TRUE, !.held = On("R9")]
       IN /\ tasks' = (tasks \ {t}) \cup {after}
          /\ IF On("R9") \/ t.kind = "put"
             THEN UNCHANGED <<free, alloc, sharedKey>>
             ELSE DropWith({t.h}, table, oslot, cache, hcache, deferred, copies, tasks')
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, trial,
                   member, valid, vis, plane, rare>>

(* A5: a task's completion is produced only after it stopped touching its  *)
(* blocks. A prefetch that found its key before its deadline, from a       *)
(* source that held the block throughout, caches the host block. The       *)
(* hazard H2 of the system as built caches it whatever the peer did.       *)
\* @type: ($task, Bool) => Bool;
PrefetchEnd(t, found) ==
    /\ t \in tasks /\ t.kind = "prefetch"
    /\ tasks' = tasks \ {t}
    /\ IF found /\ ~t.expired /\ t.held /\ hcache[t.k] = NONE /\ (t.ok \/ "H2" \in Hazards)
       THEN /\ hcache' = [hcache EXCEPT ![t.k] = t.h]
            /\ free' = free
            /\ alloc' = [alloc EXCEPT ![t.h] = "shared"]
            /\ sharedKey' = [sharedKey EXCEPT ![t.h] = t.k]
            /\ bad' = IF t.ok THEN bad ELSE bad \cup {<<t.k, "host">>}
            \* What the remote store and the peers hold is what every epoch may use.
            /\ vis' = [vis EXCEPT ![t.k].h = NONE]
       ELSE /\ IF t.held
               THEN DropWith({t.h}, table, oslot, cache, hcache, deferred, copies, tasks')
               ELSE UNCHANGED <<free, alloc, sharedKey>>
            /\ UNCHANGED <<hcache, bad, vis>>
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, deferred, copies, ncopy, trial, member,
                   valid, plane, remote, lost>>

(* The connector's put ends: the remote store holds the key's copy. A put  *)
(* whose host block was let go and reused wrote other content.             *)
\* @type: ($task, Bool) => Bool;
PutEnd(t, stored) ==
    /\ t \in tasks /\ t.kind = "put"
    /\ tasks' = tasks \ {t}
    /\ IF stored /\ ~t.expired
       THEN /\ remote' = remote \cup {t.k}
            /\ bad' = IF hcache[t.k] = t.h THEN bad ELSE bad \cup {<<t.k, "remote">>}
       ELSE UNCHANGED <<remote, bad>>
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                   trial, member, valid, vis, plane, alloc, sharedKey, lost>>

(* Hazard H11: an `Evict` is answered as done and the block stays cached.   *)
EvictNot(k) ==
    /\ "H11" \in Hazards
    /\ cache[k] # NONE /\ Evictable(cache[k])
    /\ bad' = bad \cup {<<k, "evicted">>}
    /\ UNCHANGED <<pool, plane, alloc, sharedKey, remote, lost>>

(* Hazard H4: a key is taken to be on the host when its store is asked,    *)
(* before any copy ran.                                                    *)
StoreOffered(k) ==
    /\ "H4" \in Hazards
    /\ cache[k] # NONE /\ hcache[k] = NONE
    /\ bad' = bad \cup {<<k, "stored">>}
    /\ UNCHANGED <<pool, plane, alloc, sharedKey, remote, lost>>

(* A7: an engine whose processes have exited runs no batch. The buffer     *)
(* pool sees them exit: its batches and copies are over, the copies as     *)
(* failed; every stamp of its groups counts as reached; its commits and    *)
(* transfers that waited are dropped; its owners are detached. What it     *)
(* committed stays cached, and what its tasks hold stays held (R9).        *)
Lose(e) ==
    /\ e \in Losable \ lost
    /\ lost' = lost \cup {e}
    /\ LET mine(g) == GroupEpoch[g] = e
           gone == {o \in Owners : OwnerEpoch[o] = e /\ ostate[o] = "live"}
           cps == {c \in copies : ~mine(c.g)}
           dfr == {d \in deferred :
                     /\ ~(d.kind = "finish" /\ \E c \in copies \ cps : c.id = d.c)
                     /\ ~(d.kind \in {"commit", "transfer"} /\ OwnerEpoch[d.o] = e)}
           tbl == [o \in Owners |-> IF o \in gone THEN [p \in Positions |-> Empty] ELSE table[o]]
           slt == [o \in Owners |-> IF o \in gone THEN NONE ELSE oslot[o]]
           dropped == (UNION {{table[o][p].u : p \in Positions} \cup {oslot[o]} : o \in gone})
                        \cup (UNION {{c.read, c.write} : c \in copies \ cps})
                        \cup (UNION {d.units : d \in deferred \ dfr})
       IN /\ inflight' = {b \in inflight : ~mine(b.g)}
          /\ dbt' = [g \in Groups |->
                       IF mine(g) THEN [o \in Owners |-> [p \in Rows |-> NoEntry]] ELSE dbt[g]]
          /\ handoff' = {h \in handoff : ~mine(h.g)}
          /\ requests' = {r \in requests : r.e # e}
          /\ copies' = cps
          /\ deferred' = dfr
          /\ table' = tbl
          /\ oslot' = slt
          /\ ostate' = [o \in Owners |-> IF o \in gone THEN "gone" ELSE ostate[o]]
          /\ DropWith(dropped \ {NONE}, tbl, slt, cache, hcache, dfr, cps, tasks)
    /\ UNCHANGED <<ogroup, cache, hcache, ncopy, tasks, trial, member, valid, vis, launched,
                   remote, bad>>

(* `Resize`, a shrink: a free unit leaves its region, and its memory is    *)
(* the deployment's free quota on the device. A shrink takes no block that *)
(* anything holds; a shared block without a pin leaves by `Evict` first.   *)
Shrink(u) ==
    /\ u \in free \cap Movable
    /\ free' = free \ {u}
    /\ member' = member \ {u}
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, tasks,
                   valid, vis, trial, plane, alloc, sharedKey>>

(* `Resize`, a grow: a unit joins its region. R10: its memory has left     *)
(* every other region.                                                     *)
Grow(u) ==
    /\ u \in Movable \ member
    /\ On("R10") => \A v \in member : UnitChunk[v] # UnitChunk[u]
    /\ free' = free \cup {u}
    /\ member' = member \cup {u}
    /\ UNCHANGED <<ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy, tasks,
                   valid, vis, trial, plane, alloc, sharedKey>>

(* The update of an epoch on trial is committed: what it committed is     *)
(* visible to every epoch.                                                 *)
CommitUpdate(e) ==
    /\ trial[e] = "yes"
    /\ trial' = [trial EXCEPT ![e] = "no"]
    /\ vis' = [k \in Keys |-> [d |-> IF vis[k].d = e THEN NONE ELSE vis[k].d,
                               h |-> IF vis[k].h = e THEN NONE ELSE vis[k].h]]
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                   tasks, valid, member, plane, alloc, sharedKey>>

(* The update is rolled back once the epoch's engine is gone: what it      *)
(* committed stays visible to it alone until it is evicted.                *)
Rollback(e) ==
    /\ trial[e] = "yes"
    /\ \A o \in Owners : OwnerEpoch[o] = e => ostate[o] # "live"
    /\ trial' = [trial EXCEPT ![e] = "rolled"]
    /\ UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                   tasks, valid, vis, member, plane, alloc, sharedKey>>

-----------------------------------------------------------------------------

Init ==
    /\ free = InitialUnits \cup Slots \cup HostUnits
    /\ member = InitialUnits
    /\ ostate = [o \in Owners |-> "new"]
    /\ ogroup = [o \in Owners |-> NONE]
    /\ oslot = [o \in Owners |-> NONE]
    /\ table = [o \in Owners |-> [p \in Positions |-> Empty]]
    /\ cache = [k \in Keys |-> NONE]
    /\ hcache = [k \in Keys |-> NONE]
    /\ deferred = {}
    /\ copies = {}
    /\ ncopy = 0
    /\ tasks = {}
    /\ valid = [u \in Units |-> {}]
    /\ vis = [k \in Keys |-> [d |-> NONE, h |-> NONE]]
    /\ trial = [e \in Epochs |-> IF e \in TrialEpochs THEN "yes" ELSE "no"]
    /\ dbt = [g \in Groups |-> [o \in Owners |-> [p \in Rows |-> NoEntry]]]
    /\ launched = [g \in Groups |-> 0]
    /\ inflight = {}
    /\ handoff = {}
    /\ requests = {}
    /\ alloc = [u \in AllUnits |-> NONE]
    /\ sharedKey = [u \in AllUnits |-> NONE]
    /\ remote = {}
    /\ lost = {}
    /\ bad = {}

(* The requests, the control plane and the deferred work.                   *)
Serving ==
    \/ \E o \in Owners, g \in Groups : Attach(o, g)
    \/ \E o \in Owners, p \in Positions : Allocate(o, p)
    \/ \E o \in Owners, p \in Positions, k \in Keys : Pin(o, p, k) \/ Clone(o, p, k) \/ Load(o, p, k)
    \/ \E o \in Owners, k \in Keys : Save(o, k) \/ Restore(o, k)
    \/ \E g \in Groups, k \in Keys : Store(g, k)
    \/ \E k \in Keys : Evict(k) \/ EvictHost(k)
    \/ Issue
    \/ \E g \in Groups, r \in requests : ExecStamp(g, r)
    \/ \E h \in handoff : ExecMap(h) \/ DropCompletion(h)
    \/ \E g \in Groups : Launch(g)
    \/ \E b \in inflight : EndBatch(b)
    \/ \E c \in copies, ok \in BOOLEAN : ExecFinish(c, ok)
    \/ \E r \in requests :
         TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r) \/ TakeTransfer(r)
    \/ \E d \in deferred : ReleaseDue(d) \/ CommitDue(d) \/ FinishDue(d) \/ TransferDue(d)
    \/ \E t \in tasks, ok \in BOOLEAN : TransferEnd(t, ok)
    \/ \E e \in Epochs : CommitUpdate(e) \/ Rollback(e)
    \/ \E u \in Units : Shrink(u) \/ Grow(u)

Next ==
    \/ Serving /\ UNCHANGED rare
    \/ \E k \in Keys : Prefetch(k, "remote") \/ Prefetch(k, "peer") \/ StoreRemote(k)
    \/ \E t \in tasks : PeerLost(t) \/ Deadline(t)
    \/ \E t \in tasks, ok \in BOOLEAN : PrefetchEnd(t, ok) \/ PutEnd(t, ok)
    \/ \E e \in Epochs : Lose(e)
    \/ \E k \in Keys : EvictNot(k) \/ StoreOffered(k)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* What a property may speak about (specs/buffer-pool.md §17.1): who holds *)
(* each unit, and the values the variables take. The properties are in the *)
(* modules KvP1 to KvP6, one each.                                         *)

Held(u) == HeldIn(u, table, oslot, cache, hcache, deferred, copies, tasks)
TableHolders(u) ==
    {x \in Owners \X Positions : table[x[1]][x[2]] = Private(u)}
        \cup {x \in Owners \X {SLOT} : oslot[x[1]] = u}
CopyWriters(u) == {c \in copies : c.write = u /\ ~c.fill}
CacheKeys(u) == {k \in Keys : cache[k] = u \/ hcache[k] = u}
InRegion(u) == u \in Units => u \in member

TypeOK ==
    /\ free \subseteq AllUnits
    /\ member \subseteq Units
    /\ ostate \in [Owners -> {"new", "live", "gone"}]
    /\ ogroup \in [Owners -> Groups \cup {NONE}]
    /\ oslot \in [Owners -> Slots \cup {NONE}]
    /\ \A o \in Owners, p \in Positions :
         /\ table[o][p].k \in {"none", "private", "shared"}
         /\ table[o][p].u \in Units \cup {NONE}
    /\ cache \in [Keys -> Units \cup {NONE}]
    /\ hcache \in [Keys -> HostUnits \cup {NONE}]
    /\ ncopy \in 0 .. MaxCopies
    /\ valid \in [Units -> SUBSET Memories]
    /\ \A k \in Keys : vis[k].d \in Epochs \cup {NONE} /\ vis[k].h \in Epochs \cup {NONE}
    /\ trial \in [Epochs -> {"no", "yes", "rolled"}]
    /\ launched \in [Groups -> 0 .. MaxBatch]
    /\ alloc \in [AllUnits -> Owners \cup {NONE, "shared", "copy", "task"}]
    /\ remote \subseteq Keys
    /\ lost \subseteq Epochs
    /\ sharedKey \in [AllUnits -> Keys \cup {NONE}]
=============================================================================
