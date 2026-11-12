----------------------------- MODULE KvInductive -----------------------------
(***************************************************************************)
(* The inductive invariant of KvCore under the design: every rule in force *)
(* and no hazard switched on. It states, beyond the properties, what the   *)
(* buffer pool's state and the control plane's state say about each other: *)
(* whom every unit is allocated to, what each deferred item, copy and task *)
(* holds, what an executor's table may still name, and what covers every   *)
(* unit a batch in flight addresses.                                       *)
(***************************************************************************)
EXTENDS KvSafety

Live(o) == ostate[o] = "live"
TableUnits(o) == {table[o][p].u : p \in Positions} \ {NONE}
HoldsAt(o, p, u) == IF p = SLOT THEN oslot[o] = u ELSE table[o][p].u = u

(* Whether stamped request r, of owner o's session, names row p of o: to   *)
(* take it away, or as the entry it commits.                               *)
\* @type: ($req, Str, Str) => Bool;
TakesRow(r, o, p) ==
    /\ r.e = OwnerEpoch[o] /\ r.o = o
    /\ CASE r.kind = "release" -> p \in r.ps
         [] r.kind \in {"detach", "transfer"} -> TRUE
         [] OTHER -> FALSE
\* @type: ($req, Str, Str) => Bool;
CommitsRow(r, o, p) == r.kind = "commit" /\ r.e = OwnerEpoch[o] /\ r.o = o /\ r.p = p

(* The values every variable takes.                                        *)
Stamps == [Groups -> 0 .. MaxBatch]
TypeInv ==
    /\ TypeOK
    /\ DOMAIN table = Owners
    /\ \A o \in Owners :
         /\ DOMAIN table[o] = Positions
         /\ \A p \in Positions :
              table[o][p] \in {Empty} \cup {Private(u) : u \in Units} \cup {Shared(u) : u \in Units}
    /\ DOMAIN dbt = Groups
    /\ \A g \in Groups :
         /\ DOMAIN dbt[g] = Owners
         /\ \A o \in Owners :
              /\ DOMAIN dbt[g][o] = Rows
              /\ \A p \in Rows : dbt[g][o][p].u \in AllUnits \cup {NONE}
    /\ vis \in [Keys -> [d : Epochs \cup {NONE}, h : Epochs \cup {NONE}]]
    /\ trial \in [Epochs -> {"no", "yes", "rolled"}]
    /\ remote \subseteq Keys
    /\ lost \subseteq Epochs
    /\ \A d \in deferred :
         /\ d.units \subseteq AllUnits /\ d.after \in Stamps
         /\ d.o \in Owners \cup {NONE} /\ d.p \in Positions \cup {NONE} /\ d.k \in Keys \cup {NONE}
         /\ d.c \in 0 .. MaxCopies /\ d.g \in Groups \cup {NONE} /\ d.to \in Groups \cup {NONE}
    /\ \A c \in copies :
         /\ c.seq \in 0 .. MaxBatch /\ c.k \in Keys
         /\ c.o \in Owners \cup {NONE} /\ c.p \in Positions \cup {NONE}
         /\ c.read \in AllUnits /\ c.write \in AllUnits
    /\ \A t \in tasks :
         /\ t.units \subseteq AllUnits /\ t.writes \subseteq AllUnits
         /\ t.k \in Keys \cup {NONE} /\ t.h \in HostUnits \cup {NONE}
         /\ t.slot \in Slots \cup {NONE} /\ t.o \in Owners \cup {NONE} /\ t.to \in Groups \cup {NONE}
         /\ t.src \in {NONE, "remote", "peer"}
         /\ t.kind = "transfer" <=> t.k = NONE
    /\ \A r \in requests :
         /\ r.e \in Epochs /\ r.after \in Stamps /\ r.ps \subseteq Positions
         /\ r.o \in Owners \cup {NONE} /\ r.p \in Positions \cup {NONE} /\ r.k \in Keys \cup {NONE}
         /\ r.c \in 0 .. MaxCopies /\ r.to \in Groups \cup {NONE}
         /\ r.kind # "finish" => r.o \in Owners
         /\ r.kind = "commit" => r.p \in Positions /\ r.k \in Keys
         /\ r.kind = "transfer" => r.to \in Groups
    /\ \A b \in inflight :
         /\ b.reads \subseteq Owners \X AllUnits
         /\ b.cr \subseteq AllUnits /\ b.cw \subseteq AllUnits
    /\ \A h \in handoff :
         h.g \in Groups /\ h.o \in Owners /\ h.p \in Rows /\ h.u \in AllUnits

-----------------------------------------------------------------------------
(* The buffer pool: whom each unit is allocated to.                        *)

FreeClean == \A u \in free : alloc[u] = NONE /\ sharedKey[u] = NONE /\ ~Held(u)
Outside == \A u \in Units \ member : alloc[u] = NONE /\ sharedKey[u] = NONE
Regions == member \subseteq Units /\ (free \cap Units) \subseteq member /\ OneRegion

(* A unit is allocated only to an owner that was attached.                 *)
AllocOwner == \A u \in AllUnits : alloc[u] \in Owners => ostate[alloc[u]] # "new"

SharedIff == \A u \in AllUnits : (sharedKey[u] # NONE) <=> (alloc[u] = "shared")
SharedCached ==
    \A u \in AllUnits :
        alloc[u] = "shared" => cache[sharedKey[u]] = u \/ hcache[sharedKey[u]] = u
CacheShared ==
    \A k \in Keys :
        /\ cache[k] # NONE => cache[k] \in Units /\ sharedKey[cache[k]] = k
        /\ hcache[k] # NONE => hcache[k] \in HostUnits /\ sharedKey[hcache[k]] = k

Tables ==
    \A o \in Owners, p \in Positions :
        /\ table[o][p].k = "none" => table[o][p].u = NONE
        /\ table[o][p].k = "private" => table[o][p].u \in Units /\ alloc[table[o][p].u] = o
        /\ table[o][p].k = "shared" => table[o][p].u \in Units /\ alloc[table[o][p].u] = "shared"
        /\ table[o][p].k # "none" => Live(o)
        \* A private block is at one position of its owner's table.
        /\ \A q \in Positions :
             (table[o][p].k = "private" /\ table[o][q].u = table[o][p].u) => p = q
        \* A block is valid in its holder's memory.
        /\ table[o][p].u # NONE => Mem(ogroup[o]) \in valid[table[o][p].u]

OwnersOk ==
    \A o \in Owners :
        /\ Live(o) =>
             /\ ogroup[o] \in Groups
             /\ GroupEpoch[ogroup[o]] = OwnerEpoch[o]
             /\ OwnerEpoch[o] \notin lost
             /\ trial[OwnerEpoch[o]] # "rolled"
             /\ oslot[o] \in Slots
             /\ alloc[oslot[o]] = o
             /\ SlotGroup[oslot[o]] = ogroup[o]
        /\ ~Live(o) => oslot[o] = NONE /\ \A p \in Positions : table[o][p] = Empty

DeferredOk ==
    \A d \in deferred :
        /\ d.kind \in {"release", "commit", "finish", "transfer"}
        /\ d.kind = "commit" =>
             /\ d.o \in Owners /\ d.p \in Positions /\ d.k \in Keys
             /\ OwnerEpoch[d.o] \notin lost
             /\ \E u \in Units : d.units = {u} /\ alloc[u] = d.o
             \* The block is at its position in its owner's table, or in no table.
             /\ \A o \in Owners, p \in Positions :
                  table[o][p].u \in d.units => o = d.o /\ p = d.p /\ table[o][p].k = "private"
             /\ \A e \in deferred : (e.kind = "commit" /\ e.units = d.units) => e = d
        /\ d.kind = "finish" =>
             /\ \E c \in copies :
                  /\ c.id = d.c
                  \* The stamp covers the batch the copy ran with; a copy that
                  \* did not start never will.
                  /\ c.seq > 0 => d.after[c.g] >= c.seq
                  /\ c.seq = 0 => c.stopped
             /\ \A e \in deferred : (e.kind = "finish" /\ e.c = d.c) => e = d
        /\ d.kind = "transfer" =>
             /\ d.o \in Owners /\ d.to \in Groups /\ GroupEpoch[d.to] = OwnerEpoch[d.o]
             /\ ostate[d.o] # "new"

CopiesOk ==
    \A c \in copies :
        /\ c.id \in 1 .. ncopy
        /\ \A x \in copies : x.id = c.id => x = c
        /\ c.g \in Groups /\ GroupEpoch[c.g] \notin lost
        /\ c.seq <= launched[c.g]
        /\ c.done => c.seq > 0
        /\ c.kind \in {"clone", "load", "store", "save", "restore"}
        \* What it reads is shared, or the slot of the owner that saves.
        /\ alloc[c.read] = "shared" \/ (c.kind = "save" /\ alloc[c.read] = c.o)
        /\ ~c.fill =>
             /\ c.kind \in {"clone", "load", "store", "save"}
             /\ c.kind \in {"clone", "load"} => c.write \in Units /\ alloc[c.write] = c.o
             /\ c.kind \in {"store", "save"} => c.write \in HostUnits /\ alloc[c.write] = "copy"
             /\ \A o \in Owners, p \in Positions : table[o][p].u # c.write
             /\ \A x \in copies : x.write = c.write => x = c
        /\ c.fill =>
             \/ /\ c.kind = "load"
                /\ cache[c.k] = c.write
                /\ Mem(c.g) \notin valid[c.write]
                \* Nothing else fills that memory.
                /\ \A x \in copies : (x.write = c.write /\ Mem(x.g) = Mem(c.g)) => x = c
                /\ \A t \in tasks :
                     (t.kind = "transfer" /\ Mem(t.to) = Mem(c.g)) => c.write \notin t.writes
             \/ c.kind = "restore" /\ c.write \in Slots /\ alloc[c.write] = c.o
        /\ c.kind \in {"clone", "load"} =>
             /\ c.o \in Owners /\ c.p \in Positions
             /\ table[c.o][c.p] = Empty
             /\ \A x \in copies : (x.o = c.o /\ x.p = c.p) => x = c
        /\ c.kind \in {"save", "restore"} => c.o \in Owners
        /\ c.o \in Owners => ostate[c.o] # "new"
        \* An owner a copy writes for does not move.
        /\ (c.o \in Owners /\ Live(c.o)) => ogroup[c.o] = c.g /\ ~Moving(c.o)

TasksOk ==
    \A t \in tasks :
        /\ t.held
        /\ t.kind \in {"transfer", "prefetch", "put"}
        /\ t.kind = "transfer" =>
             /\ t.o \in Owners /\ t.to \in Groups /\ GroupEpoch[t.to] = OwnerEpoch[t.o]
             /\ t.slot \in Slots /\ alloc[t.slot] = t.o /\ SlotGroup[t.slot] = t.to
             /\ \A o \in Owners : oslot[o] # t.slot
             /\ t.writes \subseteq t.units \cap Units
             /\ \A u \in t.writes :
                  Mem(t.to) \notin valid[u] /\ (alloc[u] = t.o \/ alloc[u] = "shared")
             /\ \A x \in tasks : (x.kind = "transfer" /\ (x.o = t.o \/ x.slot = t.slot)) => x = t
             /\ ~\E d \in deferred : d.kind = "transfer" /\ d.o = t.o
             /\ Live(t.o) =>
                  /\ t.units = TableUnits(t.o) \cup {oslot[t.o]}
                  /\ ~\E c \in copies : c.o = t.o
        /\ t.kind = "prefetch" =>
             /\ t.h \in HostUnits /\ alloc[t.h] = "task"
             /\ \A x \in tasks : (x.kind = "prefetch" /\ x.h = t.h) => x = t
        /\ t.kind = "put" => t.h \in HostUnits /\ hcache[t.k] = t.h /\ vis[t.k].h = NONE

(* An owner whose transfer waits or runs is named in no executor's table.  *)
MovingUnmapped ==
    \A o \in Owners :
        (\/ \E t \in tasks : t.kind = "transfer" /\ t.o = o
         \/ \E d \in deferred : d.kind = "transfer" /\ d.o = o)
            => /\ \A g \in Groups, p \in Rows : dbt[g][o][p].u = NONE
               /\ \A h \in handoff : h.o # o
               /\ \A d, e \in deferred :
                    (d.kind = "transfer" /\ e.kind = "transfer" /\ d.o = o /\ e.o = o) => d = e

LostOk ==
    /\ \A g \in Groups :
         GroupEpoch[g] \in lost =>
             /\ \A o \in Owners, p \in Rows : dbt[g][o][p].u = NONE
             /\ ~\E b \in inflight : b.g = g
             /\ ~\E h \in handoff : h.g = g
    /\ \A r \in requests : r.e \notin lost

-----------------------------------------------------------------------------
(* The control plane: what an executor's table may name, and the stamps.   *)

(* A1: an entry of an executor's table is its owner's, in the group the    *)
(* owner is attached to, and writable only while the block is private and  *)
(* no commit of it waits.                                                  *)
TablesMapped ==
    \A g \in Groups, o \in Owners, p \in Rows :
        dbt[g][o][p].u # NONE =>
            /\ Live(o) /\ ogroup[o] = g
            /\ HoldsAt(o, p, dbt[g][o][p].u)
            /\ (p \in Positions /\ dbt[g][o][p].w) =>
                 /\ table[o][p].k = "private"
                 /\ ~\E d \in deferred : d.kind = "commit" /\ dbt[g][o][p].u \in d.units

(* A completion not yet handed on that no request under way names is as    *)
(* good as an entry.                                                       *)
HandoffOk ==
    \A h \in handoff :
        /\ h.o \in Owners /\ h.g \in Groups /\ GroupEpoch[h.g] = OwnerEpoch[h.o]
        /\ (~\E r \in requests : Names(r, h)) =>
             /\ Live(h.o) /\ ogroup[h.o] = h.g
             /\ HoldsAt(h.o, h.p, h.u)
             /\ (h.p \in Positions /\ h.w) =>
                  /\ table[h.o][h.p].k = "private"
                  /\ ~\E d \in deferred : d.kind = "commit" /\ h.u \in d.units

(* A2: a request is stamped by the groups fixed at its issue; a group that *)
(* stamped it applied it; an entry it names is in the table of a group     *)
(* that stamps it.                                                         *)
RequestsOk ==
    \A r \in requests :
        /\ r.by \subseteq r.sg /\ r.sg \subseteq Groups
        /\ \A g \in r.sg : GroupEpoch[g] = r.e
        /\ \A g \in Groups : r.after[g] <= launched[g] /\ (g \notin r.by => r.after[g] = 0)
        /\ r.kind \in {"release", "detach", "commit", "transfer", "finish"}
        /\ r.kind # "finish" =>
             \A g \in Groups, p \in Rows :
                 (dbt[g][r.o][p].u # NONE /\ (TakesRow(r, r.o, p) \/ CommitsRow(r, r.o, p))) =>
                     /\ g \in r.sg
                     /\ g \in r.by => CommitsRow(r, r.o, p) /\ ~dbt[g][r.o][p].w
        /\ r.kind = "finish" =>
             \A c \in copies :
                 (c.id = r.c /\ GroupEpoch[c.g] = r.e) =>
                     /\ c.g \in r.sg
                     /\ c.g \in r.by =>
                          /\ c.seq > 0 => r.after[c.g] >= c.seq
                          /\ c.seq = 0 => c.stopped

(* A3, A4: a batch in flight. Every unit it addresses is still the block   *)
(* its row named: its owner holds it at a row that no stamped request has  *)
(* taken away without a stamp that covers the batch, or a release that     *)
(* waits for the batch holds it.                                           *)
\* @type: ($batch, Str, Str) => Bool;
Covered(b, o, u) ==
    \/ /\ Live(o) /\ ogroup[o] = b.g
       /\ \E p \in Rows :
            /\ HoldsAt(o, p, u)
            /\ \A r \in requests :
                 TakesRow(r, o, p) => b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
       /\ \A d \in deferred : (d.kind = "transfer" /\ d.o = o) => d.after[b.g] >= b.seq
       /\ ~\E t \in tasks : t.kind = "transfer" /\ t.o = o
    \/ \E d \in deferred : d.kind = "release" /\ u \in d.units /\ d.after[b.g] >= b.seq

BatchesOk ==
    \A b \in inflight :
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
             \* A commit of the block waits for the batch.
             /\ \A d \in deferred :
                  (d.kind = "commit" /\ x[2] \in d.units) => d.after[b.g] >= b.seq
             /\ \A r \in requests, p \in Positions :
                  (CommitsRow(r, x[1], p) /\ table[x[1]][p].u = x[2]) =>
                      b.g \in r.sg /\ (b.g \in r.by => r.after[b.g] >= b.seq)
        \* The copies it runs are still in flight.
        /\ \A u \in b.cw : \E c \in copies : c.write = u /\ c.g = b.g /\ c.seq = b.seq
        /\ \A u \in b.cr : \E c \in copies : c.read = u /\ c.g = b.g /\ c.seq = b.seq

-----------------------------------------------------------------------------

NoBad == bad = {}

IndInv ==
    /\ TypeInv
    /\ FreeClean /\ NoLeak /\ Outside /\ Regions
    /\ AllocOwner /\ SharedIff /\ SharedCached /\ CacheShared
    /\ Tables /\ OwnersOk /\ DeferredOk /\ CopiesOk /\ TasksOk
    /\ MovingUnmapped /\ LostOk
    /\ TablesMapped /\ HandoffOk /\ RequestsOk /\ BatchesOk
    /\ TrialVisibility
    /\ NoBad
=============================================================================
