-------------------------------- MODULE KvP1 --------------------------------
(* P1 Isolation of specs/buffer-pool.md §17, as formulas over KvCore. *)
EXTENDS KvCore

(* P1 Isolation, the buffer pool's part: a free unit has no holder, and a  *)
(* private block has one owner or one copy that writes it and is neither   *)
(* pinned nor cached.                                                      *)
FreeUnheld == \A u \in free : ~Held(u)
Exclusive ==
    \A u \in AllUnits :
        /\ Cardinality(TableHolders(u)) + Cardinality(CopyWriters(u)) <= 1
        /\ (TableHolders(u) # {} \/ CopyWriters(u) # {}) => ~Pinned(u) /\ CacheKeys(u) = {}

(* P1 Isolation, with A1 to A4: a write lands only on a block allocated to *)
(* the writing request, nothing a batch in flight addresses was reclaimed  *)
(* or given to another owner, and a copy in flight still holds what it     *)
(* reads and writes.                                                       *)
WriteIsolation ==
    \A b \in inflight : \A w \in b.writes : w[2] \notin free /\ alloc[w[2]] = w[1]
ReadIsolation ==
    \A b \in inflight : \A r \in b.reads :
        /\ r[2] \notin free
        /\ alloc[r[2]] \in {r[1], "shared"}
        /\ r[2] \in Units => Mem(b.g) \in valid[r[2]]
CopyIsolation ==
    \A b \in inflight :
        /\ \A u \in b.cw : \E c \in copies : c.write = u /\ c.g = b.g /\ c.seq = b.seq
        /\ \A u \in b.cr : \E c \in copies : c.read = u /\ c.g = b.g /\ c.seq = b.seq

(* A task in flight writes only blocks of its owner, and of a shared block *)
(* only a memory in which the block is not valid.                          *)
TaskIsolation ==
    \A t \in tasks :
        CASE t.kind = "import" -> \A u \in t.writes : u \notin free /\ alloc[u] = t.o
          \* A lend writes nothing; a borrow reads the lent blocks for another host.
          [] t.kind \in {"lend", "borrow"} -> t.writes = {}
          \* A prefetch writes a host block that is its own until it ends.
          [] t.kind = "prefetch" -> t.h \notin free /\ alloc[t.h] = "task"
          \* A remote put reads a host block that stays its key's copy.
          [] t.kind = "put" -> t.h \notin free /\ hcache[t.k] = t.h
          \* A crossing writes only its owner's block, and of a shared block only
          \* the reading group's memory while the block is not valid there.
          [] t.kind = "replicate" ->
               \A u \in t.writes :
                  /\ u \notin free
                  /\ alloc[u] = t.o \/ (alloc[u] = "shared" /\ Mem(t.to) \notin valid[u])

P1 == FreeUnheld /\ Exclusive /\ WriteIsolation /\ ReadIsolation /\ CopyIsolation /\ TaskIsolation
=============================================================================
