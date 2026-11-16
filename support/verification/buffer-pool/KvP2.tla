-------------------------------- MODULE KvP2 --------------------------------
(* P2 Integrity of specs/buffer-pool.md §17, as formulas over KvCore. *)
EXTENDS KvCore

(* P2 Integrity: a shared block is never private again, keeps its key, and *)
(* is written neither by a batch nor by a copy; a copy reads a shared      *)
(* block.                                                                  *)
KeyStable ==
    \A u \in AllUnits :
        sharedKey[u] # NONE =>
            /\ u \notin free
            /\ cache[sharedKey[u]] = u \/ hcache[sharedKey[u]] = u
            /\ TableHolders(u) = {} /\ CopyWriters(u) = {}
PinsCached ==
    \A o \in Owners, p \in Positions :
        table[o][p].k = "shared" => sharedKey[table[o][p].u] # NONE
OneKey == \A u \in AllUnits : Cardinality(CacheKeys(u)) <= 1
(* A block is committed only after `completed` reached the stamp of its    *)
(* commit, and the destination of a copy becomes shared only after the     *)
(* stamp of its `Finish`: nothing in flight writes memory in which a       *)
(* shared block is valid.                                                  *)
SharedUnwritten ==
    \A b \in inflight :
        /\ \A w \in b.writes : alloc[w[2]] # "shared"
        /\ \A u \in b.cw : alloc[u] = "shared" => Mem(b.g) \notin valid[u]
(* A block is stored, loaded, cloned or saved only as a shared block or as *)
(* its owner's private one.                                                *)
CopiesReadShared ==
    \A b \in inflight : \A u \in b.cr :
        \E c \in copies :
            /\ c.read = u /\ c.g = b.g /\ c.seq = b.seq
            /\ alloc[u] = "shared" \/ alloc[u] = c.o
(* No copy is cached from a source that was not held throughout its read.  *)
SourcesHeld == ~\E x \in bad : x[2] \in {"host", "remote"}

P2 == KeyStable /\ PinsCached /\ OneKey /\ SharedUnwritten /\ CopiesReadShared /\ SourcesHeld
=============================================================================
