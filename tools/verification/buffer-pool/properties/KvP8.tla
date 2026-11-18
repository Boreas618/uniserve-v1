-------------------------------- MODULE KvP8 --------------------------------
(* P8 Placement of specs/buffer-pool.md §17, as formulas over KvCore.      *)
EXTENDS KvCore

(* P8 Placement: a block is read only where its content is. A copy reads a *)
(* device block in its group's memory and a host block on the host parts    *)
(* its group copies with, and both hold the block's content; a copy fills  *)
(* only host parts that do not hold its block.                             *)
CopiesPlaced ==
    \A c \in copies :
        /\ c.read \in Units => Mem(c.g) \in valid[c.read]
        /\ c.read \in HostUnits => GroupParts[c.g] \subseteq valid[c.read]
        /\ (c.fill /\ c.write \in HostUnits) => c.parts \cap valid[c.write] = {}

(* A crossing reads its block in its owner's group's memory, which holds    *)
(* it, and no batch in flight writes the block meanwhile.                   *)
CrossingsPlaced ==
    \A t \in tasks : t.kind = "replicate" =>
        \A u \in t.units :
            /\ Mem(ogroup[t.o]) \in valid[u]
            /\ \A b \in inflight : \A w \in b.writes : w[2] # u

(* An import, and another host's import through a borrow, reads the blocks  *)
(* of a lend, which no batch in flight writes.                              *)
ImportsPlaced ==
    \A t \in tasks : t.kind \in {"import", "borrow"} =>
        \A u \in t.units \ t.writes : \A b \in inflight : \A w \in b.writes : w[2] # u

P8 == CopiesPlaced /\ CrossingsPlaced /\ ImportsPlaced
=============================================================================
