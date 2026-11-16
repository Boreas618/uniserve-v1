-------------------------------- MODULE KvP6 --------------------------------
(* P6 Exactness of specs/buffer-pool.md §17, as a formula over KvCore. *)
EXTENDS KvCore

(* P6 Exactness: a completion states what the buffer pool holds when it is *)
(* produced. The handlers of the design answer from the state they leave;  *)
(* `bad` records the completions of the system as built that did not.      *)
Exactness == ~\E x \in bad : x[2] \in {"evicted", "stored"}

P6 == Exactness
=============================================================================
