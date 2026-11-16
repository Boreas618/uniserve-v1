---------------------------- MODULE MC_PoolCross ----------------------------
(* MC_Pool over two memories, the decode group gb reading in its own memory *)
(* what owners of the prefill group ga hold: crossings arrive into gb.     *)
(* The configuration `cross` of the Rust buffer pool's enumeration.        *)
EXTENDS MC_Pool

MCCrossReaders == [g \in Groups |-> IF g = ga THEN {gb} ELSE {}]
=============================================================================
