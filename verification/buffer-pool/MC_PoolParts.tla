---------------------------- MODULE MC_PoolParts ----------------------------
(* MC_Pool over two host parts: ga's ranks copy with part pa of one host,  *)
(* gb's with pa and pb, the same part index on a second host. The          *)
(* configuration `parts` of the Rust buffer pool's enumeration.            *)
EXTENDS MC_Pool

CONSTANTS pa, pb

MCSplitParts == [g \in Groups |-> IF g = ga THEN {pa} ELSE {pa, pb}]
=============================================================================
