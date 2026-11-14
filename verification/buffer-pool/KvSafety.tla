------------------------------ MODULE KvSafety ------------------------------
(* The safety properties of the buffer pool's design model together.      *)
EXTENDS KvP1, KvP2, KvP3, KvP5, KvP6, KvP8

Safety == TypeOK /\ P1 /\ P2 /\ P3 /\ P5 /\ P6 /\ P8
=============================================================================
