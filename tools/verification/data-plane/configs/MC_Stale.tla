------------------------------ MODULE MC_Stale ------------------------------
(* Two requests of two prefixes through one prefill replica and one decode  *)
(* replica on one node whose host tier has one page: the directory reports  *)
(* the page under one key, and the page is then evicted and given the       *)
(* other; the store's events lag too. Full pages alone (the unkeyed items  *)
(* never meet the directory), one seed, no switch.                          *)
EXTENDS PlaneWitness, TLC

CONSTANTS r1, r2, n1, d1, d2, d3, d4, h1, q1, q2, f1, a, b, s1

MCReplicaNode == (r1 :> n1) @@ (r2 :> n1)
MCFrameReplica == (d1 :> r1) @@ (d2 :> r1) @@ (d3 :> r2) @@ (d4 :> r2)
MCSlotNode == h1 :> n1
MCFull == (q1 :> {f1}) @@ (q2 :> {f1})
MCPrefix == (q1 :> (f1 :> a)) @@ (q2 :> (f1 :> b))
MCLast == (q1 :> f1) @@ (q2 :> f1)
MCSeed == (r1 :> s1) @@ (r2 :> s1)
MCRole == (r1 :> "prefill") @@ (r2 :> "decode")

Symmetry == Permutations({d1, d2}) \cup Permutations({d3, d4})
=============================================================================
