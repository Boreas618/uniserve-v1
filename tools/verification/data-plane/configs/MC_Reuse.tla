------------------------------ MODULE MC_Reuse ------------------------------
(* Two requests of one prefix through one prefill replica and one decode    *)
(* replica on one node: the second reuses the first's cached page on either *)
(* side, its pages may take the pages the first released, and the decode    *)
(* may pull from the store, where one move can put the prefix's page; no   *)
(* host tier, one seed, no switch.                                         *)
EXTENDS PlaneWitness, TLC

CONSTANTS r1, r2, n1, d1, d2, d3, d4, d5, d6, q1, q2, f1, a, s1

MCReplicaNode == (r1 :> n1) @@ (r2 :> n1)
MCFrameReplica == (d1 :> r1) @@ (d2 :> r1) @@ (d3 :> r1) @@ (d4 :> r2) @@ (d5 :> r2) @@ (d6 :> r2)
MCSlotNode == [h \in {} |-> n1]
MCFull == (q1 :> {f1}) @@ (q2 :> {f1})
MCPrefix == (q1 :> (f1 :> a)) @@ (q2 :> (f1 :> a))
MCLast == (q1 :> f1) @@ (q2 :> f1)
MCSeed == (r1 :> s1) @@ (r2 :> s1)
MCRole == (r1 :> "prefill") @@ (r2 :> "decode")

(* The pages of one replica are interchangeable, as are the two requests.   *)
Symmetry == Permutations({d1, d2, d3}) \cup Permutations({d4, d5, d6}) \cup Permutations({q1, q2})
=============================================================================
