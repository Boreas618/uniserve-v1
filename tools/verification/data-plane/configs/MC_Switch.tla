------------------------------ MODULE MC_Switch ------------------------------
(* Two requests of one prefix on two replicas on two nodes, while the      *)
(* configuration's convergence switches roles and epochs: up to two        *)
(* switches, of any replica's role to any role or of its seed to the other *)
(* seed, at any step. The device tier alone: no host tier and no move, so  *)
(* every page a request reads was written by a prefill or a decode.         *)
EXTENDS PlaneWitness, TLC

CONSTANTS r1, r2, n1, n2, d1, d2, d3, d4, d5, d6, q1, q2, f1, a, s1, s2

MCReplicaNode == (r1 :> n1) @@ (r2 :> n2)
MCFrameReplica == (d1 :> r1) @@ (d2 :> r1) @@ (d3 :> r1) @@ (d4 :> r2) @@ (d5 :> r2) @@ (d6 :> r2)
MCSlotNode == [h \in {} |-> n1]
MCFull == (q1 :> {f1}) @@ (q2 :> {f1})
MCPrefix == (q1 :> (f1 :> a)) @@ (q2 :> (f1 :> a))
MCLast == (q1 :> f1) @@ (q2 :> f1)
MCSeed == (r1 :> s1) @@ (r2 :> s1)
MCRole == (r1 :> "prefill") @@ (r2 :> "decode")

Symmetry == Permutations({d1, d2, d3}) \cup Permutations({d4, d5, d6}) \cup Permutations({q1, q2})
=============================================================================
