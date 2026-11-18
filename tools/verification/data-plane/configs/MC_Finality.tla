----------------------------- MODULE MC_Finality -----------------------------
(* The configuration of MC_Handoff, with no symmetry (which liveness        *)
(* checking does not admit), under the fairness of PlaneFinality.           *)
EXTENDS PlaneFinality, TLC

CONSTANTS r1, r2, n1, n2, d1, d2, d3, d4, h1, h2, q1, f1, a, s1

MCReplicaNode == (r1 :> n1) @@ (r2 :> n2)
MCFrameReplica == (d1 :> r1) @@ (d2 :> r1) @@ (d3 :> r2) @@ (d4 :> r2)
MCSlotNode == (h1 :> n1) @@ (h2 :> n2)
MCFull == q1 :> {f1}
MCPrefix == q1 :> (f1 :> a)
MCLast == q1 :> f1
MCSeed == (r1 :> s1) @@ (r2 :> s1)
MCRole == (r1 :> "prefill") @@ (r2 :> "decode")
=============================================================================
