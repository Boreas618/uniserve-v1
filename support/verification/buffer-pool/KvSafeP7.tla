------------------------------ MODULE KvSafeP7 ------------------------------
(***************************************************************************)
(* P7 Capacity identity holds in every reachable state of the design      *)
(* model: Spec => []IndInv (KvStep) and IndInv => P7 (KvProofP7).          *)
(***************************************************************************)
EXTENDS KvStep, KvProofP7

THEOREM SafeP7 == Spec => []P7
  <1>1. IndInv => P7  BY InvP7
  <1>. QED  BY <1>1, Invariant, PTL
=============================================================================
