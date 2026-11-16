-------------------------- MODULE MC_TransferPair --------------------------
(* MC_Transfer with two slot rows in each group, so that two owners can be *)
(* in the prefill group at once and in transfer to the decode group at     *)
(* once.                                                                   *)
EXTENDS MC_Transfer

CONSTANTS sc, sd

PairSlotGroup == [s \in Slots |-> IF s \in {sa, sc} THEN ga ELSE gb]
=============================================================================
