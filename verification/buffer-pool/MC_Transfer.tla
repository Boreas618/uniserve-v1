---------------------------- MODULE MC_Transfer ----------------------------
(* Two worker groups of one epoch over one unit space: a prefill group and *)
(* a decode group, with transfers between them.                            *)
EXTENDS KvCore, TLC

CONSTANTS ga, gb, sa, sb, e1

MCSlotGroup == [s \in Slots |-> IF s = sa THEN ga ELSE gb]
MCGroupEpoch == [g \in Groups |-> e1]
MCOwnerEpoch == [o \in Owners |-> e1]
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"}
Without(rule) == AllRules \ {rule}
NoR1 == Without("R1")
NoR2 == Without("R2")
NoR3 == Without("R3")
NoR9 == Without("R9")

(* Units and owners are interchangeable in this configuration.             *)
Symmetry == Permutations(Units) \cup Permutations(Owners)

(* Witness: false in some reachable state, which shows that the search     *)
(* reaches an owner attached to the group it was transferred to.           *)
NeverTransferred == \A o \in Owners : ~(ostate[o] = "live" /\ oslot[o] = sb /\ dbt[gb][o]["slot"].u = sb
                                         /\ \E p \in Positions : table[o][p] # Empty /\ dbt[gb][o][p].u # "none")
=============================================================================
