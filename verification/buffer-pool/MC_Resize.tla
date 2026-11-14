------------------------------ MODULE MC_Resize -----------------------------
(* Two epochs of different layouts: each has a region of its own over the  *)
(* same device memory, which moves from the serving epoch's region to the  *)
(* target's by `Resize`.                                                   *)
EXTENDS KvInductive, KvP7, TLC

CONSTANTS ga, gb, sa, sb, e1, e2, o1, o2, a1, a2, b1, b2

MCSlotGroup == [s \in Slots |-> IF s = sa THEN ga ELSE gb]
MCGroupEpoch == [g \in Groups |-> IF g = ga THEN e1 ELSE e2]
MCOwnerEpoch == [o \in Owners |-> IF o = o1 THEN e1 ELSE e2]
(* The two layouts address the device differently: no block of one is      *)
(* valid for the other.                                                    *)
MCGroupMemory == [g \in Groups |-> g]
MCUnitRegion == [u \in Units |-> IF u \in {a1, a2} THEN "A" ELSE "B"]
MCGroupRegion == [g \in Groups |-> IF g = ga THEN "A" ELSE "B"]
(* Unit i of either region occupies chunk i of the device memory.          *)
MCUnitChunk == [u \in Units |-> IF u \in {a1, b1} THEN 1 ELSE 2]
MCMovable == Units
MCInitialUnits == {a1, a2}
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
NoR10 == AllRules \ {"R10"}

(* Witness: false in some reachable state, which shows that the search     *)
(* reaches the target's owner holding a block of its region while the      *)
(* serving epoch's owner still holds one of its own.                       *)
NeverBoth == ~(\E p \in Positions : table[o1][p].k = "private")
             \/ ~(\E p \in Positions : table[o2][p].k = "private")
=============================================================================
