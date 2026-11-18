----------------------------- MODULE MC_Handoff -----------------------------
(* Two worker groups of one epoch over one unit space, each with a memory  *)
(* of its own: a prefill group and a decode group, between whose owners    *)
(* requests are handed off. An owner is lent, and another owner imports    *)
(* the lent blocks, or the blocks of a lend of another host's pool; another *)
(* host's server borrows a lend; a lend ends at any time.                   *)
EXTENDS KvInductive, KvP7, TLC

CONSTANTS ga, gb, sa, sb, e1

MCSlotGroup == [s \in Slots |-> IF s = sa THEN ga ELSE gb]
MCGroupEpoch == [g \in Groups |-> e1]
MCOwnerEpoch == [o \in Owners |-> e1]
MCGroupMemory == [g \in Groups |-> g]
(* One KV region; every unit has memory of its own and is in the region.     *)
MCUnitRegion == [u \in Units |-> "kv"]
MCUnitChunk == [u \in Units |-> u]
MCGroupRegion == [g \in Groups |-> "kv"]
MCMovable == {}
MCInitialUnits == AllUnits
AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
Without(rule) == AllRules \ {rule}
NoR1 == Without("R1")
NoR2 == Without("R2")
NoR3 == Without("R3")
NoR9 == Without("R9")

(* Units and owners are interchangeable in this configuration.             *)
Symmetry == Permutations(Units) \cup Permutations(Owners)

(* Witnesses, each false in some reachable state: an import reads the      *)
(* blocks of a lend of this pool that has ended, which its task still      *)
(* holds; a borrow holds the blocks of a lend that has ended.              *)
NeverReadAfterLend ==
    ~\E t \in tasks : t.kind = "import" /\ Reads(t) # {} /\ ~\E x \in tasks : x.kind = "lend"
NeverBorrowedAfterLend ==
    ~\E t \in tasks : t.kind = "borrow" /\ t.units # {} /\ ostate[t.o] = "gone"
=============================================================================
