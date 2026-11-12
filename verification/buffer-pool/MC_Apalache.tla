---------------------------- MODULE MC_Apalache ----------------------------
(* The inductiveness of IndInv for fixed constants and collections of      *)
(* bounded size, checked by Apalache: IndInv is the initial predicate over *)
(* generated values, one step of Next is taken, and IndInv is checked.     *)
EXTENDS KvInductive, Apalache

AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"}

(* Two epochs; the first has two groups over memories m1 and m2, the       *)
(* second one group over m1; one KV region; one host unit.                 *)
ConstInit ==
    /\ Units = {"u1", "u2"}
    /\ Slots = {"s1", "s2", "s3"}
    /\ HostUnits = {"h1"}
    /\ Keys = {"k1"}
    /\ Owners = {"o1", "o2"}
    /\ Groups = {"g1", "g2", "g3"}
    /\ Positions = {"p1"}
    /\ Epochs = {"e1", "e2"}
    /\ SlotGroup = [s \in {"s1", "s2", "s3"} |->
                      IF s = "s1" THEN "g1" ELSE IF s = "s2" THEN "g2" ELSE "g3"]
    /\ GroupEpoch = [g \in {"g1", "g2", "g3"} |-> IF g = "g3" THEN "e2" ELSE "e1"]
    /\ OwnerEpoch = [o \in {"o1", "o2"} |-> IF o = "o1" THEN "e1" ELSE "e2"]
    /\ GroupMemory = [g \in {"g1", "g2", "g3"} |-> IF g = "g2" THEN "m2" ELSE "m1"]
    /\ TrialEpochs = {"e2"}
    /\ UnitRegion = [u \in {"u1", "u2"} |-> "kv"]
    /\ UnitChunk = [u \in {"u1", "u2"} |-> u]
    /\ GroupRegion = [g \in {"g1", "g2", "g3"} |-> "kv"]
    /\ InitialUnits = {"u1", "u2"}
    /\ Movable = {"u2"}
    /\ MaxBatch = 2
    /\ MaxRequests = 2
    /\ MaxCopies = 2
    /\ MaxTasks = 1
    /\ Losable = {"e1", "e2"}
    /\ Hazards = {}
    /\ Eager \in BOOLEAN
    /\ Rules = AllRules

(* Any state that satisfies IndInv, with collections of at most three      *)
(* elements.                                                               *)
IndInit ==
    /\ free = Gen(6) /\ ostate = Gen(2) /\ ogroup = Gen(2) /\ oslot = Gen(2)
    /\ table = Gen(2) /\ cache = Gen(1) /\ hcache = Gen(1)
    /\ deferred = Gen(3) /\ copies = Gen(2) /\ ncopy = Gen(1) /\ tasks = Gen(2)
    /\ valid = Gen(2) /\ vis = Gen(1) /\ member = Gen(2) /\ trial = Gen(2)
    /\ dbt = Gen(3) /\ launched = Gen(3) /\ inflight = Gen(3) /\ handoff = Gen(3)
    /\ requests = Gen(3)
    /\ alloc = Gen(6) /\ sharedKey = Gen(6) /\ remote = Gen(1) /\ lost = Gen(2)
    /\ bad = Gen(1)
    /\ IndInv

(* Witnesses that IndInit is satisfiable by states that hold work: each is  *)
(* an invariant Apalache must refute.                                      *)
NoState == FALSE
NoWork == ~(inflight # {} /\ deferred # {} /\ requests # {} /\ copies # {} /\ tasks # {})
=============================================================================
