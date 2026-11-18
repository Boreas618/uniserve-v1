---------------------------- MODULE MC_Apalache ----------------------------
(* The inductiveness of IndInv for fixed constants and collections of      *)
(* bounded size, checked by Apalache: IndInv is the initial predicate over *)
(* generated values, one step of Next is taken, and IndInv is checked.     *)
EXTENDS KvInductive, Apalache

AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}

(* Two epochs; the first has two groups over memories m1 and m2, the       *)
(* second one group over m1; one KV region; one host unit. g2 reads the   *)
(* class owners of g1 hold, and copies host copies with both host parts,  *)
(* the other groups with pa alone. `Resize` moves u2, the host unit and   *)
(* g3's slot row.                                                          *)
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
    /\ InitialUnits = {"u1", "u2", "h1", "s1", "s2", "s3"}
    /\ Movable = {"u2", "h1", "s3"}
    /\ MaxBatch = 2
    /\ MaxRequests = 2
    /\ MaxCopies = 2
    /\ MaxTasks = 1
    /\ Losable = {"e1", "e2"}
    /\ Hazards = {}
    /\ Eager \in BOOLEAN
    /\ Rules = AllRules
    /\ HostParts = {"pa", "pb"}
    /\ GroupParts = [g \in {"g1", "g2", "g3"} |-> IF g = "g2" THEN {"pa", "pb"} ELSE {"pa"}]
    /\ Readers = [g \in {"g1", "g2", "g3"} |-> IF g = "g1" THEN {"g2"} ELSE {}]
    /\ Handoffs = TRUE

(* Any state that satisfies IndInv, with collections of at most three      *)
(* elements.                                                               *)
IndInit ==
    /\ free = Gen(6) /\ ostate = Gen(2) /\ ogroup = Gen(2) /\ oslot = Gen(2)
    /\ table = Gen(2) /\ cache = Gen(1) /\ hcache = Gen(1)
    /\ deferred = Gen(3) /\ copies = Gen(2) /\ ncopy = Gen(1) /\ tasks = Gen(2)
    /\ valid = Gen(3) /\ vis = Gen(1) /\ member = Gen(6) /\ trial = Gen(2)
    /\ dbt = Gen(3) /\ launched = Gen(3) /\ inflight = Gen(3) /\ handoff = Gen(3)
    /\ requests = Gen(3)
    /\ alloc = Gen(6) /\ sharedKey = Gen(6) /\ remote = Gen(1) /\ lost = Gen(2)
    /\ bad = Gen(1)
    /\ IndInv

(* The actions in eleven groups, so that the inductive step is checked for *)
(* each group by a process of its own: Next is their disjunction.          *)
Next1 == /\ \/ \E o \in Owners, g \in Groups : Attach(o, g)
            \/ \E o \in Owners, p \in Positions : Allocate(o, p)
            \/ \E o \in Owners, p \in Positions, k \in Keys : Pin(o, p, k)
         /\ UNCHANGED rare
Next2 == /\ \/ \E o \in Owners, p \in Positions, k \in Keys : Clone(o, p, k) \/ Load(o, p, k)
            \/ \E o \in Owners, k \in Keys : Save(o, k) \/ Restore(o, k)
            \/ \E g \in Groups, k \in Keys : Store(g, k)
         /\ UNCHANGED rare
Next3 == /\ \/ \E k \in Keys : Evict(k) \/ EvictHost(k)
            \/ \E e \in Epochs : CommitUpdate(e) \/ Rollback(e)
            \/ \E u \in AllUnits : Shrink(u) \/ Grow(u)
         /\ UNCHANGED rare
Next4 == /\ \/ Issue
            \/ \E g \in Groups, r \in requests : ExecStamp(g, r)
            \/ \E h \in handoff : ExecMap(h) \/ DropCompletion(h)
         /\ UNCHANGED rare
Next5 == /\ \/ \E g \in Groups : Launch(g)
            \/ \E b \in inflight : EndBatch(b)
            \/ \E c \in copies, ok \in BOOLEAN : ExecFinish(c, ok)
         /\ UNCHANGED rare
Next6 == /\ \E r \in requests :
              TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r) \/ TakeLend(r)
         /\ UNCHANGED rare
Next7 == /\ \E d \in deferred : ReleaseDue(d) \/ CommitDue(d) \/ FinishDue(d)
         /\ UNCHANGED rare
Next8 == /\ \/ \E d \in deferred : LendDue(d)
            \/ \E o \in Owners, ps \in SUBSET Positions, l \in Owners \cup {NONE} : Import(o, ps, l)
            \/ \E o \in Owners : Borrow(o)
            \/ \E t \in tasks : GiveBack(t) \/ ReleaseBorrow(t)
            \/ \E t \in tasks, ok \in BOOLEAN : ImportEnd(t, ok)
         /\ UNCHANGED rare
Next9 == \/ \E k \in Keys, g \in Groups : Prefetch(k, g)
         \/ \E k \in Keys : StoreRemote(k)
         \/ \E t \in tasks : Deadline(t)
         \/ \E t \in tasks, ok \in BOOLEAN : PrefetchEnd(t, ok) \/ PutEnd(t, ok)
Next10 == \E e \in Epochs : Lose(e)
Next11 == /\ \/ \E r \in requests : TakeReplicate(r)
             \/ \E d \in deferred : ReplicateDue(d)
             \/ \E t \in tasks, ok \in BOOLEAN : ReplicateEnd(t, ok)
          /\ UNCHANGED rare

(* Witnesses that IndInit is satisfiable by states that hold work: each is  *)
(* an invariant Apalache must refute.                                      *)
NoState == FALSE
NoWork == ~(inflight # {} /\ deferred # {} /\ requests # {} /\ copies # {} /\ tasks # {})
=============================================================================
