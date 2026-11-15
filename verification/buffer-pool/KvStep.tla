-------------------------------- MODULE KvStep --------------------------------
(***************************************************************************)
(* The inductive step of IndInv for every step of Spec, from the theorems *)
(* of the eleven groups of actions, and the safety of the design model:   *)
(*     Spec => []IndInv        Spec => []Safety        Spec => []P8        *)
(* under the assumptions Design and Sizes.                                 *)
(***************************************************************************)
EXTENDS KvStep1, KvStep2, KvStep3, KvStep4, KvStep5, KvStep6, KvStep7, KvStep8, KvStep9,
        KvStep10, KvStep11, TLAPS

USE Sizes, Design

(* A step that changes no variable keeps every conjunct.                   *)
LEMMA Stutter ==
  ASSUME IndInv, UNCHANGED vars
  PROVE  IndInv'
  <1>1. UNCHANGED <<free, ostate, ogroup, oslot, table, cache, hcache, deferred, copies, ncopy,
                    tasks, valid, vis, trial, member, dbt, launched, inflight, handoff, requests,
                    alloc, sharedKey, remote, lost, bad>>
    BY DEF vars, pool, plane, rare
  <1> QED
    BY <1>1, TransfersFrame, TypeInvFrame, FreeCleanFrame, NoLeakFrame, OutsideFrame,
       RegionsFrame, AllocOwnerFrame, SharedIffFrame, SharedCachedFrame, CacheSharedFrame,
       TablesFrame, OwnersOkFrame, DeferredOkFrame, CopiesOkFrame, TasksOkFrame,
       MovingUnmappedFrame, LostOkFrame, TablesMappedFrame, HandoffOkFrame, RequestsOkFrame,
       BatchesOkFrame, CrossingsOkFrame, TrialVisibilityFrame, CopiesPlacedFrame,
       CrossingsPlacedFrame, NoBadFrame DEF IndInv

(* Every step of Serving is a step of one of the groups 1 to 8 and 11;   *)
(* the action of the hazard H14 is not a step of the design.              *)
LEMMA ServingStep ==
  ASSUME IndInv, Serving, UNCHANGED rare
  PROVE  IndInv'
  <1>1. CASE \E o \in Owners, g \in Groups : Attach(o, g)
    BY <1>1, Step1 DEF Next1
  <1>2. CASE \E o \in Owners, p \in Positions : Allocate(o, p)
    BY <1>2, Step1 DEF Next1
  <1>3. CASE \E o \in Owners, p \in Positions, k \in Keys : Pin(o, p, k) \/ Clone(o, p, k) \/ Load(o, p, k)
    <2>1. CASE \E o \in Owners, p \in Positions, k \in Keys : Pin(o, p, k)
      BY <2>1, Step1 DEF Next1
    <2>2. CASE \E o \in Owners, p \in Positions, k \in Keys : Clone(o, p, k) \/ Load(o, p, k)
      BY <2>2, Step2 DEF Next2
    <2> QED
      BY <1>3, <2>1, <2>2
  <1>4. CASE \/ \E o \in Owners, k \in Keys : Save(o, k) \/ Restore(o, k)
             \/ \E g \in Groups, k \in Keys : Store(g, k)
    BY <1>4, Step2 DEF Next2
  <1>5. CASE \/ \E k \in Keys : Evict(k) \/ EvictHost(k)
             \/ \E e \in Epochs : CommitUpdate(e) \/ Rollback(e)
             \/ \E u \in Units \cup HostUnits : Shrink(u) \/ Grow(u)
    BY <1>5, Step3
  <1>6. CASE \/ Issue
             \/ \E g \in Groups, r \in requests : ExecStamp(g, r)
             \/ \E h \in handoff : ExecMap(h) \/ DropCompletion(h)
    BY <1>6, Step4
  <1>7. CASE \/ \E g \in Groups : Launch(g)
             \/ \E b \in inflight : EndBatch(b)
             \/ \E c \in copies, ok \in BOOLEAN : ExecFinish(c, ok)
    BY <1>7, Step5
  <1>8. CASE \E r \in requests :
               \/ TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r) \/ TakeTransfer(r)
               \/ TakeReplicate(r)
    <2>1. CASE \E r \in requests :
                 TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r) \/ TakeTransfer(r)
      BY <2>1, Step6
    <2>2. CASE \E r \in requests : TakeReplicate(r)
      BY <2>2, Step11
    <2> QED
      BY <1>8, <2>1, <2>2
  <1>9. CASE \E d \in deferred :
               ReleaseDue(d) \/ CommitDue(d) \/ FinishDue(d) \/ TransferDue(d) \/ ReplicateDue(d)
    <2>1. CASE \E d \in deferred : ReleaseDue(d) \/ CommitDue(d) \/ FinishDue(d)
      BY <2>1, Step7
    <2>2. CASE \E d \in deferred : TransferDue(d)
      BY <2>2, Step8
    <2>3. CASE \E d \in deferred : ReplicateDue(d)
      BY <2>3, Step11
    <2> QED
      BY <1>9, <2>1, <2>2, <2>3
  <1>10. CASE \E t \in tasks, ok \in BOOLEAN : TransferEnd(t, ok) \/ ReplicateEnd(t, ok)
    <2>1. CASE \E t \in tasks, ok \in BOOLEAN : TransferEnd(t, ok)
      BY <2>1, Step8
    <2>2. CASE \E t \in tasks, ok \in BOOLEAN : ReplicateEnd(t, ok)
      BY <2>2, Step11
    <2> QED
      BY <1>10, <2>1, <2>2
  <1>11. ~\E g \in Groups, o \in Owners, p \in Positions : MapEarly(g, o, p)
    BY DEF MapEarly
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11 DEF Serving

THEOREM Step == IndInv /\ [Next]_vars => IndInv'
  <1> SUFFICES ASSUME IndInv, [Next]_vars PROVE IndInv'
    OBVIOUS
  <1>1. CASE Serving /\ UNCHANGED rare
    BY <1>1, ServingStep
  <1>2. CASE \/ \E k \in Keys, g \in Groups : Prefetch(k, "remote", g) \/ Prefetch(k, "peer", g)
             \/ \E k \in Keys : StoreRemote(k)
             \/ \E t \in tasks : PeerLost(t) \/ Deadline(t)
             \/ \E t \in tasks, ok \in BOOLEAN : PrefetchEnd(t, ok) \/ PutEnd(t, ok)
    BY <1>2, Step9 DEF Next9
  <1>3. CASE \E e \in Epochs : Lose(e)
    BY <1>3, Step10
  (* The actions of the hazards are not steps of the design.               *)
  <1>4. ~\E k \in Keys : EvictNot(k) \/ StoreOffered(k)
    BY DEF EvictNot, StoreOffered
  <1>5. CASE UNCHANGED vars
    BY <1>5, Stutter
  <1> QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5 DEF Next

THEOREM Invariant == Spec => []IndInv
  <1>1. Init => IndInv
    BY InitInv
  <1>2. IndInv /\ [Next]_vars => IndInv'
    BY Step
  <1> QED
    BY <1>1, <1>2, PTL DEF Spec

THEOREM Safe == Spec => []Safety
  <1>1. IndInv => Safety
    BY InvSafe
  <1> QED
    BY <1>1, Invariant, PTL

THEOREM SafeP8 == Spec => []P8
  <1>1. IndInv => P8
    BY InvSafe DEF Safety
  <1> QED
    BY <1>1, Invariant, PTL
=============================================================================
