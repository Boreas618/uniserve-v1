----------------------------- MODULE KvFinality -----------------------------
(***************************************************************************)
(* P4 Finality of specs/buffer-pool.md §17, as a property of the design    *)
(* model that TLC checks on a small configuration: a request that waits on *)
(* a copy or on a task goes on. It is checked, not proved.                 *)
(*                                                                         *)
(* It holds under what the design relies on: the scheduler gives a copy up *)
(* at its own bound, the requests it issues are stamped and taken, a batch *)
(* ends, deferred work that is due is done, and a task's deadline passes.  *)
(* It does not rely on a copy's executor returning its `Finish` or on a    *)
(* task ever completing.                                                   *)
(***************************************************************************)
EXTENDS KvCore

(* The scheduler gives copy c up: the `Finish { ok: false }` of `Issue`.   *)
GiveUp(c) ==
    /\ Cardinality(requests) < MaxRequests
    /\ "H13" \notin Hazards
    /\ \E x \in copies :
         /\ x.id = c
         /\ GroupEpoch[x.g] \notin lost
         /\ requests' = requests \cup
              {Issued(Request("finish", GroupEpoch[x.g], NONE, {}, NONE, NONE, c, FALSE))}
    /\ UNCHANGED <<pool, dbt, launched, inflight, handoff, alloc, sharedKey, rare>>

Fairness ==
    /\ \A c \in 1 .. MaxCopies : SF_vars(GiveUp(c))
    /\ WF_vars((\E g \in Groups, r \in requests : ExecStamp(g, r)) /\ UNCHANGED rare)
    /\ WF_vars((\E r \in requests :
                  TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r)
                    \/ TakeLend(r)) /\ UNCHANGED rare)
    /\ WF_vars((\E d \in deferred : FinishDue(d)) /\ UNCHANGED rare)
    /\ WF_vars((\E b \in inflight : EndBatch(b)) /\ UNCHANGED rare)
    /\ WF_vars(\E t \in tasks : Deadline(t))

LiveSpec == Spec /\ Fairness

(* A copy in flight ends: its blocks are let go.                           *)
CopyFinality ==
    \A i \in 1 .. MaxCopies :
        (\E c \in copies : c.id = i) ~> ~(\E c \in copies : c.id = i)

(* A request that waits on a task is answered: the task ends or passes its *)
(* deadline.                                                               *)
Waits(kind, k) == \E t \in tasks : t.kind = kind /\ t.k = k /\ ~t.expired
TaskFinality ==
    \A k \in Keys :
        /\ Waits("prefetch", k) ~> ~Waits("prefetch", k)
        /\ Waits("put", k) ~> ~Waits("put", k)
=============================================================================
