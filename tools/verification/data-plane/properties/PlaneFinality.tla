---------------------------- MODULE PlaneFinality ----------------------------
(* D8 Finality: a routed request ends, whatever the directory states. A     *)
(* stale account makes a pull miss or a move skip, and the decode computes  *)
(* what was missed; it never stops a handoff. Checked under weak fairness   *)
(* of each request's own steps, a lease's expiry among them; the            *)
(* environment, cancellation, eviction, moves and reports need no fairness. *)
EXTENDS PlaneSafety

Steps(q) ==
    \/ Compute(q) \/ Reserve(q) \/ Choose(q) \/ Decode(q) \/ Finish(q) \/ Release(q)
    \/ \E i \in ItemsOf(q) : StartPush(q, i) \/ StartRead(q, i) \/ StartGet(q, i) \/ Recompute(q, i)
    \/ \E t \in xfers : t.q = q /\ (FinishPush(t) \/ FinishRead(t) \/ Ack(t) \/ FinishGet(t))
    \/ \E f \in Frames : Expire(q, f)

(* Moves in flight end, so the pages they hold are given back.              *)
Ends == \E m \in moves : FinishMove(m)

FairSpec == Spec /\ (\A q \in Requests : WF_vars(Steps(q))) /\ WF_vars(Ends)

Finality == \A q \in Requests : phase[q] = "routed" ~> phase[q] = "done"

(* The prefill's pages are released once the request has ended.             *)
Released == \A q \in Requests : phase[q] = "done" /\ pstate[q] = "computed" ~> pstate[q] = "released"
=============================================================================
