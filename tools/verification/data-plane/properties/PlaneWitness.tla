---------------------------- MODULE PlaneWitness ----------------------------
(* Reachability: each formula states that a behaviour of the design never  *)
(* happens, and TLC must report it violated, which shows the safety        *)
(* properties are not met by a model that cannot reach the behaviour.      *)
EXTENDS PlaneSafety

Decoded(q, i, way) == phase[q] = "decode" /\ i \in ItemsOf(q) /\ via[q][i] = way

(* Each path delivers a full page, and push and read an unkeyed item.       *)
NoPush == ~\E q \in Requests : \E i \in Items : Decoded(q, i, "push") /\ fill[q][i] = "ok"
NoRead == ~\E q \in Requests : \E i \in Items : Decoded(q, i, "read") /\ fill[q][i] = "ok"
NoPullHost == ~\E q \in Requests : \E i \in Items : Decoded(q, i, "pull_host") /\ fill[q][i] = "ok"
NoPullStore == ~\E q \in Requests : \E i \in Items : Decoded(q, i, "pull_store") /\ fill[q][i] = "ok"
NoUnkeyedRead == ~\E q \in Requests : \E i \in Unkeyed : Decoded(q, i, "read") /\ fill[q][i] = "ok"

(* A pull the directory's stale account chose misses and is computed.       *)
NoStaleMiss ==
    ~\E q \in Requests : \E i \in Items :
        (Decoded(q, i, "pull_host") \/ Decoded(q, i, "pull_store")) /\ fill[q][i] = "redone"

(* A read whose lease expired before its acknowledgement is computed.       *)
NoExpiredRead == ~\E q \in Requests : \E i \in Items : Decoded(q, i, "read") /\ fill[q][i] = "redone"

(* A handoff between replicas of two seeds decodes: the decode computed     *)
(* what the prefill's pages would have given.                               *)
NoSeedMix == ~\E q \in Requests : phase[q] = "decode" /\ ps[q] # ds[q]

(* A request decodes on a replica whose role no longer takes decodes.       *)
NoRoleDrain == ~\E q \in Requests : phase[q] = "decode" /\ ~Decodes(role[rd[q]])

(* A request reuses its prefill's cached page of an earlier request.        *)
NoReuse == ~\E q \in Requests : pstate[q] # "none" /\ \E i \in Full[q] : \E r \in Requests :
                r # q /\ pf[r][i] = pf[q][i] /\ pf[q][i] # NONE /\ pstate[r] # "none"
=============================================================================
