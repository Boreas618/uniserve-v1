----------------------------- MODULE PlaneSafety -----------------------------
(* The safety properties of the KV data plane (specs/kv-data-plane.md §3),  *)
(* as formulas over KvPlane.                                                *)
EXTENDS KvPlane

Active(q) == phase[q] \in {"routed", "handoff", "decode"}

(* D1 Faithful reads, the data plane's one invariant: every content a       *)
(* request reads, from its prefill's device cache or in its decode's pages, *)
(* is the content it needs.                                                 *)
Faithful == misread = {}

(* D1, in the state: every item the decode holds as delivered, from its own *)
(* cache, a push, a read, a pull or its own computation, holds the content  *)
(* the request needs, until the request ends.                               *)
Filled ==
    \A q \in Requests : Active(q) =>
        \A i \in ItemsOf(q) : Delivered(q, i) => content[df[q][i]] = Want(q, i, ds[q])

(* D2 Integrity: every page a tier indexes by a key holds that key's        *)
(* content: a device cache's, a host tier's and the store's.                *)
Integrity ==
    /\ \A f \in Frames : fkey[f] # NoKey => Matches(fkey[f], content[f])
    /\ \A h \in Slots : hkey[h] # NoKey => Matches(hkey[h], hcontent[h])
    /\ \A k \in Keys : store[k] # Absent => Matches(k, store[k])

(* D3 Reservation: every copy of a handoff writes a page its request's      *)
(* decode reserved for it and still holds.                                  *)
Reserved ==
    \A t \in xfers : owner[t.dst] = t.q /\ FrameReplica[t.dst] = rd[t.q]

(* D4 Lease: a page leased to a request stays that request's page, private  *)
(* or cached and pinned by it, until the lease ends.                        *)
LeaseHeld ==
    \A f \in Frames : \A q \in lease[f] : owner[f] = q \/ q \in pins[f]

(* D5 Holding: a request under way holds the decode's pages of its items,   *)
(* and the prefill's until it releases them; a copy holds what it reads.    *)
Holding ==
    /\ \A q \in Requests : Active(q) =>
          \A i \in ItemsOf(q) : df[q][i] # NONE => owner[df[q][i]] = q \/ q \in pins[df[q][i]]
    /\ \A q \in Requests : pstate[q] = "computed" =>
          \A i \in ItemsOf(q) : pf[q][i] # NONE => owner[pf[q][i]] = q \/ q \in pins[pf[q][i]]
    /\ \A t \in xfers : t.held =>
          CASE t.kind = "push" -> owner[t.src] = t.q \/ t.q \in pins[t.src]
            [] t.src = STORE -> TRUE
            [] OTHER -> hkey[t.src] = t.k
    /\ \A m \in moves : m.held =>
          CASE m.from = "device" -> fkey[m.src] = m.k
            [] m.from = "host" -> hkey[m.src] = m.k
            [] OTHER -> TRUE

(* D6 Unkeyed: an item without a key is never pulled.                       *)
UnkeyedPushed == \A t \in xfers : t.kind = "get" => t.i \in Full[t.q]

(* D7 One key, one page: a device cache indexes a key by one page per       *)
(* replica, a host tier by one page per node, and a private page is not     *)
(* cached.                                                                  *)
OneKey ==
    /\ \A r \in Replicas, k \in Keys : Cardinality(Cached(r, k)) <= 1
    /\ \A n \in Nodes, k \in Keys : Cardinality(Hosted(n, k)) <= 1
    /\ \A f \in Frames : owner[f] # NONE => fkey[f] = NoKey

Safety == Faithful /\ Filled /\ Integrity /\ Reserved /\ LeaseHeld /\ Holding /\ UnkeyedPushed /\ OneKey

(* G1, the guarantee to the convergence model: only the environment's       *)
(* switches change a role or a seed; no step of the data plane does.        *)
ConfigUntouched == [][(role' # role \/ seed' # seed) => switches' = switches + 1]_vars
=============================================================================
