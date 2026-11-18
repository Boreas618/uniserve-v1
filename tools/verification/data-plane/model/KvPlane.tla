------------------------------- MODULE KvPlane -------------------------------
(***************************************************************************)
(* The design model of the KV data plane of the KV cache stack: pages      *)
(* addressed by content key on three tiers (the replicas' device memory,   *)
(* each node's host tier, Mooncake Store), the cluster directory that      *)
(* summarises what each node holds and may lag it, the prefill-to-decode   *)
(* handoff by its three paths (push into pages the decode reserved, read   *)
(* of the prefill's device pages under a lease, pull by key from the host  *)
(* tier or the store), the tiering moves a KV program decides from the     *)
(* directory, and the configuration's convergence as the environment:      *)
(* role switches and epoch switches of replicas, under the assumptions E1  *)
(* to E4.                                                                  *)
(*                                                                         *)
(* A page's content is the value its key names: what the weights of a seed *)
(* compute for a prefix of tokens. A request's items are its full pages,   *)
(* which have keys, and its unkeyed items (the tokens after its last full  *)
(* page and the live recurrent state), which have none. The one property   *)
(* the data plane keeps (`specs/kv-data-plane.md` §3) is that every        *)
(* content a request reads is the content it needs, through role switches, *)
(* epoch switches, a stale directory and moves in flight.                  *)
(*                                                                         *)
(* Each hazard is behaviour switched on by its name in `Hazards`: a guard  *)
(* of the design taken away, which the configurations show violates a      *)
(* property.                                                               *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Replicas,     \* the replicas of the deployment, each with its scheduler
    Nodes,        \* the nodes, each with one Storage Server and its host tier
    ReplicaNode,  \* [Replicas -> Nodes]: the node a replica's device memory is on
    Frames,       \* device pages: the pages of every replica's device memory
    FrameReplica, \* [Frames -> Replicas]: the replica a device page belongs to
    Slots,        \* host pages: the pages of every node's host tier
    SlotNode,     \* [Slots -> Nodes]: the node a host page is on
    Requests,     \* the requests, each routed once
    Full,         \* [Requests -> SUBSET FullNames]: a request's full pages
    Prefix,       \* [Requests -> [FullNames -> Prefixes]]: the token prefix each full page ends
    Last,         \* [Requests -> FullNames]: a request's last full page
    Unkeyed,      \* the items every request has without a key: "tail", "state"
    Prefixes,     \* token prefixes, from the start of a prompt to the end of a page
    Seeds,        \* page seeds: the weights, precision and format that compute pages
    InitialSeed,  \* [Replicas -> Seeds]
    InitialRole,  \* [Replicas -> Roles]
    MaxSwitches,  \* the role and epoch switches the environment makes (a bound)
    MaxMoves,     \* the tiering moves the KV program asks for (a bound)
    Hazards       \* the hazards switched on

VARIABLES
    \* Device memory, per page.
    content,  \* [Frames -> Content]: what the page holds
    fkey,     \* [Frames -> Keys \cup {NoKey}]: the key its replica's device cache indexes it by
    owner,    \* [Frames -> Requests \cup {NONE}]: the request a private page belongs to
    pins,     \* [Frames -> SUBSET Requests]: the requests that use a cached page
    lease,    \* [Frames -> SUBSET Requests]: the requests a direct read lease holds it for
    \* Host tiers and the store.
    hkey,     \* [Slots -> Keys \cup {NoKey}]: the key a host tier's index caches the page by
    hcontent, \* [Slots -> Content]
    store,    \* [Keys -> Content \cup {Absent}]: Mooncake Store's objects
    \* The cluster directory.
    dir,      \* [Nodes -> SUBSET Entry]: what each node last reported it holds
    seen,     \* SUBSET Keys: the keys Mooncake's object events last stated stored
    \* The environment: what the configuration's convergence sets.
    role,     \* [Replicas -> Roles]
    seed,     \* [Replicas -> Seeds]: the seed of the replica's serving epoch
    switches, \* the role and epoch switches made
    \* Requests.
    phase,    \* [Requests -> {"new", "routed", "handoff", "decode", "done"}]
    rp,       \* [Requests -> Replicas \cup {NONE}]: the replica that computes the prompt
    rd,       \* [Requests -> Replicas \cup {NONE}]: the replica that generates the tokens
    ps,       \* [Requests -> Seeds \cup {NONE}]: the seed of the prefill's epoch at routing
    ds,       \* [Requests -> Seeds \cup {NONE}]: the seed of the decode's epoch at routing
    pstate,   \* [Requests -> {"none", "computed", "released"}]: the prefill's pages
    pf,       \* [Requests -> [Items -> Frames \cup {NONE}]]: the prefill's page of each item
    df,       \* [Requests -> [Items -> Frames \cup {NONE}]]: the decode's page of each item
    via,      \* [Requests -> [Items -> Ways]]: how each item reaches the decode
    fill,     \* [Requests -> [Items -> Fills]]: whether the decode's page holds it yet
    \* Copies in flight.
    xfers,    \* the handoff's copies: pushes, reads and gets
    moves,    \* the tiering moves
    nmoves,   \* the tiering moves asked for
    \* History, assigned by actions and read by no guard.
    misread   \* the reads that returned content other than the reader needs

device == <<content, fkey, owner, pins, lease>>
tiers == <<hkey, hcontent, store>>
known == <<dir, seen>>
env == <<role, seed, switches>>
reqs == <<phase, rp, rd, ps, ds, pstate, pf, df, via, fill>>
flight == <<xfers, moves, nmoves>>
vars == <<device, tiers, known, env, reqs, flight, misread>>

-----------------------------------------------------------------------------
NONE == "none"
STORE == "store"
Roles == {"prefill", "decode", "mixed"}
Prefills(x) == x \in {"prefill", "mixed"}
Decodes(x) == x \in {"decode", "mixed"}

FullNames == UNION {Full[q] : q \in Requests}
Items == FullNames \cup Unkeyed
ItemsOf(q) == Full[q] \cup Unkeyed

(* A key names the seed and the token prefix: the chain of a request's      *)
(* pages from the root of its seed (`page_root`, `page_key`). The hazard    *)
(* SeedlessKey roots the chain at no seed, so pages two epochs of different *)
(* weights compute for one prefix share a key.                              *)
Key(s, p) == [s |-> IF "SeedlessKey" \in Hazards THEN "any" ELSE s, p |-> p]
Keys == {Key(s, p) : s \in Seeds, p \in Prefixes}
NoKey == [s |-> NONE, p |-> NONE]

(* Content. A full page's is what seed s computes for prefix p; an unkeyed  *)
(* item's is the live state request q's prompt leaves under seed s; Zero is *)
(* what a page never written holds, Absent marks a key the store lacks.     *)
Page(s, p) == [s |-> s, p |-> p, q |-> NONE, u |-> NONE]
Live(s, q, u) == [s |-> s, p |-> NONE, q |-> q, u |-> u]
Zero == [s |-> NONE, p |-> NONE, q |-> NONE, u |-> NONE]
Absent == [s |-> NONE, p |-> NONE, q |-> NONE, u |-> "absent"]

(* What request q needs in its page of item i, computed under seed s.       *)
Want(q, i, s) == IF i \in Full[q] THEN Page(s, Prefix[q][i]) ELSE Live(s, q, i)

(* Whether content c is what key k names.                                   *)
Matches(k, c) == c.q = NONE /\ c.p = k.p /\ c.p # NONE /\ (k.s = "any" \/ c.s = k.s)

(* The key the decode names item i of q by: its full page's under the      *)
(* decode's seed. An unkeyed item has none; the hazard KeyedTail names it   *)
(* by the nearest key, the last full page's, whose stored copy is the       *)
(* checkpoint at that page's boundary.                                      *)
ItemKey(q, i) ==
    IF i \in Full[q] THEN Key(ds[q], Prefix[q][i]) ELSE Key(ds[q], Prefix[q][Last[q]])

FramesOf(r) == {f \in Frames : FrameReplica[f] = r}
SlotsOf(n) == {h \in Slots : SlotNode[h] = n}
Cached(r, k) == {f \in FramesOf(r) : fkey[f] = k}
Hosted(n, k) == {h \in SlotsOf(n) : hkey[h] = k}

(* What copies in flight hold: the sources their holders keep for them and  *)
(* the destinations they write. A read's source is held by its lease, not   *)
(* here. The hazards StaleAddress and UnreservedPush leave out what their   *)
(* copies would have held.                                                  *)
Held ==
    {t.src : t \in {x \in xfers : x.held}} \cup
    {t.dst : t \in {x \in xfers : x.kind # "push" \/ "UnreservedPush" \notin Hazards}} \cup
    {m.src : m \in {x \in moves : x.held}} \cup {m.dst : m \in moves}

(* A lease keeps a page for its reader; EarlyRelease releases without it.  *)
Unleased(f) == lease[f] = {} \/ "EarlyRelease" \in Hazards

Free(f) == owner[f] = NONE /\ fkey[f] = NoKey /\ pins[f] = {} /\ Unleased(f) /\ f \notin Held
FreeFrames(r) == {f \in FramesOf(r) : Free(f)}
FreeSlots(n) == {h \in SlotsOf(n) : hkey[h] = NoKey /\ h \notin Held}

Injective(a) == \A i, j \in DOMAIN a : i # j => a[i] # a[j]

-----------------------------------------------------------------------------
Init ==
    /\ content = [f \in Frames |-> Zero]
    /\ fkey = [f \in Frames |-> NoKey]
    /\ owner = [f \in Frames |-> NONE]
    /\ pins = [f \in Frames |-> {}]
    /\ lease = [f \in Frames |-> {}]
    /\ hkey = [h \in Slots |-> NoKey]
    /\ hcontent = [h \in Slots |-> Zero]
    /\ store = [k \in Keys |-> Absent]
    /\ dir = [n \in Nodes |-> {}]
    /\ seen = {}
    /\ role = InitialRole
    /\ seed = InitialSeed
    /\ switches = 0
    /\ phase = [q \in Requests |-> "new"]
    /\ rp = [q \in Requests |-> NONE]
    /\ rd = [q \in Requests |-> NONE]
    /\ ps = [q \in Requests |-> NONE]
    /\ ds = [q \in Requests |-> NONE]
    /\ pstate = [q \in Requests |-> "none"]
    /\ pf = [q \in Requests |-> [i \in Items |-> NONE]]
    /\ df = [q \in Requests |-> [i \in Items |-> NONE]]
    /\ via = [q \in Requests |-> [i \in Items |-> NONE]]
    /\ fill = [q \in Requests |-> [i \in Items |-> NONE]]
    /\ xfers = {}
    /\ moves = {}
    /\ nmoves = 0
    /\ misread = {}

-----------------------------------------------------------------------------
(* The routing program picks a prefill replica and a decode replica by     *)
(* their roles. It may read the directory's hits, which may be stale; any  *)
(* pair of the right roles is a route it may give, so the model lets it    *)
(* give each. The request is bound to each replica's serving epoch (E3).   *)
Route(q) ==
    /\ phase[q] = "new"
    /\ \E p, d \in Replicas :
        /\ p # d
        /\ Prefills(role[p])
        /\ Decodes(role[d])
        /\ rp' = [rp EXCEPT ![q] = p]
        /\ rd' = [rd EXCEPT ![q] = d]
        /\ ps' = [ps EXCEPT ![q] = seed[p]]
        /\ ds' = [ds EXCEPT ![q] = seed[d]]
    /\ phase' = [phase EXCEPT ![q] = "routed"]
    /\ UNCHANGED <<device, tiers, known, env, pstate, pf, df, via, fill, flight, misread>>

(* The prefill computes the prompt: a full page its device cache holds     *)
(* under the page's key is reused (a read: what it holds must be what the  *)
(* request needs), every other item is written into a free page; the new  *)
(* full pages are cached under their keys, the unkeyed items stay private. *)
Compute(q) ==
    /\ phase[q] = "routed"
    /\ pstate[q] = "none"
    /\ LET r == rp[q]
           s == ps[q]
           hit == {i \in Full[q] : Cached(r, Key(s, Prefix[q][i])) # {}}
           need == ItemsOf(q) \ hit
           at(i) == CHOOSE f \in Cached(r, Key(s, Prefix[q][i])) : TRUE
       IN \E a \in [need -> FreeFrames(r)] :
            /\ Injective(a)
            /\ LET new == {a[i] : i \in need}
                   item(f) == CHOOSE i \in need : a[i] = f
                   reused == {at(i) : i \in hit}
               IN /\ pf' = [pf EXCEPT ![q] = [i \in Items |->
                                IF i \in hit THEN at(i) ELSE IF i \in need THEN a[i] ELSE NONE]]
                  /\ content' = [f \in Frames |->
                                IF f \in new THEN Want(q, item(f), s) ELSE content[f]]
                  /\ fkey' = [f \in Frames |->
                                IF f \in new /\ item(f) \in Full[q]
                                THEN Key(s, Prefix[q][item(f)]) ELSE fkey[f]]
                  /\ owner' = [f \in Frames |->
                                IF f \in new /\ item(f) \in Unkeyed THEN q ELSE owner[f]]
                  /\ pins' = [f \in Frames |->
                                IF f \in reused \/ (f \in new /\ item(f) \in Full[q])
                                THEN pins[f] \cup {q} ELSE pins[f]]
                  /\ misread' = misread \cup
                                {[q |-> q, i |-> i, at |-> "prefill"] :
                                    i \in {j \in hit : content[at(j)] # Want(q, j, s)}}
    /\ pstate' = [pstate EXCEPT ![q] = "computed"]
    /\ UNCHANGED <<lease, tiers, known, env, phase, rp, rd, ps, ds, df, via, fill, flight>>

(* The decode reserves a page for every item its device cache does not     *)
(* hold under the item's key, and pins the ones it holds. A reserved page   *)
(* holds whatever it held before until the handoff fills it.               *)
Reserve(q) ==
    /\ phase[q] = "routed"
    /\ \A i \in Items : fill[q][i] = NONE
    /\ LET r == rd[q]
           s == ds[q]
           hit == {i \in Full[q] : Cached(r, Key(s, Prefix[q][i])) # {}}
           need == ItemsOf(q) \ hit
           at(i) == CHOOSE f \in Cached(r, Key(s, Prefix[q][i])) : TRUE
       IN \E a \in [need -> FreeFrames(r)] :
            /\ Injective(a)
            /\ LET new == {a[i] : i \in need}
                   reused == {at(i) : i \in hit}
               IN /\ df' = [df EXCEPT ![q] = [i \in Items |->
                                IF i \in hit THEN at(i) ELSE IF i \in need THEN a[i] ELSE NONE]]
                  /\ owner' = [f \in Frames |-> IF f \in new THEN q ELSE owner[f]]
                  /\ pins' = [f \in Frames |-> IF f \in reused THEN pins[f] \cup {q} ELSE pins[f]]
                  /\ fill' = [fill EXCEPT ![q] = [i \in Items |->
                                IF i \in hit THEN "ok" ELSE IF i \in need THEN "wait" ELSE NONE]]
                  /\ via' = [via EXCEPT ![q] = [i \in Items |->
                                IF i \in hit THEN "hit" ELSE NONE]]
    /\ UNCHANGED <<content, fkey, lease, tiers, known, env, phase, rp, rd, ps, ds, pstate, pf,
                   flight, misread>>

(* The KV program's handoff hook, on the decode side's Storage Server,     *)
(* chooses how the full pages reach the decode, from the directory's       *)
(* account of what the decode's node and the store hold: a pull from a     *)
(* tier only when that account holds every page still to come (the        *)
(* server's refusal). The unkeyed items are pushed or read: they have no   *)
(* key to pull by. A handoff names the root of the prefill's pages; when   *)
(* the decode's seed differs (an epoch switch between the two replicas'    *)
(* convergences), the decode refuses the prefill's pages and computes them *)
(* itself, while a pull, by keys of its own seed, stays. A page to be read *)
(* is leased by the prefill.                                               *)
Paths == {"push", "read", "pull_host", "pull_store"}
TailPaths == IF "KeyedTail" \in Hazards THEN Paths ELSE {"push", "read"}

Found(q, path) ==
    \A i \in Full[q] :
        fill[q][i] = "wait" =>
            CASE path = "pull_host" ->
                    \E e \in dir[ReplicaNode[rd[q]]] : e.tier = "host" /\ e.k = ItemKey(q, i)
              [] path = "pull_store" -> ItemKey(q, i) \in seen
              [] OTHER -> TRUE

Choose(q) ==
    /\ phase[q] = "routed"
    /\ pstate[q] = "computed"
    /\ \E i \in Items : fill[q][i] # NONE
    /\ \E full \in Paths, tail \in TailPaths :
        /\ Found(q, full)
        /\ LET refused == ps[q] # ds[q] /\ "SeedBlind" \notin Hazards
               path(i) == IF i \in Full[q] THEN full ELSE tail
               way(i) == IF fill[q][i] # "wait" THEN via[q][i]
                         ELSE IF refused /\ path(i) \in {"push", "read"} THEN "compute"
                         ELSE path(i)
               leased == {pf[q][i] : i \in {j \in ItemsOf(q) : fill[q][j] = "wait" /\ way(j) = "read"}}
           IN /\ via' = [via EXCEPT ![q] = [i \in Items |-> IF i \in ItemsOf(q) THEN way(i) ELSE NONE]]
              /\ fill' = [fill EXCEPT ![q] = [i \in Items |->
                            IF i \in ItemsOf(q) /\ fill[q][i] = "wait" /\ way(i) = "compute"
                            THEN "miss" ELSE fill[q][i]]]
              /\ lease' = [f \in Frames |-> IF f \in leased THEN lease[f] \cup {q} ELSE lease[f]]
    /\ phase' = [phase EXCEPT ![q] = "handoff"]
    /\ UNCHANGED <<content, fkey, owner, pins, tiers, known, env, rp, rd, ps, ds, pstate, pf, df,
                   flight, misread>>

Copy(kind, q, i, src, dst, k, held) ==
    [kind |-> kind, q |-> q, i |-> i, src |-> src, dst |-> dst, k |-> k, held |-> held, done |-> FALSE]

(* Push: the prefill writes its page into the page the decode reserved,    *)
(* which the decode keeps until the push ends (it frees a reserved page    *)
(* only with no copy writing it, `Abort`); the prefill keeps its page      *)
(* until the push ends (`Release`).                                        *)
StartPush(q, i) ==
    /\ phase[q] = "handoff"
    /\ via[q][i] = "push"
    /\ fill[q][i] = "wait"
    /\ xfers' = xfers \cup {Copy("push", q, i, pf[q][i], df[q][i], NoKey, TRUE)}
    /\ fill' = [fill EXCEPT ![q][i] = "moving"]
    /\ UNCHANGED <<device, tiers, known, env, phase, rp, rd, ps, ds, pstate, pf, df, via, moves,
                   nmoves, misread>>

FinishPush(t) ==
    /\ t \in xfers
    /\ t.kind = "push"
    /\ content' = [content EXCEPT ![t.dst] = content[t.src]]
    /\ xfers' = xfers \ {t}
    /\ fill' = IF phase[t.q] = "handoff" THEN [fill EXCEPT ![t.q][t.i] = "ok"] ELSE fill
    /\ UNCHANGED <<fkey, owner, pins, lease, tiers, known, env, phase, rp, rd, ps, ds, pstate, pf, df,
                   via, moves, nmoves, misread>>

(* Direct read: the decode reads the prefill's leased page into its own.   *)
(* The read's end is its copy; its acknowledgement tells the prefill the   *)
(* read is done and learns whether the lease still held, which ends the    *)
(* lease. A lease held at the acknowledgement was held through the read,   *)
(* since a lease that ends is never renewed; a read acknowledged after its *)
(* lease expired is discarded and the item computed by the decode. The     *)
(* hazard Unfenced takes every read.                                       *)
StartRead(q, i) ==
    /\ phase[q] = "handoff"
    /\ via[q][i] = "read"
    /\ fill[q][i] = "wait"
    /\ xfers' = xfers \cup {Copy("read", q, i, pf[q][i], df[q][i], NoKey, FALSE)}
    /\ fill' = [fill EXCEPT ![q][i] = "moving"]
    /\ UNCHANGED <<device, tiers, known, env, phase, rp, rd, ps, ds, pstate, pf, df, via, moves,
                   nmoves, misread>>

FinishRead(t) ==
    /\ t \in xfers
    /\ t.kind = "read"
    /\ ~t.done
    /\ content' = [content EXCEPT ![t.dst] = content[t.src]]
    /\ xfers' = (xfers \ {t}) \cup {[t EXCEPT !.done = TRUE]}
    /\ UNCHANGED <<fkey, owner, pins, lease, tiers, known, env, reqs, moves, nmoves, misread>>

Ack(t) ==
    /\ t \in xfers
    /\ t.kind = "read"
    /\ t.done
    /\ LET kept == t.q \in lease[t.src] \/ "Unfenced" \in Hazards
       IN fill' = IF phase[t.q] = "handoff"
                  THEN [fill EXCEPT ![t.q][t.i] = IF kept THEN "ok" ELSE "miss"] ELSE fill
    /\ lease' = [lease EXCEPT ![t.src] = @ \ {t.q}]
    /\ xfers' = xfers \ {t}
    /\ UNCHANGED <<content, fkey, owner, pins, tiers, known, env, phase, rp, rd, ps, ds, pstate, pf,
                   df, via, moves, nmoves, misread>>

(* Pull: the decode asks a tier for the item's key. The host tier of its    *)
(* node answers from its own index: the page cached under the key, kept     *)
(* for the copy, or a miss, which the decode computes; the store answers    *)
(* by key, a get that fails when the object is gone (M1). The hazard        *)
(* StaleAddress reads the host page the directory's entry names, without    *)
(* the holder's index and without the holder keeping it.                    *)
StartGet(q, i) ==
    /\ phase[q] = "handoff"
    /\ via[q][i] \in {"pull_host", "pull_store"}
    /\ fill[q][i] = "wait"
    /\ LET k == ItemKey(q, i)
           n == ReplicaNode[rd[q]]
           sources ==
             IF via[q][i] = "pull_store" THEN
                 IF store[k] # Absent THEN {<<STORE, TRUE>>} ELSE {}
             ELSE IF "StaleAddress" \in Hazards THEN
                 {<<e.at, FALSE>> : e \in {x \in dir[n] : x.tier = "host" /\ x.k = k}}
             ELSE {<<h, TRUE>> : h \in Hosted(n, k)}
       IN IF sources = {}
          THEN /\ fill' = [fill EXCEPT ![q][i] = "miss"]
               /\ xfers' = xfers
          ELSE \E src \in sources :
                 /\ xfers' = xfers \cup {Copy("get", q, i, src[1], df[q][i], k, src[2])}
                 /\ fill' = [fill EXCEPT ![q][i] = "moving"]
    /\ UNCHANGED <<device, tiers, known, env, phase, rp, rd, ps, ds, pstate, pf, df, via, moves,
                   nmoves, misread>>

FinishGet(t) ==
    /\ t \in xfers
    /\ t.kind = "get"
    /\ LET c == IF t.src = STORE THEN store[t.k] ELSE hcontent[t.src]
           got == c # Absent
       IN /\ content' = IF got THEN [content EXCEPT ![t.dst] = c] ELSE content
          /\ fill' = IF phase[t.q] = "handoff"
                     THEN [fill EXCEPT ![t.q][t.i] = IF got THEN "ok" ELSE "miss"] ELSE fill
    /\ xfers' = xfers \ {t}
    /\ UNCHANGED <<fkey, owner, pins, lease, tiers, known, env, phase, rp, rd, ps, ds, pstate, pf,
                   df, via, moves, nmoves, misread>>

(* An item no path delivered is computed by the decode: the reuse lost.     *)
Recompute(q, i) ==
    /\ phase[q] = "handoff"
    /\ fill[q][i] = "miss"
    /\ content' = [content EXCEPT ![df[q][i]] = Want(q, i, ds[q])]
    /\ fill' = [fill EXCEPT ![q][i] = "redone"]
    /\ UNCHANGED <<fkey, owner, pins, lease, tiers, known, env, phase, rp, rd, ps, ds, pstate, pf,
                   df, via, flight, misread>>

(* The decode starts once every item is in its pages: it reads them all.    *)
Delivered(q, i) == fill[q][i] \in {"ok", "redone"}

Decode(q) ==
    /\ phase[q] = "handoff"
    /\ \A i \in ItemsOf(q) : Delivered(q, i)
    /\ \A t \in xfers : t.q # q
    /\ phase' = [phase EXCEPT ![q] = "decode"]
    /\ misread' = misread \cup
                  {[q |-> q, i |-> i, at |-> "decode"] :
                      i \in {j \in ItemsOf(q) : content[df[q][j]] # Want(q, j, ds[q])}}
    /\ UNCHANGED <<device, tiers, known, env, rp, rd, ps, ds, pstate, pf, df, via, fill, flight>>

(* The request ends: its decode's full pages are cached under their keys   *)
(* where the device cache lacks them, its other pages freed.               *)
Finish(q) ==
    /\ phase[q] = "decode"
    /\ LET r == rd[q]
           mine == {f \in FramesOf(r) : owner[f] = q}
           item(f) == CHOOSE i \in ItemsOf(q) : df[q][i] = f
           publish == {f \in mine : item(f) \in Full[q] /\ Cached(r, ItemKey(q, item(f))) = {}}
       IN /\ fkey' = [f \in Frames |-> IF f \in publish THEN ItemKey(q, item(f)) ELSE fkey[f]]
          /\ owner' = [f \in Frames |-> IF f \in mine THEN NONE ELSE owner[f]]
          /\ pins' = [f \in Frames |-> IF FrameReplica[f] = r THEN pins[f] \ {q} ELSE pins[f]]
    /\ phase' = [phase EXCEPT ![q] = "done"]
    /\ UNCHANGED <<content, lease, tiers, known, env, rp, rd, ps, ds, pstate, pf, df, via, fill,
                   flight, misread>>

(* The prefill releases its pages once the handoff is chosen, every push   *)
(* the handoff chose has ended (the request ended, or each pushed item is   *)
(* delivered or refused) and no lease holds them: a read's acknowledgement  *)
(* or the lease's expiry ends a lease. EarlyRelease releases with leases    *)
(* held.                                                                    *)
Pushing(q) ==
    \/ \E t \in xfers : t.kind = "push" /\ t.q = q
    \/ phase[q] = "handoff" /\ \E i \in ItemsOf(q) : via[q][i] = "push" /\ fill[q][i] = "wait"

Release(q) ==
    /\ pstate[q] = "computed"
    /\ phase[q] \in {"handoff", "decode", "done"}
    /\ ~Pushing(q)
    /\ "EarlyRelease" \in Hazards \/ \A f \in Frames : q \notin lease[f]
    /\ LET r == rp[q]
       IN /\ owner' = [f \in Frames |-> IF owner[f] = q /\ FrameReplica[f] = r THEN NONE ELSE owner[f]]
          /\ pins' = [f \in Frames |-> IF FrameReplica[f] = r THEN pins[f] \ {q} ELSE pins[f]]
    /\ pstate' = [pstate EXCEPT ![q] = "released"]
    /\ UNCHANGED <<content, fkey, lease, tiers, known, env, phase, rp, rd, ps, ds, pf, df, via, fill,
                   flight, misread>>

(* A lease expires at any time: the prefill's clock, abstracted.           *)
Expire(q, f) ==
    /\ q \in lease[f]
    /\ lease' = [lease EXCEPT ![f] = @ \ {q}]
    /\ UNCHANGED <<content, fkey, owner, pins, tiers, known, env, reqs, flight, misread>>

(* A request ends before it decodes: cancelled, or migrated, which starts  *)
(* it anew as another request. The decode frees its pages once no copy     *)
(* writes them; UnreservedPush frees them with pushes in flight.            *)
Abort(q) ==
    /\ phase[q] \in {"routed", "handoff"}
    /\ \A t \in xfers : t.q = q => (t.kind = "push" /\ "UnreservedPush" \in Hazards)
    /\ LET r == rd[q]
       IN /\ owner' = [f \in Frames |-> IF owner[f] = q /\ FrameReplica[f] = r THEN NONE ELSE owner[f]]
          /\ pins' = [f \in Frames |-> IF FrameReplica[f] = r THEN pins[f] \ {q} ELSE pins[f]]
    /\ phase' = [phase EXCEPT ![q] = "done"]
    /\ UNCHANGED <<content, fkey, lease, tiers, known, env, rp, rd, ps, ds, pstate, pf, df, via, fill,
                   flight, misread>>

-----------------------------------------------------------------------------
(* The tiering hook's moves: a page copied from a device to its node's host *)
(* tier (a demotion), from a device or a host tier into the store (a put,   *)
(* the first hop of a pull through the store), or from the store into a     *)
(* host tier (a prefetch). The program names pages by what the directory    *)
(* states; the source's holder resolves the key in its own index when the   *)
(* move starts and keeps the page until it ends, and skips a move whose key *)
(* it no longer holds. A host page is written privately and cached under    *)
(* its key when the copy ends; a node caches a key in one page. The hazard  *)
(* StaleAddress copies the page at the address the directory's entry names. *)
Move(src, from, dst, to, k, held) ==
    [src |-> src, from |-> from, dst |-> dst, to |-> to, k |-> k, held |-> held]

Arriving(n, k) == \E m \in moves : m.to = "host" /\ SlotNode[m.dst] = n /\ m.k = k

(* The sources a move of key k from node n's tier may read.                *)
Sources(n, tier, k) ==
    IF "StaleAddress" \in Hazards
    THEN {e.at : e \in {x \in dir[n] : x.tier = tier /\ x.k = k}}
    ELSE IF tier = "device" THEN {f \in Frames : ReplicaNode[FrameReplica[f]] = n /\ fkey[f] = k}
    ELSE Hosted(n, k)

StartMove ==
    /\ nmoves < MaxMoves
    /\ \E k \in Keys :
        \/ \E n \in Nodes, tier \in {"device", "host"} :
             \E src \in Sources(n, tier, k) :
                \/ /\ tier = "device"
                   /\ Hosted(n, k) = {}
                   /\ ~Arriving(n, k)
                   /\ \E h \in FreeSlots(n) :
                        moves' = moves \cup {Move(src, tier, h, "host", k, "StaleAddress" \notin Hazards)}
                \/ /\ store[k] = Absent
                   /\ moves' = moves \cup {Move(src, tier, STORE, "store", k, "StaleAddress" \notin Hazards)}
        \/ /\ store[k] # Absent
           /\ \E n \in Nodes :
                /\ Hosted(n, k) = {}
                /\ ~Arriving(n, k)
                /\ \E h \in FreeSlots(n) : moves' = moves \cup {Move(STORE, "store", h, "host", k, TRUE)}
    /\ nmoves' = nmoves + 1
    /\ UNCHANGED <<device, tiers, known, env, reqs, xfers, misread>>

FinishMove(m) ==
    /\ m \in moves
    /\ LET c == CASE m.from = "device" -> content[m.src]
                  [] m.from = "host" -> hcontent[m.src]
                  [] OTHER -> store[m.k]
       IN IF c = Absent
          THEN UNCHANGED <<hkey, hcontent, store>>
          ELSE IF m.to = "host"
          THEN /\ hkey' = [hkey EXCEPT ![m.dst] = m.k]
               /\ hcontent' = [hcontent EXCEPT ![m.dst] = c]
               /\ store' = store
          ELSE /\ store' = [store EXCEPT ![m.k] = c]
               /\ UNCHANGED <<hkey, hcontent>>
    /\ moves' = moves \ {m}
    /\ UNCHANGED <<device, known, env, reqs, xfers, nmoves, misread>>

(* Eviction: a device cache evicts a page no request uses or leases and no *)
(* copy holds; a host tier a page no copy holds; Mooncake's master any     *)
(* object at any time. An evicted page keeps its bytes until rewritten.    *)
EvictFrame(f) ==
    /\ fkey[f] # NoKey
    /\ pins[f] = {}
    /\ Unleased(f)
    /\ f \notin Held
    /\ fkey' = [fkey EXCEPT ![f] = NoKey]
    /\ UNCHANGED <<content, owner, pins, lease, tiers, known, env, reqs, flight, misread>>

EvictSlot(h) ==
    /\ hkey[h] # NoKey
    /\ h \notin Held
    /\ hkey' = [hkey EXCEPT ![h] = NoKey]
    /\ UNCHANGED <<device, hcontent, store, known, env, reqs, flight, misread>>

EvictStore(k) ==
    /\ store[k] # Absent
    /\ store' = [store EXCEPT ![k] = Absent]
    /\ UNCHANGED <<device, hkey, hcontent, known, env, reqs, flight, misread>>

-----------------------------------------------------------------------------
(* The directory: a node's Storage Server reports what its device caches   *)
(* and its host tier hold; the leader's Storage Server replaces that node's *)
(* entries. The report and every change after it are separate steps, so    *)
(* the directory lags. A Storage Server that restarts or reconnects starts  *)
(* with no entries of its node until it reports. Mooncake's object events   *)
(* state the store's keys as of when they were sent.                        *)
Entry(tier, k, at) == [tier |-> tier, k |-> k, at |-> at]

Report(n) ==
    /\ dir' = [dir EXCEPT ![n] =
                 {Entry("device", fkey[f], f) :
                     f \in {x \in Frames : ReplicaNode[FrameReplica[x]] = n /\ fkey[x] # NoKey}} \cup
                 {Entry("host", hkey[h], h) : h \in {x \in SlotsOf(n) : hkey[x] # NoKey}}]
    /\ UNCHANGED <<device, tiers, seen, env, reqs, flight, misread>>

Restart(n) ==
    /\ dir[n] # {}
    /\ dir' = [dir EXCEPT ![n] = {}]
    /\ UNCHANGED <<device, tiers, seen, env, reqs, flight, misread>>

Events ==
    /\ seen' = {k \in Keys : store[k] # Absent}
    /\ seen' # seen
    /\ UNCHANGED <<device, tiers, dir, env, reqs, flight, misread>>

-----------------------------------------------------------------------------
(* The environment: the configuration's convergence, which the data plane  *)
(* relies on for E1 to E4.                                                  *)
(*                                                                         *)
(* E1: a role switch changes only which requests a replica takes from the  *)
(*     next step on; its requests under way run to their end, and no KV    *)
(*     moves. The hazard RoleFlush frees the replica's pages instead.       *)
(* E2: an epoch switch of a replica changes the seed of its serving epoch   *)
(*     (a change of weights, precision or format; one that keeps them      *)
(*     keeps the seed and changes nothing here). The epochs of a replica    *)
(*     share its device pages through its Storage Server, which carries    *)
(*     the device cache's index across the switch.                         *)
(* E3: a request keeps the epochs it was routed to: the old epoch drains   *)
(*     its requests (`ps`, `ds`), and one that migrates is a new request.  *)
(* E4: convergence switches replicas one at a time, so replicas of two      *)
(*     seeds serve together; a switch never waits for the data plane.      *)
(*                                                                         *)
(* The data plane's guarantee to the convergence model: its steps never    *)
(* change a role or a seed (`ConfigUntouched`), and every read gives the   *)
(* content its reader needs whatever the convergence does (`Faithful`).    *)
RoleSwitch(r, x) ==
    /\ switches < MaxSwitches
    /\ x # role[r]
    /\ role' = [role EXCEPT ![r] = x]
    /\ switches' = switches + 1
    /\ IF "RoleFlush" \in Hazards
       THEN /\ owner' = [f \in Frames |-> IF FrameReplica[f] = r THEN NONE ELSE owner[f]]
            /\ fkey' = [f \in Frames |-> IF FrameReplica[f] = r THEN NoKey ELSE fkey[f]]
            /\ pins' = [f \in Frames |-> IF FrameReplica[f] = r THEN {} ELSE pins[f]]
            /\ lease' = [f \in Frames |-> IF FrameReplica[f] = r THEN {} ELSE lease[f]]
       ELSE UNCHANGED <<owner, fkey, pins, lease>>
    /\ UNCHANGED <<content, tiers, known, seed, reqs, flight, misread>>

EpochSwitch(r, s) ==
    /\ switches < MaxSwitches
    /\ s # seed[r]
    /\ seed' = [seed EXCEPT ![r] = s]
    /\ switches' = switches + 1
    /\ UNCHANGED <<device, tiers, known, role, reqs, flight, misread>>

-----------------------------------------------------------------------------
Environment ==
    \/ \E r \in Replicas, x \in Roles : RoleSwitch(r, x)
    \/ \E r \in Replicas, s \in Seeds : EpochSwitch(r, s)

Plane ==
    \/ \E q \in Requests :
        \/ Route(q) \/ Compute(q) \/ Reserve(q) \/ Choose(q) \/ Decode(q) \/ Finish(q)
        \/ Release(q) \/ Abort(q)
        \/ \E i \in ItemsOf(q) : StartPush(q, i)
        \/ \E i \in ItemsOf(q) : StartRead(q, i)
        \/ \E i \in ItemsOf(q) : StartGet(q, i)
        \/ \E i \in ItemsOf(q) : Recompute(q, i)
    \/ \E t \in xfers : FinishPush(t)
    \/ \E t \in xfers : FinishRead(t)
    \/ \E t \in xfers : Ack(t)
    \/ \E t \in xfers : FinishGet(t)
    \/ \E q \in Requests, f \in Frames : Expire(q, f)
    \/ StartMove
    \/ \E m \in moves : FinishMove(m)
    \/ \E f \in Frames : EvictFrame(f)
    \/ \E h \in Slots : EvictSlot(h)
    \/ \E k \in Keys : EvictStore(k)
    \/ \E n \in Nodes : Report(n)
    \/ \E n \in Nodes : Restart(n)
    \/ Events

Next == Plane \/ Environment

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
Content == {Zero, Absent} \cup {Page(s, p) : s \in Seeds, p \in Prefixes}
                         \cup {Live(s, q, u) : s \in Seeds, q \in Requests, u \in Unkeyed}

TypeOK ==
    /\ content \in [Frames -> Content]
    /\ fkey \in [Frames -> Keys \cup {NoKey}]
    /\ owner \in [Frames -> Requests \cup {NONE}]
    /\ pins \in [Frames -> SUBSET Requests]
    /\ lease \in [Frames -> SUBSET Requests]
    /\ hkey \in [Slots -> Keys \cup {NoKey}]
    /\ hcontent \in [Slots -> Content]
    /\ store \in [Keys -> Content]
    /\ seen \subseteq Keys
    /\ role \in [Replicas -> Roles]
    /\ seed \in [Replicas -> Seeds]
    /\ phase \in [Requests -> {"new", "routed", "handoff", "decode", "done"}]
    /\ pstate \in [Requests -> {"none", "computed", "released"}]
=============================================================================
