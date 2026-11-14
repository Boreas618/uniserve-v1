------------------------------ MODULE KvTrace ------------------------------
(* Steps of the Rust buffer pool checked against the steps of the design   *)
(* model's buffer pool (specs/buffer-pool.md §18). A window is a state the *)
(* Rust buffer pool reached, the input it took and the state that followed *)
(* (or the part of each that the transition touches, steps.py), as the     *)
(* enumeration of L1 or the replay of a trace wrote them; TLC must find,   *)
(* from the first state, a path of MC_Pool's steps to the second. Each     *)
(* initial state is one window's first state, with the window's input,    *)
(* next state and bound in variables. A state is compared as the Rust      *)
(* buffer pool writes it in the model's terms (agreement.py's projection,  *)
(* here as `Matches`), which states every field of waiting work, copies    *)
(* and tasks a step reads, so a window's start is one state of the model   *)
(* (`Start`): TLC enumerates no choice there, however many copies are in   *)
(* flight.                                                                 *)
(*                                                                         *)
(* Every step of a path is a step of `PoolNext`: each disjunct of `Step`   *)
(* is one of its disjuncts with a guard that the input allows it, or with  *)
(* its choices narrowed to the ones the input names, the standard pruning  *)
(* of a trace's validation by its events. Pruning removes paths and never  *)
(* adds one, so it can make a window fail and never pass; so do a          *)
(* window's start that fixes what no state states and `Toward`, which      *)
(* keeps a step from undoing what the next state settles. steps.py writes  *)
(* the windows and the constants of a run as MC_Trace.                     *)
EXTENDS MC_Pool, Sequences

CONSTANT Windows   \* <<[start |-> ..., input |-> ..., state |-> ..., bound |-> ...], ...>>

VARIABLES window,  \* the window searched
          input,   \* its input
          goal,    \* its next state
          bound,   \* the steps a path to it may take
          walked,  \* the model's steps taken
          reached  \* whether `goal` was reached

trace == <<window, input, goal, bound, walked, reached>>

-----------------------------------------------------------------------------
(* A state as the Rust buffer pool writes it.                              *)

Shown(u) == IF u = NONE THEN "-" ELSE u

\* The stamps work still waits for: those `completed` has not reached.
Pending(a) == [g \in {x \in Groups : a[x] > launched[x]} |-> a[g]]

\* A copy in flight, as a `copy` line states it.
CopyLine(c) ==
    [kind |-> c.kind,
     o |-> IF \/ c.kind \in {"clone", "load", "restore"}
              \/ c.kind = "save" /\ c.o \in Owners /\ ostate[c.o] = "live" /\ oslot[c.o] = c.read
           THEN c.o ELSE "-",
     p |-> IF c.kind \in {"clone", "load"} THEN c.p ELSE "-",
     read |-> c.read, write |-> c.write,
     fill |-> IF c.fill /\ c.kind \in {"load", "store"} THEN 1 ELSE 0,
     mem |-> IF c.kind \in {"clone", "load"} THEN GroupMemory[c.g] ELSE "-",
     \* The key a copy caches its block under; a clone and a restore cache none.
     k |-> IF c.kind \in {"load", "store", "save"} THEN c.k ELSE "-",
     \* The host parts a store or a save writes.
     parts |-> CASE c.kind = "store" -> IF c.fill THEN c.parts ELSE GroupParts[c.g]
                 [] c.kind = "save" -> GroupParts[c.g]
                 [] OTHER -> {},
     \* The session whose request asked for the copy.
     s |-> GroupEpoch[c.g]]

\* A task in flight, as a `task` line states it.
TaskLine(t) ==
    CASE t.kind = "transfer" ->
           [kind |-> "transfer", o |-> t.o, p |-> "-", to |-> t.to, slot |-> t.slot,
            units |-> t.units, writes |-> t.writes, h |-> "-", expired |-> 0, k |-> "-", parts |-> {},
            s |-> "-"]
      [] t.kind = "replicate" ->
           [kind |-> "replicate", o |-> t.o, p |-> t.p, to |-> t.to, slot |-> "-",
            units |-> t.units, writes |-> t.writes, h |-> "-", expired |-> 0, k |-> "-", parts |-> {},
            s |-> "-"]
      [] OTHER ->
           [kind |-> t.kind, o |-> "-", p |-> "-", to |-> "-", slot |-> "-", units |-> {},
            writes |-> {}, h |-> t.h, expired |-> IF t.expired THEN 1 ELSE 0, k |-> t.k,
            \* The host parts of the group a prefetch writes for, and its session.
            parts |-> IF t.kind = "prefetch" THEN GroupParts[t.to] ELSE {},
            s |-> IF t.kind = "prefetch" THEN GroupEpoch[t.to] ELSE "-"]

Waiting(kind) == {d \in deferred : d.kind = kind}

\* Whether the model's state is the recorded state `st`.
Matches(st) ==
    /\ ncopy = st.ncopy
    /\ launched = st.completed
    /\ free = st.free
    \* Each epoch's trial, and whether its engine is gone (its session closed).
    /\ \A e \in Epochs : trial[e] = st.epochs[e].trial /\ (e \in lost) = (st.epochs[e].is = "closed")
    /\ \A o \in Owners :
         IF ostate[o] = "live"
         THEN st.owners[o] =
                [is |-> "live", g |-> ogroup[o], slot |-> oslot[o],
                 t |-> {<<q, table[o][q].k, table[o][q].u>> :
                          q \in {x \in Positions : table[o][x].u # NONE}}]
         \* The Rust buffer pool takes the owner numbers of a queue pair in
         \* increasing order: a number below one taken is gone, whether or
         \* not it was attached, and its state cannot tell which. A model
         \* owner never attached is such a number.
         ELSE st.owners[o].is \in {ostate[o], "gone"}
    \* Each cached key with its copies, and the epoch on trial each copy is
    \* visible to alone (R8).
    /\ st.keys =
         {[k |-> y, d |-> Shown(cache[y]), h |-> Shown(hcache[y]), r |-> IF y \in remote THEN 1 ELSE 0,
           vd |-> Shown(vis[y].d), vh |-> Shown(vis[y].h)] :
            y \in {x \in Keys : cache[x] # NONE \/ hcache[x] # NONE \/ x \in remote}}
    \* The memories of every device block an owner, the cache or a waiting
    \* commit holds, and the host parts of every cached host block.
    /\ LET live == {o \in Owners : ostate[o] = "live"}
           blocks == ({table[o][q].u : o \in live, q \in Positions} \cup {cache[y] : y \in Keys}
                        \cup {hcache[y] : y \in Keys} \cup UNION {d.units : d \in Waiting("commit")})
                       \ {NONE}
       IN st.valid = {<<b, valid[b]>> : b \in blocks}
    /\ st.copies = {CopyLine(c) : c \in copies}
    /\ st.commits =
         {[o |-> d.o, p |-> d.p, u |-> CHOOSE b \in d.units : TRUE, k |-> d.k, after |-> Pending(d.after)] :
            d \in Waiting("commit")}
    /\ st.transfers = {[o |-> d.o, to |-> d.to, after |-> Pending(d.after)] : d \in Waiting("transfer")}
    /\ st.replicates =
         {[o |-> d.o, p |-> d.p, to |-> d.to, u |-> CHOOSE b \in d.units : TRUE, after |-> Pending(d.after)] :
            d \in Waiting("replicate")}
    /\ st.finishes =
         UNION {{[ok |-> IF d.ok THEN 1 ELSE 0, after |-> Pending(d.after), copy |-> CopyLine(c)] :
                    c \in {x \in copies : x.id = d.c}} :
                  d \in Waiting("finish")}
    \* A release waits once per group whose stamp it has not reached.
    /\ st.releases =
         UNION {{[after |-> [x \in {g} |-> d.after[g]], units |-> d.units] :
                    g \in DOMAIN Pending(d.after)} :
                  d \in Waiting("release")}
    /\ st.tasks = {TaskLine(t) : t \in tasks}

-----------------------------------------------------------------------------
(* A recorded state as a state of the model: `x.state` with what it leaves *)
(* out chosen. A stamp it states for the groups whose `completed` has not  *)
(* reached it is 0 for the others; a release line is a release of its own; *)
(* a device block only a release, a copy or a task holds, and a host block *)
(* nothing caches, is valid nowhere (each is written before it is read, or *)
(* is reclaimed); a clone's key is the one its source is cached under, and *)
(* a restore's the one its host block is (the copy holds either, so the    *)
(* cache keeps it, R2); a copy's group is one of the session its line     *)
(* states: the owner's, the one whose memory a clone or a load writes, one *)
(* that copies with the host parts a store or a save writes, any for a     *)
(* restore of a gone owner (a copy reads no more of its group than these); *)
(* a prefetch's the one of its session whose host parts it writes; the     *)
(* owner of a save whose owner is gone a gone owner of its session. Each   *)
(* epoch's trial and whether its session is closed, and each copy's        *)
(* visibility (R8), are as the state states them. `x.member` is the device *)
(* units in their region, `x.copies` the copies in flight in an order that *)
(* numbers them, and `x.finishes` the waiting `Finish`es by the index of   *)
(* their copy.                                                             *)

Start(x) ==
    LET st == x.state
        n == Len(x.copies)
        live(o) == st.owners[o].is = "live"
        entry(o, q) == {e \in st.owners[o].t : e[1] = q}
        keyed(y) == {e \in st.keys : e.k = y}
        ident(j) == st.ncopy - n + j
        stated == {v[1] : v \in st.valid}
        Stamp(a) == [g \in Groups |-> IF g \in DOMAIN a THEN a[g] ELSE 0]
        cachedAs(b) == {e.k : e \in {f \in st.keys : f.d = b \/ f.h = b}}
        gone == {o \in Owners : st.owners[o].is = "gone"}
        \* A copy's group: the one whose memory a clone or a load writes, a
        \* live owner's, or one that copies with the host parts a store or a
        \* save writes (groups alike in these are alike to a copy).
        ofSession(l) == {g \in Groups : GroupEpoch[g] = l.s}
        groupOf(l) ==
            CASE l.o # "-" /\ live(l.o) -> st.owners[l.o].g
              [] l.mem # "-" -> CHOOSE g \in ofSession(l) : GroupMemory[g] = l.mem
              [] l.kind = "store" /\ l.fill = 1 -> CHOOSE g \in ofSession(l) : l.parts \subseteq GroupParts[g]
              [] l.kind \in {"store", "save"} -> CHOOSE g \in ofSession(l) : GroupParts[g] = l.parts
              [] OTHER -> CHOOSE g \in ofSession(l) : TRUE
        \* A copy's owner: the one its line states; none for a store; for a
        \* save whose owner is gone, a gone owner, or none when the window has
        \* none (a save of none is one no request can name, as one of a gone owner).
        ownerOf(l) ==
            IF l.o # "-" THEN l.o
            ELSE IF l.kind = "save" /\ \E o \in gone : OwnerEpoch[o] = l.s
            THEN CHOOSE o \in gone : OwnerEpoch[o] = l.s
            ELSE NONE
        Copied(l, j) ==
            [id |-> ident(j), kind |-> l.kind, o |-> ownerOf(l), p |-> IF l.p = "-" THEN NONE ELSE l.p,
             k |-> IF l.k # "-" THEN l.k
                   ELSE IF cachedAs(l.read) # {} THEN CHOOSE y \in cachedAs(l.read) : TRUE ELSE NONE,
             g |-> groupOf(l), read |-> l.read, write |-> l.write,
             fill |-> (l.kind \in {"load", "store"} /\ l.fill = 1) \/ l.kind = "restore",
             \* A store's fill writes the host parts its block lacks.
             parts |-> IF l.kind = "store" /\ l.fill = 1 THEN l.parts ELSE {},
             seq |-> 0, done |-> FALSE, stopped |-> FALSE]
    IN /\ free = st.free
       /\ member = x.member
       /\ ostate = [o \in Owners |-> st.owners[o].is]
       /\ ogroup = [o \in Owners |-> IF live(o) THEN st.owners[o].g ELSE NONE]
       /\ oslot = [o \in Owners |-> IF live(o) THEN st.owners[o].slot ELSE NONE]
       /\ table = [o \in Owners |-> [q \in Positions |->
                     IF entry(o, q) = {} THEN Empty
                     ELSE LET e == CHOOSE f \in entry(o, q) : TRUE IN [k |-> e[2], u |-> e[3]]]]
       /\ cache = [y \in Keys |-> IF keyed(y) = {} THEN NONE
                                  ELSE LET d == (CHOOSE e \in keyed(y) : TRUE).d
                                       IN IF d = "-" THEN NONE ELSE d]
       /\ hcache = [y \in Keys |-> IF keyed(y) = {} THEN NONE
                                   ELSE LET h == (CHOOSE e \in keyed(y) : TRUE).h
                                        IN IF h = "-" THEN NONE ELSE h]
       /\ remote = {e.k : e \in {f \in st.keys : f.r = 1}}
       /\ valid = [b \in Units \cup HostUnits |->
                     IF b \in stated THEN (CHOOSE v \in st.valid : v[1] = b)[2] ELSE {}]
       /\ ncopy = st.ncopy
       /\ launched = st.completed
       /\ inflight = {}
       /\ handoff = {}
       /\ requests = {}
       /\ copies = {Copied(x.copies[j], j) : j \in 1 .. n}
       /\ deferred =
            {[kind |-> "commit", units |-> {cm.u}, after |-> Stamp(cm.after), o |-> cm.o, p |-> cm.p,
              k |-> cm.k, c |-> 0, ok |-> FALSE, g |-> ogroup[cm.o], to |-> NONE] :
               cm \in st.commits}
            \cup {[kind |-> "transfer", units |-> {}, after |-> Stamp(tr.after), o |-> tr.o, p |-> NONE,
                   k |-> NONE, c |-> 0, ok |-> FALSE, g |-> NONE, to |-> tr.to] :
                    tr \in st.transfers}
            \cup {[kind |-> "replicate", units |-> {rp.u}, after |-> Stamp(rp.after), o |-> rp.o,
                   p |-> rp.p, k |-> NONE, c |-> 0, ok |-> FALSE,
                   g |-> IF live(rp.o) THEN st.owners[rp.o].g ELSE NONE, to |-> rp.to] :
                    rp \in st.replicates}
            \cup {[kind |-> "finish", units |-> {}, after |-> Stamp(fi.after), o |-> NONE, p |-> NONE,
                   k |-> NONE, c |-> ident(fi.at), ok |-> fi.ok = 1, g |-> NONE, to |-> NONE] :
                    fi \in x.finishes}
            \cup {[kind |-> "release", units |-> rl.units, after |-> Stamp(rl.after), o |-> NONE,
                   p |-> NONE, k |-> NONE, c |-> 0, ok |-> FALSE, g |-> NONE, to |-> NONE] :
                    rl \in st.releases}
       /\ tasks =
              {CASE tl.kind = "transfer" ->
                      [kind |-> "transfer", k |-> NONE, h |-> NONE, src |-> NONE, ok |-> TRUE,
                       expired |-> FALSE, p |-> NONE, o |-> tl.o, to |-> tl.to, slot |-> tl.slot,
                       held |-> TRUE, units |-> tl.units, writes |-> tl.writes]
                 [] tl.kind = "replicate" ->
                      [kind |-> "replicate", k |-> NONE, h |-> NONE, src |-> NONE, ok |-> TRUE,
                       expired |-> FALSE, p |-> tl.p, o |-> tl.o, to |-> tl.to, slot |-> NONE,
                       held |-> TRUE, units |-> tl.units, writes |-> tl.writes]
                 [] OTHER ->
                      [kind |-> tl.kind, k |-> tl.k, h |-> tl.h, src |-> "remote", ok |-> TRUE,
                       expired |-> tl.expired = 1, p |-> NONE, o |-> NONE,
                       \* The group whose host parts a prefetch writes.
                       to |-> IF tl.kind = "prefetch"
                              THEN CHOOSE g \in Groups : GroupEpoch[g] = tl.s /\ GroupParts[g] = tl.parts
                              ELSE NONE,
                       slot |-> NONE,
                       held |-> TRUE, units |-> {}, writes |-> {}] :
                 tl \in st.tasks}
       /\ vis = [y \in Keys |-> IF keyed(y) = {} THEN [d |-> NONE, h |-> NONE]
                                ELSE LET e == CHOOSE f \in keyed(y) : TRUE
                                     IN [d |-> IF e.vd = "-" THEN NONE ELSE e.vd,
                                         h |-> IF e.vh = "-" THEN NONE ELSE e.vh]]
       /\ trial = [e \in Epochs |-> st.epochs[e].trial]
       /\ dbt = [g \in Groups |-> [o \in Owners |-> [q \in Rows |-> NoEntry]]]
       /\ alloc = [b \in AllUnits |-> IF \E y \in Keys : cache[y] = b \/ hcache[y] = b
                                      THEN "shared" ELSE NONE]
       \* The key each cached block is shared under, which a crossing's R8 reads.
       /\ sharedKey = [b \in AllUnits |->
                        IF \E y \in Keys : cache[y] = b \/ hcache[y] = b
                        THEN CHOOSE y \in Keys : cache[y] = b \/ hcache[y] = b
                        ELSE NONE]
       /\ lost = {e \in Epochs : st.epochs[e].is = "closed"}
       /\ bad = {}

-----------------------------------------------------------------------------
(* The steps an input allows: `in.acts` names the actions, and `in.os`,    *)
(* `in.gs`, `in.ps` and `in.ks` the owners, groups, positions and keys it  *)
(* names. A stamped request arrives as `in.req` states it. The work that   *)
(* comes due and the requests taken may follow any input.                  *)

Asked(q) ==
    \A r \in requests' :
        /\ r.kind = q.kind
        /\ r.after = q.after
        /\ r.kind # "finish" => r.o = q.o
        /\ r.kind = "release" => r.ps = q.ps
        /\ r.kind = "commit" => r.p = q.p /\ r.k = q.k
        /\ r.kind = "transfer" => r.to = q.to
        /\ r.kind = "replicate" => r.p = q.p /\ r.to = q.to
        /\ r.kind = "finish" => r.ok = q.ok

(* MC_Pool's `Arrive` with each of its choices narrowed to the one the     *)
(* input names, the session of its queue pair among them: every step of it *)
(* is an `Arrive` step. TLC enumerates the domain of a choice before       *)
(* `Asked` can reject it, and `Arrive` chooses a release's positions among *)
(* all sets of positions. A `Finish` names its copy by nothing a state     *)
(* states, so it may name any copy.                                        *)
ArriveAs(q) ==
    /\ requests = {}
    /\ \E e \in (Epochs \ lost) \cap {q.e},
          after \in {a \in {q.after} : /\ a \in [Groups -> 0 .. MaxBatch]
                                       /\ \A g \in Groups \ Stamping : a[g] = 0} :
         \/ \E o \in Owners \cap {q.o} :
              \/ \E ps \in {x \in {q.ps} : x \in (SUBSET Positions) \ {{}}} :
                   requests' = {[Request("release", e, o, ps, NONE, NONE, 0, FALSE)
                                   EXCEPT !.after = after]}
              \/ requests' = {[Request("detach", e, o, {}, NONE, NONE, 0, FALSE)
                                 EXCEPT !.after = after]}
              \/ \E p \in Positions \cap {q.p}, k \in Keys \cap {q.k} :
                   requests' = {[Request("commit", e, o, {}, p, k, 0, FALSE)
                                   EXCEPT !.after = after]}
              \/ \E g \in Groups \cap {q.to} :
                   /\ Transfers
                   /\ requests' = {[Request("transfer", e, o, {}, NONE, NONE, 0, FALSE)
                                      EXCEPT !.after = after, !.to = g]}
              \/ \E p \in Positions \cap {q.p}, g \in Groups \cap {q.to} :
                   /\ Crosses
                   /\ requests' = {[Request("replicate", e, o, {}, p, NONE, 0, FALSE)
                                      EXCEPT !.after = after, !.to = g]}
         \/ \E c \in 1 .. ncopy, ok \in BOOLEAN \cap {q.ok} :
              requests' = {[Request("finish", e, NONE, {}, NONE, NONE, c, ok)
                              EXCEPT !.after = after]}
    /\ UNCHANGED <<pool, dbt, launched, inflight, handoff, alloc, sharedKey>>

\* The index of the first position from the `i`-th of `order` that owner
\* o's table has empty, or 0: one pass over a request's positions, which
\* may be hundreds.
RECURSIVE FirstEmpty(_, _, _)
FirstEmpty(o, order, i) ==
    IF i > Len(order) THEN 0
    ELSE IF table[o][order[i]] = Empty THEN i
    ELSE FirstEmpty(o, order, i + 1)

\* The index of the first key from the `i`-th of `keys` that no host block
\* and no prefetch holds, or 0: the next key a prefetch of several brings.
RECURSIVE FirstToFetch(_, _)
FirstToFetch(keys, i) ==
    IF i > Len(keys) THEN 0
    ELSE IF hcache[keys[i]] = NONE /\ ~\E t \in tasks : t.kind = "prefetch" /\ t.k = keys[i] THEN i
    ELSE FirstToFetch(keys, i + 1)

\* The index of the first key from the `i`-th of `keys` whose device copy
\* (`f` = cache) or host copy (`f` = hcache) is cached, or 0.
RECURSIVE FirstCached(_, _, _)
FirstCached(f, keys, i) ==
    IF i > Len(keys) THEN 0
    ELSE IF f[keys[i]] # NONE THEN i
    ELSE FirstCached(f, keys, i + 1)

Step(in) ==
    \/ "Attach" \in in.acts /\ \E o \in in.os, g \in in.gs : Attach(o, g) /\ UNCHANGED rare
    \* A request of several positions takes them in order (`in.order`), a
    \* pin each with its key (`in.korder`), as the Rust buffer pool does:
    \* the first position of the order that the owner's table has empty.
    \/ "Allocate" \in in.acts /\ \E o \in in.os :
         LET i == FirstEmpty(o, in.order, 1)
         IN /\ i > 0
            /\ Allocate(o, in.order[i])
            /\ UNCHANGED rare
    \/ "Pin" \in in.acts /\ \E o \in in.os :
         LET i == FirstEmpty(o, in.order, 1)
         IN /\ i > 0
            /\ Pin(o, in.order[i], in.korder[i])
            /\ UNCHANGED rare
    \/ "Clone" \in in.acts /\ \E o \in in.os, q \in in.ps, y \in in.ks : Clone(o, q, y) /\ UNCHANGED rare
    \/ "Load" \in in.acts /\ \E o \in in.os, q \in in.ps, y \in in.ks : Load(o, q, y) /\ UNCHANGED rare
    \/ "Save" \in in.acts /\ \E o \in in.os, y \in in.ks : Save(o, y) /\ UNCHANGED rare
    \/ "Restore" \in in.acts /\ \E o \in in.os, y \in in.ks : Restore(o, y) /\ UNCHANGED rare
    \/ "Store" \in in.acts /\ \E g \in in.gs, y \in in.ks : Store(g, y) /\ UNCHANGED rare
    \* Evictions in the order of `in.evict` and `in.evicthost`, the keys whose
    \* device or host copy the next state no longer caches (`Toward` lets no
    \* other go): evictions of different keys commute, and in every order a
    \* search would try each set of them.
    \/ "Evict" \in in.acts /\ LET i == FirstCached(cache, in.evict, 1)
                               IN i > 0 /\ Evict(in.evict[i]) /\ UNCHANGED rare
    \/ "EvictHost" \in in.acts /\ LET i == FirstCached(hcache, in.evicthost, 1)
                                   IN i > 0 /\ EvictHost(in.evicthost[i]) /\ UNCHANGED rare
    \/ "Arrive" \in in.acts /\ ArriveAs(in.req) /\ Asked(in.req) /\ UNCHANGED rare
    \/ "Advance" \in in.acts /\ \E g \in Groups : Advance(g) /\ launched'[g] <= in.completed[g]
                                                 /\ UNCHANGED rare
    \/ \E r \in requests :
         (TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r) \/ TakeTransfer(r)
          \/ TakeReplicate(r))
         /\ UNCHANGED rare
    \/ \E d \in deferred :
         (ReleaseDue(d) \/ CommitDue(d) \/ FinishDue(d) \/ TransferDue(d) \/ ReplicateDue(d))
         /\ UNCHANGED rare
    \/ "TransferEnd" \in in.acts /\ \E t \in tasks, ok \in BOOLEAN : TransferEnd(t, ok) /\ UNCHANGED rare
    \/ "ReplicateEnd" \in in.acts /\ \E t \in tasks, ok \in BOOLEAN : ReplicateEnd(t, ok)
                                    /\ UNCHANGED rare
    \/ "CommitUpdate" \in in.acts /\ \E e \in Epochs : CommitUpdate(e) /\ UNCHANGED rare
    \/ "Rollback" \in in.acts /\ \E e \in Epochs : Rollback(e) /\ UNCHANGED rare
    \/ "Shrink" \in in.acts /\ \E u \in Units : Shrink(u) /\ UNCHANGED rare
    \/ "Grow" \in in.acts /\ \E u \in Units : Grow(u) /\ UNCHANGED rare
    \* A prefetch of several keys brings them in order, as the Rust buffer
    \* pool's task does.
    \/ "Prefetch" \in in.acts /\ \E g \in in.gs :
         LET i == FirstToFetch(in.korder, 1)
         IN i > 0 /\ Prefetch(in.korder[i], "remote", g)
    \/ "StoreRemote" \in in.acts /\ \E y \in in.ks : StoreRemote(y)
    \/ "Deadline" \in in.acts /\ \E t \in tasks : Deadline(t)
    \/ "PrefetchEnd" \in in.acts /\ \E t \in tasks, ok \in BOOLEAN : PrefetchEnd(t, ok)
    \/ "PutEnd" \in in.acts /\ \E t \in tasks, ok \in BOOLEAN : PutEnd(t, ok)

-----------------------------------------------------------------------------

TraceInit ==
    /\ window \in 1 .. Len(Windows)
    /\ input = Windows[window].input
    /\ goal = Windows[window].state
    /\ bound = Windows[window].bound
    /\ walked = 0
    /\ reached = FALSE
    /\ Start(Windows[window].start)

\* The window's next state is reached, with no request under way (the
\* Rust buffer pool takes each request with the input it arrives in);
\* steps.py reads the line printed.
Reach ==
    /\ ~reached
    /\ requests = {}
    /\ Matches(goal)
    /\ PrintT(<<"reached", window>>)
    /\ reached' = TRUE
    /\ UNCHANGED <<vars, window, input, goal, bound, walked>>

(* A step changes what the next recorded state `st` settles only toward   *)
(* it: it takes no unit that `st` has free (so the free units steps.py     *)
(* keeps in a window as witnesses stay free), writes an owner's entry only *)
(* as `st` has it or empties it, caches a key's device or host block, or   *)
(* lets it go, only as `st` does, and starts a copy or a task only as `st` *)
(* has it in flight. The model chooses the unit an allocation, a copy or a *)
(* task takes; without this a search tries every unit for each. The Rust   *)
(* buffer pool sets each of these once in a step.                          *)
Toward(st) ==
    LET keyed(y) == {e \in st.keys : e.k = y}
        device(y) == IF keyed(y) = {} THEN "-" ELSE (CHOOSE e \in keyed(y) : TRUE).d
        host(y) == IF keyed(y) = {} THEN "-" ELSE (CHOOSE e \in keyed(y) : TRUE).h
    IN /\ \A u \in free \ free' : u \notin st.free
       /\ \A o \in Owners, q \in Positions :
            table'[o][q] # table[o][q] =>
              \/ table'[o][q] = Empty
              \/ st.owners[o].is = "live" /\ <<q, table'[o][q].k, table'[o][q].u>> \in st.owners[o].t
       /\ \A y \in Keys :
            /\ cache'[y] # cache[y] => Shown(cache'[y]) = device(y)
            /\ hcache'[y] # hcache[y] => Shown(hcache'[y]) = host(y)
       /\ \A c \in copies' \ copies : CopyLine(c)' \in st.copies
       /\ \A t \in tasks' \ tasks : TaskLine(t)' \in st.tasks

\* A step of the model's buffer pool on the way to the window's next state.
Walk ==
    /\ ~reached
    /\ walked < bound
    /\ Step(input)
    /\ Toward(goal)
    /\ walked' = walked + 1
    /\ UNCHANGED <<window, input, goal, bound, reached>>

TraceNext == Reach \/ Walk

\* What the search tells states apart by: MC_Pool's view, the window, and
\* whether its next state was reached. Not the steps taken: the search is
\* breadth first, so a state is first found by its shortest path, which
\* leaves the most steps of the bound.
TraceView == <<View, window, reached>>
=============================================================================
