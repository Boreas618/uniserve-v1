------------------------------ MODULE KvTrace ------------------------------
(* Steps of the Rust buffer pool checked against the steps of the design   *)
(* model's buffer pool (specs/buffer-pool.md §18). A window is a state the *)
(* Rust buffer pool reached and the inputs and states that followed it, as *)
(* the enumeration of L1 or the replay of a trace wrote them; TLC must     *)
(* find, from the window's first state, a path of MC_Pool's steps through  *)
(* each of its states in turn. A state is compared as the Rust buffer pool *)
(* writes it in the model's terms (agreement.py's projection, here as      *)
(* `Matches`); what that leaves out, the stamps and keys of deferred work  *)
(* and the keys of copies and tasks, TLC chooses at the window's start.    *)
(*                                                                         *)
(* Every step of a path is a step of `PoolNext`: each disjunct of `Step`   *)
(* is one of its disjuncts with a guard that the input allows it, the      *)
(* standard pruning of a trace's validation by its events. Pruning removes *)
(* paths and never adds one, so it can make a window fail and never pass.  *)
(* steps.py writes the windows and the constants of a run as MC_Trace.     *)
EXTENDS MC_Pool, Sequences

CONSTANT Windows   \* <<[start |-> ..., steps |-> <<[input, state, bound], ...>>], ...>>

VARIABLES window,  \* the window searched
          matched, \* how many of its states were reached
          walked   \* the model's steps since the last state reached

trace == <<window, matched, walked>>

-----------------------------------------------------------------------------
(* A state as the Rust buffer pool writes it.                              *)

Shown(u) == IF u = NONE THEN "-" ELSE u

\* A copy in flight, as a `copy` line states it.
CopyLine(c) ==
    [kind |-> c.kind,
     o |-> IF \/ c.kind \in {"clone", "load", "restore"}
              \/ c.kind = "save" /\ ostate[c.o] = "live" /\ oslot[c.o] = c.read
           THEN c.o ELSE "-",
     p |-> IF c.kind \in {"clone", "load"} THEN c.p ELSE "-",
     read |-> c.read, write |-> c.write,
     fill |-> IF c.fill /\ c.kind = "load" THEN 1 ELSE 0,
     mem |-> IF c.kind \in {"clone", "load"} THEN GroupMemory[c.g] ELSE "-"]

\* A task in flight, as a `task` line states it.
TaskLine(t) ==
    IF t.kind = "transfer"
    THEN [kind |-> "transfer", o |-> t.o, to |-> t.to, slot |-> t.slot, units |-> t.units,
          writes |-> t.writes, h |-> "-", expired |-> 0]
    ELSE [kind |-> t.kind, o |-> "-", to |-> "-", slot |-> "-", units |-> {}, writes |-> {},
          h |-> t.h, expired |-> IF t.expired THEN 1 ELSE 0]

Waiting(kind) == {d \in deferred : d.kind = kind}

\* Whether the model's state is the recorded state `st`.
Matches(st) ==
    /\ ncopy = st.ncopy
    /\ launched = st.completed
    /\ free = st.free
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
    /\ st.keys =
         {[k |-> y, d |-> Shown(cache[y]), h |-> Shown(hcache[y]), r |-> IF y \in remote THEN 1 ELSE 0] :
            y \in {x \in Keys : cache[x] # NONE \/ hcache[x] # NONE \/ x \in remote}}
    /\ LET live == {o \in Owners : ostate[o] = "live"}
           blocks == ({table[o][q].u : o \in live, q \in Positions} \cup {cache[y] : y \in Keys})
                       \ {NONE}
       IN st.valid = {<<b, valid[b]>> : b \in blocks}
    /\ st.copies = {CopyLine(c) : c \in copies}
    /\ st.commits = {[o |-> d.o, p |-> d.p, u |-> CHOOSE b \in d.units : TRUE] : d \in Waiting("commit")}
    /\ st.transfers = {[o |-> d.o, to |-> d.to] : d \in Waiting("transfer")}
    /\ st.finishes =
         UNION {{[ok |-> IF d.ok THEN 1 ELSE 0, copy |-> CopyLine(c)] : c \in {x \in copies : x.id = d.c}} :
                  d \in Waiting("finish")}
    /\ st.released = UNION {d.units : d \in Waiting("release")}
    /\ st.tasks = {TaskLine(t) : t \in tasks}

-----------------------------------------------------------------------------
(* A recorded state as a state of the model: `x.state` with what it leaves *)
(* out chosen, among which the memories that hold a block only waiting    *)
(* work, a copy or a task holds. `x.member` is the device units in their  *)
(* region, `x.copies` the copies in flight in an order that numbers them, *)
(* and `x.finishes` the waiting `Finish`es by the index of their copy.    *)

\* The stamps of work that waits: some group's is not reached.
Waits == {a \in [Groups -> 0 .. MaxBatch] : ~Reached(a)}

Start(x) ==
    LET st == x.state
        n == Len(x.copies)
        live(o) == st.owners[o].is = "live"
        entry(o, q) == {e \in st.owners[o].t : e[1] = q}
        keyed(y) == {e \in st.keys : e.k = y}
        ident(j) == st.ncopy - n + j
        stated == {v[1] : v \in st.valid}
        unstated == (({cm.u : cm \in st.commits} \cup st.released
                      \cup {l.read : l \in st.copies} \cup {l.write : l \in st.copies}
                      \cup UNION {tl.units \cup tl.writes : tl \in st.tasks}) \cap Units) \ stated
        Copied(l, j, y, g, o) ==
            [id |-> ident(j), kind |-> l.kind, o |-> o, p |-> IF l.p = "-" THEN NONE ELSE l.p,
             k |-> y, g |-> g, read |-> l.read, write |-> l.write,
             fill |-> (l.kind = "load" /\ l.fill = 1) \/ l.kind = "restore",
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
       /\ \E vs \in [unstated -> SUBSET Memories] :
            valid = [b \in Units |-> IF b \in stated THEN (CHOOSE v \in st.valid : v[1] = b)[2]
                                    ELSE IF b \in unstated THEN vs[b] ELSE {}]
       /\ ncopy = st.ncopy
       /\ launched = st.completed
       /\ inflight = {}
       /\ handoff = {}
       /\ requests = {}
       \* A copy's key, and the group and owner its line does not state.
       /\ \E ys \in [1 .. n -> Keys], gs \in [1 .. n -> Groups], os \in [1 .. n -> Owners \cup {NONE}] :
            /\ \A j \in 1 .. n :
                 LET l == x.copies[j] IN
                 /\ IF l.o # "-" THEN os[j] = l.o ELSE (os[j] = NONE) = (l.kind = "store")
                 /\ l.mem # "-" => GroupMemory[gs[j]] = l.mem
                 /\ (l.o # "-" /\ live(l.o)) => gs[j] = st.owners[l.o].g
            /\ copies = {Copied(x.copies[j], j, ys[j], gs[j], os[j]) : j \in 1 .. n}
       \* The stamps and keys of what waits.
       /\ \E ca \in [st.commits -> Waits], ck \in [st.commits -> Keys],
             ta \in [st.transfers -> [Groups -> 0 .. MaxBatch]], fa \in [x.finishes -> Waits],
             ra \in IF st.released = {} THEN {NoStamp} ELSE Waits :
            deferred =
              {[kind |-> "commit", units |-> {cm.u}, after |-> ca[cm], o |-> cm.o, p |-> cm.p,
                k |-> ck[cm], c |-> 0, ok |-> FALSE, g |-> ogroup[cm.o], to |-> NONE] :
                 cm \in st.commits}
              \cup {[kind |-> "transfer", units |-> {}, after |-> ta[tr], o |-> tr.o, p |-> NONE,
                     k |-> NONE, c |-> 0, ok |-> FALSE, g |-> NONE, to |-> tr.to] :
                      tr \in st.transfers}
              \cup {[kind |-> "finish", units |-> {}, after |-> fa[fi], o |-> NONE, p |-> NONE,
                     k |-> NONE, c |-> ident(fi.at), ok |-> fi.ok = 1, g |-> NONE, to |-> NONE] :
                      fi \in x.finishes}
              \cup (IF st.released = {} THEN {}
                    ELSE {[kind |-> "release", units |-> st.released, after |-> ra, o |-> NONE,
                           p |-> NONE, k |-> NONE, c |-> 0, ok |-> FALSE, g |-> NONE, to |-> NONE]})
       \* The key of a prefetch or a put.
       /\ \E ks \in [st.tasks -> Keys] :
            tasks =
              {IF tl.kind = "transfer"
               THEN [kind |-> "transfer", k |-> NONE, h |-> NONE, src |-> NONE, ok |-> TRUE,
                     expired |-> FALSE, o |-> tl.o, to |-> tl.to, slot |-> tl.slot, held |-> TRUE,
                     units |-> tl.units, writes |-> tl.writes]
               ELSE [kind |-> tl.kind, k |-> ks[tl], h |-> tl.h, src |-> "remote", ok |-> TRUE,
                     expired |-> tl.expired = 1, o |-> NONE, to |-> NONE, slot |-> NONE,
                     held |-> TRUE, units |-> {}, writes |-> {}] :
                 tl \in st.tasks}
       /\ vis = [y \in Keys |-> [d |-> NONE, h |-> NONE]]
       /\ trial = [e \in Epochs |-> "no"]
       /\ dbt = [g \in Groups |-> [o \in Owners |-> [q \in Rows |-> NoEntry]]]
       /\ alloc = [b \in AllUnits |-> IF \E y \in Keys : cache[y] = b \/ hcache[y] = b
                                      THEN "shared" ELSE NONE]
       /\ sharedKey = [b \in AllUnits |-> NONE]
       /\ lost = {}
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
        /\ r.kind = "finish" => r.ok = q.ok

Step(in) ==
    \/ "Attach" \in in.acts /\ \E o \in in.os, g \in in.gs : Attach(o, g) /\ UNCHANGED rare
    \/ "Allocate" \in in.acts /\ \E o \in in.os, q \in in.ps : Allocate(o, q) /\ UNCHANGED rare
    \/ "Pin" \in in.acts /\ \E o \in in.os, q \in in.ps, y \in in.ks : Pin(o, q, y) /\ UNCHANGED rare
    \/ "Clone" \in in.acts /\ \E o \in in.os, q \in in.ps, y \in in.ks : Clone(o, q, y) /\ UNCHANGED rare
    \/ "Load" \in in.acts /\ \E o \in in.os, q \in in.ps, y \in in.ks : Load(o, q, y) /\ UNCHANGED rare
    \/ "Save" \in in.acts /\ \E o \in in.os, y \in in.ks : Save(o, y) /\ UNCHANGED rare
    \/ "Restore" \in in.acts /\ \E o \in in.os, y \in in.ks : Restore(o, y) /\ UNCHANGED rare
    \/ "Store" \in in.acts /\ \E g \in in.gs, y \in in.ks : Store(g, y) /\ UNCHANGED rare
    \/ "Evict" \in in.acts /\ \E y \in in.ks : Evict(y) /\ UNCHANGED rare
    \/ "EvictHost" \in in.acts /\ \E y \in Keys : EvictHost(y) /\ UNCHANGED rare
    \/ "Arrive" \in in.acts /\ Arrive /\ Asked(in.req) /\ UNCHANGED rare
    \/ "Advance" \in in.acts /\ \E g \in Groups : Advance(g) /\ launched'[g] <= in.completed[g]
                                                 /\ UNCHANGED rare
    \/ \E r \in requests :
         (TakeRelease(r) \/ TakeDetach(r) \/ TakeCommit(r) \/ TakeFinish(r) \/ TakeTransfer(r))
         /\ UNCHANGED rare
    \/ \E d \in deferred :
         (ReleaseDue(d) \/ CommitDue(d) \/ FinishDue(d) \/ TransferDue(d)) /\ UNCHANGED rare
    \/ "TransferEnd" \in in.acts /\ \E t \in tasks, ok \in BOOLEAN : TransferEnd(t, ok) /\ UNCHANGED rare
    \/ "CommitUpdate" \in in.acts /\ \E e \in Epochs : CommitUpdate(e) /\ UNCHANGED rare
    \/ "Rollback" \in in.acts /\ \E e \in Epochs : Rollback(e) /\ UNCHANGED rare
    \/ "Shrink" \in in.acts /\ \E u \in Units : Shrink(u) /\ UNCHANGED rare
    \/ "Grow" \in in.acts /\ \E u \in Units : Grow(u) /\ UNCHANGED rare
    \/ "Prefetch" \in in.acts /\ \E y \in in.ks : Prefetch(y, "remote")
    \/ "StoreRemote" \in in.acts /\ \E y \in in.ks : StoreRemote(y)
    \/ "Deadline" \in in.acts /\ \E t \in tasks : Deadline(t)
    \/ "PrefetchEnd" \in in.acts /\ \E t \in tasks, ok \in BOOLEAN : PrefetchEnd(t, ok)
    \/ "PutEnd" \in in.acts /\ \E t \in tasks, ok \in BOOLEAN : PutEnd(t, ok)

-----------------------------------------------------------------------------

TraceInit ==
    /\ window \in 1 .. Len(Windows)
    /\ matched = 0
    /\ walked = 0
    /\ Start(Windows[window].start)

\* The window's next state is reached; steps.py reads the line printed.
Reach ==
    LET steps == Windows[window].steps
    IN /\ matched < Len(steps)
       /\ Matches(steps[matched + 1].state)
       /\ PrintT(<<"reached", window, matched + 1>>)
       /\ matched' = matched + 1
       /\ walked' = 0
       /\ UNCHANGED <<vars, window>>

\* A step of the model's buffer pool on the way to the window's next state.
Walk ==
    LET next == Windows[window].steps[matched + 1]
    IN /\ matched < Len(Windows[window].steps)
       /\ walked < next.bound
       /\ Step(next.input)
       /\ matched' = matched
       /\ walked' = walked + 1
       /\ UNCHANGED window

TraceNext == Reach \/ Walk

\* What the search tells states apart by: MC_Pool's view and the window's progress.
TraceView == <<View, trace>>
=============================================================================
