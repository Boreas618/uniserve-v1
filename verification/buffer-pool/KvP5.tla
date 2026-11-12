-------------------------------- MODULE KvP5 --------------------------------
(* P5 Trial visibility of specs/buffer-pool.md §17, as a formula over KvCore. *)
EXTENDS KvCore

(* P5 Trial visibility: until its update is committed, what an epoch on    *)
(* trial committed is pinned and copied only for that epoch.               *)
TrialVisibility ==
    \A k \in Keys :
        /\ (cache[k] # NONE /\ vis[k].d # NONE) =>
             /\ \A o \in Owners, p \in Positions :
                  table[o][p] = Shared(cache[k]) => OwnerEpoch[o] = vis[k].d
             /\ \A c \in copies : c.read = cache[k] => GroupEpoch[c.g] = vis[k].d
        /\ (hcache[k] # NONE /\ vis[k].h # NONE) =>
             /\ \A c \in copies : c.read = hcache[k] => GroupEpoch[c.g] = vis[k].h
             /\ \A t \in tasks : t.kind = "put" => t.h # hcache[k]
        \* An epoch on trial fills no memory of a block others may use.
        /\ \A c \in copies :
             (c.kind = "load" /\ c.fill /\ c.k = k) => vis[k].d = MadeBy(GroupEpoch[c.g])

P5 == TrialVisibility
=============================================================================
