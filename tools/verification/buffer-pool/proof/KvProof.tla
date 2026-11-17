------------------------------- MODULE KvProof -------------------------------
(***************************************************************************)
(* The proof that the design model keeps its safety properties:            *)
(*     Spec => []Safety                                                    *)
(* under the assumptions Design (every rule in force, no hazard switched   *)
(* on) and Sizes (what the configuration states about the constants).      *)
(***************************************************************************)
EXTENDS KvInductive, FiniteSetTheorems, TLAPS

AllRules == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}

ASSUME Design == Rules = AllRules /\ Hazards = {}

ASSUME Sizes ==
    /\ MaxBatch \in Nat /\ MaxRequests \in Nat /\ MaxCopies \in Nat /\ MaxTasks \in Nat
    /\ Units \cap Slots = {} /\ Units \cap HostUnits = {} /\ Slots \cap HostUnits = {}
    /\ NONE \notin Units /\ NONE \notin Slots /\ NONE \notin HostUnits /\ NONE \notin Keys
    /\ NONE \notin Owners /\ NONE \notin Groups /\ NONE \notin Positions /\ NONE \notin Epochs
    /\ "shared" \notin Owners /\ "copy" \notin Owners /\ "task" \notin Owners
    /\ SLOT \notin Positions
    /\ SlotGroup \in [Slots -> Groups]
    /\ GroupEpoch \in [Groups -> Epochs]
    /\ OwnerEpoch \in [Owners -> Epochs]
    /\ InitialUnits \subseteq AllUnits /\ Movable \subseteq AllUnits
    /\ TrialEpochs \subseteq Epochs /\ Losable \subseteq Epochs
    /\ Eager \in BOOLEAN
    /\ \A u, v \in InitialUnits \cap Units : u # v => UnitChunk[u] # UnitChunk[v]
    /\ GroupParts \in [Groups -> SUBSET HostParts]
    \* A memory is not a host part.
    /\ \A g \in Groups : GroupMemory[g] \notin HostParts
    \* A group reads the classes of owners of its own session.
    /\ Readers \in [Groups -> SUBSET Groups]
    /\ \A g \in Groups : \A x \in Readers[g] : GroupEpoch[x] = GroupEpoch[g]

LEMMA RulesOn == \A rule \in AllRules : On(rule)
  BY Design DEF On, AllRules

THEOREM InitInv == Init => IndInv
<1> SUFFICES ASSUME Init PROVE IndInv
  OBVIOUS
<1> USE Sizes, Design DEF Init, NONE, SLOT, AllUnits, Rows, Empty, Private, Shared, NoEntry, NoStamp
<1>1. TypeInv
  <2>1. TypeOK
    <3>1. free \subseteq AllUnits /\ member \subseteq AllUnits
      OBVIOUS
    <3>2. ostate \in [Owners -> {"new", "live", "gone"}]
          /\ ogroup \in [Owners -> Groups \cup {NONE}]
          /\ oslot \in [Owners -> Slots \cup {NONE}]
      OBVIOUS
    <3>3. \A o \in Owners, p \in Positions :
            /\ table[o][p].k \in {"none", "private", "shared"}
            /\ table[o][p].u \in Units \cup {NONE}
      OBVIOUS
    <3>4. cache \in [Keys -> Units \cup {NONE}] /\ hcache \in [Keys -> HostUnits \cup {NONE}]
      OBVIOUS
    <3>5. ncopy \in 0 .. MaxCopies /\ launched \in [Groups -> 0 .. MaxBatch]
      OBVIOUS
    <3>6. /\ valid \in [Units \cup HostUnits -> SUBSET (Memories \cup HostParts)]
          /\ \A u \in Units : valid[u] \subseteq Memories
          /\ \A h \in HostUnits : valid[h] \subseteq HostParts
      OBVIOUS
    <3>7. \A k \in Keys : vis[k].d \in Epochs \cup {NONE} /\ vis[k].h \in Epochs \cup {NONE}
      OBVIOUS
    <3>8. trial \in [Epochs -> {"no", "yes", "rolled"}]
      OBVIOUS
    <3>9. alloc \in [AllUnits -> Owners \cup {NONE, "shared", "copy", "task"}]
          /\ sharedKey \in [AllUnits -> Keys \cup {NONE}]
          /\ remote \subseteq Keys /\ lost \subseteq Epochs
      OBVIOUS
    <3> QED
      BY <3>1, <3>2, <3>3, <3>4, <3>5, <3>6, <3>7, <3>8, <3>9 DEF TypeOK
  <2>2. DOMAIN table = Owners
        /\ \A o \in Owners :
             /\ DOMAIN table[o] = Positions
             /\ \A p \in Positions :
                  table[o][p] \in {Empty} \cup {Private(u) : u \in Units} \cup {Shared(u) : u \in Units}
    OBVIOUS
  <2>3. DOMAIN dbt = Groups
        /\ \A g \in Groups :
             /\ DOMAIN dbt[g] = Owners
             /\ \A o \in Owners :
                  /\ DOMAIN dbt[g][o] = Rows
                  /\ \A p \in Rows : dbt[g][o][p].u \in AllUnits \cup {NONE}
    OBVIOUS
  <2>4. vis \in [Keys -> [d : Epochs \cup {NONE}, h : Epochs \cup {NONE}]]
    OBVIOUS
  <2> QED
    BY <2>1, <2>2, <2>3, <2>4 DEF TypeInv, Stamps
<1>2. FreeClean /\ NoLeak /\ Outside /\ Regions
  BY DEF FreeClean, NoLeak, Outside, Regions, Held, HeldIn, InRegion, OneRegion
<1>3. AllocOwner /\ SharedIff /\ SharedCached /\ CacheShared
  BY DEF AllocOwner, SharedIff, SharedCached, CacheShared
<1>4. Tables /\ OwnersOk /\ DeferredOk /\ CopiesOk /\ TasksOk
  BY DEF Tables, OwnersOk, DeferredOk, CopiesOk, TasksOk, Live
<1>5. MovingUnmapped /\ LostOk
  BY DEF MovingUnmapped, LostOk
<1>6. TablesMapped /\ HandoffOk /\ RequestsOk /\ BatchesOk
  BY DEF TablesMapped, HandoffOk, RequestsOk, BatchesOk
<1>7. TrialVisibility /\ NoBad
  BY DEF TrialVisibility, NoBad
<1>8. CrossingsOk /\ CopiesPlaced /\ CrossingsPlaced
  BY DEF CrossingsOk, CopiesPlaced, CrossingsPlaced
<1> QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8 DEF IndInv

THEOREM InvSafe == IndInv => Safety
<1> SUFFICES ASSUME IndInv PROVE Safety
  OBVIOUS
<1> USE Sizes, Design DEF NONE, SLOT, AllUnits, Rows, Empty, Private, Shared, NoEntry, Live
<1>1. TypeOK
  BY DEF IndInv, TypeInv
<1>2. FreeUnheld
  BY DEF IndInv, FreeClean, FreeUnheld
<1>3. Exclusive
  <2> SUFFICES ASSUME NEW u \in AllUnits
               PROVE  /\ Cardinality(TableHolders(u)) + Cardinality(CopyWriters(u)) <= 1
                      /\ (TableHolders(u) # {} \/ CopyWriters(u) # {}) => ~Pinned(u) /\ CacheKeys(u) = {}
    BY DEF Exclusive
  <2>1. \A x, y \in TableHolders(u) : x = y
    <3> DEFINE InTable(x) == x \in Owners \X Positions /\ table[x[1]][x[2]] = Private(u)
               InSlot(x) == x \in Owners \X {SLOT} /\ oslot[x[1]] = u
    <3> SUFFICES ASSUME NEW x \in TableHolders(u), NEW y \in TableHolders(u) PROVE x = y
      OBVIOUS
    <3>1. (InTable(x) \/ InSlot(x)) /\ (InTable(y) \/ InSlot(y))
      BY DEF TableHolders
    <3>2. ASSUME NEW z, InTable(z)
          PROVE  u \in Units /\ alloc[u] = z[1] /\ z = <<z[1], z[2]>> /\ z[1] \in Owners
                 /\ z[2] \in Positions
      BY <3>2 DEF IndInv, Tables
    <3>3. ASSUME NEW z, InSlot(z)
          PROVE  u \in Slots /\ alloc[u] = z[1] /\ z = <<z[1], SLOT>> /\ z[1] \in Owners
      BY <3>3 DEF IndInv, OwnersOk
    <3>4. CASE InTable(x) /\ InTable(y)
      <4>1. x[1] = y[1]
        BY <3>4, <3>2
      <4>2. x[2] = y[2]
        BY <3>4, <3>2, <4>1 DEF IndInv, Tables
      <4> QED
        BY <3>4, <3>2, <4>1, <4>2
    <3>5. CASE InSlot(x) /\ InSlot(y)
      BY <3>5, <3>3
    <3>6. CASE (InTable(x) /\ InSlot(y)) \/ (InSlot(x) /\ InTable(y))
      BY <3>6, <3>2, <3>3
    <3> QED
      BY <3>1, <3>4, <3>5, <3>6
  <2>2. \A c, d \in CopyWriters(u) : c = d
    <3> SUFFICES ASSUME NEW c \in CopyWriters(u), NEW d \in CopyWriters(u) PROVE c = d
      OBVIOUS
    <3>1. c \in copies /\ d \in copies /\ ~c.fill /\ c.write = u /\ d.write = u
      BY DEF CopyWriters
    <3>2. \A x \in copies : x.write = c.write => x = c
      BY <3>1 DEF IndInv, CopiesOk
    <3> QED
      BY <3>1, <3>2
  <2>3. TableHolders(u) = {} \/ CopyWriters(u) = {}
    <3> SUFFICES ASSUME NEW x \in TableHolders(u), NEW c \in CopyWriters(u) PROVE FALSE
      OBVIOUS
    <3>1. c \in copies /\ ~c.fill /\ c.write = u
      BY DEF CopyWriters
    <3>2. /\ \A o \in Owners, p \in Positions : table[o][p].u # u
          /\ u \in Units \cup HostUnits
      BY <3>1 DEF IndInv, CopiesOk
    <3>3. CASE x \in Owners \X Positions /\ table[x[1]][x[2]] = Private(u)
      BY <3>2, <3>3 DEF Private
    <3>4. CASE x \in Owners \X {SLOT} /\ oslot[x[1]] = u
      <4>1. oslot[x[1]] \in Slots \cup {NONE}
        BY <3>4 DEF IndInv, TypeInv, TypeOK
      <4> QED
        BY <3>2, <3>4, <4>1
    <3> QED
      BY <3>3, <3>4 DEF TableHolders
  <2>4. Cardinality(TableHolders(u)) + Cardinality(CopyWriters(u)) <= 1
    <3>1. \A S : (\A x, y \in S : x = y) => IsFiniteSet(S) /\ Cardinality(S) \in {0, 1}
      <4> SUFFICES ASSUME NEW S, \A x, y \in S : x = y
                   PROVE  IsFiniteSet(S) /\ Cardinality(S) \in {0, 1}
        OBVIOUS
      <4>1. CASE S = {}
        BY <4>1, FS_EmptySet
      <4>2. CASE S # {}
        <5>1. PICK x \in S : TRUE
          BY <4>2
        <5>2. S = {x}
          BY <5>1
        <5> QED
          BY <5>2, FS_Singleton
      <4> QED
        BY <4>1, <4>2
    <3>2. CASE TableHolders(u) = {}
      BY <3>1, <3>2, <2>2, FS_EmptySet
    <3>3. CASE CopyWriters(u) = {}
      BY <3>1, <3>3, <2>1, FS_EmptySet
    <3> QED
      BY <2>3, <3>2, <3>3
  <2>5. ASSUME TableHolders(u) # {} \/ CopyWriters(u) # {}
        PROVE  ~Pinned(u) /\ CacheKeys(u) = {}
    BY <2>5 DEF IndInv, Tables, OwnersOk, CopiesOk, CacheShared, SharedIff, TableHolders, CopyWriters,
                Pinned, CacheKeys, TypeInv, TypeOK
  <2> QED
    BY <2>4, <2>5
<1>4. WriteIsolation /\ ReadIsolation /\ CopyIsolation
  BY DEF IndInv, BatchesOk, WriteIsolation, ReadIsolation, CopyIsolation
<1>5. TaskIsolation
  BY DEF IndInv, TasksOk, CrossingsOk, FreeClean, Held, HeldIn, TaskIsolation, CacheShared,
         SharedIff, TypeInv
<1>6. KeyStable
  BY DEF IndInv, SharedIff, SharedCached, FreeClean, Tables, OwnersOk, CopiesOk, KeyStable,
         TableHolders, CopyWriters, TypeInv, TypeOK
<1>7. PinsCached
  BY DEF IndInv, Tables, SharedIff, PinsCached, TypeInv, TypeOK
<1>8. OneKey
  <2> SUFFICES ASSUME NEW u \in AllUnits PROVE Cardinality(CacheKeys(u)) <= 1
    BY DEF OneKey
  <2>1. \A k, l \in CacheKeys(u) : k = l
    BY DEF IndInv, CacheShared, CacheKeys
  <2>2. CASE CacheKeys(u) = {}
    BY <2>2, FS_EmptySet
  <2>3. CASE CacheKeys(u) # {}
    <3>1. PICK k \in CacheKeys(u) : TRUE
      BY <2>3
    <3>2. CacheKeys(u) = {k}
      BY <3>1, <2>1
    <3> QED
      BY <3>2, FS_Singleton
  <2> QED
    BY <2>2, <2>3
<1>9. SharedUnwritten /\ CopiesReadShared
  <2>1. \A b \in inflight : \A u \in b.cw :
          alloc[u] = "shared" => Mem(b.g) \notin valid[u]
    <3> SUFFICES ASSUME NEW b \in inflight, NEW u \in b.cw, alloc[u] = "shared"
                 PROVE  Mem(b.g) \notin valid[u]
      OBVIOUS
    <3>1. PICK c \in copies : c.write = u /\ c.g = b.g /\ c.seq = b.seq
      BY DEF IndInv, BatchesOk
    <3>2. c.fill
      BY <3>1 DEF IndInv, CopiesOk
    <3>3. CASE c.kind = "load"
      BY <3>1, <3>2, <3>3 DEF IndInv, CopiesOk
    <3>4. CASE c.kind = "restore"
      BY <3>1, <3>2, <3>4 DEF IndInv, CopiesOk, OwnersOk, AllocOwner
    <3>5. CASE c.kind = "store"
      <4>1. u \in HostUnits /\ b.g \in Groups
        BY <3>1, <3>2, <3>5 DEF IndInv, CopiesOk, BatchesOk
      <4>2. valid[u] \subseteq HostParts
        BY <4>1 DEF IndInv, TypeInv, TypeOK
      <4> QED
        BY <4>1, <4>2 DEF Mem
    <3> QED
      BY <3>1, <3>2, <3>3, <3>4, <3>5 DEF IndInv, CopiesOk
  <2> QED
    BY <2>1 DEF IndInv, BatchesOk, CopiesOk, SharedUnwritten, CopiesReadShared, TypeInv
<1>10. SourcesHeld /\ Exactness
  BY DEF IndInv, NoBad, SourcesHeld, Exactness
<1>11. NoLeak /\ OneRegion /\ TrialVisibility
  BY DEF IndInv, Regions
<1>12. Quiescent
  <2> SUFFICES ASSUME \A o \in Owners : ostate[o] # "live",
                      deferred = {}, copies = {}, tasks = {},
                      NEW u \in AllUnits, InRegion(u), u \notin free
               PROVE  CacheKeys(u) # {} /\ ~Pinned(u)
    BY DEF Quiescent
  <2>1. \A o \in Owners : oslot[o] = NONE /\ \A p \in Positions : table[o][p] = Empty
    BY DEF IndInv, OwnersOk
  <2>2. Held(u)
    BY DEF IndInv, NoLeak
  <2>3. \E k \in Keys : cache[k] = u \/ hcache[k] = u
    BY <2>1, <2>2 DEF Held, HeldIn
  <2>4. ~Pinned(u)
    BY <2>1 DEF Pinned
  <2> QED
    BY <2>3, <2>4 DEF CacheKeys
<1>13. P8
  BY DEF IndInv, P8
<1> QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13
     DEF Safety, P1, P2, P3, P5, P6
=============================================================================
