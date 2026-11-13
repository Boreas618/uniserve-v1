------------------------------ MODULE KvProofP7 ------------------------------
(***************************************************************************)
(* P7 Capacity identity follows from the inductive invariant:              *)
(*     IndInv => P7                                                        *)
(* so that Spec => []P7 holds wherever Spec => []IndInv does, under the    *)
(* assumptions of KvProof and Finite.                                      *)
(***************************************************************************)
EXTENDS KvProof, KvP7

(* The configuration's sets of units are finite.                           *)
ASSUME Finite == IsFiniteSet(Units) /\ IsFiniteSet(Slots) /\ IsFiniteSet(HostUnits)

(* What is observed as shared or private is held.                          *)
LEMMA SharedHeld == \A u : ObservedShared(u) => Held(u)
  BY DEF ObservedShared, CacheKeys, Held, HeldIn

LEMMA PrivateHeld == \A u : ObservedPrivate(u) => Held(u)
<1> SUFFICES ASSUME NEW u, ObservedPrivate(u) PROVE Held(u)
  OBVIOUS
<1>1. CASE TableHolders(u) # {}
  <2>1. PICK x \in TableHolders(u) : TRUE
    BY <1>1
  <2>2. CASE x \in Owners \X Positions /\ table[x[1]][x[2]] = Private(u)
    BY <2>2 DEF Held, HeldIn, Private
  <2>3. CASE x \in Owners \X {SLOT} /\ oslot[x[1]] = u
    BY <2>3 DEF Held, HeldIn
  <2> QED
    BY <2>1, <2>2, <2>3 DEF TableHolders
<1>2. CASE \E d \in deferred : d.kind = "commit" /\ u \in d.units
  BY <1>2 DEF Held, HeldIn
<1>3. CASE \E c \in copies : c.write = u /\ (c.kind \in {"store", "save"} \/ ostate[c.o] = "live")
  BY <1>3 DEF Held, HeldIn
<1>4. CASE \E t \in tasks :
             /\ t.held
             /\ \/ t.kind = "transfer" /\ t.slot = u /\ ostate[t.o] = "live"
                \/ t.kind = "prefetch" /\ t.h = u
  BY <1>4 DEF Held, HeldIn
<1> QED
  BY <1>1, <1>2, <1>3, <1>4 DEF ObservedPrivate

(* The units of a region are units of the model, and finitely many; each   *)
(* is in its region in the sense of NoLeak.                                *)
LEMMA UnitsOfRegions ==
  IndInv => \A r \in Capacities :
              /\ UnitsIn(r) \subseteq AllUnits
              /\ IsFiniteSet(UnitsIn(r))
              /\ \A u \in UnitsIn(r) : InRegion(u)
<1> SUFFICES ASSUME IndInv, NEW r \in Capacities
             PROVE  /\ UnitsIn(r) \subseteq AllUnits
                    /\ IsFiniteSet(UnitsIn(r))
                    /\ \A u \in UnitsIn(r) : InRegion(u)
  OBVIOUS
<1>1. member \subseteq Units
  BY DEF IndInv, Regions
<1>2. CASE r \in {<<"device", UnitRegion[u]>> : u \in Units}
  <2>1. PICK v \in Units : r = <<"device", UnitRegion[v]>>
    BY <1>2
  <2>2. r[1] = "device" /\ r[2] = UnitRegion[v]
    BY <2>1
  <2>3. UnitsIn(r) = {u \in member : UnitRegion[u] = r[2]}
    BY <2>2 DEF UnitsIn
  <2>4. UnitsIn(r) \subseteq Units
    BY <1>1, <2>3
  <2>5. IsFiniteSet(UnitsIn(r))
    BY <2>4, Finite, FS_Subset
  <2>6. \A u \in UnitsIn(r) : u \in member
    BY <2>3
  <2> QED
    BY <2>4, <2>5, <2>6 DEF AllUnits, InRegion
<1>3. CASE r \in {<<"slots", g>> : g \in Groups}
  <2>1. PICK g \in Groups : r = <<"slots", g>>
    BY <1>3
  <2>2. r[1] = "slots" /\ r[2] = g
    BY <2>1
  <2>3. UnitsIn(r) = {s \in Slots : SlotGroup[s] = r[2]}
    BY <2>2 DEF UnitsIn
  <2>4. UnitsIn(r) \subseteq Slots
    BY <2>3
  <2>5. IsFiniteSet(UnitsIn(r))
    BY <2>4, Finite, FS_Subset
  <2>6. \A u \in UnitsIn(r) : u \notin Units
    BY <2>4, Sizes
  <2> QED
    BY <2>4, <2>5, <2>6 DEF AllUnits, InRegion
<1>4. CASE r \in {<<"host", "host">>}
  <2>1. r[1] = "host"
    BY <1>4
  <2>2. UnitsIn(r) = HostUnits
    BY <2>1 DEF UnitsIn
  <2>3. \A u \in UnitsIn(r) : u \notin Units
    BY <2>2, Sizes
  <2> QED
    BY <2>2, <2>3, Finite DEF AllUnits, InRegion
<1> QED
  BY <1>2, <1>3, <1>4 DEF Capacities

(* Two disjoint finite sets: the cardinality of their union is the sum.    *)
LEMMA DisjointUnion ==
  ASSUME NEW X, NEW Y, IsFiniteSet(X), IsFiniteSet(Y), X \cap Y = {}
  PROVE  /\ IsFiniteSet(X \cup Y)
         /\ Cardinality(X \cup Y) = Cardinality(X) + Cardinality(Y)
<1>1. /\ IsFiniteSet(X \cup Y)
      /\ Cardinality(X \cup Y) = Cardinality(X) + Cardinality(Y) - Cardinality(X \cap Y)
  BY FS_Union
<1>2. Cardinality(X \cap Y) = 0
  BY FS_EmptySet
<1>3. Cardinality(X) \in Nat /\ Cardinality(Y) \in Nat
  BY FS_CardinalityType
<1> QED
  BY <1>1, <1>2, <1>3

(* Four finite sets, pairwise disjoint, whose union is U.                  *)
LEMMA Partition ==
  ASSUME NEW U, IsFiniteSet(U), NEW A \in SUBSET U, NEW S \in SUBSET U,
         NEW P \in SUBSET U, NEW R \in SUBSET U,
         A \cap S = {}, A \cap P = {}, A \cap R = {},
         S \cap P = {}, S \cap R = {}, P \cap R = {},
         A \cup S \cup P \cup R = U
  PROVE  Cardinality(A) + Cardinality(P) + Cardinality(S) + Cardinality(R) = Cardinality(U)
<1>1. IsFiniteSet(A) /\ IsFiniteSet(S) /\ IsFiniteSet(P) /\ IsFiniteSet(R)
  BY FS_Subset
<1>2. Cardinality(A) \in Nat /\ Cardinality(S) \in Nat /\ Cardinality(P) \in Nat /\ Cardinality(R) \in Nat
  BY <1>1, FS_CardinalityType
<1>3. IsFiniteSet(A \cup S) /\ Cardinality(A \cup S) = Cardinality(A) + Cardinality(S)
  BY <1>1, DisjointUnion
<1>4. (A \cup S) \cap P = {}
  OBVIOUS
<1>5. IsFiniteSet(A \cup S \cup P)
      /\ Cardinality(A \cup S \cup P) = Cardinality(A \cup S) + Cardinality(P)
  BY <1>1, <1>3, <1>4, DisjointUnion
<1>6. (A \cup S \cup P) \cap R = {}
  OBVIOUS
<1>7. Cardinality(A \cup S \cup P \cup R) = Cardinality(A \cup S \cup P) + Cardinality(R)
  BY <1>1, <1>5, <1>6, DisjointUnion
<1> QED
  BY <1>2, <1>3, <1>5, <1>7

THEOREM InvP7 == IndInv => P7
<1> SUFFICES ASSUME IndInv, NEW r \in Capacities
             PROVE  Cardinality(FreeIn(r)) + Cardinality(PrivateIn(r))
                      + Cardinality(SharedIn(r)) + Cardinality(RetiredIn(r))
                      = Cardinality(UnitsIn(r))
  BY DEF P7, CapacityIdentity
<1>1. /\ UnitsIn(r) \subseteq AllUnits
      /\ IsFiniteSet(UnitsIn(r))
      /\ \A u \in UnitsIn(r) : InRegion(u)
  BY UnitsOfRegions
<1>2. \A u \in free : ~Held(u)
  BY DEF IndInv, FreeClean
<1>3. \A u \in UnitsIn(r) : u \in free \/ Held(u)
  BY <1>1 DEF IndInv, NoLeak
<1>4. /\ FreeIn(r) \in SUBSET UnitsIn(r)
      /\ SharedIn(r) \in SUBSET UnitsIn(r)
      /\ PrivateIn(r) \in SUBSET UnitsIn(r)
      /\ RetiredIn(r) \in SUBSET UnitsIn(r)
  BY DEF FreeIn, SharedIn, PrivateIn, RetiredIn
<1>5. /\ FreeIn(r) \cap SharedIn(r) = {}
      /\ FreeIn(r) \cap PrivateIn(r) = {}
      /\ FreeIn(r) \cap RetiredIn(r) = {}
  BY <1>2, SharedHeld, PrivateHeld DEF FreeIn, SharedIn, PrivateIn, RetiredIn
<1>6. /\ SharedIn(r) \cap PrivateIn(r) = {}
      /\ SharedIn(r) \cap RetiredIn(r) = {}
      /\ PrivateIn(r) \cap RetiredIn(r) = {}
  BY DEF SharedIn, PrivateIn, RetiredIn
<1>7. FreeIn(r) \cup SharedIn(r) \cup PrivateIn(r) \cup RetiredIn(r) = UnitsIn(r)
  BY <1>3, <1>4 DEF FreeIn, SharedIn, PrivateIn, RetiredIn
<1> QED
  BY <1>1, <1>4, <1>5, <1>6, <1>7, Partition
=============================================================================
