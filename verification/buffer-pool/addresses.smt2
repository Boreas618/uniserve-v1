; The address arithmetic of the buffer pool (specs/buffer-pool.md §16): an
; address computed from a run lies inside its region, distinct runs do not
; overlap, splitting a run loses and duplicates nothing, a token's slot names
; one unit, and an aligned product run stays inside the space it was given.
;
; Integers are mathematical; the assumption A8 (the sizes the configuration
; states) bounds a region's end by 2^64, so every value below is a u64 and the
; implementation's unsigned arithmetic does not wrap. Every goal asserts the
; negation of its claim: z3 answers `unsat` for each of the thirteen.
;
;   z3 addresses.smt2
;
; The goals are linear. A product of two unknowns is a constant of its own,
; related to the others only by the four facts about multiplication that the
; first four goals prove.

(set-logic ALL)

; --- Multiplication: the facts the goals below use.
(push)                                    ; M1. It distributes over a sum.
(declare-const a Int) (declare-const b Int) (declare-const c Int) (declare-const u Int)
(assert (not (= (* (+ a b c) u) (+ (* a u) (* b u) (* c u)))))
(check-sat)
(pop)
(push)                                    ; M2. And over a difference.
(declare-const a Int) (declare-const b Int) (declare-const u Int)
(assert (not (= (* (- a b) u) (- (* a u) (* b u)))))
(check-sat)
(pop)
(push)                                    ; M3. A product of non-negatives is non-negative.
(declare-const a Int) (declare-const u Int)
(assert (and (>= a 0) (> u 0)))
(assert (not (>= (* a u) 0)))
(check-sat)
(pop)
(push)                                    ; M4. A positive multiple of u is at least u.
(declare-const a Int) (declare-const u Int)
(assert (and (> a 0) (> u 0)))
(assert (not (>= (* a u) u)))
(check-sat)
(pop)

; --- A region and a run of it.
(define-fun two64 () Int 18446744073709551616)
(declare-const base Int)          ; the region's first byte
(declare-const unit Int)          ; a unit's bytes
(declare-const capacity_bytes Int) ; capacity * unit
(declare-const first_bytes Int)   ; first * unit: the run's first unit
(declare-const count_bytes Int)   ; count * unit: the run's units, at least one
(declare-const slack_bytes Int)   ; (capacity - first - count) * unit
(assert (and (>= base 0) (> unit 0)))
(assert (<= (+ base capacity_bytes) two64))                               ; A8
(assert (= capacity_bytes (+ first_bytes count_bytes slack_bytes)))       ; M1
(assert (and (>= first_bytes 0) (>= slack_bytes 0)))                      ; M3
(assert (>= count_bytes unit))                                            ; M4

(declare-const offset Int)        ; a byte of the run
(assert (and (>= offset 0) (< offset count_bytes)))
(define-fun address () Int (+ base first_bytes offset))

(push)                                    ; 1. The address lies inside the region.
(assert (not (and (>= address base) (< address (+ base capacity_bytes)))))
(check-sat)
(pop)

(push)                                    ; 2. The address is a u64.
(assert (not (< address two64)))
(check-sat)
(pop)

(push)                                    ; 3. A run that begins at or after this run's end
(declare-const gap_bytes Int)             ;    holds no byte of it.
(declare-const offset2 Int)
(assert (and (>= gap_bytes 0) (>= offset2 0)))                            ; M3
(define-fun first2_bytes () Int (+ first_bytes count_bytes gap_bytes))    ; M1
(assert (not (< address (+ base first2_bytes offset2))))
(check-sat)
(pop)

(push)                                    ; 4. Split after `left` units: a byte of the run
(declare-const left_bytes Int)            ;    is in the left part or in the right part.
(assert (and (>= left_bytes unit) (<= left_bytes (- count_bytes unit))))  ; M4
(define-fun right_bytes () Int (- count_bytes left_bytes))                ; M2
(assert (not (or (< offset left_bytes)
                 (and (>= offset left_bytes) (< (- offset left_bytes) right_bytes)))))
(check-sat)
(pop)

(push)                                    ; 5. And not in both: the right part begins where
(declare-const left_bytes Int)            ;    the left part ends.
(declare-const in_left Int)
(declare-const in_right Int)
(assert (and (>= in_left 0) (< in_left left_bytes) (>= in_right 0)))
(define-fun right_first_bytes () Int (+ first_bytes left_bytes))          ; M1
(assert (not (< (+ base first_bytes in_left) (+ base right_first_bytes in_right))))
(check-sat)
(pop)

(push)                                    ; 6. A token's slot, unit * block + token with
(declare-const block Int)                 ;    token < block, names its unit: tokens of
(declare-const slots_a Int)               ;    different units have different slots.
(declare-const apart_slots Int)           ; (unit_b - unit_a) * block, unit_b > unit_a
(declare-const token_a Int)
(declare-const token_b Int)
(assert (> block 0))
(assert (and (>= token_a 0) (< token_a block) (>= token_b 0) (< token_b block)))
(assert (>= apart_slots block))                                           ; M4
(define-fun slots_b () Int (+ slots_a apart_slots))                       ; M1
(assert (not (< (+ slots_a token_a) (+ slots_b token_b))))
(check-sat)
(pop)

; --- A product run is aligned to 256 bytes: from `start` it begins at the
;     next multiple of 256, (start + 255) rounded down.
(declare-const start Int)
(declare-const quotient Int)
(declare-const remainder Int)
(assert (>= start 0))
(assert (and (= (+ start 255) (+ (* 256 quotient) remainder)) (>= remainder 0) (< remainder 256)))
(define-fun aligned () Int (* 256 quotient))

(push)                                    ; 7. It is a multiple of 256 at or after `start`.
(assert (not (and (>= aligned start) (= (mod aligned 256) 0))))
(check-sat)
(pop)

(push)                                    ; 8. It wastes fewer than 256 bytes.
(assert (not (< (- aligned start) 256)))
(check-sat)
(pop)

(push)                                    ; 9. A run of `size` bytes ends by `limit` whenever
(declare-const size Int)                  ;    `start + size + 255` does.
(declare-const limit Int)
(assert (and (>= size 0) (<= (+ start size 255) limit)))
(assert (not (<= (+ aligned size) limit)))
(check-sat)
(pop)
