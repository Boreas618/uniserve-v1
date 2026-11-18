;; A policy module of interface version 7 that speaks every export of the scheduling policy
;; interface and decides nothing: `sched_init` replies nothing, `sched_limits` replies a
;; 256-token prefetch threshold and a 30 s starvation bound, `sched_dispatch` a decision that names nothing, `sched_observe` and
;; `sched_preempt` reply nothing, and `sched_place` chooses the first candidate.
;;
;; Every message is written at 1024, and the replies lie at the end of the one page, past any
;; message a test sends, each packed as `ptr << 32 | len`: the limits at 64768 (278176441827377), a
;; decision that names nothing at 65024 (279275953455177).
(module
  (memory (export "memory") 1)
  (data (i32.const 64768) "{\"prefetch_min_tokens\":256,\"timeout\":30000000000}")
  (data (i32.const 65024) "{\"admit\":[],\"reject\":[],\"work\":[],\"offload\":[],\"prefetch\":[],\"refill\":[]}")
  (func (export "sched_interface_version") (result i32) (i32.const 7))
  (func (export "sched_alloc") (param i32) (result i32) (i32.const 1024))
  (func (export "sched_free") (param i32 i32))
  (func (export "sched_init") (param i32 i32) (result i64) (i64.const 0))
  (func (export "sched_observe") (param i32 i32) (result i64) (i64.const 0))
  (func (export "sched_dispatch") (param i32 i32) (result i64) (i64.const 279275953455177))
  (func (export "sched_preempt") (param i32 i32) (result i64) (i64.const 0))
  (func (export "sched_place") (param i32 i32) (result i32) (i32.const 0))
  (func (export "sched_limits") (result i64) (i64.const 278176441827377)))
