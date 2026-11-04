//! A fixture policy package: the arrival-order body, for the program layer's
//! delivery test. It is not a deployment policy; `fcfs` is.

type Fixture = scheduling_guest::Arrival<scheduling_guest::ArrivalOrder>;

scheduling_guest::export_policy!(Fixture);
