//! A fixture policy package: the arrival-order body, for the generation layer's
//! delivery test. It is not a deployment policy; `fcfs` is.

type Fixture = scheduling::Arrival<scheduling::ArrivalOrder>;

scheduling::export_policy!(Fixture);
