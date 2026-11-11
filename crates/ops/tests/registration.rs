//! The operator names this build installs.
//!
//! `ops::install` resolves a module's operator through `proto::module::operator::named`, so the
//! install table is `proto::module::operator::ALL` itself and an operator's `Op::NAME` is one of
//! its constants. What remains a property of the table is that its names are distinct: a package's
//! module is keyed by `(operator, descriptor_key)`, and two operators answering one name would make
//! the second unloadable and the first ambiguous.

use std::collections::BTreeSet;

#[test]
fn the_names_this_build_installs_are_distinct() {
    let names = proto::module::operator::ALL;
    let distinct: BTreeSet<&&str> = names.iter().collect();
    assert_eq!(
        names.len(),
        distinct.len(),
        "{} names, {} distinct",
        names.len(),
        distinct.len()
    );
}
