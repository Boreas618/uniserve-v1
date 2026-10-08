//! NVTX ranges of the `uniserve.worker` domain, enabled by `UNISERVE_NVTX`.
//! No language callback or device operation is involved.

use std::ffi::CStr;
use std::sync::OnceLock;

use nvtx::{Domain, domain::LocalRange};

/// Mark a synchronous phase on its executing thread. Batch phases carry the
/// scheduler's batch ID; shared resource reaping has no single batch owner.
pub(crate) fn range(name: &'static CStr, batch_id: Option<u64>) -> Option<LocalRange<'static>> {
    static DOMAIN: OnceLock<Option<Domain>> = OnceLock::new();

    let domain = DOMAIN.get_or_init(|| {
        let enabled = super::flag(std::env::var("UNISERVE_NVTX").ok().as_deref());
        enabled.then(|| Domain::new(c"uniserve.worker"))
    });

    domain.as_ref().map(|domain| {
        let mut attributes = domain.event_attributes_builder().message(name);
        if let Some(batch_id) = batch_id {
            attributes = attributes.payload(batch_id);
        }

        domain.local_range(attributes.build())
    })
}
