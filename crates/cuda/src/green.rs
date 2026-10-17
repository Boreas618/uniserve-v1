//! SM partitions through the CUDA green-context API.

use std::sync::Arc;

use crate::driver::{CU_DEV_RESOURCE_TYPE_SM, DevResource, Handle, driver};
use crate::{DeviceGuard, DeviceId, Error, Result, Stream};

const CU_GREEN_CTX_DEFAULT_STREAM: u32 = 1;
const CU_STREAM_NON_BLOCKING: u32 = 1;

/// A green context: a disjoint set of a device's SMs.
///
/// Streams created in it ([`Stream::partition`]) and events recorded on those
/// streams retain it, so it is destroyed only after its last stream and event
/// are dropped. The owner never makes it current on a thread.
pub struct GreenContext {
    raw: Handle,
    sm_count: u32,
}

// SAFETY: this immutable owner only passes its handle to thread-safe driver
// calls (stream creation, resource queries, destruction).
unsafe impl Send for GreenContext {}
// SAFETY: as for Send above; the owner is immutable, so shared references only
// read its handle and SM count.
unsafe impl Sync for GreenContext {}

impl GreenContext {
    fn new(device: i32, mut resource: DevResource) -> Result<Self> {
        let driver = driver()?;
        let mut descriptor = std::ptr::null_mut();
        let mut raw = std::ptr::null_mut();

        driver.check(
            // SAFETY: the one resource passed was produced by the driver's split
            // API; the driver writes the descriptor to the local `descriptor`.
            unsafe { (driver.dev_resource_generate_desc)(&mut descriptor, &mut resource, 1) },
            "cuDevResourceGenerateDesc",
        )?;
        driver.check(
            // SAFETY: the descriptor was generated above for this device; the
            // driver writes the context handle to the local `raw`.
            unsafe {
                (driver.green_ctx_create)(&mut raw, descriptor, device, CU_GREEN_CTX_DEFAULT_STREAM)
            },
            "cuGreenCtxCreate",
        )?;

        Ok(Self {
            raw,
            sm_count: resource.sm_count,
        })
    }

    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }

    fn resource(&self) -> Result<DevResource> {
        let driver = driver()?;
        let mut resource = DevResource::empty();
        driver.check(
            // SAFETY: self is live; the driver fills the local `resource`.
            unsafe {
                (driver.green_ctx_get_dev_resource)(
                    self.raw,
                    &mut resource,
                    CU_DEV_RESOURCE_TYPE_SM,
                )
            },
            "cuGreenCtxGetDevResource",
        )?;
        Ok(resource)
    }
}

impl Drop for GreenContext {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            // SAFETY: the last stream and event retaining this context are gone.
            let _ = unsafe { (driver.green_ctx_destroy)(self.raw) };
        }
    }
}

/// The SM count of a (primary or green) context's SM resource.
pub(crate) fn context_sm_count(context: Handle) -> Result<u32> {
    let driver = driver()?;
    let mut resource = DevResource::empty();
    driver.check(
        // SAFETY: the caller's stream keeps the context alive; the driver fills
        // the local `resource`.
        unsafe { (driver.ctx_get_dev_resource)(context, &mut resource, CU_DEV_RESOURCE_TYPE_SM) },
        "cuCtxGetDevResource",
    )?;
    Ok(resource.sm_count)
}

/// Splits exactly `count` SMs off `resource`, returning the group and the
/// remainder. The driver may round a count up to its partition granularity;
/// lane capacities require the requested allocation, so that is an error.
fn split(resource: &DevResource, count: u32) -> Result<(DevResource, DevResource)> {
    let driver = driver()?;
    let mut group = DevResource::empty();
    let mut remainder = DevResource::empty();
    let mut groups = 1;
    driver.check(
        // SAFETY: all pointers address live resources; one group is requested,
        // so the driver writes one group to the local `group`.
        unsafe {
            (driver.dev_sm_resource_split_by_count)(
                &mut group,
                &mut groups,
                resource,
                &mut remainder,
                0,
                count,
            )
        },
        "cuDevSmResourceSplitByCount",
    )?;

    if groups != 1 || group.sm_count != count {
        return Err(Error::invalid(format!(
            "the device cannot partition exactly {count} SMs (the driver offers {})",
            group.sm_count
        )));
    }
    Ok((group, remainder))
}

pub(crate) fn partition(device: DeviceId, counts: &[u32]) -> Result<Vec<Stream>> {
    if counts.is_empty() {
        return Ok(Vec::new());
    }
    if counts.contains(&0) {
        return Err(Error::invalid("SM partition counts must be positive"));
    }
    let requested = counts
        .iter()
        .try_fold(0_u32, |total, count| total.checked_add(*count))
        .ok_or_else(|| Error::invalid("SM partition counts overflow"))?;

    let _guard = DeviceGuard::new(device)?;
    let driver = driver()?;
    let ordinal = device.handle()?;
    let mut resource = DevResource::empty();
    driver.check(
        // SAFETY: the driver fills the local `resource` with the SM resource of
        // the valid device `ordinal`.
        unsafe {
            (driver.device_get_dev_resource)(ordinal, &mut resource, CU_DEV_RESOURCE_TYPE_SM)
        },
        "cuDeviceGetDevResource",
    )?;
    if requested > resource.sm_count {
        return Err(Error::invalid(format!(
            "SM partitions require {requested} SMs but {device} exposes {}",
            resource.sm_count
        )));
    }

    // All partitions come from one split tree: an aggregate of the requested
    // SMs is split lane by lane. A split result can be split again only after
    // the driver has materialized it as a context resource, hence the chain of
    // parent contexts, all kept until every lane exists. Streams retain only
    // their own lane.
    let (aggregate, _) = split(&resource, requested)?;
    let mut parents = vec![GreenContext::new(ordinal, aggregate)?];
    let mut streams = Vec::with_capacity(counts.len());

    for (index, count) in counts.iter().enumerate() {
        let available = parents[index].resource()?;
        let (lane, remainder) = if index + 1 == counts.len() {
            (available, None)
        } else {
            let (lane, remainder) = split(&available, *count)?;
            (lane, Some(remainder))
        };

        let green = Arc::new(GreenContext::new(ordinal, lane)?);
        let mut raw = std::ptr::null_mut();
        driver.check(
            // SAFETY: the green context is live and retained by the new stream;
            // the driver writes the stream handle to the local `raw`.
            unsafe {
                (driver.green_ctx_stream_create)(&mut raw, green.raw, CU_STREAM_NON_BLOCKING, 0)
            },
            "cuGreenCtxStreamCreate",
        )?;
        streams.push(Stream::from_green(device, raw, green)?);

        if let Some(remainder) = remainder {
            parents.push(GreenContext::new(ordinal, remainder)?);
        }
    }
    Ok(streams)
}
