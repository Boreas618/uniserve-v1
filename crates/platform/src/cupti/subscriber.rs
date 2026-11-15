//! The process's one CUPTI subscription, and the dispatch of what CUPTI
//! delivers to every collector.
//!
//! [`subscribe`] subscribes the process to CUPTI's callback API with one
//! callback and registers one pair of activity buffer callbacks. Collectors
//! never subscribe themselves: they ask this module to enable the callbacks
//! and the activity kinds they need, and receive what CUPTI delivers through
//! the dispatch below. Enabling is reference-counted per callback and per
//! activity kind, so collectors that share one start and stop independently.
//!
//! Activity buffers are delivered as CUPTI completes them; CUPTI's own worker
//! thread delivers every buffer whose records completed every
//! [`FLUSH_PERIOD`] (`cuptiActivityFlushPeriod`), which issues no CUDA
//! synchronization, and a collector that needs the records of operations in
//! flight forces a flush ([`flush`]).

use std::alloc::Layout;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use super::{check, cupti};
use crate::sys::cupti as sys;
use crate::{Error, Result, lock};

/// How often CUPTI's worker thread delivers the buffers whose records
/// completed.
pub const FLUSH_PERIOD: Duration = Duration::from_millis(100);

/// `CUPTI_CB_DOMAIN_DRIVER_API`.
pub(crate) const DRIVER_DOMAIN: sys::CUpti_CallbackDomain = sys::CUPTI_CB_DOMAIN_DRIVER_API;
/// `CUPTI_CB_DOMAIN_RUNTIME_API`.
pub(crate) const RUNTIME_DOMAIN: sys::CUpti_CallbackDomain = sys::CUPTI_CB_DOMAIN_RUNTIME_API;

/// `CUPTI_API_ENTER`.
pub(crate) const API_ENTER: sys::CUpti_ApiCallbackSite = sys::CUPTI_API_ENTER;

/// `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`.
pub(crate) const GRAPH_TRACE: sys::CUpti_ActivityKind = sys::CUPTI_ACTIVITY_KIND_GRAPH_TRACE;

/// Each activity buffer CUPTI fills: 1 MiB (about 4,800 kernel records),
/// aligned as CUPTI requires. While a stream runs without pause, the buffer
/// it fills always holds the record of an operation in flight, and a flush
/// delivers it only once it is full; a small buffer bounds how long its
/// operations wait to be delivered.
const BUFFER: Layout = match Layout::from_size_align(1 << 20, 8) {
    Ok(layout) => layout,
    Err(_) => panic!("the activity buffer layout is invalid"),
};

/// The subscriber handle, once subscribed.
struct Subscription {
    handle: usize,
}

static SUBSCRIPTION: OnceLock<Result<Subscription>> = OnceLock::new();

/// What is enabled, by how many collectors.
#[derive(Default)]
struct Enabled {
    /// Activity kinds.
    kinds: HashMap<u32, u32>,
    /// Single callbacks, by domain and callback id.
    callbacks: HashMap<(sys::CUpti_CallbackDomain, u32), u32>,
    /// Whole callback domains.
    domains: HashMap<sys::CUpti_CallbackDomain, u32>,
}

static ENABLED: Mutex<Option<Enabled>> = Mutex::new(None);

/// Records CUPTI dropped for lack of buffer space since the last take.
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Subscribe the process to CUPTI, once: the callback API with the dispatch
/// below, the activity buffer callbacks and CUPTI's delivery period. Later
/// calls return the first one's outcome.
///
/// # Errors
///
/// Fails when `libcupti.so.13` is missing ([`Error::Load`]), when the loaded
/// CUPTI is older than [`super::MINIMUM_VERSION`] ([`Error::Version`]), or
/// when another CUPTI client (a profiler that injected CUPTI) holds the
/// process's subscription, with [`Error::Cupti`] code
/// `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`.
pub fn subscribe() -> Result<()> {
    SUBSCRIPTION
        .get_or_init(|| {
            let cupti = cupti()?;
            let mut handle = std::ptr::null_mut();
            check(
                // SAFETY: CUPTI writes the subscriber handle; the dispatch
                // never unwinds.
                unsafe { (cupti.subscribe)(&mut handle, dispatch, std::ptr::null_mut()) },
                "cuptiSubscribe",
            )?;
            *lock(&ENABLED) = Some(Enabled::default());
            let subscription = Subscription {
                handle: handle as usize,
            };
            check(
                // SAFETY: both callbacks match CUPTI's prototypes and never
                // unwind.
                unsafe { (cupti.register_buffers)(on_buffer_requested, on_buffer_completed) },
                "cuptiActivityRegisterCallbacks",
            )?;
            check(
                // SAFETY: takes the period in milliseconds by value.
                unsafe { (cupti.flush_period)(FLUSH_PERIOD.as_millis() as u32) },
                "cuptiActivityFlushPeriod",
            )?;
            Ok(subscription)
        })
        .as_ref()
        .map(drop)
        .map_err(Clone::clone)
}

/// Whether [`subscribe`] succeeded in this process.
pub fn subscribed() -> bool {
    matches!(SUBSCRIPTION.get(), Some(Ok(_)))
}

fn handle() -> Result<*mut c_void> {
    match SUBSCRIPTION.get() {
        Some(Ok(subscription)) => Ok(subscription.handle as *mut c_void),
        Some(Err(error)) => Err(error.clone()),
        None => Err(Error::InvalidArgument(
            "the process has not subscribed to CUPTI".into(),
        )),
    }
}

/// Enable the activity `kind` for one more collector.
pub(crate) fn enable_kind(kind: u32) -> Result<()> {
    let cupti = cupti()?;
    let mut enabled = lock(&ENABLED);
    let enabled = enabled
        .as_mut()
        .ok_or_else(|| Error::InvalidArgument("the process has not subscribed to CUPTI".into()))?;
    let count = enabled.kinds.entry(kind).or_default();
    if *count == 0 {
        check(
            // SAFETY: the kind is a `CUpti_ActivityKind`.
            unsafe { (cupti.activity_enable)(kind) },
            "cuptiActivityEnable",
        )?;
    }
    *count += 1;
    Ok(())
}

/// Release one collector's enabling of the activity `kind`; the last
/// disables it.
pub(crate) fn disable_kind(kind: u32) {
    let Ok(cupti) = cupti() else {
        return;
    };
    let mut enabled = lock(&ENABLED);
    let Some(count) = enabled
        .as_mut()
        .and_then(|enabled| enabled.kinds.get_mut(&kind))
    else {
        return;
    };
    *count = count.saturating_sub(1);
    if *count == 0 {
        // SAFETY: disabling a kind is harmless.
        let _ = unsafe { (cupti.activity_disable)(kind) };
    }
}

/// Enable the callback `cbid` of `domain` for one more collector.
pub(crate) fn enable_callback(domain: sys::CUpti_CallbackDomain, cbid: u32) -> Result<()> {
    let (cupti, subscriber) = (cupti()?, handle()?);
    let mut enabled = lock(&ENABLED);
    let enabled = enabled
        .as_mut()
        .ok_or_else(|| Error::InvalidArgument("the process has not subscribed to CUPTI".into()))?;
    let count = enabled.callbacks.entry((domain, cbid)).or_default();
    if *count == 0 {
        check(
            // SAFETY: the subscriber is live; the domain and id are CUPTI's.
            unsafe { (cupti.enable_callback)(1, subscriber, domain, cbid) },
            "cuptiEnableCallback",
        )?;
    }
    *count += 1;
    Ok(())
}

/// Release one collector's enabling of the callback `cbid` of `domain`.
pub(crate) fn disable_callback(domain: sys::CUpti_CallbackDomain, cbid: u32) {
    let (Ok(cupti), Ok(subscriber)) = (cupti(), handle()) else {
        return;
    };
    let mut enabled = lock(&ENABLED);
    let Some(enabled) = enabled.as_mut() else {
        return;
    };
    let Some(count) = enabled.callbacks.get_mut(&(domain, cbid)) else {
        return;
    };
    *count = count.saturating_sub(1);
    let whole = enabled.domains.get(&domain).is_some_and(|&count| count > 0);
    if *count == 0 && !whole {
        // SAFETY: as in `enable_callback`.
        let _ = unsafe { (cupti.enable_callback)(0, subscriber, domain, cbid) };
    }
}

/// Enable every callback of `domain` for one more collector.
pub(crate) fn enable_domain(domain: sys::CUpti_CallbackDomain) -> Result<()> {
    let (cupti, subscriber) = (cupti()?, handle()?);
    let mut enabled = lock(&ENABLED);
    let enabled = enabled
        .as_mut()
        .ok_or_else(|| Error::InvalidArgument("the process has not subscribed to CUPTI".into()))?;
    let count = enabled.domains.entry(domain).or_default();
    if *count == 0 {
        check(
            // SAFETY: the subscriber is live; the domain is CUPTI's.
            unsafe { (cupti.enable_domain)(1, subscriber, domain) },
            "cuptiEnableDomain",
        )?;
    }
    *count += 1;
    Ok(())
}

/// Release one collector's enabling of `domain`. The last disables the
/// domain and enables again the single callbacks other collectors hold in
/// it.
pub(crate) fn disable_domain(domain: sys::CUpti_CallbackDomain) {
    let (Ok(cupti), Ok(subscriber)) = (cupti(), handle()) else {
        return;
    };
    let mut enabled = lock(&ENABLED);
    let Some(enabled) = enabled.as_mut() else {
        return;
    };
    let Some(count) = enabled.domains.get_mut(&domain) else {
        return;
    };
    *count = count.saturating_sub(1);
    if *count > 0 {
        return;
    }
    // SAFETY: the subscriber is live; the domain and ids are CUPTI's.
    unsafe {
        let _ = (cupti.enable_domain)(0, subscriber, domain);
        for (&(held, cbid), &count) in &enabled.callbacks {
            if held == domain && count > 0 {
                let _ = (cupti.enable_callback)(1, subscriber, domain, cbid);
            }
        }
    }
}

/// Have CUPTI deliver every buffer whose records completed; `forced` also
/// delivers the records of operations in flight, whose end is unknown.
pub(crate) fn flush(forced: bool) {
    if let Ok(cupti) = cupti() {
        let flags = if forced {
            sys::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED
        } else {
            0
        };
        // SAFETY: the buffer callbacks stay registered for the process's
        // life.
        let _ = unsafe { (cupti.flush_all)(flags) };
    }
}

/// Activity records CUPTI dropped for lack of buffer space since the last
/// take.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::AcqRel)
}

/// The CUPTI callback: hands each callback to the collectors that use it.
/// Every collector's part only touches atomics, thread locals and its own
/// locks, and the range profiler's launch ranges also push and pop a CUPTI
/// range.
unsafe extern "C" fn dispatch(
    _userdata: *mut c_void,
    domain: sys::CUpti_CallbackDomain,
    cbid: u32,
    data: *const c_void,
) {
    if data.is_null() {
        return;
    }
    // Both API domains supply `CUpti_CallbackData`; no other domain is enabled.
    let data = data.cast::<sys::CUpti_CallbackData>();
    if domain == DRIVER_DOMAIN {
        // SAFETY: CUPTI supplies CUpti_CallbackData for this API domain.
        unsafe {
            super::blocking::on_driver_call(cbid, data);
            super::range_profiler::on_launch(cbid, data);
        }
    }
    if domain == DRIVER_DOMAIN || domain == RUNTIME_DOMAIN {
        // SAFETY: both API domains supply CUpti_CallbackData.
        unsafe { super::timeline::on_api(domain, cbid, data) };
    }
}

/// CUPTI asks for an empty buffer; a null one makes it drop records.
unsafe extern "C" fn on_buffer_requested(
    buffer: *mut *mut u8,
    size: *mut usize,
    records: *mut usize,
) {
    // SAFETY: the layout has a nonzero size.
    let allocated = unsafe { std::alloc::alloc(BUFFER) };
    // SAFETY: CUPTI passes valid out-pointers; zero records means as many
    // as fit.
    unsafe {
        *buffer = allocated;
        *size = if allocated.is_null() {
            0
        } else {
            BUFFER.size()
        };
        *records = 0;
    }
}

/// CUPTI delivers a buffer from [`on_buffer_requested`]: every record goes
/// to the collectors that take its kind, and the buffer is freed.
unsafe extern "C" fn on_buffer_completed(
    context: *mut c_void,
    stream: u32,
    buffer: *mut u8,
    _size: usize,
    valid: usize,
) {
    if buffer.is_null() {
        return;
    }
    if let Ok(cupti) = cupti() {
        let mut accounting = super::activity::Delivery::begin();
        let mut timeline = super::timeline::Delivery::begin();
        let mut record = std::ptr::null_mut();
        // SAFETY: CUPTI walks the valid bytes of its own buffer and points
        // `record` at each record, until it reports none left.
        while unsafe { (cupti.next_record)(buffer, valid, &mut record) } == 0 {
            // SAFETY: the record lies in the buffer and starts with its
            // kind; its kind gives its layout.
            unsafe {
                accounting.record(record);
                timeline.record(record);
            }
        }
        accounting.end();
        timeline.end();
        let mut dropped = 0;
        // SAFETY: the context and stream are those CUPTI passed; it writes
        // one count.
        if unsafe { (cupti.dropped_records)(context, stream, &mut dropped) } == 0 {
            DROPPED.fetch_add(dropped as u64, Ordering::AcqRel);
        }
    }
    // SAFETY: the buffer came from `on_buffer_requested` with this layout.
    unsafe { std::alloc::dealloc(buffer, BUFFER) };
}
