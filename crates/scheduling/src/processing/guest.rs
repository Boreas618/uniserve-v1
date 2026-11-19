//! The module side of the processing exports and host functions, which
//! [`export_processing!`](crate::export_processing) expands into calls of.
//!
//! An export decodes its message, calls the bound [`Processing`] value and encodes its reply, as
//! the program exports do ([`crate::program::guest`]), in the processing interface's call
//! convention ([`crate::processing`]): the export frees its message once it decoded it, and
//! writes its reply into the module's one reply buffer, which lives until the module's next
//! call. A message that does not decode, or a call before `bind`, is a trap: host and program are
//! built from the same types, so neither is a request's fault.

use std::cell::RefCell;

use serde::Serialize;
use serde::de::DeserializeOwned;

use super::video::{InputError, Process, VideoInput, VideoSpec};
use super::{Bind, Bound, Event, Open, Processing, Prompt, Refusal, Render};
use crate::abi::{message, release};
use crate::wire::{decode, pack};

/// The bytes of the module's message buffer ([`inbox`]): what one decoded update's text and the
/// most of the parser's other messages take, beyond which a message takes a buffer of `alloc`.
pub const INBOX_BYTES: usize = 64 << 10;

thread_local! {
    /// The module's reply buffer: the last export's reply, which the host reads in place before
    /// its next call. wasm32 without threads has one thread, so this is the module's one buffer,
    /// reused by every call.
    static REPLY: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };

    /// The module's message buffer, which the host writes a message of at most [`INBOX_BYTES`]
    /// into in place of a buffer of `alloc`.
    static INBOX: Box<[u8]> = vec![0; INBOX_BYTES].into_boxed_slice();
}

/// The `inbox` export: the place and length of the module's message buffer, packed.
pub fn inbox() -> u64 {
    INBOX.with(|inbox| {
        let (Ok(ptr), Ok(len)) = (
            u32::try_from(inbox.as_ptr() as usize),
            u32::try_from(inbox.len()),
        ) else {
            panic!("a wasm32 buffer's place fits in a u32");
        };
        pack(ptr, len)
    })
}

/// Whether `ptr` is the module's message buffer, which no export frees.
fn in_inbox(ptr: u32) -> bool {
    INBOX.with(|inbox| inbox.as_ptr() as usize == ptr as usize)
}

/// The bound program in `slot`, a trap before `bind` filled it.
///
/// # Safety
///
/// `slot` is the module's one slot, and no other reference to it is live.
unsafe fn bound<'a, P>(slot: *mut Option<P>) -> &'a mut P {
    // SAFETY: the caller's obligation.
    let slot = unsafe { &mut *slot };
    slot.as_mut()
        .unwrap_or_else(|| panic!("the host binds a processing program before any other call"))
}

/// The message of this call, decoded as a `T`, its buffer freed unless it is the module's
/// message buffer; a trap when it is not one.
///
/// # Safety
///
/// `ptr` and `len` name the host's message for this call: a buffer of the module's `alloc`,
/// which nothing reads after this call, or the module's message buffer.
unsafe fn read<T: DeserializeOwned>(ptr: u32, len: u32, what: &str) -> T {
    // SAFETY: the caller's obligation.
    let bytes = unsafe { message(ptr, len) };
    let value = decode(bytes).unwrap_or_else(|error| panic!("the {what} does not decode: {error}"));
    if !in_inbox(ptr) {
        // SAFETY: the buffer came from `alloc` with this length and is no longer read.
        unsafe { release(ptr, len) };
    }
    value
}

/// Encodes `value` into the module's reply buffer, a trap when it does not encode, and packs
/// the buffer's place.
fn reply<T: Serialize>(value: &T, what: &str) -> u64 {
    REPLY.with_borrow_mut(|buffer| {
        buffer.clear();
        serde_json::to_writer(&mut *buffer, value)
            .unwrap_or_else(|error| panic!("the {what} does not encode: {error}"));
        let Ok(len) = u32::try_from(buffer.len()) else {
            panic!("a reply fits in a u32 length");
        };
        let Ok(ptr) = u32::try_from(buffer.as_ptr() as usize) else {
            panic!("a wasm32 pointer fits in a u32");
        };
        pack(ptr, len)
    })
}

/// The `families` export.
pub fn families<P: Processing>() -> u64 {
    crate::program::guest::install();
    reply(&P::families(), "families")
}

/// The `bind` export: binds the instance's family, replacing what it held.
///
/// # Safety
///
/// `ptr` and `len` name the host's message for this call; `slot` is the module's one slot.
pub unsafe fn bind<P: Processing>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
    crate::program::guest::install();
    // SAFETY: the caller's obligation.
    let bind: Bind = unsafe { read(ptr, len, "binding") };
    let bound: Result<Bound, String> = P::bind(bind).map(|(program, bound)| {
        // SAFETY: the caller's obligation.
        unsafe { *slot = Some(program) };
        bound
    });
    reply(&bound, "binding's facts")
}

/// The `render` export.
///
/// # Safety
///
/// As [`bind`].
pub unsafe fn render<P: Processing>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
    // SAFETY: the caller's obligation.
    let render: Render = unsafe { read(ptr, len, "render request") };
    // SAFETY: the caller's obligation.
    let prompt: Result<Prompt, Refusal> = unsafe { bound(slot) }.render(render);
    reply(&prompt, "prompt")
}

/// The `parse_open` export.
///
/// # Safety
///
/// As [`bind`].
pub unsafe fn open<P: Processing>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
    // SAFETY: the caller's obligation.
    let open: Open = unsafe { read(ptr, len, "parser's opening") };
    // SAFETY: the caller's obligation.
    let opened = unsafe { bound(slot) }.open(open);
    reply(&opened, "parser's opening")
}

/// The `parse_push` export.
///
/// # Safety
///
/// As [`bind`].
pub unsafe fn push<P: Processing>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
    // SAFETY: the caller's obligation.
    let text: String = unsafe { read(ptr, len, "decoded text") };
    // SAFETY: the caller's obligation.
    let events: Result<Vec<Event>, String> = unsafe { bound(slot) }.push(&text);
    reply(&events, "parsed events")
}

/// The `parse_finish` export.
///
/// # Safety
///
/// `slot` is the module's one slot.
pub unsafe fn finish<P: Processing>(slot: *mut Option<P>) -> u64 {
    // SAFETY: the caller's obligation.
    let events: Result<Vec<Event>, String> = unsafe { bound(slot) }.finish();
    reply(&events, "parsed events")
}

/// The `parse_save` export.
///
/// # Safety
///
/// `slot` is the module's one slot.
pub unsafe fn save<P: Processing>(slot: *mut Option<P>) -> u64 {
    // SAFETY: the caller's obligation.
    let state = unsafe { bound(slot) }.save();
    reply(&state, "parser's state")
}

/// The `parse_restore` export.
///
/// # Safety
///
/// As [`bind`].
pub unsafe fn restore<P: Processing>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
    // SAFETY: the caller's obligation.
    let state: Vec<u8> = unsafe { read(ptr, len, "parser's state") };
    // SAFETY: the caller's obligation.
    let restored = unsafe { bound(slot) }.restore(&state);
    reply(&restored, "parser's restoration")
}

/// The `video_check` export.
///
/// # Safety
///
/// As [`bind`].
pub unsafe fn check<P: Processing>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
    // SAFETY: the caller's obligation.
    let spec: VideoSpec = unsafe { read(ptr, len, "video request") };
    // SAFETY: the caller's obligation.
    let checked: Result<(), InputError> = unsafe { bound(slot) }.check_video(&spec);
    reply(&checked, "video request's check")
}

/// The `video_process` export.
///
/// # Safety
///
/// As [`bind`].
pub unsafe fn process<P: Processing>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
    // SAFETY: the caller's obligation.
    let process: Process = unsafe { read(ptr, len, "video request") };
    // SAFETY: the caller's obligation.
    let input: Result<VideoInput, InputError> = unsafe { bound(slot) }.process_video(process);
    reply(&input, "video input")
}

/// `uniserve.token`: the id of the bound checkpoint's token whose text is exactly `text`.
pub fn token(text: &str) -> Option<u32> {
    let id = imports::token(text);
    u32::try_from(id).ok()
}

/// `uniserve.encode`: `text` encoded by the bound checkpoint's tokenizer without special
/// tokens, or why the tokenizer could not encode it.
///
/// # Errors
///
/// The tokenizer's own failure, as the host states it.
pub fn encode_text(text: &str) -> Result<Vec<u32>, String> {
    crate::program::guest::exchange(imports::encode, &text, "text to encode")
}

/// The host functions. They exist only in a wasm32 module, where the host links them; a native
/// build of a program has no host to call.
#[cfg(target_arch = "wasm32")]
mod imports {
    #[link(wasm_import_module = "uniserve")]
    unsafe extern "C" {
        #[link_name = "token"]
        fn host_token(ptr: u32, len: u32) -> i64;
        #[link_name = "encode"]
        fn host_encode(ptr: u32, len: u32) -> u64;
    }

    pub(super) fn token(text: &str) -> i64 {
        // SAFETY: the host reads `len` bytes at `ptr` during the call; `text` outlives it.
        unsafe { host_token(text.as_ptr() as u32, text.len() as u32) }
    }

    pub(super) fn encode(ptr: u32, len: u32) -> u64 {
        // SAFETY: the host reads the message during the call; the caller keeps it alive.
        unsafe { host_encode(ptr, len) }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imports {
    const HOSTLESS: &str = "a processing program calls its host only as a wasm32 module in the \
                            program host; a native build has none";

    pub(super) fn token(_: &str) -> i64 {
        panic!("{HOSTLESS}")
    }

    pub(super) fn encode(_: u32, _: u32) -> u64 {
        panic!("{HOSTLESS}")
    }
}
