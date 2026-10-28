//! CUPTI's activity records and API calls, decoded into plain values.
//!
//! Each record kind keeps the fields of its CUPTI struct a client needs to
//! place it on a timeline and join it with the others: its interval or
//! timestamp in CUPTI's clock ([`super::timestamp`]), its device, context
//! and stream, and the correlation id that joins a device operation with
//! the API call that launched it (a graph's kernels carry the correlation id
//! of the graph's launch). Names are interned ([`Names`]): a record carries
//! the id of its name, and the delivery that first carries a name carries
//! its text.
//!
//! The field offsets are those of the CUPTI 13 structs the loaded library
//! writes (`cupti_activity.h` of `nvidia-cuda-cupti` 13.0): every record
//! starts with its `CUpti_ActivityKind`, which names its struct.

use std::collections::HashMap;
use std::ffi::{CStr, c_char};

/// `CUPTI_ACTIVITY_KIND_MEMCPY`.
pub const MEMCPY: u32 = 1;
/// `CUPTI_ACTIVITY_KIND_MEMSET`.
pub const MEMSET: u32 = 2;
/// `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`.
pub const CONCURRENT_KERNEL: u32 = 10;
/// `CUPTI_ACTIVITY_KIND_MARKER`.
pub const MARKER: u32 = 12;
/// `CUPTI_ACTIVITY_KIND_MARKER_DATA`.
pub const MARKER_DATA: u32 = 13;
/// `CUPTI_ACTIVITY_KIND_MEMCPY2`: a peer copy.
pub const MEMCPY2: u32 = 22;
/// `CUPTI_ACTIVITY_KIND_NVLINK`.
pub const NVLINK: u32 = 40;
/// `CUPTI_ACTIVITY_KIND_PCIE`.
pub const PCIE: u32 = 46;
/// `CUPTI_ACTIVITY_KIND_MEMORY2`.
pub const STORAGE2: u32 = 49;
/// `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`.
pub const GRAPH_TRACE: u32 = super::subscriber::GRAPH_TRACE;

/// A kernel's execution (`CUpti_ActivityKernel10`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kernel {
    pub start: u64,
    pub end: u64,
    /// When the kernel was queued and submitted; 0 when CUPTI did not
    /// record it.
    pub queued: u64,
    pub submitted: u64,
    pub device: u32,
    pub context: u32,
    pub stream: u32,
    pub correlation: u32,
    pub grid: [i32; 3],
    pub block: [i32; 3],
    pub registers: u16,
    pub static_shared: i32,
    pub dynamic_shared: i32,
    pub name: u32,
    /// The graph and graph node the kernel ran in; 0 outside a graph.
    pub graph: u32,
    pub graph_node: u64,
    pub launch_type: u8,
    pub channel: u32,
}

/// A copy (`CUpti_ActivityMemcpy6`, or `CUpti_ActivityMemcpyPtoP4` for a
/// peer copy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Copy {
    /// `CUpti_ActivityMemcpyKind`.
    pub copy_kind: u8,
    /// `CUpti_ActivityMemoryKind` of the source and destination; 0 for a
    /// peer copy, which names their devices instead.
    pub source_kind: u8,
    pub destination_kind: u8,
    pub bytes: u64,
    pub start: u64,
    pub end: u64,
    pub device: u32,
    pub context: u32,
    pub stream: u32,
    pub correlation: u32,
    pub graph: u32,
    pub graph_node: u64,
    /// The source and destination devices of a peer copy.
    pub peer: Option<(u32, u32)>,
}

/// A fill (`CUpti_ActivityMemset4`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fill {
    pub value: u32,
    pub bytes: u64,
    pub start: u64,
    pub end: u64,
    pub device: u32,
    pub context: u32,
    pub stream: u32,
    pub correlation: u32,
    pub memory_kind: u16,
    pub graph: u32,
    pub graph_node: u64,
}

/// A storage allocation or release (`CUpti_ActivityMemory4`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Memory {
    /// `CUpti_ActivityMemoryOperationType`: 1 allocation, 2 release.
    pub operation: u32,
    /// `CUpti_ActivityMemoryKind`.
    pub memory_kind: u32,
    pub correlation: u32,
    pub address: u64,
    pub bytes: u64,
    pub timestamp: u64,
    pub process: u32,
    pub device: u32,
    pub context: u32,
    pub stream: u32,
    pub asynchronous: bool,
}

/// An NVTX mark, or the start or end of an NVTX range
/// (`CUpti_ActivityMarker2`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Marker {
    /// `CUpti_ActivityFlag`: instantaneous, start or end.
    pub flags: u32,
    pub timestamp: u64,
    /// The range's id, shared by its start and end.
    pub id: u32,
    /// `CUpti_ActivityObjectKind` and the object's ids: the process and
    /// thread, or the device, context and stream.
    pub object_kind: u32,
    pub object: [u32; 3],
    pub name: u32,
    pub domain: u32,
}

/// The payload, color and category of an NVTX mark
/// (`CUpti_ActivityMarkerData2`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkerData {
    pub flags: u32,
    pub id: u32,
    /// `CUpti_MetricValueKind` of the payload, whose 8 bytes are kept as
    /// they are.
    pub payload_kind: u32,
    pub payload: u64,
    pub color: u32,
    pub category: u32,
}

/// A CUDA graph's launch recorded whole (`CUpti_ActivityGraphTrace2`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    pub correlation: u32,
    pub start: u64,
    pub end: u64,
    pub device: u32,
    pub graph: u32,
    pub context: u32,
    pub stream: u32,
}

/// An NVLink connection (`CUpti_ActivityNvLink4`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvLink {
    pub version: u32,
    /// `CUpti_DevType` of each end and its id: a GPU's UUID, or an NPU's
    /// index and domain in the first 8 bytes.
    pub types: [u32; 2],
    pub ends: [[u8; 16]; 2],
    pub flag: u32,
    pub links: u32,
    pub bandwidth: u64,
    pub nvswitch: bool,
}

/// A PCIe device or bridge (`CUpti_ActivityPcie`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcie {
    /// `CUpti_PcieDeviceType`: 0 GPU, 1 bridge.
    pub device_type: u32,
    pub id: u32,
    pub domain: u32,
    pub generation: u16,
    pub link_rate: u16,
    pub link_width: u16,
    pub upstream_bus: u16,
}

/// A runtime or driver API call, from CUPTI's callbacks at its entry and
/// exit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Api {
    /// `CUPTI_CB_DOMAIN_DRIVER_API` (1) or `CUPTI_CB_DOMAIN_RUNTIME_API` (2).
    pub domain: i32,
    pub cbid: u32,
    pub name: u32,
    pub start: u64,
    pub end: u64,
    /// The calling thread's Linux thread id.
    pub thread: u32,
    pub correlation: u32,
    /// What the call returned (`CUresult` or `cudaError_t`).
    pub result: i32,
}

/// One decoded record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Record {
    Kernel(Kernel),
    Copy(Copy),
    Fill(Fill),
    Memory(Memory),
    Marker(Marker),
    MarkerData(MarkerData),
    Graph(Graph),
    NvLink(NvLink),
    Pcie(Pcie),
    Api(Api),
}

/// Interned names: an id per distinct text, from 1 (0 is no name).
///
/// CUPTI shares one string among the records of one kernel, so the table
/// first looks a pointer up and confirms the text it stored for it.
#[derive(Default)]
pub struct Names {
    by_text: HashMap<String, u32>,
    by_pointer: HashMap<usize, u32>,
    texts: Vec<String>,
    /// Names interned since the last [`Names::take_new`].
    new: Vec<(u32, String)>,
}

impl Names {
    /// The id of `text`.
    pub fn intern(&mut self, text: &str) -> u32 {
        if let Some(&id) = self.by_text.get(text) {
            return id;
        }
        self.texts.push(text.to_owned());
        let id = self.texts.len() as u32;
        self.by_text.insert(text.to_owned(), id);
        self.new.push((id, text.to_owned()));
        id
    }

    /// The id of the C string at `pointer`; 0 for a null pointer.
    ///
    /// # Safety
    /// `pointer` is null or a NUL-terminated string valid during the call.
    pub unsafe fn intern_c(&mut self, pointer: *const c_char) -> u32 {
        if pointer.is_null() {
            return 0;
        }
        // SAFETY: forwarded from the caller.
        let text = unsafe { CStr::from_ptr(pointer) }.to_bytes();
        if let Some(&id) = self.by_pointer.get(&(pointer as usize))
            && self.texts[id as usize - 1].as_bytes() == text
        {
            return id;
        }
        let id = self.intern(&String::from_utf8_lossy(text));
        self.by_pointer.insert(pointer as usize, id);
        id
    }

    /// The names interned since the last take, with their ids.
    pub fn take_new(&mut self) -> Vec<(u32, String)> {
        std::mem::take(&mut self.new)
    }
}

/// # Safety
/// `offset` bytes past `record` lie `size_of::<T>()` readable bytes.
unsafe fn read<T: std::marker::Copy>(record: *const u8, offset: usize) -> T {
    // SAFETY: forwarded from the caller.
    unsafe { record.add(offset).cast::<T>().read_unaligned() }
}

/// The kind of the record at `record`.
///
/// # Safety
/// `record` points at an activity record.
pub(super) unsafe fn kind(record: *const u8) -> u32 {
    // SAFETY: every record starts with its kind.
    unsafe { read(record, 0) }
}

/// Decode the record at `record`, interning its names in `names`; `None`
/// for a kind this module does not decode.
///
/// # Safety
/// `record` points at a whole activity record of the loaded CUPTI.
pub(super) unsafe fn decode(record: *const u8, names: &mut Names) -> Option<Record> {
    // SAFETY: every offset below lies inside the struct the kind names.
    unsafe {
        Some(match kind(record) {
            CONCURRENT_KERNEL => Record::Kernel(Kernel {
                start: read(record, 16),
                end: read(record, 24),
                queued: read(record, 120),
                submitted: read(record, 128),
                device: read(record, 40),
                context: read(record, 44),
                stream: read(record, 48),
                correlation: read(record, 92),
                grid: [read(record, 52), read(record, 56), read(record, 60)],
                block: [read(record, 64), read(record, 68), read(record, 72)],
                registers: read(record, 6),
                static_shared: read(record, 76),
                dynamic_shared: read(record, 80),
                name: names.intern_c(read(record, 104)),
                graph: read(record, 156),
                graph_node: read(record, 144),
                launch_type: read(record, 136),
                channel: read(record, 168),
            }),
            MEMCPY => Record::Copy(Copy {
                copy_kind: read(record, 4),
                source_kind: read(record, 5),
                destination_kind: read(record, 6),
                bytes: read(record, 8),
                start: read(record, 16),
                end: read(record, 24),
                device: read(record, 32),
                context: read(record, 36),
                stream: read(record, 40),
                correlation: read(record, 44),
                graph: read(record, 72),
                graph_node: read(record, 64),
                peer: None,
            }),
            MEMCPY2 => Record::Copy(Copy {
                copy_kind: read(record, 4),
                source_kind: 0,
                destination_kind: 0,
                bytes: read(record, 8),
                start: read(record, 16),
                end: read(record, 24),
                device: read(record, 32),
                context: read(record, 36),
                stream: read(record, 40),
                correlation: read(record, 60),
                graph: read(record, 80),
                graph_node: read(record, 72),
                peer: Some((read(record, 44), read(record, 52))),
            }),
            MEMSET => Record::Fill(Fill {
                value: read(record, 4),
                bytes: read(record, 8),
                start: read(record, 16),
                end: read(record, 24),
                device: read(record, 32),
                context: read(record, 36),
                stream: read(record, 40),
                correlation: read(record, 44),
                memory_kind: read(record, 50),
                graph: read(record, 72),
                graph_node: read(record, 64),
            }),
            STORAGE2 => Record::Memory(Memory {
                operation: read(record, 4),
                memory_kind: read(record, 8),
                correlation: read(record, 12),
                address: read(record, 16),
                bytes: read(record, 24),
                timestamp: read(record, 32),
                process: read(record, 48),
                device: read(record, 52),
                context: read(record, 56),
                stream: read(record, 60),
                asynchronous: read::<u32>(record, 72) != 0,
            }),
            MARKER => Record::Marker(Marker {
                flags: read(record, 4),
                timestamp: read(record, 8),
                id: read(record, 16),
                object_kind: read(record, 20),
                object: [read(record, 24), read(record, 28), read(record, 32)],
                name: names.intern_c(read(record, 40)),
                domain: names.intern_c(read(record, 48)),
            }),
            MARKER_DATA => Record::MarkerData(MarkerData {
                flags: read(record, 4),
                id: read(record, 8),
                payload_kind: read(record, 12),
                payload: read(record, 16),
                color: read(record, 24),
                category: read(record, 28),
            }),
            GRAPH_TRACE => Record::Graph(Graph {
                correlation: read(record, 4),
                start: read(record, 8),
                end: read(record, 16),
                device: read(record, 24),
                graph: read(record, 28),
                context: read(record, 32),
                stream: read(record, 36),
            }),
            NVLINK => Record::NvLink(NvLink {
                version: read(record, 4),
                types: [read(record, 8), read(record, 12)],
                ends: [read(record, 16), read(record, 32)],
                flag: read(record, 48),
                links: read(record, 52),
                bandwidth: read(record, 120),
                nvswitch: read::<u8>(record, 128) != 0,
            }),
            PCIE => Record::Pcie(Pcie {
                device_type: read(record, 4),
                id: read(record, 8),
                domain: read(record, 12),
                generation: read(record, 16),
                link_rate: read(record, 18),
                link_width: read(record, 20),
                upstream_bus: read(record, 22),
            }),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_keeps_its_id_and_is_new_once() {
        let mut names = Names::default();
        let first = names.intern("gemm");
        assert_eq!(names.intern("gemm"), first);
        let second = names.intern("softmax");
        assert_ne!(first, second);
        assert_eq!(
            names.take_new(),
            vec![(first, "gemm".to_owned()), (second, "softmax".to_owned())]
        );
        assert!(names.take_new().is_empty());
    }

    #[test]
    fn a_pointer_whose_text_changed_is_interned_again() {
        let mut names = Names::default();
        let mut text = *b"gemm\0";
        // SAFETY: the buffer is NUL-terminated and outlives the calls.
        let first = unsafe { names.intern_c(text.as_ptr().cast()) };
        text[0] = b'h';
        // SAFETY: as above.
        let second = unsafe { names.intern_c(text.as_ptr().cast()) };
        assert_ne!(first, second);
        // SAFETY: a null pointer has no name.
        assert_eq!(unsafe { names.intern_c(std::ptr::null()) }, 0);
    }
}
