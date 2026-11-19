//! The Protobuf form of a MiniMax-H3 denoiser's description ([`super::DenoiserInfo`]): what the
//! placed denoiser serves, as the family's numerical side encodes it for its request processing.
//! The engine carries the encoding (`EngineInfo.video_denoiser`) without reading it, so the
//! message is the family's own, and it changes only by additions, as the engine interface's
//! messages do.

/// What a placed video denoiser serves: the tasks it serves, its fixed schedule, the canvases
/// the deployment prepared, the references one request may carry and how a request's
/// conditions are packed into its rows.
#[derive(Clone, PartialEq, prost::Message)]
pub struct VideoDenoiser {
    /// The names of the tasks it serves, in the family's canonical order.
    #[prost(string, repeated, tag = "1")]
    pub tasks: Vec<String>,
    /// Sigma points of the fixed schedule, the clean endpoint included: one more than the
    /// network evaluations of every request.
    #[prost(uint32, tag = "2")]
    pub schedule_points: u32,
    /// Shift of the video schedule.
    #[prost(double, tag = "3")]
    pub video_shift: f64,
    /// Shift of the audio schedule.
    #[prost(double, tag = "4")]
    pub audio_shift: f64,
    /// The canvases the deployment serves, in the order the deployment names them; the first
    /// is the canvas of a request that states no size.
    #[prost(message, repeated, tag = "5")]
    pub canvases: Vec<ServedCanvas>,
    /// The references one request may carry.
    #[prost(message, optional, tag = "6")]
    pub references: Option<ReferenceLimits>,
    /// How the denoiser packs a request's conditions into its rows.
    #[prost(message, optional, tag = "7")]
    pub condition_packing: Option<ConditionPacking>,
}

/// A canvas a deployment serves, with the pair its model names it by.
#[derive(Clone, PartialEq, prost::Message)]
pub struct ServedCanvas {
    /// The resolution class in pixels: the canvas's short edge as the model names it.
    #[prost(uint32, tag = "1")]
    pub short_edge: u32,
    /// The display ratio as `W:H`.
    #[prost(string, tag = "2")]
    pub aspect_ratio: String,
    /// The raster the pair selects.
    #[prost(message, optional, tag = "3")]
    pub canvas: Option<Canvas>,
}

/// A raster in pixels.
#[derive(Clone, Copy, PartialEq, prost::Message)]
pub struct Canvas {
    /// Width in pixels.
    #[prost(uint32, tag = "1")]
    pub width: u32,
    /// Height in pixels.
    #[prost(uint32, tag = "2")]
    pub height: u32,
}

/// The references one request may carry, of each kind and in all, and the largest short edge
/// an image reference is encoded at; zeros for a denoiser that takes none.
#[derive(Clone, Copy, PartialEq, prost::Message)]
pub struct ReferenceLimits {
    /// Image references.
    #[prost(uint32, tag = "1")]
    pub images: u32,
    /// Video references.
    #[prost(uint32, tag = "2")]
    pub videos: u32,
    /// Audio references that are not a video's soundtrack.
    #[prost(uint32, tag = "3")]
    pub audios: u32,
    /// References of every kind.
    #[prost(uint32, tag = "4")]
    pub total: u32,
    /// Short edge, in pixels, an image reference is encoded at, at most.
    #[prost(uint32, tag = "5")]
    pub image_short_edge: u32,
}

/// How a denoiser lays a request's conditions into its rows: the rows they may take, and the
/// tile each condition's rows fill whole.
#[derive(Clone, PartialEq, prost::Message)]
pub struct ConditionPacking {
    /// Rows one request's conditions may take; zero for a denoiser that takes none.
    #[prost(uint32, tag = "1")]
    pub capacity: u32,
    /// Rows of one tile, at least one.
    #[prost(uint32, tag = "2")]
    pub tile_rows: u32,
    /// Tokens of one video tile along time, height and width, each at least one.
    #[prost(uint32, repeated, tag = "3")]
    pub video_tile: Vec<u32>,
}
