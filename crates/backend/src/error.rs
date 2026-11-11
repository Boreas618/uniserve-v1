//! What the backend refuses.

/// A failure of the device backend.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The context's region cannot hold an allocation.
    #[error(
        "an allocation of {needed} bytes does not fit the region: {used} of {capacity} bytes are in use"
    )]
    OutOfMemory {
        /// The rounded size asked for.
        needed: u64,
        /// The furthest byte in use when the allocation was refused.
        used: u64,
        /// The region's size.
        capacity: u64,
    },
}

/// The backend's result.
pub type Result<T> = std::result::Result<T, Error>;
