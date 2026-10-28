//! Request media acquisition: one connector for every medium a request names
//! by URL.
//!
//! [`MediaConnector`] resolves a reference into its bytes before model
//! preprocessing runs, so network and file I/O stay on the async runtime and
//! never enter the blocking preprocessing pool. Two connectors serve the
//! server, each under its own option table: input images of chat, image
//! generation and System One readouts (`image_fetch_config`), and video
//! condition media (`media_config`). [`MediaConnector::fetch_images`]
//! additionally validates each image into an [`ImageInput`] (the encoded
//! file bytes, their header dimensions and a content digest).
//!
//! A reference is one of:
//!
//! - `data:[<type>][;<parameter>]*;base64,<payload>`, decoded locally from
//!   standard padded base64.
//! - `file://<path>`, read only below the connector's media directory and
//!   refused when it has none. A path outside the directory is refused before
//!   the filesystem is consulted, so requests cannot learn which files exist
//!   elsewhere. The path is then canonicalized, so symbolic links that lead
//!   outside the directory are refused, and the canonical path is opened
//!   without following a final symbolic link, so a link swapped in after the
//!   check is refused as well.
//! - `http(s)://`, fetched unless the connector disables remote media. These
//!   rules apply to the initial URL and to every redirect hop:
//!   - Only `http` and `https` are followed, for at most
//!     [`MediaConnector::MAX_REDIRECTS`] redirects, and a redirect may not
//!     downgrade `https` to `http`.
//!   - Unless private destinations are allowed, every destination address
//!     must be public ([`is_public_address`]). A literal address in the URL
//!     is checked before the request. A host name is checked inside the
//!     client's DNS resolver, and the client connects only to the addresses
//!     that resolver returned, so a host cannot pass the check with one DNS
//!     answer and connect with another. A host with any non-public address is
//!     refused outright.
//!   - The whole fetch, redirects and body included, completes within the
//!     connector's time limit.
//!   - Environment proxy settings are ignored: a proxy resolves the host
//!     itself, which would bypass the address check.
//!
//! Each medium may not exceed the connector's per-medium limit, and the
//! media of one request may not exceed its request total when it has one. A
//! declared `Content-Length` over the limit fails before the body is read,
//! and the streamed byte count enforces both limits whatever the headers
//! declare.

use std::io::{Cursor, Read as _};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use base64::Engine as _;
use bytes::{Bytes, BytesMut};
use reqwest::header::{ACCEPT, CONTENT_TYPE, LOCATION};
use reqwest::{StatusCode, Url};
use thiserror::Error;
use tokio::sync::Semaphore;

use crate::options::{ImageFetchOptions, MediaOptions};

/// Failure to resolve one media reference.
///
/// Messages describe the failure in terms of the reference the client sent.
/// They never name a redirect target or a resolved address, which may belong
/// to a network the client cannot otherwise observe.
#[derive(Debug, Error)]
pub enum MediaError {
    /// The reference names a scheme other than `data`, `file`, `http` or `https`.
    #[error("the URL must be a data:, file://, http:// or https:// URL")]
    UnsupportedScheme,
    /// The reference cannot be parsed as an absolute URL.
    #[error("the URL is not a valid absolute URL: {0}")]
    InvalidUrl(url::ParseError),
    /// A `data:` reference is not of the form `data:[<type>][;<parameter>]*;base64,<payload>`.
    #[error("a data URL must have the form data:[<type>][;<parameter>];base64,<payload>")]
    InvalidDataUrl,
    /// A data URL payload is not valid standard padded base64.
    #[error("the data URL payload is not standard base64: {0}")]
    InvalidBase64(base64::DecodeError),
    /// The connector does not fetch `http(s)` media.
    #[error("http(s) media is disabled on this server")]
    RemoteDisabled,
    /// The connector has no media directory to read `file://` media under.
    #[error("file:// media is disabled: the server has no media directory")]
    LocalDisabled,
    /// A `file://` reference names a host or a relative path.
    #[error("a file:// URL must name a local absolute path")]
    InvalidFileUrl,
    /// A `file://` path lies outside the media directory, or leaves it
    /// through a symbolic link.
    #[error("{path} lies outside the media directory")]
    OutsideMediaDirectory {
        /// The path the reference names.
        path: String,
    },
    /// A `file://` path below the media directory cannot be read as a
    /// non-empty regular file.
    #[error("{path}: {reason}")]
    LocalFile {
        /// The path the reference names.
        path: String,
        /// Why the file cannot be read.
        reason: String,
    },
    /// The URL's host is, or resolves to, an address outside public unicast space.
    #[error("the URL host resolves to a non-public address")]
    NonPublicAddress,
    /// A redirect leads to an address outside public unicast space.
    #[error("the URL redirects to a non-public address")]
    NonPublicRedirect,
    /// A redirect leads to a scheme other than http or https.
    #[error("the URL redirects to a scheme other than http or https")]
    UnsupportedRedirect,
    /// A redirect leads from `https` to `http`.
    #[error("the URL redirects from https to http")]
    InsecureRedirect,
    /// A redirect response carries no usable `Location` header.
    #[error("the URL returned a redirect without a valid Location header")]
    InvalidRedirect,
    /// The URL redirects more times than [`MediaConnector::MAX_REDIRECTS`].
    #[error("the URL redirects more than {limit} times")]
    TooManyRedirects {
        /// Redirects allowed per fetch.
        limit: usize,
    },
    /// The final response status is not a success.
    #[error("the URL returned HTTP status {status}")]
    Status {
        /// Response status code.
        status: u16,
    },
    /// The medium exceeds the connector's per-medium limit.
    #[error("the media exceeds the {limit}-byte size limit")]
    TooLarge {
        /// Per-medium limit in bytes.
        limit: u64,
    },
    /// The media of one request exceed the connector's request total.
    #[error("the request's media exceed {limit} bytes in total")]
    RequestTooLarge {
        /// Request total in bytes.
        limit: u64,
    },
    /// An image neither declares an image type nor starts with an image magic number.
    #[error(
        "the media is not an image: its declared type is not image/* and its bytes match no image format"
    )]
    NotImage,
    /// The bytes do not form a readable image of a supported format.
    #[error("the image cannot be decoded: {0}")]
    Undecodable(image::ImageError),
    /// The fetch did not complete within the connector's time limit.
    #[error("the URL fetch did not complete within {timeout:?}")]
    Timeout {
        /// Fetch time limit.
        timeout: Duration,
    },
    /// Connecting, TLS, or reading the response failed.
    #[error("the URL could not be fetched: {reason}")]
    Transport {
        /// Innermost cause reported by the HTTP client.
        reason: String,
    },
    /// The blocking task that decodes or reads media did not complete.
    #[error("the media decoding task failed")]
    Task(#[source] tokio::task::JoinError),
}

/// Failure of one reference in an ordered list resolved by
/// [`MediaConnector::fetch_all`] or [`MediaConnector::fetch_images`].
#[derive(Debug, Error)]
#[error("input {}: {source}", .index + 1)]
pub struct MediaListError {
    /// Zero-based position of the failing reference in the input list.
    pub index: usize,
    /// Why the reference could not be resolved.
    #[source]
    pub source: MediaError,
}

/// The bytes of one medium as its source delivered them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedMedia {
    /// The media bytes; cloning shares them.
    pub bytes: Bytes,
    /// The media type the source declared (`data:` type or `Content-Type`),
    /// lowercased without parameters. Advisory only: consumers identify the
    /// format from the bytes.
    pub media_type: Option<String>,
}

/// One encoded input image with the facts preprocessing needs before pixels are decoded.
///
/// The bytes are the complete image file exactly as received (PNG, JPEG, GIF,
/// WebP, or BMP). The dimensions come from the file header alone, so
/// construction never decodes pixels; the worker decodes them. Every value is
/// derived from `bytes` at construction, so the fields cannot disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    bytes: Bytes,
    width: u32,
    height: u32,
    digest: [u8; 32],
}

impl ImageInput {
    /// Validates an encoded image file and reads its header dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`MediaError::Undecodable`] when the bytes match no supported
    /// image format or their header cannot be read.
    pub fn from_bytes(bytes: impl Into<Bytes>) -> Result<Self, MediaError> {
        let bytes = bytes.into();

        // The format comes from the magic number, not from any declared media
        // type, so a mislabeled file is read as what it actually is.
        let (width, height) = image::ImageReader::new(Cursor::new(&bytes[..]))
            .with_guessed_format()
            .map_err(|error| MediaError::Undecodable(image::ImageError::IoError(error)))?
            .into_dimensions()
            .map_err(MediaError::Undecodable)?;

        let mut key = storage::key::Digest::new(b"media");
        key.bytes(&bytes);
        let digest = key.finish();
        Ok(Self {
            bytes,
            width,
            height,
            digest,
        })
    }

    /// Encoded image file bytes.
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }

    /// Width in pixels, from the file header.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels, from the file header.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The content key of the encoded bytes.
    ///
    /// Equal bytes always give equal digests, whether the image arrived
    /// inline or from a URL, so the value starts the identity of the
    /// image's encoder outputs (`proto::request::ImageInput::digest`).
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Resolves media references under one source policy and size limits.
///
/// One connector serves every request of its kind: it owns the HTTP client
/// (and its connection pool) and the slots that bound concurrent downloads.
/// Cloning is cheap and shares both.
#[derive(Debug, Clone)]
pub struct MediaConnector {
    client: reqwest::Client,
    allow_remote: bool,
    allow_private: bool,
    media_root: Option<MediaRoot>,
    timeout: Duration,
    max_bytes: u64,
    max_request_bytes: Option<u64>,
    slots: Arc<Semaphore>,
}

impl MediaConnector {
    /// Redirects followed before a fetch fails.
    pub const MAX_REDIRECTS: usize = 3;

    /// Largest number of http(s) downloads in progress at once on one connector.
    ///
    /// Each download buffers at most the per-medium limit, so the limit also
    /// bounds the memory in-flight downloads hold (1.28 GB at the 20 MB image
    /// default). Further fetches wait for a slot; `data:` and `file://`
    /// references never take one.
    pub const MAX_CONCURRENT_FETCHES: usize = 64;

    /// Longest time to establish one connection.
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

    /// Longest time for one fetch of video condition media, body included.
    const CONDITION_FETCH_TIMEOUT: Duration = Duration::from_secs(60);

    /// The connector of input images (`image_fetch_config`): `data:` and
    /// `http(s)` references, each image within `max_bytes`, each fetch within
    /// `timeout`, and private destinations only when `allow_private` is set.
    ///
    /// # Errors
    ///
    /// Fails when the HTTP client cannot be initialized, for example when the
    /// TLS backend cannot load its configuration.
    pub fn from_image_options(options: &ImageFetchOptions) -> anyhow::Result<Self> {
        Self::new(
            None,
            true,
            options.allow_private,
            options.timeout,
            options.max_bytes,
            None,
        )
    }

    /// The connector of video condition media (`media_config`): `data:`
    /// references, `file://` below `directory` when one is configured, and
    /// public `http(s)` destinations when `allow_remote` is set. The media of
    /// one request share `max_request_bytes`, which a single condition may
    /// use up.
    ///
    /// # Errors
    ///
    /// Fails when the media directory cannot be resolved or the HTTP client
    /// cannot be initialized.
    pub fn from_media_options(options: &MediaOptions) -> anyhow::Result<Self> {
        Self::new(
            options.directory.as_deref(),
            options.allow_remote,
            false,
            Self::CONDITION_FETCH_TIMEOUT,
            options.max_request_bytes,
            Some(options.max_request_bytes),
        )
    }

    fn new(
        directory: Option<&Path>,
        allow_remote: bool,
        allow_private: bool,
        timeout: Duration,
        max_bytes: u64,
        max_request_bytes: Option<u64>,
    ) -> anyhow::Result<Self> {
        let media_root = directory.map(MediaRoot::resolve).transpose()?;
        let client = reqwest::Client::builder()
            // Redirects are followed by `download` so each hop is validated
            // before it is requested.
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .dns_resolver(Arc::new(AddressResolver { allow_private }))
            .connect_timeout(Self::CONNECT_TIMEOUT)
            .user_agent(concat!("uniserve/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            client,
            allow_remote,
            allow_private,
            media_root,
            timeout,
            max_bytes,
            max_request_bytes,
            slots: Arc::new(Semaphore::new(Self::MAX_CONCURRENT_FETCHES)),
        })
    }

    /// Resolves an ordered list of references concurrently, preserving order,
    /// under one request total.
    ///
    /// The first failure cancels the remaining fetches and is reported with
    /// the index of its reference. Dropping the returned future cancels
    /// in-progress downloads.
    ///
    /// # Errors
    ///
    /// Returns a [`MediaListError`] for the first reference that fails, with
    /// the [`MediaError`] describing the first rule it violates (see the
    /// module documentation).
    pub async fn fetch_all(
        &self,
        references: &[&str],
    ) -> Result<Vec<FetchedMedia>, MediaListError> {
        let budget = Budget::new(self.max_request_bytes);
        futures::future::try_join_all(references.iter().enumerate().map(|(index, reference)| {
            let budget = &budget;
            async move {
                self.fetch(reference, "*/*", budget)
                    .await
                    .map_err(|source| MediaListError { index, source })
            }
        }))
        .await
    }

    /// Resolves an ordered list of image references into validated images,
    /// as [`Self::fetch_all`] resolves media.
    ///
    /// A medium is an image when its source declares an `image/*` type or
    /// its bytes begin with a recognized image magic number, and its header
    /// decodes. Header decoding and content hashing run on the blocking pool.
    ///
    /// # Errors
    ///
    /// As [`Self::fetch_all`], with [`MediaError::NotImage`] and
    /// [`MediaError::Undecodable`] for media that are not readable images.
    pub async fn fetch_images(
        &self,
        references: Vec<String>,
    ) -> Result<Vec<ImageInput>, MediaListError> {
        let budget = Budget::new(self.max_request_bytes);
        futures::future::try_join_all(references.iter().enumerate().map(|(index, reference)| {
            let budget = &budget;
            async move {
                self.fetch_image(reference, budget)
                    .await
                    .map_err(|source| MediaListError { index, source })
            }
        }))
        .await
    }

    /// Resolves one image reference, reserving its bytes against `budget`.
    async fn fetch_image(
        &self,
        reference: &str,
        budget: &Budget,
    ) -> Result<ImageInput, MediaError> {
        let media = self.fetch(reference, "image/*", budget).await?;
        tokio::task::spawn_blocking(move || accept_image(media))
            .await
            .map_err(MediaError::Task)?
    }

    /// Resolves one reference, reserving its bytes against `budget`.
    ///
    /// `accept` is the `Accept` header of an http(s) request.
    async fn fetch(
        &self,
        reference: &str,
        accept: &'static str,
        budget: &Budget,
    ) -> Result<FetchedMedia, MediaError> {
        // A data URL is matched before URL parsing, which would copy its
        // payload.
        let media = if has_prefix_ignore_case(reference, "data:") {
            let reference = reference.to_owned();
            let max_bytes = self.max_bytes;
            tokio::task::spawn_blocking(move || decode_data_url(&reference, max_bytes))
                .await
                .map_err(MediaError::Task)??
        } else {
            let url = Url::parse(reference).map_err(MediaError::InvalidUrl)?;
            match url.scheme() {
                // Remote bodies reserve their bytes as they arrive.
                "http" | "https" => return self.fetch_remote(url, accept, budget).await,
                "file" => self.read_file(&url).await?,
                _ => return Err(MediaError::UnsupportedScheme),
            }
        };
        budget.take(media.bytes.len() as u64)?;
        Ok(media)
    }

    /// Reads a `file://` reference below the media directory.
    async fn read_file(&self, url: &Url) -> Result<FetchedMedia, MediaError> {
        let Some(root) = self.media_root.clone() else {
            return Err(MediaError::LocalDisabled);
        };
        let path = url
            .to_file_path()
            .map_err(|()| MediaError::InvalidFileUrl)?;
        let max_bytes = self.max_bytes;
        tokio::task::spawn_blocking(move || read_media_file(&root, &path, max_bytes))
            .await
            .map_err(MediaError::Task)?
    }

    /// Fetches an http(s) URL within the time limit, holding one download slot.
    async fn fetch_remote(
        &self,
        url: Url,
        accept: &'static str,
        budget: &Budget,
    ) -> Result<FetchedMedia, MediaError> {
        if !self.allow_remote {
            return Err(MediaError::RemoteDisabled);
        }
        // A reference that can never be fetched fails before it waits for a slot.
        self.check_destination(&url, false)?;

        // The semaphore is never closed, so acquisition only waits.
        let Ok(_slot) = self.slots.acquire().await else {
            unreachable!("the media fetch semaphore is never closed");
        };
        let timeout = self.timeout;
        tokio::time::timeout(timeout, self.download(url, accept, budget))
            .await
            .map_err(|_| MediaError::Timeout { timeout })?
    }

    /// Checks the scheme of a URL about to be requested and, when the policy
    /// requires public destinations, its literal address.
    ///
    /// Host names are checked by [`AddressResolver`] when the client connects,
    /// against the exact addresses the connection uses.
    fn check_destination(&self, url: &Url, redirected: bool) -> Result<(), MediaError> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(if redirected {
                MediaError::UnsupportedRedirect
            } else {
                MediaError::UnsupportedScheme
            });
        }

        if self.allow_private {
            return Ok(());
        }
        let literal = match url.host() {
            Some(url::Host::Ipv4(address)) => Some(IpAddr::V4(address)),
            Some(url::Host::Ipv6(address)) => Some(IpAddr::V6(address)),
            Some(url::Host::Domain(_)) | None => None,
        };
        match literal {
            Some(address) if !is_public_address(address) => Err(non_public(redirected)),
            _ => Ok(()),
        }
    }

    /// Requests `url`, following validated redirects, and reads the final body.
    ///
    /// The caller has already checked `url` itself.
    async fn download(
        &self,
        mut url: Url,
        accept: &'static str,
        budget: &Budget,
    ) -> Result<FetchedMedia, MediaError> {
        let mut redirects = 0;
        loop {
            let redirected = redirects > 0;
            let response = self
                .client
                .get(url.clone())
                .header(ACCEPT, accept)
                .send()
                .await
                .map_err(|error| transport_error(error, redirected))?;

            let status = response.status();
            if is_followed_redirect(status) {
                if redirects == Self::MAX_REDIRECTS {
                    return Err(MediaError::TooManyRedirects {
                        limit: Self::MAX_REDIRECTS,
                    });
                }
                let location = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .ok_or(MediaError::InvalidRedirect)?;
                // A relative `Location` resolves against the URL that returned it.
                let target = url
                    .join(location)
                    .map_err(|_| MediaError::InvalidRedirect)?;
                self.check_destination(&target, true)?;
                if url.scheme() == "https" && target.scheme() == "http" {
                    return Err(MediaError::InsecureRedirect);
                }

                url = target;
                redirects += 1;
                continue;
            }
            if !status.is_success() {
                return Err(MediaError::Status {
                    status: status.as_u16(),
                });
            }
            return self.read_body(response, redirected, budget).await;
        }
    }

    /// Reads a success response body within the per-medium limit, reserving
    /// each chunk against `budget` as it arrives.
    async fn read_body(
        &self,
        mut response: reqwest::Response,
        redirected: bool,
        budget: &Budget,
    ) -> Result<FetchedMedia, MediaError> {
        let limit = self.max_bytes;
        // `content_length` is the length the HTTP framing will deliver; a
        // chunked body has none and is bounded only by the streamed count.
        let declared_length = response.content_length();
        if declared_length.is_some_and(|length| length > limit) {
            return Err(MediaError::TooLarge { limit });
        }
        let media_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(media_type);

        // The preallocation never exceeds `limit`, which was checked above.
        let mut body = BytesMut::with_capacity(declared_length.unwrap_or(0) as usize);
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| transport_error(error, redirected))?
        {
            if (body.len() + chunk.len()) as u64 > limit {
                return Err(MediaError::TooLarge { limit });
            }
            budget.take(chunk.len() as u64)?;
            body.extend_from_slice(&chunk);
        }
        Ok(FetchedMedia {
            bytes: body.freeze(),
            media_type,
        })
    }
}

/// The bytes one request may still fetch; unbounded without a request total.
struct Budget {
    limit: Option<u64>,
    remaining: AtomicU64,
}

impl Budget {
    fn new(limit: Option<u64>) -> Self {
        Self {
            limit,
            remaining: AtomicU64::new(limit.unwrap_or(u64::MAX)),
        }
    }

    /// Reserves `bytes` of the request total.
    fn take(&self, bytes: u64) -> Result<(), MediaError> {
        let Some(limit) = self.limit else {
            return Ok(());
        };
        let reserved =
            self.remaining
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                    remaining.checked_sub(bytes)
                });
        match reserved {
            Ok(_) => Ok(()),
            Err(_) => Err(MediaError::RequestTooLarge { limit }),
        }
    }
}

/// The media directory as configured and as canonicalized; a `file://`
/// path may name it either way.
#[derive(Debug, Clone)]
struct MediaRoot {
    configured: PathBuf,
    canonical: PathBuf,
}

impl MediaRoot {
    fn resolve(directory: &Path) -> anyhow::Result<Self> {
        let failed = |error| anyhow::anyhow!("media directory {}: {error}", directory.display());
        Ok(Self {
            configured: std::path::absolute(directory).map_err(failed)?,
            canonical: std::fs::canonicalize(directory).map_err(failed)?,
        })
    }
}

/// Reads a non-empty regular file below the media directory of at most
/// `max_bytes` bytes.
///
/// `path` is absolute with its `.` and `..` segments resolved, as URL
/// parsing leaves a `file://` path.
fn read_media_file(
    root: &MediaRoot,
    path: &Path,
    max_bytes: u64,
) -> Result<FetchedMedia, MediaError> {
    let shown = || path.display().to_string();
    let unreadable = |reason: String| MediaError::LocalFile {
        path: shown(),
        reason,
    };
    let outside = || MediaError::OutsideMediaDirectory { path: shown() };
    if !path.starts_with(&root.configured) && !path.starts_with(&root.canonical) {
        return Err(outside());
    }
    // Canonicalization resolves every symbolic link, so a path that leaves
    // the media directory through a link ends outside its canonical form.
    let canonical =
        std::fs::canonicalize(path).map_err(|_| unreadable("does not exist".to_owned()))?;
    if !canonical.starts_with(&root.canonical) {
        return Err(outside());
    }
    // Refuse a final symbolic link swapped in after canonicalization.
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&canonical)
        .map_err(|error| unreadable(format!("cannot be opened: {error}")))?;
    let metadata = file
        .metadata()
        .map_err(|error| unreadable(format!("cannot be read: {error}")))?;
    if !metadata.is_file() {
        return Err(unreadable("is not a regular file".to_owned()));
    }
    let too_large = MediaError::TooLarge { limit: max_bytes };
    if metadata.len() > max_bytes {
        return Err(too_large);
    }
    // The file may grow after its metadata was read; the read stops one byte
    // past the limit to detect it.
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    (&mut file)
        .take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| unreadable(format!("cannot be read: {error}")))?;
    if bytes.len() as u64 > max_bytes {
        return Err(too_large);
    }
    if bytes.is_empty() {
        return Err(unreadable("is empty".to_owned()));
    }
    Ok(FetchedMedia {
        bytes: Bytes::from(bytes),
        media_type: None,
    })
}

/// Decodes a `data:[<type>][;<parameter>]*;base64,<payload>` URL of at most
/// `max_bytes` decoded bytes.
///
/// The scheme, media type, and `base64` marker match case-insensitively. The
/// payload must be non-empty standard padded base64 without whitespace.
fn decode_data_url(reference: &str, max_bytes: u64) -> Result<FetchedMedia, MediaError> {
    let (metadata, payload) = reference
        .split_once(',')
        .ok_or(MediaError::InvalidDataUrl)?;
    // The caller matched the ASCII `data:` prefix, which holds no comma.
    let mut parameters = metadata["data:".len()..].split(';');
    let declared = parameters.next().unwrap_or_default();
    let is_base64 = parameters
        .next_back()
        .is_some_and(|marker| marker.eq_ignore_ascii_case("base64"));
    if !is_base64 || payload.is_empty() {
        return Err(MediaError::InvalidDataUrl);
    }
    let media_type = match declared {
        "" => None,
        declared => Some(media_type(declared).ok_or(MediaError::InvalidDataUrl)?),
    };

    // Padded base64 decodes to the estimate less at most two padding bytes,
    // so a payload whose smallest decoded length exceeds the limit is
    // refused before it is decoded.
    let too_large = MediaError::TooLarge { limit: max_bytes };
    if base64::decoded_len_estimate(payload.len()).saturating_sub(2) as u64 > max_bytes {
        return Err(too_large);
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(MediaError::InvalidBase64)?;
    if bytes.len() as u64 > max_bytes {
        return Err(too_large);
    }
    Ok(FetchedMedia {
        bytes: Bytes::from(bytes),
        media_type,
    })
}

/// The essence of a `<type>/<subtype>` media type: lowercased, parameters
/// dropped; `None` unless both parts are present.
fn media_type(value: &str) -> Option<String> {
    let essence = value.split(';').next()?.trim().to_ascii_lowercase();
    let (kind, subtype) = essence.split_once('/')?;
    (!kind.is_empty() && !subtype.is_empty()).then_some(essence)
}

/// Validates fetched media as an image: declared `image/*` or recognized by
/// its magic number, with a readable header.
fn accept_image(media: FetchedMedia) -> Result<ImageInput, MediaError> {
    let declares_image = media
        .media_type
        .as_deref()
        .is_some_and(|media_type| media_type.starts_with("image/"));
    if !declares_image && image::guess_format(&media.bytes).is_err() {
        return Err(MediaError::NotImage);
    }
    ImageInput::from_bytes(media.bytes)
}

/// Returns whether `value` starts with the ASCII `prefix`, ignoring case.
fn has_prefix_ignore_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

/// Returns whether a fetch may connect to `address` under the public-only policy.
///
/// Rejected: unspecified, loopback, private-use (RFC 1918), shared (CGNAT),
/// link-local (which holds the 169.254.169.254 cloud metadata service),
/// unique-local and site-local IPv6 (which hold IPv6 metadata services such
/// as `fd00:ec2::254`), documentation, benchmarking, IETF protocol
/// assignment, discard-only, segment-routing, multicast, reserved, and
/// broadcast ranges, the deprecated IPv4-compatible IPv6 range, and NAT64
/// prefixes other than the well-known one. IPv6 forms that carry an IPv4
/// destination (IPv4-mapped, IPv4-translated, well-known NAT64, and 6to4)
/// are judged by that IPv4 address.
pub fn is_public_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_public_ipv4(address),
        IpAddr::V6(address) => is_public_ipv6(address),
    }
}

/// IPv4 ranges outside public unicast space, as `(network, prefix length)`.
const NON_PUBLIC_IPV4: &[(Ipv4Addr, u8)] = &[
    // "This network", including the unspecified address 0.0.0.0, which
    // Linux connects to the local host.
    (Ipv4Addr::new(0, 0, 0, 0), 8),
    (Ipv4Addr::new(10, 0, 0, 0), 8),
    // Shared address space for carrier-grade NAT, which also holds the
    // 100.100.100.200 cloud metadata service.
    (Ipv4Addr::new(100, 64, 0, 0), 10),
    (Ipv4Addr::new(127, 0, 0, 0), 8),
    // Link-local, which holds the 169.254.169.254 and 169.254.170.2 cloud
    // metadata services.
    (Ipv4Addr::new(169, 254, 0, 0), 16),
    (Ipv4Addr::new(172, 16, 0, 0), 12),
    // IETF protocol assignments.
    (Ipv4Addr::new(192, 0, 0, 0), 24),
    // Documentation (TEST-NET-1).
    (Ipv4Addr::new(192, 0, 2, 0), 24),
    // Deprecated 6to4 relay anycast.
    (Ipv4Addr::new(192, 88, 99, 0), 24),
    (Ipv4Addr::new(192, 168, 0, 0), 16),
    // Benchmarking.
    (Ipv4Addr::new(198, 18, 0, 0), 15),
    // Documentation (TEST-NET-2 and TEST-NET-3).
    (Ipv4Addr::new(198, 51, 100, 0), 24),
    (Ipv4Addr::new(203, 0, 113, 0), 24),
    // Multicast.
    (Ipv4Addr::new(224, 0, 0, 0), 4),
    // Reserved, including the limited broadcast address 255.255.255.255.
    (Ipv4Addr::new(240, 0, 0, 0), 4),
];

/// IPv6 ranges outside public unicast space, as `(network, prefix length)`.
///
/// Ranges that embed an IPv4 destination are handled by `embedded_ipv4`
/// before this table is consulted.
const NON_PUBLIC_IPV6: &[(Ipv6Addr, u8)] = &[
    // Deprecated IPv4-compatible addresses, including the unspecified
    // address `::` and loopback `::1`.
    (Ipv6Addr::UNSPECIFIED, 96),
    // NAT64 prefixes other than the well-known `64:ff9b::/96`, including
    // local-use `64:ff9b:1::/48`, which translate to destinations the
    // embedded IPv4 address does not determine.
    (Ipv6Addr::new(0x64, 0xff9b, 0, 0, 0, 0, 0, 0), 32),
    // Discard-only.
    (Ipv6Addr::new(0x100, 0, 0, 0, 0, 0, 0, 0), 64),
    // IETF protocol assignments, including Teredo.
    (Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0), 23),
    // Documentation.
    (Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32),
    (Ipv6Addr::new(0x3fff, 0, 0, 0, 0, 0, 0, 0), 20),
    // Segment-routing SIDs.
    (Ipv6Addr::new(0x5f00, 0, 0, 0, 0, 0, 0, 0), 16),
    // Unique-local.
    (Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 0), 7),
    // Link-local and deprecated site-local.
    (Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 0), 10),
    (Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 0), 10),
    // Multicast.
    (Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0), 8),
];

/// Returns whether an IPv4 address lies outside every [`NON_PUBLIC_IPV4`] range.
fn is_public_ipv4(address: Ipv4Addr) -> bool {
    let bits = u32::from(address);
    !NON_PUBLIC_IPV4.iter().any(|(network, length)| {
        let mask = u32::MAX.checked_shl(32 - u32::from(*length)).unwrap_or(0);
        bits & mask == u32::from(*network) & mask
    })
}

/// Returns whether an IPv6 address is public, judging embedded IPv4
/// destinations by their IPv4 address.
fn is_public_ipv6(address: Ipv6Addr) -> bool {
    if let Some(ipv4) = embedded_ipv4(address) {
        return is_public_ipv4(ipv4);
    }
    let bits = u128::from(address);
    !NON_PUBLIC_IPV6.iter().any(|(network, length)| {
        let mask = u128::MAX.checked_shl(128 - u32::from(*length)).unwrap_or(0);
        bits & mask == u128::from(*network) & mask
    })
}

/// Returns the IPv4 destination an IPv6 address carries, if any.
///
/// Covers IPv4-mapped (`::ffff:0:0/96`), IPv4-translated
/// (`::ffff:0:0:0/96`), well-known NAT64 (`64:ff9b::/96`), and 6to4
/// (`2002::/16`, IPv4 in bits 16..48).
fn embedded_ipv4(address: Ipv6Addr) -> Option<Ipv4Addr> {
    let bits = u128::from(address);

    // The three /96 forms carry the IPv4 address in their low 32 bits; the
    // match is on the 96-bit prefix above it.
    const IPV4_MAPPED: u128 = 0xffff;
    const IPV4_TRANSLATED: u128 = 0xffff_0000;
    const NAT64: u128 = 0x0064_ff9b_0000_0000_0000_0000;
    if matches!(bits >> 32, IPV4_MAPPED | IPV4_TRANSLATED | NAT64) {
        return Some(Ipv4Addr::from(bits as u32));
    }

    // 6to4 carries it in the 32 bits after the 16-bit `2002` prefix.
    if bits >> 112 == 0x2002 {
        return Some(Ipv4Addr::from((bits >> 80) as u32));
    }
    None
}

/// DNS resolution that admits only addresses the fetch policy allows.
///
/// The client connects to exactly the addresses this resolver returns, so the
/// address check and the connection share one resolution. A host with any
/// non-public address is refused outright rather than filtered, so a DNS
/// answer that mixes public and internal addresses cannot select either.
struct AddressResolver {
    allow_private: bool,
}

impl reqwest::dns::Resolve for AddressResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let allow_private = self.allow_private;
        Box::pin(async move {
            // The client replaces the port on every returned address.
            let addresses = tokio::net::lookup_host((name.as_str(), 0))
                .await?
                .collect::<Vec<SocketAddr>>();
            if !allow_private
                && addresses
                    .iter()
                    .any(|address| !is_public_address(address.ip()))
            {
                return Err(Box::new(NonPublicHost) as Box<dyn std::error::Error + Send + Sync>);
            }
            Ok(Box::new(addresses.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// Resolver refusal, recovered from the client's connect error chain by
/// `transport_error`.
#[derive(Debug, Error)]
#[error("host resolves to a non-public address")]
struct NonPublicHost;

/// Returns the non-public destination error for the initial URL or a redirect.
fn non_public(redirected: bool) -> MediaError {
    if redirected {
        MediaError::NonPublicRedirect
    } else {
        MediaError::NonPublicAddress
    }
}

/// Maps an HTTP client error without exposing the URL it was requesting.
///
/// A resolver refusal becomes the non-public destination error; any other
/// failure is described by its innermost cause (for example "Connection
/// refused" or a TLS certificate error), which names no URL or address.
fn transport_error(error: reqwest::Error, redirected: bool) -> MediaError {
    let error = error.without_url();
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    loop {
        if cause.is::<NonPublicHost>() {
            return non_public(redirected);
        }
        match cause.source() {
            Some(next) => cause = next,
            None => break,
        }
    }
    MediaError::Transport {
        reason: cause.to_string(),
    }
}

/// Returns whether a status is a redirect `download` follows with a GET.
fn is_followed_redirect(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::MOVED_PERMANENTLY
            | StatusCode::FOUND
            | StatusCode::SEE_OTHER
            | StatusCode::TEMPORARY_REDIRECT
            | StatusCode::PERMANENT_REDIRECT
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::AtomicUsize;

    use image::ImageFormat;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::{TcpListener, TcpStream};

    use super::*;

    /// Encodes a black `width` x `height` RGB image in `format`.
    fn encoded_image(format: ImageFormat, width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(width, height)
            .write_to(&mut bytes, format)
            .unwrap();
        bytes.into_inner()
    }

    /// Returns the `kind` data URL carrying `bytes`.
    fn data_url_of(kind: &str, bytes: &[u8]) -> String {
        format!(
            "data:{kind};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    /// Returns the image data URL carrying `bytes`.
    fn data_url(bytes: &[u8]) -> String {
        data_url_of("image/png", bytes)
    }

    /// Default image limits with loopback destinations allowed, which the
    /// local test server needs.
    fn loopback_policy() -> ImageFetchOptions {
        ImageFetchOptions {
            allow_private: true,
            ..ImageFetchOptions::default()
        }
    }

    fn image_connector(options: &ImageFetchOptions) -> MediaConnector {
        MediaConnector::from_image_options(options).unwrap()
    }

    /// Resolves one image reference through `fetch_images`.
    async fn fetch_image(
        connector: &MediaConnector,
        reference: impl Into<String>,
    ) -> Result<ImageInput, MediaError> {
        connector
            .fetch_images(vec![reference.into()])
            .await
            .map(|mut images| images.remove(0))
            .map_err(|error| {
                assert_eq!(error.index, 0);
                error.source
            })
    }

    /// Resolves one reference through `fetch_all`.
    async fn fetch_one(
        connector: &MediaConnector,
        reference: &str,
    ) -> Result<FetchedMedia, MediaError> {
        connector
            .fetch_all(&[reference])
            .await
            .map(|mut media| media.remove(0))
            .map_err(|error| {
                assert_eq!(error.index, 0);
                error.source
            })
    }

    /// A condition-media connector whose request total is `max_request_bytes`.
    fn condition_connector(directory: Option<&Path>, max_request_bytes: u64) -> MediaConnector {
        MediaConnector::from_media_options(&MediaOptions {
            directory: directory.map(Path::to_path_buf),
            max_request_bytes,
            ..MediaOptions::default()
        })
        .unwrap()
    }

    /// One scripted answer of [`TestServer`].
    enum Reply {
        /// Writes the bytes, then closes the connection.
        Close(Vec<u8>),
        /// Writes the bytes, then holds the connection open without sending more.
        Hold(Vec<u8>),
    }

    /// Complete HTTP/1.1 response with a `Content-Length` body.
    fn response(status: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
        let mut head = format!(
            "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\n",
            body.len()
        );
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("\r\n");

        let mut bytes = head.into_bytes();
        bytes.extend_from_slice(body);
        bytes
    }

    /// Complete `200 OK` response carrying `body` as `content_type`.
    fn ok(content_type: &str, body: &[u8]) -> Reply {
        Reply::Close(response("200 OK", &[("Content-Type", content_type)], body))
    }

    /// Redirect response with the given status line and `Location`.
    fn redirect(status: &str, location: &str) -> Reply {
        Reply::Close(response(status, &[("Location", location)], b""))
    }

    /// HTTP/1.1 server on an ephemeral 127.0.0.1 port. It answers each
    /// request path with a scripted reply and counts accepted connections.
    struct TestServer {
        address: SocketAddr,
        connections: Arc<AtomicUsize>,
    }

    impl TestServer {
        /// Starts serving on the current runtime; `route` maps a request path
        /// to its reply.
        async fn start(route: impl Fn(&str) -> Reply + Send + Sync + 'static) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let connections = Arc::new(AtomicUsize::new(0));
            let accepted = Arc::clone(&connections);
            let route = Arc::new(route);

            tokio::spawn(async move {
                while let Ok((mut socket, _)) = listener.accept().await {
                    accepted.fetch_add(1, Ordering::SeqCst);
                    let route = Arc::clone(&route);
                    tokio::spawn(async move {
                        let path = request_path(&mut socket).await;
                        match route(&path) {
                            Reply::Close(bytes) => {
                                let _ = socket.write_all(&bytes).await;
                                let _ = socket.shutdown().await;
                            }
                            Reply::Hold(bytes) => {
                                let _ = socket.write_all(&bytes).await;
                                std::future::pending::<()>().await;
                            }
                        }
                    });
                }
            });
            Self {
                address,
                connections,
            }
        }

        /// Returns the http URL of `path` on this server.
        fn url(&self, path: &str) -> String {
            format!("http://{}{path}", self.address)
        }

        /// Number of connections accepted so far.
        fn connections(&self) -> usize {
            self.connections.load(Ordering::SeqCst)
        }
    }

    /// Reads a request head and returns its request-target path.
    async fn request_path(socket: &mut TcpStream) -> String {
        let mut head = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !head.windows(4).any(|window| window == b"\r\n\r\n") {
            match socket.read(&mut buffer).await {
                Ok(0) | Err(_) => break,
                Ok(read) => head.extend_from_slice(&buffer[..read]),
            }
        }
        String::from_utf8_lossy(&head)
            .split_whitespace()
            .nth(1)
            .unwrap_or("/")
            .to_owned()
    }

    /// Each row is an address and whether the public-only policy lets a fetch
    /// connect to it.
    #[test]
    fn public_addresses_are_distinguished_from_internal_ones() {
        let cases = [
            // Public unicast, including IPv6 forms that carry a public IPv4
            // destination, and the neighbors of internal ranges.
            ("8.8.8.8", true),
            ("93.184.216.34", true),
            ("100.63.255.255", true),
            ("100.128.0.0", true),
            ("172.15.255.255", true),
            ("172.32.0.0", true),
            ("2606:4700:4700::1111", true),
            ("2a00:1450:4001:80b::200e", true),
            ("::ffff:8.8.8.8", true),
            ("64:ff9b::808:808", true),
            ("2002:808:808::1", true),
            // Unspecified, "this network", and loopback.
            ("0.0.0.0", false),
            ("0.1.2.3", false),
            ("127.0.0.1", false),
            ("127.255.255.254", false),
            ("::", false),
            ("::1", false),
            // Private use and shared address space.
            ("10.0.0.1", false),
            ("172.16.0.1", false),
            ("172.31.255.255", false),
            ("192.168.1.1", false),
            ("100.64.0.1", false),
            // Link-local, unique-local, site-local, and the cloud metadata
            // services within them.
            ("169.254.0.1", false),
            ("169.254.169.254", false),
            ("169.254.170.2", false),
            ("100.100.100.200", false),
            ("fe80::1", false),
            ("fc00::1", false),
            ("fd00:ec2::254", false),
            ("fec0::1", false),
            // Protocol assignments, documentation, benchmarking, discard,
            // and segment routing.
            ("192.0.0.8", false),
            ("192.0.2.1", false),
            ("198.18.0.1", false),
            ("198.19.255.255", false),
            ("198.51.100.1", false),
            ("203.0.113.1", false),
            ("2001::1", false),
            ("2001:db8::1", false),
            ("3fff::1", false),
            ("100::1", false),
            ("5f00::1", false),
            // NAT64 prefixes other than the well-known one.
            ("64:ff9b:1::1", false),
            ("64:ff9b:2::808:808", false),
            // Multicast, reserved, and broadcast.
            ("224.0.0.1", false),
            ("239.255.255.250", false),
            ("240.0.0.1", false),
            ("255.255.255.255", false),
            ("ff02::1", false),
            // IPv6 forms that carry an internal IPv4 destination, and the
            // deprecated IPv4-compatible range whatever it carries.
            ("::ffff:127.0.0.1", false),
            ("::ffff:169.254.169.254", false),
            ("::ffff:10.0.0.1", false),
            ("::127.0.0.1", false),
            ("::8.8.8.8", false),
            ("::ffff:0:192.168.0.1", false),
            ("64:ff9b::a00:1", false),
            ("2002:7f00:1::", false),
            ("2002:a9fe:a9fe::1", false),
        ];
        for (address, public) in cases {
            let parsed = address.parse::<IpAddr>().unwrap();
            assert_eq!(is_public_address(parsed), public, "{address}");
        }
    }

    /// A data URL resolves to its decoded bytes and header dimensions, and
    /// the same bytes fetched over http resolve to an equal image with the
    /// same content digest, while different bytes digest differently.
    #[tokio::test]
    async fn fetched_and_inline_images_resolve_identically() {
        let png = encoded_image(ImageFormat::Png, 3, 2);
        let body = png.clone();
        let server = TestServer::start(move |_| ok("image/png", &body)).await;
        let connector = image_connector(&loopback_policy());

        let inline = fetch_image(&connector, data_url(&png)).await.unwrap();
        assert_eq!(inline.bytes().as_ref(), png.as_slice());
        assert_eq!((inline.width(), inline.height()), (3, 2));

        let fetched = fetch_image(&connector, server.url("/image.png"))
            .await
            .unwrap();
        assert_eq!(fetched, inline);

        let other = encoded_image(ImageFormat::Png, 2, 3);
        let other = fetch_image(&connector, data_url(&other)).await.unwrap();
        assert_ne!(other.digest(), inline.digest());
    }

    /// Lists resolve in reference order and name the index of a reference
    /// that fails.
    #[tokio::test]
    async fn lists_keep_their_order_and_name_the_failing_entry() {
        let wide = encoded_image(ImageFormat::Png, 4, 1);
        let tall = encoded_image(ImageFormat::Png, 1, 4);
        let server = TestServer::start(move |_| ok("image/png", &tall)).await;
        let connector = image_connector(&loopback_policy());

        let images = connector
            .fetch_images(vec![server.url("/tall.png"), data_url(&wide)])
            .await
            .unwrap();
        let dimensions = images
            .iter()
            .map(|image| (image.width(), image.height()))
            .collect::<Vec<_>>();
        assert_eq!(dimensions, [(1, 4), (4, 1)]);

        let error = connector
            .fetch_images(vec![data_url(&wide), "ftp://example.com/x.png".to_owned()])
            .await
            .unwrap_err();
        assert_eq!(error.index, 1);
        assert!(matches!(error.source, MediaError::UnsupportedScheme));
    }

    /// An image served without an image `Content-Type` is accepted when its
    /// magic number identifies PNG, JPEG, GIF, WebP, or BMP, and its header
    /// dimensions are read.
    #[tokio::test]
    async fn images_are_recognized_by_magic_number() {
        let formats = [
            ("/png", ImageFormat::Png),
            ("/jpeg", ImageFormat::Jpeg),
            ("/gif", ImageFormat::Gif),
            ("/webp", ImageFormat::WebP),
            ("/bmp", ImageFormat::Bmp),
        ];
        let bodies = formats
            .iter()
            .map(|(path, format)| (path.to_string(), encoded_image(*format, 5, 3)))
            .collect::<HashMap<_, _>>();
        let server =
            TestServer::start(move |path| ok("application/octet-stream", &bodies[path])).await;
        let connector = image_connector(&loopback_policy());

        for (path, _) in formats {
            let image = fetch_image(&connector, server.url(path)).await.unwrap();
            assert_eq!((image.width(), image.height()), (5, 3), "{path}");
        }
        let untyped = encoded_image(ImageFormat::Png, 5, 3);
        let image = fetch_image(
            &connector,
            data_url_of("application/octet-stream", &untyped),
        )
        .await
        .unwrap();
        assert_eq!((image.width(), image.height()), (5, 3));
    }

    /// Media that carry no usable image are refused as images: an
    /// unsuccessful status, a body neither declared nor recognizable as an
    /// image, and a declared image whose bytes do not decode.
    #[tokio::test]
    async fn media_without_an_image_are_refused_as_images() {
        let server = TestServer::start(|path| match path {
            "/missing" => Reply::Close(response("404 Not Found", &[], b"")),
            "/page" => ok("text/html; charset=utf-8", b"<html><body>cat</body></html>"),
            "/unlabeled" => Reply::Close(response("200 OK", &[], b"plain text")),
            _ => ok("image/png", b"these bytes are not a PNG file"),
        })
        .await;
        let connector = image_connector(&loopback_policy());

        assert!(matches!(
            fetch_image(&connector, server.url("/missing")).await,
            Err(MediaError::Status { status: 404 })
        ));
        for reference in [
            server.url("/page"),
            server.url("/unlabeled"),
            "data:text/plain;base64,aGVsbG8=".to_owned(),
        ] {
            assert!(
                matches!(
                    fetch_image(&connector, reference.clone()).await,
                    Err(MediaError::NotImage)
                ),
                "{reference}"
            );
        }
        assert!(matches!(
            fetch_image(&connector, server.url("/broken")).await,
            Err(MediaError::Undecodable(_))
        ));
    }

    /// The size limit admits a medium of exactly the limit and refuses one
    /// byte more, whether the response declares its length, is delimited by
    /// connection close, or declares a `Content-Length` its chunked framing
    /// overrides.
    #[tokio::test]
    async fn media_over_the_size_limit_are_refused() {
        let png = encoded_image(ImageFormat::Png, 8, 8);
        let body = png.clone();
        let server = TestServer::start(move |path| {
            let head = match path {
                "/declared" => return ok("image/png", &body),
                "/close-delimited" => {
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n"
                        .to_owned()
                }
                _ => format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 16\r\n\
                     Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n",
                    body.len()
                ),
            };
            let mut bytes = head.into_bytes();
            bytes.extend_from_slice(&body);
            if path != "/close-delimited" {
                bytes.extend_from_slice(b"\r\n0\r\n\r\n");
            }
            Reply::Close(bytes)
        })
        .await;
        let paths = ["/declared", "/close-delimited", "/chunked"];

        let exact = image_connector(&ImageFetchOptions {
            max_bytes: png.len() as u64,
            ..loopback_policy()
        });
        for path in paths {
            let image = fetch_image(&exact, server.url(path)).await.unwrap();
            assert_eq!(image.bytes().as_ref(), png.as_slice(), "{path}");
        }

        let limit = png.len() as u64 - 1;
        let smaller = image_connector(&ImageFetchOptions {
            max_bytes: limit,
            ..loopback_policy()
        });
        for path in paths {
            let result = fetch_image(&smaller, server.url(path)).await;
            assert!(
                matches!(result, Err(MediaError::TooLarge { limit: refused }) if refused == limit),
                "{path}: {result:?}"
            );
        }
    }

    /// A declared length over the limit is refused as soon as the response
    /// head arrives. The server never sends the body, so a fetch that waited
    /// for it would end at the time limit instead.
    #[tokio::test]
    async fn declared_oversize_is_refused_before_the_body_arrives() {
        let server = TestServer::start(|_| {
            Reply::Hold(
                b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 1000000\r\n\r\n"
                    .to_vec(),
            )
        })
        .await;
        let connector = image_connector(&ImageFetchOptions {
            max_bytes: 1000,
            ..loopback_policy()
        });

        assert!(matches!(
            fetch_image(&connector, server.url("/image.png")).await,
            Err(MediaError::TooLarge { limit: 1000 })
        ));
    }

    /// A fetch that stalls before the response head or in the middle of the
    /// body fails at the time limit.
    #[tokio::test]
    async fn stalled_fetches_end_at_the_time_limit() {
        let server = TestServer::start(|path| match path {
            "/silent" => Reply::Hold(Vec::new()),
            _ => Reply::Hold(
                b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 100\r\n\r\n\x89PNG"
                    .to_vec(),
            ),
        })
        .await;
        let timeout = Duration::from_millis(200);
        let connector = image_connector(&ImageFetchOptions {
            timeout,
            ..loopback_policy()
        });

        for path in ["/silent", "/partial"] {
            let result = fetch_image(&connector, server.url(path)).await;
            assert!(
                matches!(result, Err(MediaError::Timeout { timeout: limit }) if limit == timeout),
                "{path}: {result:?}"
            );
        }
    }

    /// Only `data:`, `file://` and http(s) URLs are accepted, `file://` only
    /// with a media directory. Other schemes, relative references, malformed
    /// data URLs, and data URLs over the size limit are refused.
    #[tokio::test]
    async fn unsupported_references_are_refused() {
        let connector = image_connector(&ImageFetchOptions {
            max_bytes: 256,
            ..ImageFetchOptions::default()
        });
        let fetch = |reference: &str| {
            let reference = reference.to_owned();
            let connector = &connector;
            async move { fetch_one(connector, &reference).await }
        };

        for reference in [
            "ftp://example.com/image.png",
            "gopher://example.com/image",
            "javascript:alert(1)",
            "blob:https://example.com/image",
        ] {
            assert!(
                matches!(fetch(reference).await, Err(MediaError::UnsupportedScheme)),
                "{reference}"
            );
        }
        assert!(matches!(
            fetch("file:///etc/passwd").await,
            Err(MediaError::LocalDisabled)
        ));
        assert!(matches!(
            fetch("images/cat.png").await,
            Err(MediaError::InvalidUrl(_))
        ));

        for reference in [
            "data:image/png,rawbytes",
            "data:image/;base64,AAAA",
            "data:image/png;base64,",
            "data:image/png;base64",
        ] {
            assert!(
                matches!(fetch(reference).await, Err(MediaError::InvalidDataUrl)),
                "{reference}"
            );
        }
        // Whitespace and missing padding are not standard base64.
        for reference in [
            "data:image/png;base64,not*base64",
            "data:image/png;base64,cGl4\nZWxz",
            "data:image/png;base64,cGl4ZWw",
        ] {
            assert!(
                matches!(fetch(reference).await, Err(MediaError::InvalidBase64(_))),
                "{reference}"
            );
        }

        // An uncompressed 16x16 BMP is 822 bytes, over the 256-byte limit.
        let large = encoded_image(ImageFormat::Bmp, 16, 16);
        assert!(matches!(
            fetch(&data_url(&large)).await,
            Err(MediaError::TooLarge { limit: 256 })
        ));
    }

    /// A data URL decodes to its bytes and its declared type, lowercased
    /// without parameters; a URL without a type declares none.
    #[tokio::test]
    async fn data_urls_decode_with_their_declared_type() {
        let connector = condition_connector(None, 64);

        let media = fetch_one(&connector, "data:Audio/WAV;rate=8000;base64,cGl4ZWxz")
            .await
            .unwrap();
        assert_eq!(media.bytes.as_ref(), b"pixels");
        assert_eq!(media.media_type.as_deref(), Some("audio/wav"));

        let media = fetch_one(&connector, "data:;base64,cGl4ZWxz")
            .await
            .unwrap();
        assert_eq!(media.media_type, None);
    }

    /// The media of one request share the request total, which a single
    /// medium may use up.
    #[tokio::test]
    async fn a_request_shares_one_total() {
        let connector = condition_connector(None, 64);

        let whole = data_url_of("video/mp4", &[7; 64]);
        assert!(fetch_one(&connector, &whole).await.is_ok());

        let video = data_url_of("video/mp4", &[1; 40]);
        let image = data_url_of("image/png", &[2; 40]);
        let error = connector.fetch_all(&[&video, &image]).await.unwrap_err();
        assert!(
            matches!(error.source, MediaError::RequestTooLarge { limit: 64 }),
            "{error}"
        );
    }

    /// `file://` resolves only below the media directory: traversal and
    /// symbolic links out of it are refused, links within it are followed,
    /// and whether a file exists outside the directory is not revealed.
    #[tokio::test]
    async fn files_resolve_below_the_media_directory() {
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.wav");
        std::fs::write(&secret, b"secret").unwrap();
        let root = tempfile::tempdir().unwrap();
        let media = root.path().join("media");
        std::fs::create_dir(&media).unwrap();
        std::fs::write(media.join("voice.wav"), b"voice").unwrap();
        std::fs::write(media.join("large.wav"), [0; 65]).unwrap();
        std::fs::write(media.join("empty.wav"), b"").unwrap();
        std::os::unix::fs::symlink(&secret, media.join("escape.wav")).unwrap();
        std::os::unix::fs::symlink(media.join("voice.wav"), media.join("alias.wav")).unwrap();

        let connector = condition_connector(Some(&media), 64);
        let uri = |path: &Path| format!("file://{}", path.display());
        let read = |path: &Path| {
            let reference = uri(path);
            let connector = &connector;
            async move { fetch_one(connector, &reference).await }
        };
        for path in [media.join("voice.wav"), media.join("alias.wav")] {
            assert_eq!(read(&path).await.unwrap().bytes.as_ref(), b"voice");
        }

        for outside_path in [
            media.join("escape.wav"),
            secret.clone(),
            outside.path().join("missing.wav"),
        ] {
            assert!(
                matches!(
                    read(&outside_path).await,
                    Err(MediaError::OutsideMediaDirectory { .. })
                ),
                "{}",
                outside_path.display()
            );
        }
        let traversal = format!("file://{}/../../{}", media.display(), secret.display());
        assert!(matches!(
            fetch_one(&connector, &traversal).await,
            Err(MediaError::OutsideMediaDirectory { .. })
        ));
        for refused in [
            media.join("missing.wav"),
            media.clone(),
            media.join("empty.wav"),
        ] {
            assert!(
                matches!(read(&refused).await, Err(MediaError::LocalFile { .. })),
                "{}",
                refused.display()
            );
        }
        assert!(matches!(
            read(&media.join("large.wav")).await,
            Err(MediaError::TooLarge { limit: 64 })
        ));
        let remote_host = format!("file://example.com{}", media.join("voice.wav").display());
        assert!(matches!(
            fetch_one(&connector, &remote_host).await,
            Err(MediaError::InvalidFileUrl)
        ));

        // The directory may be named through a link to it.
        let linked = root.path().join("linked");
        std::os::unix::fs::symlink(&media, &linked).unwrap();
        let through_link = condition_connector(Some(&linked), 64);
        for path in [linked.join("voice.wav"), media.join("voice.wav")] {
            let read = fetch_one(&through_link, &uri(&path)).await.unwrap();
            assert_eq!(read.bytes.as_ref(), b"voice");
        }

        let without_directory = condition_connector(None, 64);
        assert!(matches!(
            fetch_one(&without_directory, &uri(&media.join("voice.wav"))).await,
            Err(MediaError::LocalDisabled)
        ));
    }

    /// Redirects, relative or absolute, are followed for up to three hops of
    /// any followed status. Every hop is validated again: a fourth redirect,
    /// a redirect to a scheme other than http(s), and a redirect without a
    /// location are refused.
    #[tokio::test]
    async fn redirects_are_limited_and_revalidated() {
        let png = encoded_image(ImageFormat::Png, 2, 2);
        let server = TestServer::start(move |path| match path {
            "/image.png" => ok("image/png", &png),
            "/hop1" => redirect("302 Found", "/image.png"),
            "/hop2" => redirect("301 Moved Permanently", "hop1"),
            "/hop3" => redirect("307 Temporary Redirect", "/hop2"),
            "/hop4" => redirect("308 Permanent Redirect", "/hop3"),
            "/to-ftp" => redirect("302 Found", "ftp://example.com/image.png"),
            "/to-file" => redirect("303 See Other", "file:///etc/passwd"),
            _ => Reply::Close(response("302 Found", &[], b"")),
        })
        .await;
        let connector = image_connector(&loopback_policy());

        let image = fetch_image(&connector, server.url("/hop3")).await.unwrap();
        assert_eq!((image.width(), image.height()), (2, 2));

        assert!(matches!(
            fetch_image(&connector, server.url("/hop4")).await,
            Err(MediaError::TooManyRedirects { limit: 3 })
        ));
        for path in ["/to-ftp", "/to-file"] {
            assert!(
                matches!(
                    fetch_image(&connector, server.url(path)).await,
                    Err(MediaError::UnsupportedRedirect)
                ),
                "{path}"
            );
        }
        assert!(matches!(
            fetch_image(&connector, server.url("/no-location")).await,
            Err(MediaError::InvalidRedirect)
        ));
    }

    /// A connection failure on a redirect hop is reported without naming the
    /// redirect target, which the client never sent.
    #[tokio::test]
    async fn transport_failures_do_not_reveal_redirect_targets() {
        // Keep the destination reserved while refusing every response. Other
        // tests can bind ephemeral ports without receiving this redirect.
        let target_server = TestServer::start(|_| Reply::Close(Vec::new())).await;
        let target = target_server.url("/internal-target");
        let server = TestServer::start(move |_| redirect("302 Found", &target)).await;
        let connector = image_connector(&loopback_policy());

        let error = fetch_image(&connector, server.url("/image.png"))
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(matches!(error, MediaError::Transport { .. }), "{message}");
        assert!(
            !message.contains(&target_server.address.port().to_string()),
            "{message}"
        );
        assert!(!message.contains("internal-target"), "{message}");
    }

    /// Under the public-only policy, a URL whose host is or resolves to a
    /// loopback address is refused before any connection, however the host
    /// is spelled; allowing private destinations reaches the same server. A
    /// connector without remote media refuses every http(s) URL.
    #[tokio::test]
    async fn non_public_hosts_are_refused_before_connecting() {
        let png = encoded_image(ImageFormat::Png, 1, 1);
        let server = TestServer::start(move |_| ok("image/png", &png)).await;
        let port = server.address.port();
        let public_only = [
            image_connector(&ImageFetchOptions::default()),
            condition_connector(None, 64),
        ];

        for connector in &public_only {
            for host in [
                "127.0.0.1",
                "127.1",
                "2130706433",
                "0x7f.0.0.1",
                "0.0.0.0",
                "[::1]",
                "[::ffff:127.0.0.1]",
                "localhost",
            ] {
                let result = fetch_one(connector, &format!("http://{host}:{port}/image.png")).await;
                assert!(
                    matches!(result, Err(MediaError::NonPublicAddress)),
                    "{host}: {result:?}"
                );
            }
        }
        let remote_disabled = MediaConnector::from_media_options(&MediaOptions {
            allow_remote: false,
            ..MediaOptions::default()
        })
        .unwrap();
        assert!(matches!(
            fetch_one(&remote_disabled, &server.url("/image.png")).await,
            Err(MediaError::RemoteDisabled)
        ));
        assert_eq!(server.connections(), 0);

        fetch_image(
            &image_connector(&loopback_policy()),
            server.url("/image.png"),
        )
        .await
        .unwrap();
        assert_eq!(server.connections(), 1);
    }
}
