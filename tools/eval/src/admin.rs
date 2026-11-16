//! Calls to a deployment's admin listener, an HTTP/1.1 server on a Unix socket.

use std::path::Path;

use anyhow::{Context, Result};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;

/// One request to the listener at `socket`: its status and its body as text
/// (invalid UTF-8 replaced). A body is sent as JSON.
///
/// # Errors
///
/// When the socket does not answer or the exchange fails.
pub async fn request(
    socket: &Path,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> Result<(u16, String)> {
    let stream = tokio::net::UnixStream::connect(socket)
        .await
        .with_context(|| format!("connecting to {}", socket.display()))?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(stream)).await?;
    let driver = tokio::spawn(connection);
    let mut builder = hyper::Request::builder()
        .method(method)
        .uri(path)
        .header(hyper::header::HOST, "localhost");
    if body.is_some() {
        builder = builder.header(hyper::header::CONTENT_TYPE, "application/json");
    }
    let request = builder.body(Full::new(Bytes::copy_from_slice(body.unwrap_or_default())))?;
    let response = sender.send_request(request).await?;
    let status = response.status().as_u16();
    let data = response.into_body().collect().await?.to_bytes();
    driver.abort();
    Ok((status, String::from_utf8_lossy(&data).into_owned()))
}
