//! Dynamo's TCP request plane, the worker's side (`lib/runtime/src/pipeline/network`): the
//! request frames a frontend sends to a worker's endpoint, acknowledged one by one, and the
//! response stream the worker opens back to the frontend's response server.
//!
//! A request frame is the endpoint's path (`u16` length, big-endian), the request's headers as a
//! JSON object (`u16` length) and its payload (`u32` length): a two-part message whose header is
//! the control message ([`wire::Control`]) and whose body is the request in the codec the
//! instance states, JSON here. Every frame is acknowledged as it is read, with an empty `u32`
//! length, or with the reason it is refused (`Server unavailable: ...`).
//!
//! A two-part message is its header's and its body's lengths and a checksum (`u64` each,
//! big-endian; the checksum 0, which a receiver does not check) followed by the header and the
//! body. The response stream is one TCP connection to the frontend's response server: the
//! handshake naming the stream's subject, the prologue, one data message per output, the data
//! message that completes the stream, and the sentinel. The frontend cancels the request on the
//! same connection, with a header message of `"stop"` or `"kill"`, after which the stream sends
//! nothing more.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use super::wire;

/// The largest request frame a worker reads (Dynamo's default `DYN_TCP_MAX_MESSAGE_SIZE`).
const MAX_FRAME: usize = 32 << 20;

/// A request a frontend sent to an endpoint.
pub(super) struct Incoming {
    /// The endpoint's path, which names the worker ([`wire::endpoint_path`]).
    pub(super) path: String,
    pub(super) control: wire::Control,
    pub(super) request: wire::Request,
}

/// What takes each request: `Err` with the reason a frame is refused, which its acknowledgement
/// states.
pub(super) type Dispatch = Arc<dyn Fn(Incoming) -> Result<(), String> + Send + Sync>;

/// Serves `listener`'s connections, each frame of each handed to `dispatch`, until `stop`.
pub(super) async fn serve(listener: TcpListener, dispatch: Dispatch, stop: CancellationToken) {
    loop {
        let accepted = tokio::select! {
            () = stop.cancelled() => return,
            accepted = listener.accept() => accepted,
        };
        let Ok((stream, peer)) = accepted else {
            continue;
        };
        let dispatch = Arc::clone(&dispatch);
        let stop = stop.clone();
        tokio::spawn(async move {
            tokio::select! {
                () = stop.cancelled() => {}
                result = connection(stream, &dispatch) => {
                    if let Err(error) = result {
                        debug!(%peer, "a request-plane connection ended: {error:#}");
                    }
                }
            }
        });
    }
}

/// Reads the frames of one frontend connection and acknowledges each.
async fn connection(mut stream: TcpStream, dispatch: &Dispatch) -> Result<()> {
    loop {
        let mut length = [0u8; 2];
        if stream.read_exact(&mut length).await.is_err() {
            return Ok(());
        }
        let path = read_text(&mut stream, usize::from(u16::from_be_bytes(length))).await?;
        // The headers (the path again, the request's id, its trace) state nothing the worker
        // serves by: they are read past.
        stream.read_exact(&mut length).await?;
        read_bytes(&mut stream, usize::from(u16::from_be_bytes(length))).await?;
        let mut payload_length = [0u8; 4];
        stream.read_exact(&mut payload_length).await?;
        let payload_length = u32::from_be_bytes(payload_length) as usize;
        if payload_length > MAX_FRAME {
            bail!("a request frame of {payload_length} bytes is over {MAX_FRAME}");
        }
        let payload = read_bytes(&mut stream, payload_length).await?;

        let refused = match parts(&payload) {
            Ok((header, body)) => match incoming(path, header, body) {
                Ok(incoming) => dispatch(incoming).err(),
                Err(error) => Some(format!("{error:#}")),
            },
            Err(error) => Some(format!("{error:#}")),
        };
        match refused {
            None => stream.write_all(&0u32.to_be_bytes()).await?,
            Some(reason) => {
                let reason = format!("Server unavailable: {reason}");
                let length = u32::try_from(reason.len()).unwrap_or(u32::MAX);
                stream.write_all(&length.to_be_bytes()).await?;
                stream.write_all(reason.as_bytes()).await?;
            }
        }
    }
}

async fn read_bytes(stream: &mut TcpStream, length: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; length];
    stream.read_exact(&mut bytes).await?;
    Ok(bytes)
}

async fn read_text(stream: &mut TcpStream, length: usize) -> Result<String> {
    String::from_utf8(read_bytes(stream, length).await?).context("an endpoint path is not UTF-8")
}

/// The header and body of a two-part message.
fn parts(message: &[u8]) -> Result<(&[u8], &[u8])> {
    let length = |at: usize| -> Result<usize> {
        let bytes: [u8; 8] = message
            .get(at..at + 8)
            .context("a two-part message shorter than its lengths")?
            .try_into()?;
        usize::try_from(u64::from_be_bytes(bytes)).context("a two-part length")
    };
    let (header, body) = (length(0)?, length(8)?);
    let rest = &message[24..];
    if rest.len() != header + body {
        bail!(
            "a two-part message states {header} and {body} bytes and holds {}",
            rest.len()
        );
    }
    Ok(rest.split_at(header))
}

fn incoming(path: String, header: &[u8], body: &[u8]) -> Result<Incoming> {
    let control: wire::Control =
        serde_json::from_slice(header).context("the request's control message")?;
    if control
        .payload_codec
        .as_deref()
        .is_some_and(|codec| codec != "json")
    {
        bail!("the worker reads JSON requests, as its instance states");
    }
    let request: wire::Request = serde_json::from_slice(body).context("the request")?;
    Ok(Incoming {
        path,
        control,
        request,
    })
}

/// A two-part message of `header` and `body`.
pub(super) fn message(header: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 + header.len() + body.len());
    out.extend_from_slice(&(header.len() as u64).to_be_bytes());
    out.extend_from_slice(&(body.len() as u64).to_be_bytes());
    out.extend_from_slice(&0u64.to_be_bytes());
    out.extend_from_slice(header);
    out.extend_from_slice(body);
    out
}

/// Streams `outputs` (each a data frame's JSON) to the frontend at `control`'s response
/// server until the channel closes, then completes the stream. A `"stop"` or `"kill"` from the
/// frontend cancels `cancel` and ends the stream there, as Dynamo's streams end; the stream also
/// ends, unfinished, when `cancel` is cancelled otherwise.
pub(super) async fn respond(
    control: &wire::Control,
    mut outputs: mpsc::Receiver<Value>,
    cancel: CancellationToken,
) -> Result<()> {
    if control.connection_info.transport != "tcp_server" {
        bail!(
            "the response plane `{}` is not TCP",
            control.connection_info.transport
        );
    }
    let home: wire::Home = serde_json::from_str(&control.connection_info.info)
        .context("the response stream's connection")?;
    if home.stream_type != "response" || home.context != control.id {
        bail!("the response connection names another stream");
    }
    let stream = TcpStream::connect(&home.address).await.with_context(|| {
        format!(
            "connecting to the frontend's response server {}",
            home.address
        )
    })?;
    let (mut reader, mut writer) = stream.into_split();

    // The frontend's control messages: a cancellation, which ends the stream.
    let watching = cancel.clone();
    let watch = tokio::spawn(async move {
        loop {
            let mut lengths = [0u8; 24];
            if reader.read_exact(&mut lengths).await.is_err() {
                return;
            }
            let header = u64::from_be_bytes(lengths[..8].try_into().unwrap_or_default());
            let body = u64::from_be_bytes(lengths[8..16].try_into().unwrap_or_default());
            let mut bytes = vec![0u8; (header + body) as usize];
            if reader.read_exact(&mut bytes).await.is_err() {
                return;
            }
            let control: Option<String> = serde_json::from_slice(&bytes[..header as usize]).ok();
            if matches!(control.as_deref(), Some("stop" | "kill")) {
                watching.cancel();
                return;
            }
        }
    });

    let handshake = json!({"subject": home.subject, "stream_type": "response"});
    writer
        .write_all(&message(&serde_json::to_vec(&handshake)?, &[]))
        .await?;
    writer
        .write_all(&message(&serde_json::to_vec(&json!({"error": null}))?, &[]))
        .await?;
    loop {
        let output = tokio::select! {
            () = cancel.cancelled() => {
                watch.abort();
                return Ok(());
            }
            output = outputs.recv() => output,
        };
        match output {
            Some(output) => {
                writer
                    .write_all(&message(&[], &serde_json::to_vec(&output)?))
                    .await?;
            }
            None => break,
        }
    }
    writer
        .write_all(&message(&[], &serde_json::to_vec(&wire::complete())?))
        .await?;
    writer
        .write_all(&message(&serde_json::to_vec(&json!("sentinel"))?, &[]))
        .await?;
    writer.flush().await?;
    // The frontend closes the stream once it read the sentinel.
    let closed = tokio::time::timeout(std::time::Duration::from_secs(10), watch).await;
    if closed.is_err() {
        warn!(
            subject = home.subject,
            "the frontend kept the response stream open after its end"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{message, parts};

    #[test]
    fn a_two_part_message_splits_into_its_header_and_body() {
        let encoded = message(b"head", b"body!");
        assert_eq!(&encoded[16..24], &[0; 8], "no checksum");
        assert_eq!(parts(&encoded).unwrap(), (&b"head"[..], &b"body!"[..]));
        assert!(parts(&encoded[..encoded.len() - 1]).is_err());
    }
}
