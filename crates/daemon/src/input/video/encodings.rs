//! Cache keys for video encoder inputs resolved by request preprocessing.

use models::video::{ConditionMedia, ConditionRole, ConditionVision, Conditioning, VideoCondition};
use proto::request::EncoderInputs;
use serde::Serialize;
use storage::key::Digest;

/// The digest of what the condition's encodings read: the media's
/// content (`content`, the fetched media content key) and the
/// condition's plan, but not where the media are published.
fn condition_key(
    condition: &VideoCondition,
    content: &[u8; 32],
) -> Result<[u8; 32], serde_json::Error> {
    /// The plan of a condition without its media's object name.
    #[derive(Serialize)]
    struct Plan<'a> {
        role: &'a ConditionRole,
        bytes: u64,
        media: &'a ConditionMedia,
        vision: &'a Option<ConditionVision>,
        latent_units: &'a [u32],
        audio_rows: u32,
    }
    let plan = serde_json::to_vec(&Plan {
        role: &condition.role,
        bytes: condition.source.bytes,
        media: &condition.media,
        vision: &condition.vision,
        latent_units: &condition.latent_units,
        audio_rows: condition.audio_rows,
    })?;
    let mut hasher = Digest::new(b"video-condition");
    hasher.bytes(content);
    hasher.bytes(&plan);
    Ok(hasher.finish())
}

/// The digests of what each encoder computation of the request reads,
/// from its presentation's tokens (`prompt_token_ids`) and the content key
/// of each condition's fetched media in request order (`contents`).
///
/// A condition's digest covers its media's content and its whole plan
/// but where the media are published (the locator's object name, which
/// differs between publications of the same media): its role, how its
/// media are decoded, what the conditioner reads of it and the rows it
/// encodes to. The vision encoding reads the conditions the conditioner
/// reads, in presentation order; the text encoding the presentation's
/// tokens and those conditions' vision features; a visual unit's latent
/// encoding its condition and its index among the condition's units;
/// the audio encoding every condition with an audio track.
///
/// # Errors
///
/// Fails only when serializing a condition's plan fails, which these
/// plain values do not.
///
/// The caller supplies one content key per condition, in condition order.
pub(super) fn encoder_inputs(
    conditioning: &Conditioning,
    prompt_token_ids: &[u32],
    contents: &[[u8; 32]],
) -> Result<EncoderInputs, serde_json::Error> {
    let conditions: Vec<[u8; 32]> = conditioning
        .conditions
        .iter()
        .zip(contents)
        .map(|(condition, content)| condition_key(condition, content))
        .collect::<Result<_, _>>()?;

    let combined = |label: &str, digests: &mut dyn Iterator<Item = [u8; 32]>| {
        let mut hasher = Digest::new(label.as_bytes());
        let mut any = false;
        for digest in digests {
            hasher.bytes(&digest);
            any = true;
        }
        any.then(|| hasher.finish())
    };
    let vision = combined(
        "uniserve-video-vision/1\0",
        &mut conditioning
            .conditions
            .iter()
            .zip(&conditions)
            .filter(|(condition, _)| condition.vision.is_some())
            .map(|(_, digest)| *digest),
    );
    let audio = combined(
        "uniserve-video-audio/1\0",
        &mut conditioning
            .conditions
            .iter()
            .zip(&conditions)
            .filter(|(condition, _)| condition.audio_rows > 0)
            .map(|(_, digest)| *digest),
    );

    let mut text = Digest::new(b"video-text");
    text.number(prompt_token_ids.len() as u64);
    for token in prompt_token_ids {
        text.number(u64::from(*token));
    }
    match vision {
        Some(digest) => {
            text.number(1);
            text.bytes(&digest);
        }
        None => text.number(0),
    }

    let video_units = conditioning
        .conditions
        .iter()
        .zip(&conditions)
        .flat_map(|(condition, digest)| {
            (0..condition.latent_units.len() as u32).map(move |unit| {
                let mut hasher = Digest::new(b"video-unit");
                hasher.bytes(digest);
                hasher.number(u64::from(unit));
                hasher.finish()
            })
        })
        .collect();
    Ok(EncoderInputs {
        text: text.finish(),
        vision,
        video_units,
        audio,
    })
}
