//! The arena floor: the model's plan is the least a device's share may offer.
//!
//! Two runners ask this question - a denoise bucket and a decode bucket - and one rule answers it,
//! so the two cannot drift into reading an absent plan and a refused one differently.

use crate::{Error, Result};

use crate::plan::Bucket;

/// What a model's answer to its own arena question means for the device's share.
///
/// Three outcomes, and the first two are the ones that must not be confused:
///
/// - **the model states no plan** - the trait's `Unsupported` - so there is no floor, and the pool
///   is the only figure; a capture that outgrows it refuses at its own launch by name. `Ok(None)`.
/// - **the model's plan fits the share**: `Ok(Some(needed))`.
/// - **the plan refuses for a reason of its own** (a canvas the deployment does not serve, rows past
///   its ladder) **or does not fit**: a refusal naming the bucket and the numbers involved. A
///   refusal is never turned into an absence, because an absent floor and a floor the share cannot
///   meet are opposite facts about the deployment.
pub(crate) fn checked(
    bucket: Bucket,
    answer: nn::Result<u64>,
    offered: u64,
) -> Result<Option<u64>> {
    let needed = match answer {
        Ok(needed) => needed,
        Err(nn::Error::Unsupported(_)) => return Ok(None),
        Err(other) => {
            return Err(Error::Invalid(format!(
                "bucket {bucket} states no arena for its own warm request: {other}"
            )));
        }
    };
    if needed > offered {
        return Err(Error::Invalid(format!(
            "bucket {bucket} needs {needed} B of arena for one evaluation and this device's share \
             is {offered} B: the model's plan is the floor, so a share below it serves nothing - \
             raise the device's graph share or drop a layout from the deployment"
        )));
    }
    Ok(Some(needed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn::capability::{LatentShape, Layout};

    /// The floor rule's four answers, without a device, a pool or a model: what a plan states and
    /// what the share offers are the rule's only inputs.
    #[test]
    fn a_plan_that_fits_a_share_that_does_not_a_refusal_and_an_absence_are_four_answers() {
        let bucket = Bucket::denoise(
            1,
            Layout {
                video: LatentShape {
                    channels: 16,
                    frames: 1,
                    height: 2,
                    width: 2,
                },
                audio: None,
                text_rows: 64,
                condition_rows: 0,
                condition_elements: 0,
            },
        );
        // The trait's own absence: there is no floor, and the pool is the only figure.
        assert_eq!(
            checked(
                bucket,
                Err(nn::Error::Unsupported("no plan".into())),
                1 << 30
            )
            .unwrap(),
            None
        );
        // A plan the share meets answers the number it needed.
        assert_eq!(
            checked(bucket, Ok(1 << 20), 1 << 30).unwrap(),
            Some(1 << 20)
        );
        // A plan the share does not meet is a refusal naming both numbers.
        let short = checked(bucket, Ok(1 << 31), 1 << 30).unwrap_err();
        let text = short.to_string();
        assert!(text.contains("2147483648"), "{text}");
        assert!(text.contains("1073741824"), "{text}");
        assert!(text.contains("floor"), "{text}");
        // **A refusal with a subject is not an absence**: a deployment whose own warm request the
        // model does not serve must not be read as one whose arena fits.
        let refused = checked(bucket, Err(nn::Error::Unsupported("no plan".into())), 0).unwrap();
        assert_eq!(refused, None, "an absence is not a shortfall, even at zero");
        let other = checked(
            bucket,
            Err(nn::Error::Shape {
                owner: "minimax-h3",
                problem: "the canvas is not one this deployment serves".into(),
            }),
            u64::MAX,
        )
        .unwrap_err();
        let text = other.to_string();
        assert!(text.contains("not one this deployment serves"), "{text}");
    }
}
