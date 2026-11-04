//! MiniMax-H3's noise schedule: the sigma points of each modality, their shifts, and the heads a
//! parallel-decoding student fuses.
//!
//! A checkpoint *states* one schedule for all of its DiT partitions and two shifts, one per
//! modality ([`ScheduleSpec`], read at the loading boundary). This module *evaluates* that statement
//! into the sequences a step loop advances. The base release evaluates the released full-step grid:
//! 50 sigma points with video shift 12 and audio shift 3. A FastH3 DMD student evaluates its trained
//! rungs on the 1000-step training schedule, and an OmniRef student (a parallel-decoding, "PDD",
//! student) evaluates the block nodes of a fine grid. The schedule is fixed by the checkpoint: no
//! request can change it.
//!
//! ## The arithmetic
//!
//! The formulas are the released ones (`uniserve/diffusion/schedule.py`), evaluated in the same
//! precision so the host-side sequences agree with them:
//!
//! ```text
//! uniform grid   u_i = linspace(1, 0, points) in FP32;  sigma_i = shift(u_i), deduplicated in FP32
//! DMD ladder     u_i = rung / num_train_timesteps in FP64;  sigma_i = shift(u_i), then the clean
//!                endpoint 0, materialized in FP32
//! block grid     u   = linspace(max_t, 0, intervals + 1) in FP64;  sigma_j = f_shift(u[node_j])
//!                rounded once to FP32; the fine intervals' integration weights stay FP64
//! shift(u)       s u / (1 + (s - 1) u)          (the rectified-flow time shift)
//! fixed point    f_s(u) = s u M / (u (s - 1) + M),  M = max_t, which maps M to itself
//! network time   t_i = 1 - sigma_i in FP32
//! ```
//!
//! Every resulting sequence decreases to the clean endpoint `0`, so a schedule of `n` sigma points
//! has `n - 1` network evaluations. Consecutive equal FP32 values collapse into one point, as the
//! released grid's `unique_consecutive` does, so both modalities can be checked for one step count.
//!
//! ## What this module is not
//!
//! It is host arithmetic: it holds no tensor and reaches no operator. The latents a step advances
//! belong to the executor, which is why the crate also publishes [`Schedule::steps`] - the
//! executor has to be told which evaluations to schedule, and `nn::Denoiser` has no way to ask
//! (`TO exec`).

use nn::Step;

use crate::config::{ScheduleGrid, ScheduleSpec};
use crate::{Error, Result};

/// One of the two timelines a denoiser advances.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Modality {
    /// The video latents.
    Video,
    /// The audio latents.
    Audio,
}

impl Modality {
    /// Both modalities, video first, as the released denoiser enumerates them.
    pub const ALL: [Self; 2] = [Self::Video, Self::Audio];

    /// The name the checkpoint's sidecars and the schedule's tables use.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

/// One checkpoint's schedule, evaluated: its sigma points per modality and the block boundaries of
/// a parallel-decoding student.
///
/// Construction validates the statement once ([`ScheduleSpec`] checks its own domain) and checks
/// that the two modalities evaluate the same number of times. Every later use - the steps a runner
/// schedules, the level a draw starts from, the integration weights of a student's heads - is then
/// total.
#[derive(Clone, PartialEq, Debug)]
pub struct Schedule {
    spec: ScheduleSpec,
}

impl Schedule {
    /// Evaluate the schedule a checkpoint states.
    ///
    /// Fails when the two modalities do not evaluate the same number of times: one runner advances
    /// both latents with one step counter, and the released checkpoint aligns them exactly.
    pub fn new(spec: &ScheduleSpec) -> Result<Self> {
        let schedule = Self { spec: spec.clone() };
        schedule.num_steps()?;
        Ok(schedule)
    }

    /// The statement this schedule was evaluated from.
    pub const fn spec(&self) -> &ScheduleSpec {
        &self.spec
    }

    /// The shift of one modality.
    pub fn shift(&self, modality: Modality) -> f32 {
        match modality {
            Modality::Video => self.spec.shifts.video,
            Modality::Audio => self.spec.shifts.audio,
        }
    }

    /// The sigma points of one modality: decreasing, starting from its noise level and ending at
    /// the clean endpoint `0`.
    ///
    /// Consecutive equal FP32 values - which a strong shift can produce - collapse into one point,
    /// as the released `unique_consecutive` does.
    pub fn sigmas(&self, modality: Modality) -> Vec<f32> {
        let shift = f64::from(self.shift(modality));
        match &self.spec.grid {
            ScheduleGrid::Uniform { points } => uniform_sigmas(*points, shift),
            ScheduleGrid::Dmd {
                rungs,
                num_train_timesteps,
            } => ladder_sigmas(rungs, f64::from(*num_train_timesteps), shift),
            ScheduleGrid::Blocks {
                intervals,
                nodes,
                max_t,
            } => block_sigmas(*intervals, nodes, f64::from(*max_t), shift),
        }
    }

    /// The network times `1 - sigma` of one modality, in FP32.
    pub fn timesteps(&self, modality: Modality) -> Vec<f32> {
        self.sigmas(modality)
            .into_iter()
            .map(|sigma| 1.0 - sigma)
            .collect()
    }

    /// How many sigma points one modality has.
    pub fn points(&self, modality: Modality) -> usize {
        self.sigmas(modality).len()
    }

    /// How many network evaluations this schedule takes; the same for both modalities.
    ///
    /// The statement's own count ([`ScheduleSpec::steps`]) is what a checkpoint declares; this is
    /// what the evaluated grid actually runs after collapsed points, and the number a runner reads.
    pub fn num_steps(&self) -> Result<u32> {
        let video = self.points(Modality::Video);
        let audio = self.points(Modality::Audio);
        if video != audio {
            return Err(Error::checkpoint(format!(
                "H3 video and audio schedules must align: {video} video sigma points against \
                 {audio} audio sigma points"
            )));
        }
        Ok((video - 1) as u32)
    }

    /// The noise level one modality starts from: its first sigma point.
    ///
    /// A parallel-decoding student scales its initial draw by this level, because its first node is
    /// not unit noise; the released and distilled grids start from unit noise.
    pub fn start_sigma(&self, modality: Modality) -> Result<f32> {
        self.sigmas(modality)
            .first()
            .copied()
            .ok_or_else(|| Error::checkpoint("H3 schedule has no sigma point".to_owned()))
    }

    /// The steps of one modality, as the executor schedules them: evaluation `i` runs from
    /// `sigma[i]` to `sigma[i + 1]`.
    pub fn steps(&self, modality: Modality) -> Result<Vec<Step>> {
        let sigmas = self.sigmas(modality);
        let count = self.num_steps()?;
        Ok(sigmas
            .windows(2)
            .enumerate()
            .map(|(index, pair)| Step {
                index: index as u32,
                count,
                sigma: pair[0],
                sigma_next: pair[1],
            })
            .collect())
    }

    /// The normalized FP64 integration weights of one modality's evaluation `block`.
    ///
    /// A parallel-decoding student predicts one output head per fine interval of its block; the
    /// block's weights, normalized by their sum, combine the heads into the block's one prediction.
    /// The sum is the block's node-sigma increment, so an ordinary Euler step between the two node
    /// sigmas applies that prediction. The other grids have one head, so a block of them is refused.
    pub fn block_weights(&self, modality: Modality, block: u32) -> Result<Vec<f64>> {
        let ScheduleGrid::Blocks {
            intervals,
            nodes,
            max_t,
        } = &self.spec.grid
        else {
            return Err(Error::checkpoint(format!(
                "an H3 {} schedule has one output head per evaluation, so it has no block weights",
                self.spec.grid.kind()
            )));
        };
        let blocks = nodes.len() - 1;
        if block as usize >= blocks {
            return Err(Error::checkpoint(format!(
                "H3 parallel-decoding block {block} is outside the schedule's {blocks} blocks"
            )));
        }
        let weights = block_increments(
            *intervals,
            f64::from(*max_t),
            f64::from(self.shift(modality)),
        );
        let start = nodes[block as usize] as usize;
        let end = nodes[block as usize + 1] as usize;
        let total: f64 = weights[start..end].iter().sum();
        if !total.is_finite() || total == 0.0 {
            return Err(Error::checkpoint(
                "H3 parallel-decoding block integration weight must be nonzero and finite"
                    .to_owned(),
            ));
        }
        Ok(weights[start..end]
            .iter()
            .map(|weight| weight / total)
            .collect())
    }
}

/// The rectified-flow time shift in FP64.
fn shifted(value: f64, shift: f64) -> f64 {
    shift * value / (1.0 + (shift - 1.0) * value)
}

/// The rectified-flow time shift in FP32, the arithmetic the released grid evaluates.
fn shifted32(value: f32, shift: f32) -> f32 {
    shift * value / (1.0 + (shift - 1.0) * value)
}

/// Drop consecutive equal FP32 values, as `torch.unique_consecutive` does.
fn deduplicate(values: Vec<f32>) -> Vec<f32> {
    let mut points: Vec<f32> = Vec::with_capacity(values.len());
    for value in values {
        if points
            .last()
            .is_none_or(|previous| previous.to_bits() != value.to_bits())
        {
            points.push(value);
        }
    }
    points
}

/// `linspace(start, end, count)` in FP64: `start + i · (end - start) / (count - 1)`.
fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    let step = (end - start) / (count - 1) as f64;
    (0..count)
        .map(|index| start + index as f64 * step)
        .collect()
}

/// The released uniform grid's sigma points for one modality.
fn uniform_sigmas(points: u32, shift: f64) -> Vec<f32> {
    let shift32 = shift as f32;
    let values = linspace(1.0, 0.0, points as usize)
        .into_iter()
        .map(|value| shifted32(value as f32, shift32))
        .collect();
    deduplicate(values)
}

/// A DMD ladder's sigma points for one modality: the shifted rungs, then the clean endpoint.
fn ladder_sigmas(rungs: &[u32], num_train_timesteps: f64, shift: f64) -> Vec<f32> {
    let mut sigmas: Vec<f32> = rungs
        .iter()
        .map(|rung| shifted(f64::from(*rung) / num_train_timesteps, shift) as f32)
        .collect();
    sigmas.push(0.0);
    deduplicate(sigmas)
}

/// A parallel-decoding student's node sigmas for one modality.
fn block_sigmas(intervals: u32, nodes: &[u32], max_t: f64, shift: f64) -> Vec<f32> {
    let grid = linspace(max_t, 0.0, intervals as usize + 1);
    nodes
        .iter()
        .map(|node| fixed_point_shift(grid[*node as usize], shift, max_t) as f32)
        .collect()
}

/// The shift `f_s(u) = s u M / (u (s - 1) + M)` whose fixed point is `M`.
fn fixed_point_shift(value: f64, shift: f64, max_t: f64) -> f64 {
    if shift == 1.0 {
        return value;
    }
    shift * value * max_t / (value * (shift - 1.0) + max_t)
}

/// The fine grid's integration weights `f(u[j + 1]) - f(u[j])`, without the cancellation the direct
/// difference suffers: the rational difference is `s M² (b - a) / (D(a) D(b))` with
/// `D(u) = M + (s - 1) u`.
fn block_increments(intervals: u32, max_t: f64, shift: f64) -> Vec<f64> {
    let grid = linspace(max_t, 0.0, intervals as usize + 1);
    grid.windows(2)
        .map(|pair| {
            let (start, end) = (pair[0], pair[1]);
            if shift == 1.0 {
                return end - start;
            }
            shift * max_t * max_t * (end - start)
                / ((max_t + (shift - 1.0) * start) * (max_t + (shift - 1.0) * end))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Shifts;

    const VIDEO_SHIFT: f32 = 12.0;
    const AUDIO_SHIFT: f32 = 3.0;

    fn shifts() -> Shifts {
        Shifts {
            video: VIDEO_SHIFT,
            audio: AUDIO_SHIFT,
        }
    }

    fn base() -> Schedule {
        let spec = ScheduleSpec::uniform(50, shifts()).expect("the base checkpoint's schedule");
        Schedule::new(&spec).expect("an aligned base schedule")
    }

    fn dmd() -> Schedule {
        // FastVideo-FastH3-8-Step-V2's contract: eight rungs, video shift 10, audio shift 3.
        let spec = ScheduleSpec::dmd(
            vec![999, 874, 749, 624, 500, 375, 250, 125],
            1000.0,
            Shifts {
                video: 10.0,
                audio: 3.0,
            },
        )
        .expect("a DMD ladder");
        Schedule::new(&spec).expect("an aligned DMD schedule")
    }

    fn blocks() -> Schedule {
        // FastVideo-FastH3-Omni-8-Step-V1's contract: 32 intervals, nine nodes, max_t 0.999.
        let spec = ScheduleSpec::blocks(
            32,
            vec![0, 4, 8, 12, 16, 20, 24, 28, 32],
            0.999,
            Shifts {
                video: 12.0,
                audio: 3.0,
            },
        )
        .expect("a block grid");
        Schedule::new(&spec).expect("an aligned block schedule")
    }

    #[test]
    fn the_base_grid_has_fifty_sigma_points_and_unit_start() {
        let schedule = base();
        let video = schedule.sigmas(Modality::Video);
        assert_eq!(video.len(), 50);
        assert_eq!(video[0], 1.0);
        assert_eq!(*video.last().expect("the clean endpoint"), 0.0);
        assert_eq!(schedule.num_steps().expect("aligned modalities"), 49);
        let times = schedule.timesteps(Modality::Video);
        assert_eq!(times[0], 0.0);
        assert_eq!(*times.last().expect("the clean endpoint"), 1.0);
    }

    #[test]
    fn sigma_decreases_from_the_start_to_the_clean_endpoint() {
        for schedule in [base(), dmd(), blocks()] {
            for modality in Modality::ALL {
                let sigmas = schedule.sigmas(modality);
                assert!(
                    sigmas.windows(2).all(|pair| pair[0] > pair[1]),
                    "{} sigma points are not strictly decreasing",
                    modality.as_str()
                );
            }
        }
    }

    #[test]
    fn a_shift_moves_the_first_points_apart_while_the_endpoints_stay() {
        let schedule = base();
        let video = schedule.sigmas(Modality::Video);
        let audio = schedule.sigmas(Modality::Audio);
        // The first point is 1 for every shift, and the last is the clean endpoint.
        assert_eq!(video[0], audio[0]);
        assert_eq!(video.last(), audio.last());
        // A stronger shift concentrates the trajectory nearer the start.
        assert!(video[1] > audio[1]);
    }

    #[test]
    fn a_uniform_shift_of_one_keeps_the_linear_grid() {
        let spec = ScheduleSpec::uniform(
            5,
            Shifts {
                video: 1.0,
                audio: 1.0,
            },
        )
        .expect("the identity shift");
        let schedule = Schedule::new(&spec).expect("an aligned schedule");
        assert_eq!(
            schedule.sigmas(Modality::Video),
            vec![1.0, 0.75, 0.5, 0.25, 0.0]
        );
    }

    #[test]
    fn a_strong_shift_collapses_consecutive_equal_points() {
        // `s u / (1 + (s - 1) u)` tends to 1 as `s` grows, so the grid's interior points collapse
        // into one, exactly as the released `unique_consecutive` collapses them.
        let spec = ScheduleSpec::uniform(
            4,
            Shifts {
                video: 1e30,
                audio: 1e30,
            },
        )
        .expect("a strong but valid shift");
        let schedule = Schedule::new(&spec).expect("an aligned schedule");
        assert_eq!(schedule.sigmas(Modality::Video), vec![1.0, 0.0]);
        assert_eq!(schedule.num_steps().expect("aligned modalities"), 1);
    }

    #[test]
    fn a_dmd_ladder_evaluates_its_rungs_and_the_clean_endpoint() {
        let schedule = dmd();
        assert_eq!(schedule.num_steps().expect("aligned modalities"), 8);
        let video = schedule.sigmas(Modality::Video);
        assert_eq!(video.len(), 9);
        assert_eq!(*video.last().expect("the clean endpoint"), 0.0);
        // A rung is unshifted on the training schedule and shifted once here.
        assert_eq!(video[0], shifted(999.0 / 1000.0, 10.0) as f32);
    }

    #[test]
    fn a_block_grid_has_one_evaluation_per_block() {
        let schedule = blocks();
        assert_eq!(schedule.num_steps().expect("aligned modalities"), 8);
        // The shift's fixed point is max_t, so the first node is the same level for both
        // modalities.
        let video = schedule
            .start_sigma(Modality::Video)
            .expect("a start level");
        let audio = schedule
            .start_sigma(Modality::Audio)
            .expect("a start level");
        assert!((video - 0.999).abs() < 1e-6);
        assert!((audio - 0.999).abs() < 1e-6);
    }

    #[test]
    fn a_block_grids_weights_normalize_to_one_per_block() {
        let schedule = blocks();
        for modality in Modality::ALL {
            for block in 0..8 {
                let weights = schedule
                    .block_weights(modality, block)
                    .expect("a block of the grid");
                assert_eq!(weights.len(), 4);
                let total: f64 = weights.iter().sum();
                assert!((total - 1.0).abs() < 1e-12);
                assert!(weights.iter().all(|weight| weight.is_finite()));
            }
            assert!(schedule.block_weights(modality, 8).is_err());
        }
    }

    #[test]
    fn the_other_grids_have_no_block_weights() {
        assert!(base().block_weights(Modality::Video, 0).is_err());
        assert!(dmd().block_weights(Modality::Video, 0).is_err());
    }

    #[test]
    fn steps_carry_each_evaluation_interval() {
        let steps = base().steps(Modality::Video).expect("the base steps");
        assert_eq!(steps.len(), 49);
        for (index, step) in steps.iter().enumerate() {
            assert_eq!(step.index, index as u32);
            assert_eq!(step.count, 49);
            assert!(step.sigma > step.sigma_next);
        }
        assert_eq!(steps[48].sigma_next, 0.0);
    }

    #[test]
    fn the_schedule_keeps_the_statement_it_was_built_from() {
        let schedule = dmd();
        assert_eq!(schedule.spec().steps(), 8);
        assert_eq!(schedule.shift(Modality::Video), 10.0);
        assert_eq!(schedule.points(Modality::Audio), 9);
    }
}
