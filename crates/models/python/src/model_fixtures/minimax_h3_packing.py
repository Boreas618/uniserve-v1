"""Generates the packed-sequence fixtures from the reference implementation.

The reference is vLLM-Omni's MiniMax-H3 packing (`packed_sequence`), its
denoise branch (`denoise_loop`, for the timestep group of every row and the
prefix segments of the native layouts) and its video sparse attention tiling
(`attention.fastvideo_h3`), imported from this project's `reference`
dependency group and run on `--device`. The packing draws nothing, so its
tensors are those of the reference functions alone:

    minimax-h3-packing-fixtures \
        --out crates/models/tests/it/minimax_h3/fixtures/packing

It writes `cases.json` (the requests, in the model's own terms) and
`expected.safetensors` (the reference's tensors of each request) under
`--out`.
"""

from __future__ import annotations

import json
import math

import torch
from safetensors.torch import save_file
from vllm_omni.diffusion.models.minimax_h3 import denoise_loop, packed_sequence
from vllm_omni.diffusion.models.minimax_h3.attention import fastvideo_h3

from model_fixtures import run

# The request rules the cases are stated in (`processing::minimax_h3`): 24
# frames and 40 audio latent steps a second, 800 samples a step, `17n + 5`
# frames of which each 17-frame clip encodes to five latent frames and the last
# clip drops three, a 16x spatial compression.
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_HOP = 800
CLIP_FRAMES = 17
CLIP_LATENTS = 5
TOKEN_DROP = 3
LATENT_COMPRESSION = 16


def latent_frames(frames: int) -> int:
    """Latent frames of `frames` frames (`FrameClips::latent_frames`)."""
    if frames <= 1:
        return frames
    return math.ceil(frames / CLIP_FRAMES) * CLIP_LATENTS - TOKEN_DROP


def audio_latents(frames: int) -> int:
    """Audio latent steps of `frames` generated frames, half to even."""
    return round(frames * AUDIO_LATENTS_PER_SECOND / FPS)


def condition(
    role: str,
    *,
    frames: int = 0,
    height: int = 0,
    width: int = 0,
    samples: int = 0,
):
    """One condition of a request, as `nn::ConditionShape` states it."""
    return {
        "role": role,
        "frames": frames,
        "height": height,
        "width": width,
        "audio_samples": samples,
    }


def reference_block(shape: dict) -> dict:
    """The reference block of a condition (`packed_sequence.py:357-361`)."""
    steps = math.ceil(shape["audio_samples"] / AUDIO_HOP)
    if not shape["width"]:
        return {"kind": "audio", "ref_audio_t": steps}

    frames = latent_frames(shape["frames"])
    raster = {
        "latent_h": shape["height"] // LATENT_COMPRESSION,
        "latent_w": shape["width"] // LATENT_COMPRESSION,
    }
    if frames == 1:
        return {"kind": "image", **raster}
    return {
        "kind": "video_audio" if steps else "video",
        "ref_audio_t": steps,
        "latent_t": frames,
        **raster,
    }


def pack(case: dict) -> dict:
    """The reference's packed sequence of `case`."""
    target = {
        "text_len": case["text_tokens"],
        "latent_t": latent_frames(case["frames"]),
        "latent_h": case["height"] // LATENT_COMPRESSION,
        "latent_w": case["width"] // LATENT_COMPRESSION,
        "audio_t": audio_latents(case["frames"]),
    }
    keyframes = [
        shape for shape in case["conditions"] if shape["role"] != "reference"
    ]
    references = [
        shape for shape in case["conditions"] if shape["role"] == "reference"
    ]
    indices = [
        0 if shape["role"] == "first_frame" else -1 for shape in keyframes
    ]

    # `t2va` and `fl2va` are the native layout; a request with references is
    # the block layout, whose keyframes are timeline guides at the frame they
    # anchor.
    if not references:
        return packed_sequence.minimax_h3_packed_sequence(
            **target,
            include_keyframe_cond=bool(keyframes),
            keyframe_frame_indices=indices or None,
            frame_count=case["frames"] if keyframes else None,
        )

    guides = [
        {
            "kind": "image",
            "latent_h": target["latent_h"],
            "latent_w": target["latent_w"],
            "frame_index": case["frames"] - 1 if index else 0,
        }
        for index in indices
    ]
    return packed_sequence.minimax_h3_packed_sequence_ref2va_blocks(
        **target,
        ref_blocks=[reference_block(shape) for shape in references],
        guide_blocks=guides,
    )


def expected(case: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """The reference's tensors of `case`, over the rows its document uses."""
    packed = pack(case)
    used = int(packed["cu_seqlens"][1])
    text_len = case["text_tokens"]

    # The prompt's rows take the text encoder's tags: a row of a vision span is
    # a video row (`pipeline_minimax_h3.py:2160-2161`).
    text_tags = torch.ones(text_len, dtype=torch.long)
    for first, rows in case["vision_spans"]:
        text_tags[first : first + rows] = 0
    tags = packed["token_tags"].clone()
    tags[packed["text_pos"]] = text_tags

    branch = denoise_loop.MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(text_len, 1),
        token_tags=tags,
        device=device,
    )

    # Each row's timestep, with the four levels standing for their groups: the
    # generated video 0, the generated audio 1, a visual condition 2, an audio
    # reference 3.
    groups = torch.zeros(branch.seq_len, device=device)
    branch.fill_timesteps(
        groups,
        t_video=0.0,
        t_audio=1.0,
        imgvid_cond_timestep=2.0,
        audio_ref_cond_timestep=3.0,
    )

    # The rows of the conditioning a row gathers. The conditioning is the
    # model's own buffer: the text capacity, then the condition rows in packed
    # order, then one row of zeros, which every generated row gathers. The
    # reference states which rows are which.
    target_video = branch.img_pos[branch.update_mask]
    target_audio = branch.audio_pos[branch.audio_update_mask]
    conditions = (
        torch.cat(
            [
                branch.img_pos[~branch.update_mask],
                branch.audio_pos[~branch.audio_update_mask],
            ]
        )
        .sort()
        .values
    )
    prefix_index = torch.full(
        (used,), case["text_rows"] + case["condition_rows"], device=device
    )
    prefix_index[:text_len] = torch.arange(text_len, device=device)
    prefix_index[conditions] = case["text_rows"] + torch.arange(
        conditions.numel(), device=device
    )

    tensors = {
        "positions": packed["img_position_ids"][:used].to(torch.float32),
        "tags": tags[:used],
        "groups": groups[:used],
        "prefix_index": prefix_index,
        # The dense layout leads with the generated rows: the video, the audio,
        # then the prefix.
        "dense_rows": torch.cat(
            [
                target_video,
                target_audio,
                torch.arange(int(target_audio[0]), device=device),
            ]
        ),
        "used": torch.tensor([used]),
    }

    if case["tile_rows"] is not None:
        tensors.update(tiles(case, branch, device))
    return {
        name: tensor.to(torch.int32) if name != "positions" else tensor
        for name, tensor in tensors.items()
    }


def tiles(case: dict, branch, device: torch.device) -> dict[str, torch.Tensor]:
    """The reference tiles of `case`: every tile and each video span."""
    layout = branch.static_kwargs["video_token_layout"]
    prefix = tuple(
        branch.static_kwargs["packed_seq_params"]["vsa_prefix_segments"]
    )
    target = layout.video_spans[-1]

    if case["tile_rows"] == 64:
        partition, sizes, _, _, prefix_blocks, video_blocks = (
            fastvideo_h3._get_h3_tile_metadata(
                prefix, tuple(target.latent_grid), device
            )
        )
        spans = ((prefix_blocks, prefix_blocks + video_blocks),)
    else:
        regions = tuple(
            (span.start, tuple(span.latent_grid)) for span in layout.video_spans
        )
        partition, sizes, _, _, _, spans = (
            fastvideo_h3._get_h3_multiregion_tile_metadata(
                prefix, regions, device
            )
        )

    # The tiles a query of each video keeps
    # (`attention/fastvideo_h3.py:216-229`).
    target_rate = 1.0 - case["sparsity"]
    keep = case["reference_keep"]
    reference_rate = target_rate if keep is None else 1.0 - (1.0 - keep)
    rates = (reference_rate,) * (len(spans) - 1) + (target_rate,)
    kept = [
        max(1, min(math.ceil(rate * (end - start)), end - start))
        for (start, end), rate in zip(spans, rates, strict=True)
    ]

    return {
        "tile_partition": partition,
        "tile_sizes": sizes,
        "tile_regions": torch.tensor(
            [
                [start, end, count]
                for (start, end), count in zip(spans, kept, strict=True)
            ],
            device=device,
        ),
    }


def request(
    name: str,
    *,
    size: tuple[int, int],
    frames: int,
    text_tokens: int,
    conditions: tuple[dict, ...] = (),
    vision_spans: tuple[tuple[int, int], ...] = (),
    tile_rows: int | None = None,
    sparsity: float = 0.0,
    reference_keep: float | None = None,
) -> dict:
    """One request and the capacities of the layout it is laid out in."""
    unit = tile_rows or 64

    def capacity(rows: int) -> int:
        return math.ceil(rows / unit) * unit

    # Each condition's audio and its pixels fill capacity units of their own;
    # one more unit is left unused, as a deployment's capacity is wider than
    # most of its requests.
    taken = 0
    for shape in conditions:
        block = reference_block(shape) if shape["role"] == "reference" else None
        audio = 2 * math.ceil(shape["audio_samples"] / AUDIO_HOP)
        patches = (shape["height"] // 32) * (shape["width"] // 32)

        if (
            block is not None
            and block["kind"] in ("video", "video_audio")
            and tile_rows
        ):
            grid = (
                latent_frames(shape["frames"]),
                shape["height"] // 32,
                shape["width"] // 32,
            )
            shape_of_tile = (4, 4, 4) if tile_rows == 64 else (4, 4, 8)
            visual = tile_rows * math.prod(
                math.ceil(extent / tile)
                for extent, tile in zip(grid, shape_of_tile, strict=True)
            )
        else:
            visual = capacity(latent_frames(shape["frames"]) * patches)
        taken += capacity(audio) + visual

    return {
        "name": name,
        "width": size[0],
        "height": size[1],
        "frames": frames,
        "text_tokens": text_tokens,
        "conditions": list(conditions),
        "vision_spans": [list(span) for span in vision_spans],
        "text_rows": capacity(text_tokens) + unit,
        "condition_rows": taken + unit if conditions else 0,
        "tile_rows": tile_rows,
        "sparsity": sparsity,
        "reference_keep": reference_keep,
    }


def cases() -> list[dict]:
    """The requests: every task, reference blocks, odd sizes and limits."""
    first = condition("first_frame", frames=1, height=64, width=96)
    last = condition("last_frame", frames=1, height=64, width=96)
    image = condition("reference", frames=1, height=64, width=128)
    track = condition("reference", samples=40_000)
    clip = condition(
        "reference", frames=30, height=64, width=64, samples=40_000
    )

    return [
        request("t2va_small", size=(96, 64), frames=22, text_tokens=37),
        request("t2va_one_token", size=(160, 96), frames=39, text_tokens=1),
        # The longest output a bound is taken over: 297 latent frames, 1680
        # audio steps.
        request("t2va_longest", size=(64, 32), frames=1008, text_tokens=200),
        request(
            "fl2va_first",
            size=(96, 64),
            frames=22,
            text_tokens=50,
            conditions=(first,),
            vision_spans=((10, 6),),
        ),
        # Thirty-two latent frames: past sixteen the reference's two summation
        # orders of the timeline's span differ in the last place of FP64.
        request(
            "fl2va_last",
            size=(96, 64),
            frames=107,
            text_tokens=20,
            conditions=(last,),
        ),
        request(
            "fl2va_both",
            size=(96, 96),
            frames=362,
            text_tokens=64,
            conditions=(
                condition("first_frame", frames=1, height=96, width=96),
                condition("last_frame", frames=1, height=96, width=96),
            ),
            vision_spans=((3, 9), (20, 9)),
        ),
        request(
            "ref2va_image",
            size=(96, 64),
            frames=22,
            text_tokens=30,
            conditions=(image,),
        ),
        request(
            "ref2va_audio",
            size=(96, 64),
            frames=22,
            text_tokens=30,
            conditions=(track,),
        ),
        request(
            "ref2va_video",
            size=(96, 64),
            frames=22,
            text_tokens=30,
            conditions=(clip,),
        ),
        request(
            "ref2va_blocks",
            size=(160, 96),
            frames=56,
            text_tokens=45,
            conditions=(
                image,
                condition("reference", frames=17, height=96, width=64),
                track,
                condition(
                    "reference", frames=51, height=64, width=96, samples=8_000
                ),
                condition("reference", frames=1, height=160, width=96),
            ),
        ),
        request(
            "ref2va_guided",
            size=(96, 64),
            frames=22,
            text_tokens=30,
            conditions=(image, first, track, last),
        ),
        # The tiled layouts: partial tiles on every axis of the video's grid.
        request(
            "tiled_t2va",
            size=(352, 224),
            frames=39,
            text_tokens=70,
            tile_rows=64,
            sparsity=0.9,
        ),
        request(
            "tiled_fl2va",
            size=(160, 96),
            frames=22,
            text_tokens=64,
            conditions=(
                condition("first_frame", frames=1, height=96, width=160),
            ),
            tile_rows=64,
            sparsity=0.8,
        ),
        request(
            "tiled_ref2va",
            size=(288, 160),
            frames=39,
            text_tokens=45,
            conditions=(
                image,
                condition("reference", frames=30, height=160, width=96),
                track,
                condition(
                    "reference", frames=51, height=96, width=288, samples=8_000
                ),
            ),
            tile_rows=128,
            sparsity=0.8,
            reference_keep=0.5,
        ),
        request(
            "tiled_ref2va_one_rate",
            size=(288, 160),
            frames=22,
            text_tokens=128,
            conditions=(
                condition("reference", frames=30, height=160, width=96),
            ),
            tile_rows=128,
            sparsity=0.75,
        ),
    ]


def main() -> None:
    arguments = run.options(__doc__.splitlines()[0])
    requests = cases()

    tensors = {}
    for case in requests:
        for name, tensor in expected(case, arguments.device).items():
            tensors[f"{case['name']}.{name}"] = tensor.cpu().contiguous()

    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(requests, indent=2) + "\n")
    save_file(tensors, out / "expected.safetensors")

    print(f"{len(requests)} requests, {len(tensors)} tensors")


if __name__ == "__main__":
    main()
