#!/usr/bin/env python3
"""Check the parts of `dump_vae_blocks.py` that need no checkpoint and no device.

The dumper's own logic - the probe catalogue, the parse back from a probe name, the fixed input's
shape, the channel count's cross-check - is host arithmetic over names, and a mistake there costs a run
on the machine that has the weights. This script exercises exactly that half and nothing else, the way
`artifacts/h3-diagnosis/selfcheck/` exercises the packing dumper's own code paths:

    .venv/bin/python crates/models/minimax-h3/tools/run_dump_vae_blocks.py

It is a diagnosis of the dumper, never an acceptance gate (`specs/worker.md` 10), and it deliberately
says nothing about whether the reference's loader accepts what the dumper hands it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dump_vae_blocks as dumper  # noqa: E402


def check_catalogue() -> None:
    """Every catalogued probe parses back into its component, path and description."""
    count = 0
    for component, probes in dumper.CATALOGUE.items():
        for path, description in probes:
            probe = f"{component}:{path}"
            parsed = dumper.plan(probe)
            assert parsed == (component, path, description), probe
            count += 1
    assert count == sum(len(probes) for probes in dumper.CATALOGUE.values())
    print(f"catalogue: {count} probes parse back to their component, path and description")


def check_refusals() -> None:
    """A name outside the catalogue is refused, and the message names the catalogue as the source."""
    for bad in ("video_decoder", "video_decoder:", "audio_decoder:decoder.nope", "nothing:x"):
        try:
            dumper.plan(bad)
        except SystemExit as stop:
            assert "list" in str(stop), stop
        else:
            raise AssertionError(f"{bad!r} was accepted")
    print("refusals: four names outside the catalogue are refused, each pointing at --list")


def check_shapes() -> None:
    """The fixed inputs have the rank and the extents their component's layout states."""
    video = dumper.latent_shape("video_decoder")
    audio = dumper.latent_shape("audio_decoder")
    assert len(video) == 5 and video[2:] == (
        dumper.VIDEO_LATENT_FRAMES,
        dumper.VIDEO_LATENT_SIDE,
        dumper.VIDEO_LATENT_SIDE,
    ), video
    assert len(audio) == 3 and audio[2] == dumper.AUDIO_LATENT_FRAMES, audio
    # The channel count is saturated from the checkpoint, so the fixed input leaves it zero and the
    # dumper fills it before the run - a fixed input that named a channel count would compare the
    # wrong thing on a checkpoint that changed it.
    assert video[1] == 0 and audio[1] == 0
    print(f"shapes: video {video} and audio {audio}, channels saturated from the checkpoint")


def main() -> int:
    check_catalogue()
    check_refusals()
    check_shapes()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
