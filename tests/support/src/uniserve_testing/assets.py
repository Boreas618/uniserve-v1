"""The released checkpoints and adapters the end-to-end cases serve.

Every asset is a Hugging Face repository at a pinned commit, and a folder of
it where the repository holds several (``docs/configuration.md``: a model is
``<owner>/<name>[@<revision>][/<folder>]``). A case names the asset by its
entry in :data:`CHECKPOINTS`, and the server it starts resolves it into the
standard Hugging Face cache as every deployment does; a test that reads one
of the asset's files itself fetches it through ``huggingface_hub`` into the
same cache. A host without network access populates the cache beforehand
and sets the standard ``HF_HUB_OFFLINE``; a mirror is named by the standard
``HF_ENDPOINT``, and a private repository's token by ``HF_TOKEN``.

An asset that no repository publishes yet has no repository in its entry:
a case that needs it is skipped with the entry's reason.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import huggingface_hub
import pytest


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """A released checkpoint, adapter or input set: its repository, the
    commit it is pinned at, the folder of the repository that holds it, and
    the repository's kind (``model`` or ``dataset``, as the Hugging Face Hub
    names them).

    ``repository`` is ``None`` for an asset no repository publishes, and
    ``unpublished`` then states what the asset is and why it has no entry.
    """  # noqa: D205

    repository: str | None
    revision: str = ""
    folder: str = ""
    kind: str = "model"
    unpublished: str = ""

    @property
    def model(self) -> str:
        """The asset as a package's ``model`` and ``lora`` name it.

        Skips the calling case when no repository publishes the asset.
        """
        if self.repository is None:
            pytest.skip(self.unpublished)
        model = f"{self.repository}@{self.revision}"
        return f"{model}/{self.folder}" if self.folder else model

    def file(self, name: str) -> Path:
        """The asset's file `name`, fetched into the Hugging Face cache.

        Skips the calling case when no repository publishes the asset.
        """
        if self.repository is None:
            pytest.skip(self.unpublished)
        return Path(
            huggingface_hub.hf_hub_download(
                self.repository,
                name,
                subfolder=self.folder or None,
                revision=self.revision,
                repo_type=self.kind,
            )
        )


#: Every asset an end-to-end case serves, by the name the cases use.
CHECKPOINTS: dict[str, Checkpoint] = {
    # Qwen3-MoE in BF16, the text cases' model.
    "qwen3": Checkpoint(
        "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    ),
    # Qwen3.5-MoE in BF16, whose checkpoint carries a multi-token-prediction
    # drafter: the speculation cases' model.
    "qwen3.5": Checkpoint(
        "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754"
    ),
    # The same model's files with other weight values, which a weight
    # update moves a deployment to.
    "qwen3-other-weights": Checkpoint(
        None,
        unpublished="no repository publishes a second Qwen3-30B-A3B "
        "checkpoint with other weight values (the weight-update cases' "
        "target): publish one, then name it here",
    ),
    # FastH3, the distilled MiniMax-H3 the video cases serve.
    "fasth3": Checkpoint(
        "FastVideo/FastVideo-FastH3-8-Step-V2",
        "3da2ddfe1954d9cda4c05b643dc0f26007a655c5",
    ),
    # The MiniMax-H3 diffusers root.
    "minimax-h3": Checkpoint(
        "MiniMaxAI/MiniMax-H3", "42ed227ee7df40d41602854ae760620d6eb651fe"
    ),
    # A FastH3 OmniRef export, which reads the parts it omits from the
    # MiniMax-H3 revision it pins.
    "fasth3-omniref": Checkpoint(
        "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
        "88d512c5e7cc38dda499fbb65688ce58611f9c80",
    ),
    # A FastH3 Preview v1 adapter, and the distilled student whose inference
    # contract it is trained for: the adapter release ships the weights
    # alone (:func:`adapter_directory`).
    "fasth3-adapter": Checkpoint(
        "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
        "bcf40ca6f457ed66f8badf13514943e390205fca",
        folder="vsa-datafree",
    ),
    "fasth3-adapter-student": Checkpoint(
        "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
        "5ea076f35b84da4c3c82217112fa733d8eea2ae1",
    ),
    # The reference media of the MiniMax-H3 protocol's referenced workloads
    # (`docs/minimax_h3/evaluation/workloads`).
    "minimax-h3-inputs": Checkpoint(
        None,
        kind="dataset",
        unpublished="the MiniMax-H3 protocol's reference media are "
        "published under no dataset repository: publish them, then name "
        "the repository here",
    ),
}

#: The file a video family reads an adapter's weights from, and the student
#: contract it reads beside them (``lora``, ``docs/configuration.md``).
ADAPTER_WEIGHTS = "adapter_model.safetensors"
STUDENT_CONTRACT = "fastvideo_inference.json"


def adapter_directory(directory: Path) -> Path:
    """A directory holding the FastH3 adapter's weights beside its
    student's inference contract, which a package's ``lora_dir`` serves.

    The adapter release ships its weights alone, and the family reads the
    contract of the student the adapter is trained for beside them, so the
    directory links both files from the Hugging Face cache, each fetched at
    its pinned revision.
    """  # noqa: D205
    directory.mkdir(parents=True, exist_ok=True)
    weights = CHECKPOINTS["fasth3-adapter"].file(ADAPTER_WEIGHTS)
    contract = CHECKPOINTS["fasth3-adapter-student"].file(STUDENT_CONTRACT)
    (directory / ADAPTER_WEIGHTS).symlink_to(weights)
    (directory / STUDENT_CONTRACT).symlink_to(contract)
    return directory
