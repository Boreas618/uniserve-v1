"""Generates `data/engram.json`: the n-gram hashing DeepSeek-V4.1 is tested by.

The values are computed by the checkpoint's own reference code,
`inference/engram.py` of deepseek-ai/DeepSeek-V4.1-Flash-1017 at commit
1bae0e1088993b9b82de08f59de5be5a34e42170, run as it is:

- `EngramLayout.from_args`: the prime moduli of every head of every n-gram
  length of each Engram layer;
- `compute_hash_multipliers`: the multipliers numpy's generator draws for each
  layer;
- `build_compressed_token_map`: the normalized vocabulary, stated by its size,
  the padding token's normalized id, a SHA-256 of the map as little-endian
  32-bit integers and a few of its entries;
- `NgramHashState.forward`: the table rows of every token of a fixed token
  sequence, prefilled at once and again as a prefill of eight tokens followed
  by single-token steps, which the cache carries across.

Run with an interpreter that has torch, numpy, sympy and transformers, from
the repository root:

    cd crates/models/tests/it/deepseek_v41
    python generate_engram.py <snapshot of the release> data/engram.json
"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

SNAPSHOT = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
sys.path.insert(0, str(SNAPSHOT / "inference"))

from engram import (  # noqa: E402
    EngramLayout,
    NgramHashState,
    build_compressed_token_map,
    compute_hash_multipliers,
)

#: Text whose tokens normalize alike in several spellings, then special and
#: rare tokens of the vocabulary.
TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "THE QUICK Brown Fox!  the  quick\tbrown"
)
EXTRA = [0, 1, 2, 128799, 129279, 3, 4, 5]

#: The release the values are of.
RELEASE = (
    "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    "@1bae0e1088993b9b82de08f59de5be5a34e42170"
)


def main():
    config = json.loads((SNAPSHOT / "config.json").read_text())["text_config"]
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT)
    tokens = tokenizer.encode(TEXT, add_special_tokens=False) + EXTRA

    args = SimpleNamespace(
        engram_layer_ids=tuple(config["engram_layer_ids"]),
        engram_num_embeddings=tuple(config["engram_num_embeddings"]),
        engram_max_ngram_size=config["engram_max_ngram_size"],
        engram_vocab_size=config["engram_vocab_size"],
        engram_n_heads=config["engram_n_heads"],
        engram_head_dim=config["engram_head_dim"],
        engram_pad_id=config["engram_pad_token_id"],
        engram_compressed_vocab_size=config["engram_compressed_vocab_size"],
        max_batch_size=1,
        max_seq_len=len(tokens),
    )
    layout = EngramLayout.from_args(args)
    token_map, vocab_size = build_compressed_token_map(tokenizer)
    multipliers = compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, vocab_size
    )
    digest = hashlib.sha256(
        np.asarray(token_map, dtype="<i4").tobytes()
    ).hexdigest()

    ids = torch.tensor([tokens])
    whole = NgramHashState(args, layout, tokenizer)(ids, 0)
    stepped = NgramHashState(args, layout, tokenizer)
    chunks = [stepped(ids[:, :8], 0)]
    for position in range(8, len(tokens)):
        chunks.append(stepped(ids[:, position : position + 1], position))
    stepped = torch.cat(chunks, dim=1)
    assert torch.equal(whole, stepped), "the prefill and the steps hash alike"

    reference = {
        "release": RELEASE,
        "layer_ids": list(layout.layer_ids),
        "multipliers": multipliers.tolist(),
        "primes": [[list(heads) for heads in layer] for layer in layout.primes],
        "vocabulary": len(token_map),
        "normalized_vocabulary": vocab_size,
        "pad": token_map[args.engram_pad_id],
        "token_map_sha256": digest,
        "token_map_samples": {
            str(i): token_map[i] for i in tokens + [100, 1000, 50000, 129000]
        },
        "tokens": tokens,
        # [layer][token][column]
        "rows": whole[0].permute(1, 0, 2).tolist(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(reference, indent=1) + "\n")


if __name__ == "__main__":
    main()
