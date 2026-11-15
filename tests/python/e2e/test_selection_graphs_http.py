"""Captured selections honour every sampling constraint.

Qwen3 serves from CUDA graphs, prefilling a prompt of about a thousand tokens
in chunks of 256 tokens. A forward's token selection, the logprobs and top
candidates it reports and the scores of its prompt are captured graphs whose
inputs are tensors: token sets as bitmasks, each request's row of logit
biases, its sampler and parameters, and each prompt token's target, the prompt
token after it; a chunk's last token scores the next chunk's first.

Constraints hold exactly under graphs: allowed tokens, a dominating logit
bias, bad words of one token and of several, reported top logprobs that rank
the sampled token, and prompt scores for every prompt token after the first.
The bad words are texts of the served checkpoint's tokenizer, which the case
loads from the checkpoint's repository at its pinned revision.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from uniserve_testing import chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    # The module serves one deployment for all of its cases.
    pytest.mark.gpus(1),
]

SERVED_MODEL = "qwen3-selection"
CHUNK = 256
# About a thousand tokens of records, then a question about them: four
# prefill chunks.
PROMPT = chat.long_prompt(52)
MAX_TOKENS = 16

# Greedy, reporting five top tokens and three per prompt position.
RANKED = {
    **chat.GREEDY,
    "logprobs": True,
    "top_logprobs": 5,
    "prompt_logprobs": 3,
}
# Greedy, reporting each selected token and no prompt scores.
SELECTED = {**chat.GREEDY, "logprobs": True}
# Seeded sampling through the general sampler.
SAMPLED = {
    "temperature": 0.8,
    "top_k": 300,
    "top_p": 0.95,
    "min_p": 0.02,
    "seed": 11,
    "logprobs": True,
    "top_logprobs": 1,
}


@pytest.fixture(scope="module")
def graphs(module_deployments: Deployments) -> Iterator[str]:
    """The base URL of the module's deployment, which steps at most two
    chunks of tokens and prefills a prompt one chunk at a time.
    """  # noqa: D205
    with module_deployments.serve(
        "--served-model-name",
        SERVED_MODEL,
        "--set",
        f"max_num_batched_tokens={2 * CHUNK}",
        "--set",
        f"scheduler_config.chunked_prefill_size={CHUNK}",
        model=CHECKPOINTS["qwen3"].model,
    ) as server:
        yield server.base_url


def _chat(base_url: str, **fields: Any) -> dict[str, Any]:
    """The answer to the prompt, with `fields` beside its defaults: at most
    ``MAX_TOKENS`` tokens, each reported by its token id.
    """  # noqa: D205
    request = chat.body(
        SERVED_MODEL, PROMPT, MAX_TOKENS, return_tokens_as_token_ids=True
    )
    return chat.complete(base_url, {**request, **fields})


def _ids(answer: dict[str, Any]) -> list[int]:
    """The ids of the tokens `answer` selected (``token_id:<id>``)."""
    return [
        int(token.removeprefix("token_id:"))
        for token, _ in chat.ranked_tokens(answer)
    ]


def _word(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> str | None:
    """The text of `ids` when it encodes back to exactly them, as a bad word.

    Special tokens decode to no text, so a word holding one never encodes
    back to its ids.
    """
    text = tokenizer.decode(ids, skip_special_tokens=True)
    if tokenizer.encode(text, add_special_tokens=False) != ids:
        return None
    return text


def test_constraints_hold_under_graphs(graphs: str) -> None:
    greedy = _chat(graphs, **RANKED)
    for entry in greedy["choices"][0]["logprobs"]["content"]:
        # Greedy over the model's distribution selects its top token.
        top = entry["top_logprobs"]
        assert top[0]["token"] == entry["token"]
        assert top[0]["logprob"] == entry["logprob"]
        assert [candidate["logprob"] for candidate in top] == sorted(
            (candidate["logprob"] for candidate in top), reverse=True
        )

    # Every prompt token after the first is scored, among at most three
    # other candidates.
    prompt = greedy["prompt_logprobs"]
    assert len(prompt) == greedy["usage"]["prompt_tokens"]
    assert prompt[0] is None
    assert all(1 <= len(position) <= 4 for position in prompt[1:])
    assert all(
        value <= 0.0 for position in prompt[1:] for value in position.values()
    )

    # Tasks restricted to two tokens select only those.
    selected = _ids(greedy)
    allowed = sorted(set(selected[:4]))[:2]
    restricted = _chat(
        graphs,
        **SAMPLED,
        allowed_token_ids=allowed,
        ignore_eos=True,
        max_completion_tokens=8,
    )
    assert set(_ids(restricted)) <= set(allowed)

    # A bias of a hundred makes its token the greedy selection.
    forced = selected[-1]
    biased = _chat(
        graphs,
        **RANKED,
        logit_bias={str(forced): 100.0},
        max_completion_tokens=2,
    )
    assert _ids(biased)[0] == forced


def test_greedy_constraints_hold_under_graphs(graphs: str) -> None:
    checkpoint = CHECKPOINTS["qwen3"]
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.repository, revision=checkpoint.revision
    )
    special = set(tokenizer.added_tokens_decoder)
    greedy = _ids(_chat(graphs, **SELECTED, ignore_eos=True))

    # A greedy request restricted to one token selects it at every step.
    only = _chat(
        graphs,
        **SELECTED,
        allowed_token_ids=[greedy[-1]],
        ignore_eos=True,
        max_completion_tokens=4,
    )
    assert set(_ids(only)) == {greedy[-1]}

    # A one-token bad word: the greedy reply's first ordinary token that is a
    # word of its own is never selected.
    single = next(
        token
        for token in greedy
        if token not in special and _word(tokenizer, [token]) is not None
    )
    banned = _ids(
        _chat(
            graphs,
            **SELECTED,
            ignore_eos=True,
            bad_words=[_word(tokenizer, [single])],
        )
    )
    assert single not in banned
    assert banned != greedy

    # A two-token bad word that the greedy reply completes: its first token
    # may still be selected, never followed by its second.
    pair = next(
        list(pair)
        for pair in zip(greedy, greedy[1:])
        if _word(tokenizer, list(pair)) is not None
    )
    reply = _ids(
        _chat(
            graphs,
            **SELECTED,
            ignore_eos=True,
            bad_words=[_word(tokenizer, pair)],
        )
    )
    assert all([a, b] != pair for a, b in zip(reply, reply[1:]))
    assert reply != greedy
