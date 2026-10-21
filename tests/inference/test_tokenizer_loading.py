"""Regression tests for how the pinned transformers loads tokenizers from checkpoints."""

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

TEXT = "\nOkay, so I have this question.\nLet me think."


@pytest.fixture
def qwen2_checkpoint_with_llama_tokenizer_class(tmp_path: Path) -> Path:
    """A checkpoint shaped like deepseek-ai/DeepSeek-R1-Distill-Qwen-*.

    The model is qwen2 with a byte-level BPE tokenizer, but tokenizer_config.json
    names LlamaTokenizerFast.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator([TEXT] * 10, trainer=trainer)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "LlamaTokenizerFast"})
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    return tmp_path


def test_qwen2_checkpoint_with_llama_tokenizer_class_round_trips(
    qwen2_checkpoint_with_llama_tokenizer_class: Path,
) -> None:
    # transformers 5.6-5.7 trust the Llama class and drop the byte-level
    # pre-tokenizer and decoder, so whitespace disappears on encode and decoded
    # text keeps the Ġ/Ċ markers (issue #241). Fixed upstream in 5.8.0.
    tokenizer = AutoTokenizer.from_pretrained(qwen2_checkpoint_with_llama_tokenizer_class)
    ids = tokenizer.encode(TEXT, add_special_tokens=False)
    assert tokenizer.decode(ids) == TEXT
