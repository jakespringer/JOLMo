"""Smoke tests for HF-teacher support in
``olmo_core.train.train_module.transformer.teacher``.

Uses ``sshleifer/tiny-gpt2`` (~1 MB) to keep CI light. Assumes the
``transformers`` package is installed.
"""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.train_module.transformer.teacher import (
    HFTeacherModel,
    TeacherModelConfig,
)


TINY_HF = "sshleifer/tiny-gpt2"


def test_teacher_config_xor_validation_neither_set():
    with pytest.raises(OLMoConfigurationError):
        TeacherModelConfig()


def test_teacher_config_xor_validation_both_set():
    # We don't need a real TransformerConfig — the XOR check fires in
    # __post_init__ before the dataclass semantics trip over anything.
    # But we DO need a truthy ``model`` value to make has_jolmo True.
    class _StubTransformerConfig:  # noqa: D401 - trivial stub
        pass

    with pytest.raises(OLMoConfigurationError):
        TeacherModelConfig(
            model=_StubTransformerConfig(),
            checkpoint_path="/dev/null",
            hf_model_name=TINY_HF,
        )


def test_teacher_config_hf_mode_builds():
    cfg = TeacherModelConfig(
        hf_model_name=TINY_HF,
        autocast_precision=DType.bfloat16,
    )
    teacher = cfg.build(
        student_world_mesh=None,
        device=torch.device("cpu"),
        student_max_sequence_length=128,
        rank_microbatch_size=128,
    )
    assert isinstance(teacher, HFTeacherModel)
    assert teacher.vocab_size > 0
    assert teacher.d_model > 0
    # Auto-populated from HF config.max_position_embeddings.
    assert teacher.trained_sequence_length is not None
    assert teacher.trained_sequence_length > 0


def test_hf_teacher_forward_shapes():
    cfg = TeacherModelConfig(
        hf_model_name=TINY_HF,
        autocast_precision=None,  # plain fp32 on CPU to keep test trivial
    )
    teacher = cfg.build(
        student_world_mesh=None,
        device=torch.device("cpu"),
        student_max_sequence_length=128,
        rank_microbatch_size=128,
    )
    S = 16
    ids = torch.randint(0, teacher.vocab_size, (1, S))

    logits = teacher.logits(ids)
    assert logits.shape == (1, S, teacher.vocab_size)

    h = teacher.hidden_states(ids)
    assert h.shape == (1, S, teacher.d_model)


def test_hf_teacher_silently_drops_doc_lens_kwargs():
    """HF models don't accept JOLMo's doc_lens / max_doc_lens /
    cache_leftpad kwargs — HFTeacherModel must drop them without
    raising."""
    cfg = TeacherModelConfig(hf_model_name=TINY_HF)
    teacher = cfg.build(
        student_world_mesh=None,
        device=torch.device("cpu"),
        student_max_sequence_length=128,
        rank_microbatch_size=128,
    )
    ids = torch.randint(0, teacher.vocab_size, (1, 16))
    # Pass arbitrary kwargs; should be silently ignored.
    logits = teacher.logits(
        ids, doc_lens=torch.tensor([16]), max_doc_lens=[16], cache_leftpad=None,
    )
    assert logits.shape[:2] == (1, 16)


def test_teacher_config_hf_rejects_dp_config():
    """Setting dp_config with an HF teacher must raise — HF teachers
    stay replicated in v1."""
    from olmo_core.train.train_module.transformer.config import (
        TransformerDataParallelConfig,
    )
    from olmo_core.distributed.parallel import DataParallelType

    with pytest.raises(OLMoConfigurationError):
        TeacherModelConfig(
            hf_model_name=TINY_HF,
            dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp),
        )
