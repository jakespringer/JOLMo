"""
Frozen-teacher wrapper used by :class:`TransformerDistillTrainModule` and
:class:`TransformerRepLossTrainModule`.

Two modes are supported:

- **JOLMo-native teacher** (:class:`TeacherModel`): a JOLMo
  :class:`~olmo_core.nn.transformer.Transformer` loaded from a distributed
  checkpoint, parallelized with the same primitives the student uses, and
  held in eval mode with ``requires_grad=False``. A forward hook on the
  final block captures per-token hidden states for the rep-loss pathway.

- **HuggingFace teacher** (:class:`HFTeacherModel`): any model loadable via
  ``transformers.AutoModelForCausalLM.from_pretrained``. The HF model stays
  replicated per rank (no FSDP wrap in v1), runs inference-only, and uses
  ``output_hidden_states=True`` to expose per-token hidden states natively
  — no forward hook is needed.

Both modes present the same public surface to the train modules:
``logits(input_ids, **kwargs)``, ``hidden_states(input_ids, **kwargs)``, and
the ``vocab_size`` / ``d_model`` / ``trained_sequence_length`` attributes.
"""

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh

from olmo_core.config import Config, DType
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import gc_cuda

from .common import parallelize_model
from .config import TransformerDataParallelConfig

log = logging.getLogger(__name__)

__all__ = ["TeacherModelConfig", "TeacherModel", "HFTeacherModel"]


@dataclass
class TeacherModelConfig(Config):
    """
    Configuration for a frozen teacher model.

    Two mutually exclusive modes:

    - **JOLMo-native**: set ``model`` (a :class:`TransformerConfig`) and
      ``checkpoint_path``. The teacher is built, parallelized with the
      student's world mesh, and loaded from the distributed checkpoint.
    - **HuggingFace**: set ``hf_model_name`` (e.g.
      ``"microsoft/harrier-oss-v1-270m"``). The model is loaded via
      ``transformers.AutoModelForCausalLM.from_pretrained`` and held
      replicated per rank in v1. ``model``, ``checkpoint_path``, and
      ``dp_config`` must be ``None`` in this mode.

    Exactly one of ``model`` / ``hf_model_name`` must be set;
    ``__post_init__`` rejects configurations that violate this.
    """

    # --- JOLMo-native teacher (mode A) ---

    model: Optional[TransformerConfig] = None
    """Full transformer config for a JOLMo-native teacher. Mutually
    exclusive with :attr:`hf_model_name`."""

    checkpoint_path: Optional[str] = None
    """Local path or URL to the JOLMo-native teacher's distributed
    checkpoint. Required when :attr:`model` is set."""

    # --- HuggingFace teacher (mode B) ---

    hf_model_name: Optional[str] = None
    """HuggingFace Hub model id (e.g. ``"microsoft/harrier-oss-v1-270m"``).
    Mutually exclusive with :attr:`model` / :attr:`checkpoint_path`."""

    # --- Shared ---

    dp_config: Optional[TransformerDataParallelConfig] = None
    """FSDP/DDP config for a JOLMo-native teacher. None → replicated per
    rank. HF teachers always stay replicated in v1; setting this with an
    HF teacher is rejected."""

    autocast_precision: Optional[DType] = None
    """Autocast dtype for teacher forward (typically bfloat16). For HF
    teachers this is also passed as ``torch_dtype`` to
    ``from_pretrained``."""

    compile: bool = False
    """Whether to ``torch.compile`` the teacher's forward."""

    trained_sequence_length: Optional[int] = None
    """
    Sequence length the teacher was *trained* at. The train modules use
    this to reject configurations where the student's
    ``max_sequence_length`` exceeds the teacher's trained context window
    (RoPE extrapolation would produce meaningless teacher signal). For HF
    teachers this is auto-populated from
    ``model.config.max_position_embeddings`` if left ``None``.

    Intentionally NOT named ``max_sequence_length`` so that the config
    field does not collide with the ``max_sequence_length`` kwarg passed
    to :meth:`build` (which is the student's seq-len, used for FSDP
    sizing in the JOLMo-native path).
    """

    def __post_init__(self):
        has_jolmo = self.model is not None
        has_hf = self.hf_model_name is not None
        if has_jolmo == has_hf:
            raise OLMoConfigurationError(
                "TeacherModelConfig requires exactly one of 'model' "
                "(JOLMo-native) or 'hf_model_name' (HuggingFace)."
            )
        if has_jolmo and not self.checkpoint_path:
            raise OLMoConfigurationError(
                "JOLMo-native teacher requires 'checkpoint_path'."
            )
        if has_hf and self.dp_config is not None:
            raise OLMoConfigurationError(
                "dp_config for HF teachers is not supported in v1; HF "
                "models stay replicated per rank."
            )

    def build(
        self,
        *,
        student_world_mesh: Optional[DeviceMesh],
        device: torch.device,
        student_max_sequence_length: int,
        rank_microbatch_size: int,
    ) -> "TeacherModel":
        """
        Build and return either a :class:`TeacherModel` (JOLMo-native) or
        an :class:`HFTeacherModel`. The return type is always a
        ``TeacherModel`` subclass so the train modules see a uniform API.
        """
        if self.hf_model_name is not None:
            return self._build_hf(device)
        return self._build_jolmo(
            student_world_mesh=student_world_mesh,
            device=device,
            student_max_sequence_length=student_max_sequence_length,
            rank_microbatch_size=rank_microbatch_size,
        )

    def _build_jolmo(
        self,
        *,
        student_world_mesh: Optional[DeviceMesh],
        device: torch.device,
        student_max_sequence_length: int,
        rank_microbatch_size: int,
    ) -> "TeacherModel":
        """Build a JOLMo-native teacher (original behavior)."""
        assert self.model is not None and self.checkpoint_path is not None
        # Build on meta then parallelize + init with the same primitives
        # the student uses.
        teacher = self.model.build(init_device="meta")
        parallelize_model(
            teacher,
            world_mesh=student_world_mesh,
            device=device,
            max_sequence_length=student_max_sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=self.compile,
            dp_config=self.dp_config,
            # Teacher is inference-only: no activation checkpointing, no
            # float8, no TP/CP/EP/PP wrapping.
        )

        # Overwrite the random init with the real checkpoint weights.
        log.info(f"Loading teacher checkpoint from '{self.checkpoint_path}'...")
        load_model_and_optim_state(self.checkpoint_path, teacher, optim=None)
        gc_cuda()

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if not all(not p.requires_grad for p in teacher.parameters()):
            raise OLMoConfigurationError(
                "Teacher has parameters with requires_grad=True after freeze "
                "(possibly due to tied/shared weights)."
            )

        # Mesh consistency: by construction the teacher uses the SAME
        # ``student_world_mesh`` passed in, so ``parallelize_model``
        # resolves the DP sub-mesh via ``get_dp_model_mesh(world_mesh)``
        # against the same mesh the student used. No explicit
        # process-group check is necessary — identity follows from using
        # the same mesh object.

        tm = TeacherModel(
            model=teacher,
            device=device,
            autocast_precision=(
                self.autocast_precision.as_pt()
                if self.autocast_precision is not None
                else None
            ),
            trained_sequence_length=self.trained_sequence_length,
        )

        # Register a forward hook on the final block to capture hidden
        # states.
        last_block = teacher.blocks[str(teacher.n_layers - 1)]

        def _capture(_module, _inp, out):
            tm._hidden_state_cache[0] = out

        tm._hook_handle = last_block.register_forward_hook(_capture)
        return tm

    def _build_hf(self, device: torch.device) -> "HFTeacherModel":
        """Build an HF teacher (new path)."""
        assert self.hf_model_name is not None
        # Lazy import so `transformers` isn't a hard dependency of this
        # module's import time.
        from transformers import AutoModelForCausalLM

        torch_dtype = (
            self.autocast_precision.as_pt()
            if self.autocast_precision is not None
            else None
        )
        log.info(
            f"Loading HF teacher '{self.hf_model_name}' "
            f"(torch_dtype={torch_dtype})..."
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name, torch_dtype=torch_dtype,
        )
        hf_model.to(device)
        hf_model.eval()
        for p in hf_model.parameters():
            p.requires_grad_(False)
        if self.compile:
            hf_model = torch.compile(hf_model)

        # Auto-populate trained_sequence_length from the HF config if the
        # user didn't specify it.
        tsl = self.trained_sequence_length
        if tsl is None:
            tsl = getattr(hf_model.config, "max_position_embeddings", None)

        return HFTeacherModel(
            hf_model=hf_model,
            device=device,
            autocast_precision=torch_dtype,
            trained_sequence_length=tsl,
            hf_model_name=self.hf_model_name,
        )


class TeacherModel:
    """
    A thin inference-only wrapper around a frozen JOLMo-native
    :class:`~olmo_core.nn.transformer.Transformer`.

    Exposes :meth:`logits`, :meth:`hidden_states`, :attr:`vocab_size`,
    :attr:`d_model`, and :attr:`trained_sequence_length` — the same surface
    :class:`HFTeacherModel` exposes for HuggingFace teachers.

    :meth:`logits` and :meth:`hidden_states` wrap the forward in
    ``torch.no_grad()`` + ``torch.autocast`` (if configured) and forward
    any ``doc_lens`` / ``max_doc_lens`` / ``cache_leftpad`` kwargs so the
    teacher respects the same intra-document masking as the student.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        autocast_precision: Optional[torch.dtype],
        trained_sequence_length: Optional[int],
    ):
        # ``model`` is typed as nn.Module (rather than the more-specific
        # Transformer) so subclasses like HFTeacherModel can wrap an HF
        # PreTrainedModel. The default vocab_size / d_model properties
        # below read JOLMo Transformer attributes; subclasses override.
        self.model = model
        self.device = device
        self.autocast_precision = autocast_precision
        self.trained_sequence_length = trained_sequence_length
        self._hidden_state_cache: List[Optional[torch.Tensor]] = [None]
        self._hook_handle: Optional[Any] = None

    # -------- size properties (train modules read these) --------

    @property
    def vocab_size(self) -> int:
        # JOLMo Transformer exposes this directly.
        return self.model.vocab_size

    @property
    def d_model(self) -> int:
        return self.model.d_model

    # -------- inference context --------

    @contextlib.contextmanager
    def _infer_ctx(self):
        with torch.no_grad():
            if self.autocast_precision is not None:
                with torch.autocast(self.device.type, dtype=self.autocast_precision):
                    yield
            else:
                yield

    # -------- forward APIs --------

    def logits(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return ``(B, S, vocab_size)`` logits. ``kwargs`` are forwarded
        to ``Transformer.forward`` — pass ``doc_lens`` / ``max_doc_lens``
        / ``cache_leftpad`` to honor intra-document masking."""
        with self._infer_ctx():
            return self.model(input_ids, return_logits=True, **kwargs)

    def hidden_states(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return ``(B, S, d_teacher)`` — the final block's output,
        captured via forward hook. ``kwargs`` are forwarded to
        ``Transformer.forward`` (see :meth:`logits`)."""
        with self._infer_ctx():
            self.model(input_ids, **kwargs)
        h = self._hidden_state_cache[0]
        self._hidden_state_cache[0] = None
        if h is None:
            raise RuntimeError(
                "TeacherModel forward did not populate the hidden-state "
                "cache (hook may have been removed)."
            )
        return h

    # -------- FLOPs accounting --------

    def num_flops_per_token(self, seq_len: int) -> int:
        """Approximate forward-only FLOPs per token.

        The teacher is inference-only (no backward), so it uses 1/3 of
        the JOLMo ``num_flops_per_token`` formula (which counts
        forward + backward as ``6*params + 12*n*h*q*t``)."""
        return self.model.num_flops_per_token(seq_len) // 3


class HFTeacherModel(TeacherModel):
    """
    Inference-only wrapper around a HuggingFace ``CausalLM``.

    Exposes the same surface as :class:`TeacherModel` so the train
    modules don't care which backend is in use. Internally:

    - :attr:`vocab_size` / :attr:`d_model` read from ``model.config``.
    - :meth:`logits` uses HF's ``return_dict=True`` path.
    - :meth:`hidden_states` uses HF's ``output_hidden_states=True`` to
      expose the final block output directly — no forward hook is needed.
    - ``doc_lens`` / ``max_doc_lens`` / ``cache_leftpad`` kwargs are
      silently dropped (HF models don't accept them). For
      ``NumpyFSLDataset`` (fixed-length sliding windows, used for DCLM)
      these kwargs are not in the batch, so this is a non-issue. If a
      future dataset passes intra-document masking, revisit.
    """

    def __init__(
        self,
        *,
        hf_model: nn.Module,
        device: torch.device,
        autocast_precision: Optional[torch.dtype],
        trained_sequence_length: Optional[int],
        hf_model_name: str,
    ):
        super().__init__(
            model=hf_model,
            device=device,
            autocast_precision=autocast_precision,
            trained_sequence_length=trained_sequence_length,
        )
        self.hf_model_name = hf_model_name
        # self._hidden_state_cache / self._hook_handle are set by the
        # base __init__ but remain unused — HF exposes hidden states
        # natively via output_hidden_states=True, so no forward hook is
        # needed.

    # -------- size properties override --------

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def d_model(self) -> int:
        # Standard on modern HF configs. Older GPT-2-style configs use
        # ``n_embd``; not supported here.
        return self.model.config.hidden_size

    # -------- forward APIs override --------

    def logits(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return ``(B, S, vocab_size)`` logits via HF's
        ``PreTrainedModel.forward``. Any kwargs besides ``input_ids``
        are silently dropped (HF models don't accept JOLMo's
        ``doc_lens``/``max_doc_lens``/``cache_leftpad``)."""
        del kwargs  # dropped — see class docstring
        # HF models don't auto-move inputs to their device the way JOLMo's
        # Transformer.forward does, so we move here.
        input_ids = input_ids.to(self.device)
        with self._infer_ctx():
            return self.model(input_ids=input_ids, return_dict=True).logits

    def hidden_states(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return ``(B, S, d_teacher)`` — the final block's output,
        obtained via ``output_hidden_states=True``. Any kwargs besides
        ``input_ids`` are silently dropped."""
        del kwargs
        input_ids = input_ids.to(self.device)
        with self._infer_ctx():
            out = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        # out.hidden_states = (embeddings, block_0_out, ..., block_L_out).
        # The final entry matches what the JOLMo-native TeacherModel
        # captures via its forward hook.
        return out.hidden_states[-1]

    # -------- FLOPs accounting override --------

    def num_flops_per_token(self, seq_len: int) -> int:
        """Approximate forward-only FLOPs per token for an HF teacher.

        Computed from standard HF config attributes
        (``num_hidden_layers`` / ``num_attention_heads`` / ``hidden_size``
        / ``vocab_size``) using the forward-only analogue of JOLMo's
        formula: ``2*non_embedding_params + 4*n*h*q*t``. Embeddings are
        approximated as ``vocab_size * hidden_size``; tied output heads
        are not double-counted."""
        cfg = self.model.config
        n_layers = cfg.num_hidden_layers
        n_heads = cfg.num_attention_heads
        d_model = cfg.hidden_size
        q = d_model // n_heads
        total_params = sum(p.numel() for p in self.model.parameters())
        non_embed = max(0, total_params - cfg.vocab_size * d_model)
        return 2 * non_embed + 4 * n_layers * n_heads * q * seq_len
