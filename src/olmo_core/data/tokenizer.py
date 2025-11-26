from dataclasses import dataclass
from typing import Optional

from ..config import Config, StrEnum

__all__ = [
    "TokenizerConfig",
    "TokenizerName",
]


class TokenizerName(StrEnum):
    """
    An enumeration of tokenizer identifiers commonly used OLMo researchers.
    """

    dolma2 = "allenai/dolma2-tokenizer"
    """
    The dolma2 tokenizer.
    """

    dolma2_sigdig = "allenai/dolma2-tokenizer-sigdig"
    """
    The R2L dolma2 tokenizer.
    """

    gpt_neox_olmo_dolma_v1_5 = "allenai/gpt-neox-olmo-dolma-v1_5"
    """
    A modified GPT NeoX tokenizer.
    """

    gpt2 = "gpt2"
    """
    The base GPT2 tokenizer.
    """


@dataclass
class TokenizerConfig(Config):
    """
    A configuration class that represents a tokenizer.
    """

    vocab_size: Optional[int] = None
    """
    The vocab size.
    """

    eos_token_id: Optional[int] = None
    """
    The end-of-sentence token ID.
    """

    pad_token_id: Optional[int] = None
    """
    The padding token ID.
    """

    bos_token_id: Optional[int] = None
    """
    The begin-of-sentence token ID.
    """

    identifier: Optional[str] = None
    """
    The identifier of the tokenizer. Could be a path or HuggingFace identifier.
    """

    def __post_init__(self):
        """
        Allow specifying only the identifier (or TokenizerName) and infer the numeric fields.
        """
        # Already complete
        if self.vocab_size is not None and self.eos_token_id is not None and self.pad_token_id is not None:
            return

        if self.identifier is None:
            # Nothing to infer from
            return

        ident = str(self.identifier)
        name = None
        try:
            name = TokenizerName(ident)
        except Exception:
            name = None

        inferred: Optional["TokenizerConfig"] = None
        if name is not None:
            if name == TokenizerName.gpt2:
                inferred = TokenizerConfig.gpt2()
            elif name == TokenizerName.dolma2:
                inferred = TokenizerConfig.dolma2()
            elif name == TokenizerName.dolma2_sigdig:
                inferred = TokenizerConfig.dolma2_sigdig()
            elif name == TokenizerName.gpt_neox_olmo_dolma_v1_5:
                inferred = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
        else:
            # Try to map common aliases
            if ident == "gpt2":
                inferred = TokenizerConfig.gpt2()
            elif "dolma2-tokenizer-sigdig" in ident:
                inferred = TokenizerConfig.dolma2_sigdig()
            elif "dolma2-tokenizer" in ident:
                inferred = TokenizerConfig.dolma2()
            elif "gpt-neox-olmo-dolma-v1_5" in ident:
                inferred = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
            else:
                # Fall back to HF config if available
                try:
                    inferred = TokenizerConfig.from_hf(ident)
                except Exception:
                    inferred = None

        if inferred is not None:
            # Preserve identifier given by user; copy numeric fields
            self.vocab_size = inferred.vocab_size
            self.eos_token_id = inferred.eos_token_id
            self.pad_token_id = inferred.pad_token_id
            # Only set bos if not specified
            if self.bos_token_id is None:
                self.bos_token_id = inferred.bos_token_id

    def padded_vocab_size(self, pad_multiple: int = 128) -> int:
        """
        Returns the vocab size padded to be a multiple of ``pad_multiple``.
        This is useful to set model embeddings to this number to increase throughput.
        """
        if self.vocab_size is None:
            raise ValueError("vocab_size is not set on TokenizerConfig")
        return pad_multiple * ((self.vocab_size + pad_multiple - 1) // pad_multiple)

    @classmethod
    def dolma2(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.dolma2` tokenizer config.
        """
        return cls(
            vocab_size=100278,
            eos_token_id=100257,
            pad_token_id=100277,
            identifier=TokenizerName.dolma2,
        )

    @classmethod
    def dolma2_sigdig(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.dolma2_sigdig` tokenizer config.
        """
        return cls(
            vocab_size=100278,
            eos_token_id=100257,
            pad_token_id=100277,
            bos_token_id=100257,
            identifier=TokenizerName.dolma2_sigdig,
        )

    @classmethod
    def gpt_neox_olmo_dolma_v1_5(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.gpt_neox_olmo_dolma_v1_5` tokenizer config.
        """
        return cls(
            vocab_size=50280,
            eos_token_id=50279,
            pad_token_id=1,
            identifier=TokenizerName.gpt_neox_olmo_dolma_v1_5,
        )

    @classmethod
    def gpt2(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.gpt2` tokenizer config.
        """
        return cls(
            vocab_size=50257,
            eos_token_id=50256,
            bos_token_id=50256,
            pad_token_id=50256,
            identifier=TokenizerName.gpt2,
        )

    @classmethod
    def from_hf(cls, identifier: str) -> "TokenizerConfig":
        """
        Initialize a tokenizer config from a model on HuggingFace.

        :param identifier: The HF model identifier, e.g. "meta-llama/Llama-3.2-1B".
        """
        import json

        from cached_path import cached_path

        try:
            config_path = cached_path(f"hf://{identifier}/config.json")
        except FileNotFoundError:
            config_path = cached_path(f"hf://{identifier}/tokenizer_config.json")

        with config_path.open() as f:
            config = json.load(f)

        return cls(
            vocab_size=config["vocab_size"],
            eos_token_id=config["eos_token_id"],
            pad_token_id=config.get("pad_token_id", config["eos_token_id"]),
            bos_token_id=config.get("bos_token_id"),
            identifier=identifier,
        )
