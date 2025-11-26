import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from olmo_core.nn.transformer import TransformerConfig

# Import YamlExperimentConfig from launch_from_yaml
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))
try:
    from launch_from_yaml import YamlExperimentConfig
finally:
    sys.path.pop(0)


def _format_number_with_underscores(n: int) -> str:
    s = str(n)
    result = []
    for i, char in enumerate(reversed(s)):
        if i > 0 and i % 3 == 0:
            result.append("_")
        result.append(char)
    return "".join(reversed(result))


def _get_model_config(cfg: YamlExperimentConfig) -> TransformerConfig:
    if cfg.model is not None:
        return cfg.model
    if cfg.model_factory:
        factory = getattr(TransformerConfig, cfg.model_factory, None)
        if factory is None:
            raise ValueError(f"Unknown model factory: {cfg.model_factory}")
        kwargs = cfg.model_factory_args or {}
        return factory(**kwargs)
    raise ValueError("You must provide either 'model' or 'model_factory' in the config.")


def main():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} CONFIG.yaml",
        description="Print model parameter counts from a YAML launch config.",
    )
    parser.add_argument("config", type=str, help="Path/URL to YAML config.")
    opts = parser.parse_args()

    cfg = YamlExperimentConfig.from_file(opts.config)
    model_config = _get_model_config(cfg)

    total_params = model_config.num_params
    non_embedding_params = model_config.num_non_embedding_params
    embedding_params = total_params - non_embedding_params

    result: Dict[str, str] = {
        "total_params": _format_number_with_underscores(total_params),
        "embedding_params": _format_number_with_underscores(embedding_params),
        "non_embedding_params": _format_number_with_underscores(non_embedding_params),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

