# utils/config.py

import argparse
from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"

def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model configuration name, such as 'fcn'",
    )

    args, overrides = parser.parse_known_args()

    model_config_path = CONFIG_DIR / f"{args.model}.yaml"

    if not model_config_path.is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {model_config_path}"
        )

    default_cfg = OmegaConf.load(DEFAULT_CONFIG)

    # All valid configuration fields must exist in default.yaml.
    OmegaConf.set_struct(default_cfg, True)

    model_cfg = OmegaConf.load(model_config_path)
    cli_cfg = OmegaConf.from_dotlist(overrides)

    cfg = OmegaConf.merge(
        default_cfg,
        model_cfg,
        cli_cfg,
    )

    cfg.model_name = args.model

    return cfg