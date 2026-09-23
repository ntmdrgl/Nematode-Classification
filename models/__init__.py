from models.fcn import FCN

def create_model(cfg):
    if cfg.model_name == "fcn":
        return FCN(
            in_channels=cfg.in_channels,
            hidden_size=list(cfg.hidden_size),
            kernel_size=list(cfg.kernel_size),
            output_size=cfg.num_classes,
            dropout=cfg.dropout,
        )

    raise ValueError(
        f"Unsupported model: {cfg.model_name}"
    )