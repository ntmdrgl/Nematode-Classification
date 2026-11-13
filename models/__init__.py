from models.fcn import FCN
from models.bilstm import BiLSTM
from models.dilated_fcn import DilatedFCN

def create_model(model, model_params):
    if model == "fcn":
        return FCN(
            in_channels=model_params.in_channels,
            num_classes=model_params.num_classes
        )

    elif model == "dilated_fcn":
        return DilatedFCN(
            in_channels=model_params.in_channels,
            num_classes=model_params.num_classes,
        )

    elif model == "bilstm":
        return BiLSTM(
            in_channels=model_params.in_channels,
            hidden_size=model_params.hidden_size,
            num_classes=model_params.num_classes
        )

    raise ValueError(
        f"Unsupported model: {model}"
    )
