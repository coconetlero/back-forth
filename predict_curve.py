import os
import numpy as np
import torch

from gru_model_evaluation import (
    CurveToWModel,
    CurveDataset,
    TrainingConfig,
    load_model,
)

def predict_w_from_curve(model, new_curve, device="cpu"):
    model.eval()

    # Convert to tensor
    if not isinstance(new_curve, torch.Tensor):
        points = torch.tensor(new_curve, dtype=torch.float32)
    else:
        points = new_curve.to(torch.float32)

    # Same normalization as during training
    points = CurveDataset._normalize_curve(points)  # (n, 2)

    length = torch.tensor([points.shape[0]], dtype=torch.long)  # (1,)
    padded_points = points.unsqueeze(0)  # (1, n, 2)

    padded_points = padded_points.to(device)
    length = length.to(device)

    with torch.no_grad():
        w_hat = model(padded_points, length)  # (1, 2)

    return w_hat.squeeze(0).cpu().numpy()  # (2,)



def main():
    # Must match what you used in training
    config = TrainingConfig(
        hidden_dim=128,
        num_layers=1,
        model_dir="./models",
        model_name="curve_to_w_gru.pt",
    )
    device = config.device

    # Build model and load weights
    model = CurveToWModel(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)

    ok = load_model(model, config)
    if not ok:
        return

    # Load or construct a new curve: shape (n, 2)
    # Example dummy curve:
    n = 50
    new_curve = np.random.randn(n, 2).astype(np.float32)

    w_pred = predict_w_from_curve(model, new_curve, device=device)
    print("Predicted w:", w_pred)
 

if __name__ == "__main__":
    main()
