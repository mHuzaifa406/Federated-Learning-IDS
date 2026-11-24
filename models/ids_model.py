
"""
TRFA Intrusion Detection Model
------------------------------
Architecture copied from the verified base project model.

Input:       78 CIC-IDS-2017 features
Hidden 1:   128 neurons + ReLU + Dropout(0.2)
Hidden 2:    64 neurons + ReLU + Dropout(0.2)
Output:       7 traffic classes
"""

import torch
import torch.nn as nn


class IntrusionDetectionMLP(nn.Module):
    """Multiclass intrusion detection model."""

    def __init__(self, input_features=78, num_classes=7):
        super().__init__()

        if input_features <= 0:
            raise ValueError("input_features must be positive.")

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                f"Expected a 2D input tensor, received shape {tuple(x.shape)}"
            )

        return self.network(x)


if __name__ == "__main__":

    model = IntrusionDetectionMLP()

    print("TRFA Intrusion Detection Model")
    print("-" * 40)
    print(model)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTrainable parameters:", total_parameters)

    # Test with four sample records.
    sample_input = torch.randn(4, 78)
    model.eval()

    with torch.no_grad():
        sample_output = model(sample_input)

    print("Input shape:", tuple(sample_input.shape))
    print("Output shape:", tuple(sample_output.shape))
    print("Device: CPU")
