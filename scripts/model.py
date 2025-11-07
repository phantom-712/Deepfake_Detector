"""
model.py
Defines the Vision Transformer (ViT) binary classifier using timm.
For binary classification (real vs fake), we output a single logit
and use BCEWithLogitsLoss during training.
"""

import torch
import torch.nn as nn
import timm


class ViTBinaryClassifier(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        """
        Initializes a Vision Transformer (ViT) for binary classification.

        Args:
            model_name (str): ViT architecture from timm (e.g., "vit_base_patch16_224").
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super().__init__()

        # Create the model with a single output logit (num_classes=1)
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B,), representing fake probability before sigmoid.
        """
        logits = self.model(x)       # (B, 1)
        return logits.view(-1)       # flatten to (B,)
