import torch
import torch.nn as nn
from typing import Optional

class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # TODO: định nghĩa self.net = nn.Sequential(...)
        self.net = nn.Sequential(
            # Tầng 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Tầng 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Tầng 3
            nn.Linear(hidden_dim, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: flatten x rồi đưa qua self.net
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out
    
    def count_parameters(self) -> int:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

# Smoke test
model = SimpleMLP()
dummy = torch.randn(32, 1, 28, 28)  # batch=32, MNIST shape
out = model(dummy)
assert out.shape == (32, 10), f"Shape sai: {out.shape}"
print("✅ Pass! Output shape:", out.shape)
print(model)
print(f"SimpleMLP: {model.count_parameters()["total"]} params ({model.count_parameters()["trainable"]} trainable)")
