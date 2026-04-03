import numpy as np
from typing import Tuple
import random
class LinearLayer:
    def __init__(self, in_features: int, out_features: int) -> None:  
        # TODO: khởi tạo W (in_features × out_features) và b (out_features,)
        # Dùng np.random.randn và scale nhỏ (× 0.01)
        W = np.random.randn(in_features, out_features) *  0.01
        b = np.random.randn(out_features) * 0.01
        self.W = W
        self.b = b

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: return x @ self.W + self.b
        return x @ self.W + self.b # @ la toan tu nhan ma tran trong numpy

    def __repr__(self) -> str:
        # TODO: return "LinearLayer(in=..., out=...)"
        return f"LinearLayer(in={self.W.shape[0]}, out={self.W.shape[1]})"

class Activation:
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
        
class MLPClassifier(Activation):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        self.linear1 = LinearLayer(in_features, hidden_features)
        self.linear2 = LinearLayer(hidden_features, out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # LL1
        z1 = self.linear1.forward(x)
        a1 = self.relu(z1)

        # LL2
        z2 = self.linear2.forward(a1)
        a2 = self.sigmoid(z2)
        
        return a2
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.forward(x)
        return probs.argmax(axis=1)
# Test

# 1.1
layer = LinearLayer(4, 2)
x = np.random.randn(8, 4)   # batch=8, features=4
out = layer.forward(x)
assert out.shape == (8, 2), f"Shape sai: {out.shape}"
print("✅ Pass! Output shape:", out.shape)
print(layer)
model = MLPClassifier(in_features=4, hidden_features=8, out_features=3)
print(model)


# 1.2
x = np.random.randn(5, 4)  # batch=5, features=4
probs = model.forward(x)
print("Output probabilities:\n", probs)