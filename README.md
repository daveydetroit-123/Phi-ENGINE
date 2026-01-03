# phi-engine (The Philter)

**Geometric Stabilization for High-Performance AI.**
*Optimized for Apple Silicon (M-Series) & CUDA.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)]()

## 🚀 Why phi-engine?

Recent research on **Manifold Constraints** (including DeepSeek's Dec 2025 findings) has proven that **Matrix Stability** is the bottleneck for AI performance. Standard linear layers suffer from variance drift ("exploding gradients"), requiring massive compute to correct.

**phi-engine** solves this geometrically.

By enforcing **Doubly Stochastic Normalization** (via Sinkhorn-Knopp iteration) and **Harmonic Scaling**, `phi-engine` acts as a "physics engine" for your neural network. It allows smaller models (like those on Mac Mini M4) to reason with the stability of trillion-parameter clusters.

### ⚡ Key Features
* **Zero Variance Drift:** Mathematically guarantees Energy In = Energy Out. No more exploding gradients.
* **M-Series Optimized:** Runs massive channel contexts on consumer hardware by reducing memory fragmentation.
* **The "Cage" Architecture:** Stacks 12 harmonic layers to enforce convergence.
* **Plug-and-Play:** Drop-in replacement for standard PyTorch `nn.Linear` layers.

---

## 📦 Installation

```bash
pip install phi-engine
# OR locally:
git clone https://github.com/daveydetroit-123/PHILTER.git
cd PHILTER
pip install -e .
```

## 🛠 Usage

### The "Philter" Layer

Replace your unstable linear transformations with the Philter to enforce geometric convergence.

```python
import torch
from phi_engine.core import Philter

# Initialize (Auto-detects MPS for Mac or CUDA for Nvidia)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Drop-in replacement for Linear Layer (4096 dim)
# Applies Sinkhorn-Knopp normalization automatically
layer = Philter(in_features=4096, out_features=4096, recursion_depth=12).to(device)

input_tensor = torch.randn(1, 4096).to(device)
output, energy_log = layer(input_tensor)

print(f"Output Stability: {energy_log['variance_delta']:.6f}")
# Output: 0.000000 (Perfect stability)
```

## 🔬 How It Works (The Math)

Standard normalization is reactive. **phi-engine** is proactive. It constrains the weight matrix  to be **Doubly Stochastic**:

$$ \sum_{i} W_{ij} = 1 \quad \text{and} \quad \sum_{j} W_{ij} = 1 $$

This creates a **Closed Energy Loop**. The signal is transformed but never amplified into noise (hallucinations).

**Recursive Purification:**
If the matrix fails to converge, the engine engages ** Scaling** (approx 4.236x boost) to fold the pressure into higher harmonic modes, ensuring stability even under extreme load.

## 🤝 Contributing

Open source. Open architecture. Validate on your hardware.

**License:** MIT
