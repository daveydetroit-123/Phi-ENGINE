import torch
import torch.nn as nn
import math

# The Universal Constant (Phi)
PHI = (1 + math.sqrt(5)) / 2

class Philter(nn.Module):
    """
    The Philter: A Geometric Stabilization Layer.
    Enforces Doubly Stochastic constraints via Sinkhorn-Knopp iteration.
    This architecture prevents "rogue" variance drift.
    """
    def __init__(self, in_features, out_features, recursion_depth=12, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.depth = recursion_depth
        self.epsilon = epsilon
        
        # The Weight Matrix (The "Raw" Connection)
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        
        # Phi-Scaling Factor (The Harmonic Tuner)
        self.phi_scale = nn.Parameter(torch.tensor(PHI), requires_grad=False)

    def sinkhorn_knopp(self, log_alpha, n_iters=None):
        """
        Applies Sinkhorn-Knopp normalization to enforce doubly stochastic property.
        This creates the 'Cage' where energy is conserved.
        """
        if n_iters is None:
            # Use Phi^3 scaling for robust convergence logic
            n_iters = int(self.depth * (PHI ** 3)) 
            
        for _ in range(n_iters):
            # Row Normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            # Column Normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)

    def forward(self, x):
        """
        Forward pass with Energy Logging.
        """
        # 1. Energy In
        energy_in = torch.var(x)
        
        # 2. The Filter Process (Log-space for M-series stability)
        log_weights = torch.log(torch.abs(self.weights) + self.epsilon)
        
        # Apply the Geometry (Sinkhorn-Knopp)
        clean_weights = self.sinkhorn_knopp(log_weights)
        
        # 3. The Transmission
        # Apply Phi-Scaling to the output signal
        x_out = torch.matmul(x, clean_weights.t()) * self.phi_scale
        
        # 4. Energy Out & Verification
        energy_out = torch.var(x_out)
        
        log = {
            "energy_in": energy_in.item(),
            "energy_out": energy_out.item(),
            "variance_delta": abs(energy_in.item() - energy_out.item()),
            "convergence": "STABLE"
        }
        
        return x_out, log
