import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            # nn.GELU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)
        
class ResBlock(nn.Module):
    def __init__(self, dim, resblock_dim=None):
        super().__init__()
        if resblock_dim is None:
            resblock_dim = dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, resblock_dim),
            nn.ReLU(),
            nn.Linear(resblock_dim, dim),
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        # Residual connection
        return self.act(self.norm(x + self.net(x)))

class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3):
        super().__init__()
        
        # Initial projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Final projection to output
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Initialize weights using Kaiming initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Final projection
        return self.output_proj(x)

class SeparableApproximation(nn.Module):
    def __init__(self, num_terms=2, symm_kind=1, sub_arch='ResMLP', log_transform=False):
        super().__init__()
        self.num_terms = num_terms
        self.log_transform = log_transform
        # SubModel = ResMLP(1, hidden_dim=64, num_blocks=1)
        if sub_arch=="MLP":
            self.alpha = nn.ModuleList([MLP(1) for _ in range(num_terms)])
            self.beta = nn.ModuleList([MLP(1) for _ in range(num_terms)])
            if symm_kind in [1,3]:
                self.gamma = nn.ModuleList([MLP(1) for _ in range(num_terms)])
        elif sub_arch=='ResMLP':
            self.alpha = nn.ModuleList([ResMLP(1,hidden_dim=64, num_blocks=1) for _ in range(num_terms)])
            self.beta = nn.ModuleList([ResMLP(1,hidden_dim=64, num_blocks=1)  for _ in range(num_terms)])
            if symm_kind in [1,3]:
                self.gamma = nn.ModuleList([ResMLP(1,hidden_dim=64, num_blocks=1)  for _ in range(num_terms)])
        else:
            raise ValueError('architecture of submodules not defined')
        self.weights = nn.Parameter(torch.ones(num_terms))#+1))
        self.symm_kind = symm_kind

    def forward(self, x):
        if self.log_transform:
            f = lambda x: torch.exp(x)
            g = lambda x: torch.log(x)
        else:
            f = lambda x: x
            g = lambda x: x
        
        # Define ks
        k1, k2, k3 = g(x[:, 0]).view(-1, 1), g(x[:, 1]).view(-1, 1), g(x[:, 2]).view(-1, 1)
        
        # Symmtry kind 1: enforcing full symmetry, 3 different functions with 6 permutations
        if self.symm_kind==1:
            result = sum(self.weights[i] * f(self.alpha[i](k1)) * f(self.beta[i](k2)) * f(self.gamma[i](k3))\
                        +self.weights[i] * f(self.alpha[i](k1)) * f(self.beta[i](k3)) * f(self.gamma[i](k2))\
                        +self.weights[i] * f(self.alpha[i](k2)) * f(self.beta[i](k1)) * f(self.gamma[i](k3))\
                        +self.weights[i] * f(self.alpha[i](k2)) * f(self.beta[i](k3)) * f(self.gamma[i](k1))\
                        +self.weights[i] * f(self.alpha[i](k3)) * f(self.beta[i](k1)) * f(self.gamma[i](k2))\
                        +self.weights[i] * f(self.alpha[i](k3)) * f(self.beta[i](k2)) * f(self.gamma[i](k1))\
                              for i in range(self.num_terms))
            return result
        
        # Symmtry kind 2: assuming additional symmetry, 2 functions alpha and beta, with 3 permutations
        elif self.symm_kind==2:
            result = sum( self.weights[i] * f(self.alpha[i](k1)) * f(self.beta[i](k2)) * f(self.beta[i](k3))\
                        +self.weights[i] * f(self.alpha[i](k2)) * f(self.beta[i](k3)) * f(self.beta[i](k1))\
                        +self.weights[i] * f(self.alpha[i](k3)) * f(self.beta[i](k1)) * f(self.beta[i](k2)) for i in range(self.num_terms))
            return result
        
        # Symmtry kind 3: assuming addtional symmetry, but 3 function used
        elif self.symm_kind==3:
            result = sum( self.weights[i] * f(self.alpha[i](k1)) * f(self.beta[i](k2)) * f(self.gamma[i](k3))\
                        +self.weights[i] * f(self.alpha[i](k2)) * f(self.beta[i](k3)) * f(self.gamma[i](k1))\
                        +self.weights[i] * f(self.alpha[i](k3)) * f(self.beta[i](k1)) * f(self.gamma[i](k2)) for i in range(self.num_terms))
            return result