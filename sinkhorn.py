import torch
from torch import nn, Tensor
from math import log

def matmul_lse(A: Tensor, x: Tensor) -> Tensor:
    """
    Compute log(By) where B=exp(A) and y=exp(x) (where log and exp taken
    element-wise).
    
    Use log-sum-exp trick to avoid numerical over/underflow.
    
    Args:
        A: (Tensor) NxM dimensional matrix.
        x: (Tensor) M dimensional vector.
    Returns:
        lse: (Tensor) element-wise log(exp(A)exp(x))
    """
    y = A + x.view(1, A.shape[1])
    y_max = torch.max(y, dim=1)[0]
    y_sub = y - y_max.view(A.shape[0], 1)
    lse = y_max + y_sub.exp().sum(dim=1).log()
    return lse

class Sinkhorn(nn.Module):
    """Apply Sinkhorn algorithm with automatic differentiation."""
    
    def __init__(self, reg_param: float, num_iter: int=100, dual: bool=False) -> None:
        """
        Args:
            reg_param: (float) entropy regularisation factor.
            num_iter: (int) number of iterations before returning result.
            dual: (bool) if true, use dual form of Sinkhorn algorithm.
        """
        super().__init__()
        self.reg_param = reg_param
        self.num_iter = num_iter
        self.dual = dual
    
    def forward(self, C: Tensor) -> Tensor:
        """
        Args:
            C: (Tensor) cost matrix.
        Returns:
            W: (Tensor) estimated Wasserstein distance.
        """
        if self.dual:
            W = self.sinkhorn_dual(C)
        else:
            W = self.sinkhorn_primal(C)
        if W.isnan():
            raise ValueError('Sinkhorn outputed NaN.')
        return W
    
    def sinkhorn_primal(self, C: Tensor) -> Tensor:
        """
        Args:
            C: (Tensor) cost matrix.
        Returns:
            W: (Tensor) estimated Wasserstein distance.
        """
        n, m = C.shape
        K = torch.exp(- C / self.reg_param)
        u = torch.ones(n).to(K.device)
        
        for i in range(self.num_iter):
            v = 1 / (m * K.T.matmul(u))
            u = 1 / (n * K.matmul(v))
        
        ot_plan = u.view(n, 1) * K * v.view(1, m)
        W = (ot_plan*C).sum()
        return W

    def sinkhorn_dual(self, C: Tensor) -> Tensor:
        """
        Args:
            C: (Tensor) cost matrix.
        Returns:
            W: (Tensor) estimated Wasserstein distance.
        """
        n, m = C.shape
        A = - C / self.reg_param
        cst1 = - self.reg_param * log(n)
        cst2 = - self.reg_param * log(m)
        K = torch.exp(mCovereps)
        alpha = torch.zeros(n).to(K.device)
        
        for i in range(self.num_iter):
            beta = cst2 - self.reg_param * matmul_lse(A.T,
                                                      alpha / self.reg_param)
            alpha = cst1 - self.reg_param * matmul_lse(A,
                                                       beta / self.reg_param)

        exponent = - C + alpha.view(n, 1) + beta.view(1, m)
        exponent = exponent / self.reg_param
        exponent_max = exponent.max()
        exponent_sub = exponent - exponent_max
        W = exponent_max.exp() * (C * exponent_sub.exp()).sum()
        return W