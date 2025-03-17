import sys
sys.path.append('../')

import torch
from torch.distributions import Distribution
from supn_base.sparse_precision_cholesky import get_prec_chol_as_sparse_tensor, \
    log_prob_from_sparse_chol_prec
    
from supn_base.cholespy_solver import SUPNSolver, sparse_chol_linsolve, sample_zero_mean
from supn_base.supn_data import SUPNData, get_num_off_diag_weights

class SUPN(Distribution):
    """
    SUPN distribution class implementing the torch.distributions.Distribution interface.
    """
    
    def __init__(self, 
                 supn_data: SUPNData):
        self.supn_data = supn_data
        self.supn_solver = SUPNSolver(supn_data)
        super(SUPN, self).__init__()


    def sample(self, 
               num_samples: int = 1) -> torch.Tensor:

        zero_mean_sample = sample_zero_mean(self.supn_data, num_samples, self.supn_solver)

        return  zero_mean_sample + self.mean

    def log_prob(self, data: torch.tensor, stop_grads: bool = False) -> torch.Tensor:
        """
        Compute the log probability of the data given the distribution.
        stop_grads: blocks gradients back into the supn parameters
        """
        if stop_grads:
            return log_prob_from_sparse_chol_prec(x = data,
                                       mean = self.supn_data.mean.detach(),
                                       log_diag_weights = self.supn_data.log_diag.detach(),
                                       off_diag_weights = self.supn_data.off_diag.detach(),
                                       local_connection_dist = self.supn_data.local_connection_dist,
                                       use_transpose=True,
                                       use_3d=self.supn_data.use_3d,
                                       cross_ch=self.supn_data.cross_ch.detach())
        else:
            return log_prob_from_sparse_chol_prec(x = data,
                                       mean = self.supn_data.mean,
                                       log_diag_weights = self.supn_data.log_diag,
                                       off_diag_weights = self.supn_data.off_diag,
                                       local_connection_dist = self.supn_data.local_connection_dist,
                                       use_transpose=True,
                                       use_3d=self.supn_data.use_3d,
                                       cross_ch=self.supn_data.cross_ch)


    def cov(self) -> torch.Tensor:
        # some methods to get the covariance matrix from chol or prec (or maybe just a row)
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        return self.supn_data.mean
        
    @property
    def precision(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def precision_cholesky(self) -> torch.Tensor:
        # Return the precision is a sparse COO torch tensor
        return get_prec_chol_as_sparse_tensor(log_diag_weights = self.supn_data.log_diag,
                                    off_diag_weights = self.supn_data.off_diag,
                                    local_connection_dist = self.supn_data.local_connection_dist,
                                    use_transpose = True,
                                    use_3d = self.supn_data.use_3d,
                                    cross_ch=self.supn_data.cross_ch)
    

# Register the distribution
torch.distributions.SUPN = SUPN