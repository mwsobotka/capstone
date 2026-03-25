import numpy as np

def sample_voters_normal(N: int,
                         sigma: float = 1.0,
                         dim: int = 2) -> np.ndarray:

    mean = np.zeros(dim)
    cov = np.eye(dim) * (sigma ** 2)
    return np.random.multivariate_normal(mean, cov, size=N)

def sample_voters_mixture(N: int,
                          pi: float = 0.5,
                          mu_left = np.array([-1.5, -0.5]),
                          mu_right = np.array([1.5, 0.5]),
                          sigma_left: float = 0.8,
                          sigma_right: float = 0.8,
                          dim: int = 2) -> np.ndarray:

    assert dim == 2

    # group labels: True = left, False = right
    g = np.random.rand(N) < pi
    voters = np.zeros((N, dim))

    # left cluster
    cov_L = np.eye(dim) * (sigma_left ** 2)
    idx_L = np.where(g)[0]
    voters[idx_L, :] = np.random.multivariate_normal(mu_left, cov_L, size=len(idx_L))

    # right cluster
    cov_R = np.eye(dim) * (sigma_right ** 2)
    idx_R = np.where(~g)[0]
    voters[idx_R, :] = np.random.multivariate_normal(mu_right, cov_R, size=len(idx_R))

    return voters
