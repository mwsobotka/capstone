# parties.py
import numpy as np

def init_parties(P: int,
                 dim: int = 2,
                 center: np.ndarray | None = None,
                 spread: float = 1.0) -> np.ndarray:

    if center is None:
        center = np.zeros(dim)

    # sample around 'center' with an identity covariance * spread^2
    cov = np.eye(dim) * (spread ** 2)
    parties = np.random.multivariate_normal(center, cov, size=P)
    return parties

def update_parties_fptp(voters: np.ndarray,
                   parties: np.ndarray,
                   choices: np.ndarray,
                   eta: float = 0.2) -> np.ndarray:

    new_parties = parties.copy()
    P, dim = parties.shape

    for p in range(P):
        idx = np.where(choices == p)[0]  # voters who chose party p
        if len(idx) == 0:
            # nobody voted for this party; for now, leave it where it is
            continue

        supporters = voters[idx, :]           # (n_p x 2)
        mean_support = supporters.mean(axis=0)  # (2,)

        # move party p a fraction eta toward mean of its supporters
        new_parties[p, :] = parties[p, :] + eta * (mean_support - parties[p, :])

    return new_parties

def update_parties_approval(voters: np.ndarray,
                            parties: np.ndarray,
                            approvals: np.ndarray,
                            eta: float = 0.2) -> np.ndarray:

    new_parties = parties.copy()
    P, dim = parties.shape

    for p in range(P):
        idx = np.where(approvals[:, p] == 1)[0]   # voters who approved party p
        if len(idx) == 0:
            continue

        supporters = voters[idx, :]
        mean_support = supporters.mean(axis=0)

        new_parties[p, :] = parties[p, :] + eta * (mean_support - parties[p, :])

    return new_parties

def update_parties_irv(voters: np.ndarray,
                       parties: np.ndarray,
                       rankings: np.ndarray,
                       eta: float = 0.2) -> np.ndarray:

    new_parties = parties.copy()
    P, dim = parties.shape

    first_choices = rankings[:, 0]

    for p in range(P):
        idx = np.where(first_choices == p)[0]
        if len(idx) == 0:
            continue

        supporters = voters[idx, :]
        mean_support = supporters.mean(axis=0)

        new_parties[p, :] = parties[p, :] + eta * (mean_support - parties[p, :])

    return new_parties

def merge_close_parties(parties: np.ndarray,
                        d_merge: float = 0.4) -> np.ndarray:

    P, dim = parties.shape

    if P <= 1:
        return parties

    # 1. Compute pairwise distances between parties 

    # diff[i,j,:] = parties[i,:] - parties[j,:]
    diff = parties[:, np.newaxis, :] - parties[np.newaxis, :, :]   # (P, P, dim)

    # dist[i,j] = Euclidean distance between party i and j
    dist = np.linalg.norm(diff, axis=2)                            # (P, P)

    # 2. Clustering

    visited = np.zeros(P, dtype=bool)   
    new_positions = []                 

    for i in range(P):
        if visited[i]:
            continue

        # Start a new cluster with party i
        cluster_indices = [i]
        visited[i] = True

        # Check all parties j > i to see if they are close enough to merge with i
        for j in range(i + 1, P):
            if not visited[j] and dist[i, j] < d_merge:
                cluster_indices.append(j)
                visited[j] = True

        # Compute the position of the merged party: average of all in the cluster
        cluster_pos = parties[cluster_indices, :].mean(axis=0)
        new_positions.append(cluster_pos)

    new_parties = np.vstack(new_positions)
    return new_parties
