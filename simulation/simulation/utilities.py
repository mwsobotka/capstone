# utilities.py
import numpy as np


# 1. SPATIAL UTILITY FUNCTIONS

def compute_utilities(voters: np.ndarray,
                      parties: np.ndarray,
                      gamma: float = 0.5) -> np.ndarray:

    v = voters[:, np.newaxis, :]     # shape (N, 1, 2)
    p = parties[np.newaxis, :, :]    # shape (1, P, 2)

    diff = v - p                      # (N, P, 2)
    sq = np.sum(diff**2, axis=2)      # (N, P)

    U = -sq                           # base Euclidean utility

    if gamma != 0:
        cross = diff[:, :, 0] * diff[:, :, 1]
        U -= gamma * cross

    return U


# 2. BALLOT GENERATION RULES

# Turnout
def compute_turnout_probabilistic(voters: np.ndarray,
                                  parties: np.ndarray,
                                  alpha: float = 1.0) -> np.ndarray:

    # distances to all parties
    v = voters[:, np.newaxis, :]      # (N,1,2)
    p = parties[np.newaxis, :, :]     # (1,P,2)
    d = np.linalg.norm(v - p, axis=2) # (N,P)
    nearest = d.min(axis=1)           # (N,)

    # turnout probabilities
    probs = np.exp(-alpha * nearest)

    # probabilistic turnout decision
    draws = np.random.rand(len(voters))
    return draws < probs



#  FPTP (top choice only) 
def ballots_fptp(U: np.ndarray) -> np.ndarray:

    return np.argmax(U, axis=1)


#  Approval Voting 
def ballots_approval(U: np.ndarray, top_frac: float = 0.5) -> np.ndarray:
    N, P = U.shape
    approvals = np.zeros((N, P), dtype=int)

    k = max(1, int(np.ceil(P * top_frac)))

    for i in range(N):
        top_k = np.argsort(U[i])[::-1][:k]   # indices of top-k utilities
        approvals[i, top_k] = 1

    return approvals


#  Ranked Ballots (RCV / IRV) 
def ballots_ranked(U: np.ndarray) -> np.ndarray:

    return np.argsort(-U, axis=1)




# 3. WINNER SELECTION FOR EACH VOTING RULE

#  FPTP 
def winner_fptp(top_choices: np.ndarray,
                turnout_mask: np.ndarray,
                P: int) -> int:

    voters = top_choices[turnout_mask]
    if len(voters) == 0:
        return -1
    counts = np.bincount(voters, minlength=P)
    return int(np.argmax(counts))


#  Approval Voting 
def winner_approval(approvals: np.ndarray,
                    turnout_mask: np.ndarray) -> int:

    A = approvals[turnout_mask]
    if len(A) == 0:
        return -1
    scores = A.sum(axis=0)
    return int(np.argmax(scores))


#  IRV / RCV 
def winner_irv(rankings: np.ndarray,
               turnout_mask: np.ndarray,
               P: int) -> int:

    ballots = rankings[turnout_mask]
    if len(ballots) == 0:
        return -1

    active = set(range(P))

    while True:
        # Tally first-choice votes among active candidates
        first = []
        for ballot in ballots:
            for party in ballot:
                if party in active:
                    first.append(party)
                    break

        if len(first) == 0:
            return -1

        counts = np.bincount(first, minlength=P)
        total = counts.sum()

        # Majority?
        if counts.max() > total / 2:
            return int(np.argmax(counts))

        # Eliminate lowest-ranked active candidate
        active_list = list(active)
        active_counts = counts[active_list]
        eliminated = active_list[int(np.argmin(active_counts))]

        active.remove(eliminated)

        if len(active) == 1:
            return active.pop()
