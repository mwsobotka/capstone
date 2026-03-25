# metrics.py
import numpy as np

from utilities import (
    compute_utilities,
    ballots_fptp,
    ballots_approval,
    ballots_ranked,
    winner_fptp,
    winner_approval,
    winner_irv)


# 1. DISTANCE METRICS

def party_movements(old_parties: np.ndarray,
                    new_parties: np.ndarray) -> np.ndarray:

    return np.linalg.norm(new_parties - old_parties, axis=1)


def mean_distance_to_party(voters: np.ndarray,
                           party_pos: np.ndarray) -> float:

    d = np.linalg.norm(voters - party_pos, axis=1)
    return float(d.mean())


def mean_distance_to_nearest_party(voters: np.ndarray,
                                   parties: np.ndarray) -> float:

    v = voters[:, np.newaxis, :]      # (N,1,2)
    p = parties[np.newaxis, :, :]     # (1,P,2)
    d = np.linalg.norm(v - p, axis=2) # (N,P)
    d_min = d.min(axis=1)             # (N,)
    return float(d_min.mean())


def mean_supporter_distance_per_party(voters: np.ndarray,
                                      parties: np.ndarray,
                                      choices: np.ndarray) -> np.ndarray:

    P = parties.shape[0]
    result = np.full(P, np.nan)
    for p in range(P):
        idx = np.where(choices == p)[0]
        if len(idx) == 0:
            continue
        v_sup = voters[idx, :]
        d = np.linalg.norm(v_sup - parties[p], axis=1)
        result[p] = d.mean()
    return result


# 2. CONDORCET WINNER

def condorcet_winner_from_utilities(U: np.ndarray) -> int:
    N, P = U.shape
    # rankings[i, :] gives party order from best (0) to worst (P-1)
    rankings = np.argsort(-U, axis=1)  # (N, P)

    # ranks[i, party] = position of 'party' in voter i's ranking (0 = top)
    ranks = np.zeros((N, P), dtype=int)
    for i in range(N):
        for pos, party in enumerate(rankings[i]):
            ranks[i, party] = pos

    # pairwise majority comparisons
    def majority_prefers(p, q) -> bool:
        # True if a majority of voters prefer p to q
        return np.sum(ranks[:, p] < ranks[:, q]) > N / 2

    # candidate that beats all others pairwise
    for p in range(P):
        if all(p == q or majority_prefers(p, q) for q in range(P)):
            return p

    return -1


# 3. RULE-SPECIFIC RESULT WRAPPERS

def fptp_result(U: np.ndarray,
                voters: np.ndarray,
                parties: np.ndarray,
                turnout: np.ndarray) -> dict:

    N, P = U.shape
    top = ballots_fptp(U)

    valid = turnout
    if not valid.any():
        return {
            "winner": -1,
            "tie": False,
            "mean_dist": np.nan
        }

    votes = top[valid]
    counts = np.bincount(votes, minlength=P)
    max_votes = counts.max()
    tie_flag = (counts == max_votes).sum() > 1

    w = int(np.argmax(counts))
    mean_dist = mean_distance_to_party(voters, parties[w])

    return {
        "winner": w,
        "tie": tie_flag,
        "mean_dist": mean_dist,
        "counts": counts,
    }


def approval_result(U: np.ndarray,
                    voters: np.ndarray,
                    parties: np.ndarray,
                    turnout: np.ndarray,
                    percentile: float = 0.5) -> dict:

    from utilities import ballots_approval  
    approvals = ballots_approval(U, top_frac=percentile)
    valid = turnout
    if not valid.any():
        return {
            "winner": -1,
            "tie": False,
            "mean_dist": np.nan
        }

    A = approvals[valid]
    scores = A.sum(axis=0)
    max_score = scores.max()
    tie_flag = (scores == max_score).sum() > 1

    w = int(np.argmax(scores))
    mean_dist = mean_distance_to_party(voters, parties[w])

    return {
        "winner": w,
        "tie": tie_flag,
        "mean_dist": mean_dist,
        "scores": scores,
    }


def irv_result(U: np.ndarray,
               voters: np.ndarray,
               parties: np.ndarray,
               turnout: np.ndarray) -> dict:

    rankings = ballots_ranked(U)
    w = winner_irv(rankings, turnout_mask=turnout, P=parties.shape[0])

    if w == -1:
        return {
            "winner": -1,
            "tie": False,
            "mean_dist": np.nan}

    mean_dist = mean_distance_to_party(voters, parties[w])

    return {
        "winner": w,
        "tie": False,  # could add detailed tie-tracking later
        "mean_dist": mean_dist,}


# 4. PER-ITERATION SUMMARY

def iteration_summary(voters: np.ndarray,
                      parties_before: np.ndarray,
                      parties_after: np.ndarray,
                      U: np.ndarray,
                      turnout: np.ndarray,
                      choices: np.ndarray,
                      merges_this_round: int) -> dict:

    P_after = parties_after.shape[0]

    # movement per party 
    P_before = parties_before.shape[0]
    P_min = min(P_before, P_after)
    movement = party_movements(parties_before[:P_min, :],
                               parties_after[:P_min, :])

    # distances
    nearest_dist_mean = mean_distance_to_nearest_party(voters, parties_after)
    supporter_dist = mean_supporter_distance_per_party(voters,
                                                       parties_after,
                                                       choices)

    # Condorcet winner based on utilities
    cw = condorcet_winner_from_utilities(U)

    # rule-specific winners
    fptp = fptp_result(U, voters, parties_after, turnout)
    approval = approval_result(U, voters, parties_after, turnout,
                               percentile=0.5)
    irv = irv_result(U, voters, parties_after, turnout)

    return {
        "num_parties": P_after,
        "movement": movement,
        "nearest_dist_mean": nearest_dist_mean,
        "supporter_dist": supporter_dist,
        "merges": merges_this_round,
        "condorcet": cw,
        "fptp": fptp,
        "approval": approval,
        "irv": irv,}
