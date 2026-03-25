# plots.py
import numpy as np
import matplotlib.pyplot as plt

from utilities import (
    compute_utilities,
    ballots_fptp,
    ballots_approval,
    ballots_ranked,
    winner_fptp,
    winner_approval,
    winner_irv,
)

# 1. VOTER DISTRIBUTION PLOTS

def plot_voter_distribution(voters: np.ndarray,
                            parties: np.ndarray | None = None,
                            title: str = "Voter Distribution"):

    plt.figure(figsize=(7, 7))
    plt.scatter(voters[:, 0], voters[:, 1],
                s=5, alpha=0.3, label="Voters")

    if parties is not None:
        plt.scatter(parties[:, 0], parties[:, 1],
                    s=200, marker="X", label="Parties")

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.grid(alpha=0.2)
    plt.xlabel("Ideology Dimension 1")
    plt.ylabel("Ideology Dimension 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 2. PARTY TRAJECTORIES OVER TIME

def plot_party_evolution(voters: np.ndarray,
                         party_history: list[np.ndarray],
                         title: str = "Party Evolution"):

    
    plt.figure(figsize=(7, 7))

    # voters as background
    plt.scatter(voters[:, 0], voters[:, 1], s=5, alpha=0.2, label="Voters")

    # all historical party positions
    for t, parties_t in enumerate(party_history[:-1]):
        plt.scatter(
            parties_t[:, 0],
            parties_t[:, 1],
            s=20,
            alpha=0.4,
        )

    # final party positions
    final = party_history[-1]
    plt.scatter(
        final[:, 0],
        final[:, 1],
        s=200,
        marker="X",
        label="Final parties",
    )

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.grid(alpha=0.2)
    plt.xlabel("Ideology Dimension 1")
    plt.ylabel("Ideology Dimension 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 3. UTILITY HEATMAP 

def plot_utility_heatmap(parties: np.ndarray,
                         party_index: int = 0,
                         xlim=(-3, 3),
                         ylim=(-3, 3),
                         resolution: int = 200):
   
    # grid of hypothetical voter locations
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel()])  # (R^2 x 2)

    # compute utilities for all points to all parties
    U = compute_utilities(points, parties)  # (R^2 x P)
    U_p = U[:, party_index].reshape(XX.shape)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        U_p,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        aspect="equal",
    )
    plt.colorbar(im, label=f"Utility for party {party_index}")
    plt.scatter(parties[party_index, 0],
                parties[party_index, 1],
                s=150, marker="X")
    plt.xlabel("Ideology Dimension 1")
    plt.ylabel("Ideology Dimension 2")
    plt.title(f"Utility field for party {party_index}")
    plt.tight_layout()
    plt.show()


# 4. WINNER REGIONS UNDER DIFFERENT RULES

def plot_winner_regions(parties: np.ndarray,
                        rule: str = "fptp",
                        xlim=(-3, 3),
                        ylim=(-3, 3),
                        resolution: int = 200,
                        approval_percentile: float = 0.5):

    P = parties.shape[0]

    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel()])  # (R^2 x 2)

    # utilities for all hypothetical voters
    U = compute_utilities(points, parties)

    # turnout mask: here assume everyone "turns out"
    turnout = np.ones(points.shape[0], dtype=bool)

    winners = np.full(points.shape[0], -1, dtype=int)

    if rule == "fptp":
        top = ballots_fptp(U)
        # winner for each point is just top choice
        winners = top

    elif rule == "approval":
        approvals = ballots_approval(U, percentile=approval_percentile)
        # each point is an independent one-voter "election"
        for i in range(points.shape[0]):
            winners[i] = winner_approval(approvals[i:i+1, :],
                                         turnout=np.array([True]))
    elif rule == "irv":
        rankings = ballots_ranked(U)
        for i in range(points.shape[0]):
            winners[i] = winner_irv(rankings[i:i+1, :],
                                    turnout_mask=np.array([True]),
                                    P=P)
    else:
        raise ValueError(f"Unknown rule: {rule}")

    # reshape winners back to grid
    W = winners.reshape(XX.shape)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        W,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        aspect="equal",
    )
    plt.colorbar(im, label="Winning party index")
    plt.scatter(parties[:, 0], parties[:, 1],
                s=150, marker="X")
    plt.xlabel("Ideology Dimension 1")
    plt.ylabel("Ideology Dimension 2")
    plt.title(f"Winner regions ({rule.upper()})")
    plt.tight_layout()
    plt.show()


# 5. MERGING EVENTS TABLE / PLOT

def plot_party_count_over_time(party_counts: list[int]):

    T = len(party_counts) - 1
    xs = np.arange(T + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, party_counts, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Number of parties")
    plt.title("Party count over time (merging events)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_merging_table(party_counts: list[int]):
    
    print("Iteration | Parties")
    print("------------------")
    for t, c in enumerate(party_counts):
        print(f"{t:9d} | {c:7d}")


# 6. METRIC DISTRIBUTIONS ACROSS MONTE CARLO RUNS

def plot_metric_distribution(metric_values: list[float],
                             metric_name: str = "Mean voter-winner distance"):

    metric_values = np.array(metric_values)

    plt.figure(figsize=(6, 4))
    plt.hist(metric_values, bins=20, alpha=0.7)
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {metric_name}")
    plt.tight_layout()
    plt.show()
