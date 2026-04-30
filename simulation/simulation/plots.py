# plots.py
import numpy as np
import matplotlib.pyplot as plt

from utilities import (
    compute_utilities,
    ballots_fptp,
    ballots_approval,
    ballots_ranked,
    winner_approval,
    winner_irv,
)


# ============================================================
# 1. SINGLE-RUN DIAGNOSTIC PLOTS
# ============================================================

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


def plot_party_evolution(voters: np.ndarray,
                         party_history: list[np.ndarray],
                         title: str = "Party Evolution"):
    plt.figure(figsize=(7, 7))

    # voters as background
    plt.scatter(voters[:, 0], voters[:, 1], s=5, alpha=0.2, label="Voters")

    # historical party positions
    for parties_t in party_history[:-1]:
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


def plot_utility_heatmap(parties: np.ndarray,
                         party_index: int = 0,
                         xlim=(-3, 3),
                         ylim=(-3, 3),
                         resolution: int = 200):
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel()])

    U = compute_utilities(points, parties)
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
    points = np.column_stack([XX.ravel(), YY.ravel()])

    U = compute_utilities(points, parties)
    winners = np.full(points.shape[0], -1, dtype=int)

    if rule == "fptp":
        winners = ballots_fptp(U)

    elif rule == "approval":
        # adjust this call if your ballots_approval signature differs
        approvals = ballots_approval(U, top_frac=approval_percentile)
        for i in range(points.shape[0]):
            winners[i] = winner_approval(
                approvals[i:i+1, :],
                turnout=np.array([True])
            )

    elif rule == "irv":
        rankings = ballots_ranked(U)
        for i in range(points.shape[0]):
            winners[i] = winner_irv(
                rankings[i:i+1, :],
                turnout_mask=np.array([True]),
                P=P
            )
    else:
        raise ValueError(f"Unknown rule: {rule}")

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


def plot_party_count_over_time(party_counts: list[int],
                               title: str = "Party count over time"):
    xs = np.arange(len(party_counts))

    plt.figure(figsize=(6, 4))
    plt.plot(xs, party_counts, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Number of parties")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_merging_table(party_counts: list[int]):
    print("Iteration | Parties")
    print("------------------")
    for t, c in enumerate(party_counts):
        print(f"{t:9d} | {c:7d}")


def plot_metric_distribution(metric_values: list[float] | np.ndarray,
                             metric_name: str = "Metric",
                             bins: int = 20,
                             title: str | None = None):
    metric_values = np.asarray(metric_values, dtype=float)
    metric_values = metric_values[~np.isnan(metric_values)]

    plt.figure(figsize=(6, 4))
    plt.hist(metric_values, bins=bins, alpha=0.7)
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.title(title if title is not None else f"Distribution of {metric_name}")
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. CROSS-RUN COMPARISON PLOTS
# ============================================================

def plot_rule_boxplot(data_dict: dict[str, np.ndarray],
                      ylabel: str,
                      title: str):
    labels = list(data_dict.keys())
    values = []
    for label in labels:
        arr = np.asarray(data_dict[label], dtype=float)
        arr = arr[~np.isnan(arr)]
        values.append(arr)

    plt.figure(figsize=(7, 5))
    plt.boxplot(values, tick_labels=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rule_violinplot(data_dict: dict[str, np.ndarray],
                         ylabel: str,
                         title: str):
    labels = list(data_dict.keys())
    values = []
    for label in labels:
        arr = np.asarray(data_dict[label], dtype=float)
        arr = arr[~np.isnan(arr)]
        values.append(arr)

    plt.figure(figsize=(7, 5))
    parts = plt.violinplot(values, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rule_histograms(data_dict: dict[str, np.ndarray],
                         xlabel: str,
                         title: str,
                         bins: int = 25,
                         alpha: float = 0.5):
    plt.figure(figsize=(7, 5))
    for label, values in data_dict.items():
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        plt.hist(values, bins=bins, alpha=alpha, label=label)

    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rate_bars(rate_dict: dict[str, float],
                   ylabel: str = "Rate",
                   title: str = "Rates by category",
                   ylim=(0, 1)):
    labels = list(rate_dict.keys())
    values = [rate_dict[k] for k in labels]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_grouped_rate_bars(group_dict: dict[str, dict[str, float]],
                           ylabel: str = "Rate",
                           title: str = "Grouped rates",
                           ylim=(0, 1)):
    outer_labels = list(group_dict.keys())
    inner_labels = list(next(iter(group_dict.values())).keys())

    x = np.arange(len(outer_labels))
    width = 0.8 / len(inner_labels)

    plt.figure(figsize=(8, 5))
    for i, inner in enumerate(inner_labels):
        vals = [group_dict[o][inner] for o in outer_labels]
        plt.bar(x + i * width - (len(inner_labels) - 1) * width / 2,
                vals,
                width=width,
                label=inner)

    plt.xticks(x, outer_labels)
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_system_histogram(values,
                          xlabel: str,
                          title: str,
                          bins: int = 20):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_time_series_mean(series_list: list[list[float] | np.ndarray],
                          ylabel: str,
                          title: str):
    arr = np.array(series_list, dtype=float)
    mean_series = np.nanmean(arr, axis=0)
    xs = np.arange(arr.shape[1])

    plt.figure(figsize=(7, 4))
    plt.plot(xs, mean_series, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_time_series_mean_with_band(series_list: list[list[float] | np.ndarray],
                                    ylabel: str,
                                    title: str):
    arr = np.array(series_list, dtype=float)
    mean_series = np.nanmean(arr, axis=0)
    std_series = np.nanstd(arr, axis=0)
    xs = np.arange(arr.shape[1])

    plt.figure(figsize=(7, 4))
    plt.plot(xs, mean_series, marker="o")
    plt.fill_between(xs,
                     mean_series - std_series,
                     mean_series + std_series,
                     alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. CONVENIENCE WRAPPERS FOR YOUR CURRENT ANALYSIS OUTPUT
# ============================================================

def plot_rule_distance_comparison(fptp_dist: np.ndarray,
                                  approval_dist: np.ndarray,
                                  irv_dist: np.ndarray):
    plot_rule_boxplot(
        {
            "FPTP": fptp_dist,
            "Approval": approval_dist,
            "IRV": irv_dist,
        },
        ylabel="Mean distance to winner",
        title="Representation quality by rule"
    )


def plot_rule_welfare_comparison(fptp_util: np.ndarray,
                                 approval_util: np.ndarray,
                                 irv_util: np.ndarray):
    plot_rule_boxplot(
        {
            "FPTP": fptp_util,
            "Approval": approval_util,
            "IRV": irv_util,
        },
        ylabel="Mean utility of winner",
        title="Welfare by rule"
    )


def plot_condorcet_rates(fptp_rate: float,
                         approval_rate: float,
                         irv_rate: float):
    plot_rate_bars(
        {
            "FPTP": fptp_rate,
            "Approval": approval_rate,
            "IRV": irv_rate,
        },
        ylabel="Condorcet match rate",
        title="Condorcet efficiency by rule"
    )


def plot_best_candidate_rates(fptp_rate: float,
                              approval_rate: float,
                              irv_rate: float):
    plot_rate_bars(
        {
            "FPTP": fptp_rate,
            "Approval": approval_rate,
            "IRV": irv_rate,
        },
        ylabel="Best-candidate match rate",
        title="Best-candidate selection by rule"
    )


def plot_rule_agreement_rates(fptp_approval: float,
                              fptp_irv: float,
                              approval_irv: float):
    plot_rate_bars(
        {
            "FPTP vs Approval": fptp_approval,
            "FPTP vs IRV": fptp_irv,
            "Approval vs IRV": approval_irv,
        },
        ylabel="Agreement rate",
        title="Winner agreement across rules"
    )