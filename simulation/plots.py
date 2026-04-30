import os
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
# 0. HELPER FUNCTIONS
# ============================================================

def _clean_array(values):
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def _finalize_plot(show=True, save_path=None):
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# ============================================================
# 1. SINGLE-RUN DIAGNOSTIC PLOTS
# ============================================================

def plot_voter_distribution(voters,
                            parties=None,
                            title="Voter Distribution",
                            show=True,
                            save_path=None):
    plt.figure(figsize=(7, 7))
    plt.scatter(voters[:, 0], voters[:, 1], s=5, alpha=0.3, label="Voters")

    if parties is not None:
        plt.scatter(parties[:, 0], parties[:, 1], s=180, marker="X", label="Parties")

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.grid(alpha=0.2)
    plt.xlabel("Ideology Dimension 1")
    plt.ylabel("Ideology Dimension 2")
    plt.title(title)
    plt.legend()
    _finalize_plot(show=show, save_path=save_path)


def plot_party_evolution(voters,
                         party_history,
                         title="Party Evolution",
                         show=True,
                         save_path=None):
    plt.figure(figsize=(7, 7))
    plt.scatter(voters[:, 0], voters[:, 1], s=5, alpha=0.2, label="Voters")

    for parties_t in party_history[:-1]:
        plt.scatter(parties_t[:, 0], parties_t[:, 1], s=20, alpha=0.35)

    final = party_history[-1]
    plt.scatter(final[:, 0], final[:, 1], s=180, marker="X", label="Final parties")

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.grid(alpha=0.2)
    plt.xlabel("Ideology Dimension 1")
    plt.ylabel("Ideology Dimension 2")
    plt.title(title)
    plt.legend()
    _finalize_plot(show=show, save_path=save_path)


def plot_party_count_over_time(party_counts,
                               title="Party Count Over Time",
                               show=True,
                               save_path=None):
    xs = np.arange(len(party_counts))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, party_counts, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Parties")
    plt.title(title)
    plt.grid(alpha=0.3)
    _finalize_plot(show=show, save_path=save_path)


def plot_metric_distribution(metric_values,
                             metric_name="Metric",
                             bins=20,
                             title=None,
                             show=True,
                             save_path=None):
    vals = _clean_array(metric_values)
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=bins, alpha=0.75)
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.title(title if title is not None else f"Distribution of {metric_name}")
    _finalize_plot(show=show, save_path=save_path)


def plot_utility_heatmap(parties,
                         party_index=0,
                         xlim=(-3, 3),
                         ylim=(-3, 3),
                         resolution=200,
                         show=True,
                         save_path=None):
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
    plt.scatter(parties[party_index, 0], parties[party_index, 1], s=150, marker="X")
    plt.xlabel("Ideology Dimension 1")
    plt.ylabel("Ideology Dimension 2")
    plt.title(f"Utility Field for Party {party_index}")
    _finalize_plot(show=show, save_path=save_path)


def plot_winner_regions(parties,
                        rule="fptp",
                        xlim=(-3, 3),
                        ylim=(-3, 3),
                        resolution=200,
                        approval_percentile=0.5,
                        show=True,
                        save_path=None):
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
    plt.scatter(parties[:, 0], parties[:, 1], s=150, marker="X")
    plt.xlabel("Ideology Dimension 1")
    plt.ylabel("Ideology Dimension 2")
    plt.title(f"Winner Regions ({rule.upper()})")
    _finalize_plot(show=show, save_path=save_path)


# ============================================================
# 2. GENERIC COMPARISON PLOTS
# ============================================================

def plot_boxplot(data_dict,
                 ylabel,
                 title,
                 show=True,
                 save_path=None):
    labels = list(data_dict.keys())
    values = [_clean_array(data_dict[k]) for k in labels]

    plt.figure(figsize=(7, 5))
    plt.boxplot(values, tick_labels=labels, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    _finalize_plot(show=show, save_path=save_path)


def plot_violinplot(data_dict,
                    ylabel,
                    title,
                    show=True,
                    save_path=None):
    labels = list(data_dict.keys())
    values = [_clean_array(data_dict[k]) for k in labels]

    plt.figure(figsize=(7, 5))
    plt.violinplot(values, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    _finalize_plot(show=show, save_path=save_path)


def plot_histograms(data_dict,
                    xlabel,
                    title,
                    bins=25,
                    alpha=0.5,
                    show=True,
                    save_path=None):
    plt.figure(figsize=(7, 5))
    for label, values in data_dict.items():
        vals = _clean_array(values)
        plt.hist(vals, bins=bins, alpha=alpha, label=label)

    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    _finalize_plot(show=show, save_path=save_path)


def plot_rate_bars(rate_dict,
                   ylabel="Rate",
                   title="Rates by Category",
                   ylim=(0, 1),
                   show=True,
                   save_path=None):
    labels = list(rate_dict.keys())
    values = [rate_dict[k] for k in labels]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    _finalize_plot(show=show, save_path=save_path)


def plot_grouped_rate_bars(group_dict,
                           ylabel="Rate",
                           title="Grouped Rates",
                           ylim=(0, 1),
                           show=True,
                           save_path=None):
    outer_labels = list(group_dict.keys())
    inner_labels = list(next(iter(group_dict.values())).keys())

    x = np.arange(len(outer_labels))
    width = 0.8 / len(inner_labels)

    plt.figure(figsize=(8, 5))
    for i, inner in enumerate(inner_labels):
        vals = [group_dict[o][inner] for o in outer_labels]
        plt.bar(
            x + i * width - (len(inner_labels) - 1) * width / 2,
            vals,
            width=width,
            label=inner
        )

    plt.xticks(x, outer_labels)
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _finalize_plot(show=show, save_path=save_path)


# ============================================================
# 3. TIME-SERIES PLOTS
# ============================================================

def plot_time_series_mean(series_list,
                          ylabel,
                          title,
                          show=True,
                          save_path=None):
    arr = np.array(series_list, dtype=float)
    mean_series = np.nanmean(arr, axis=0)
    xs = np.arange(arr.shape[1])

    plt.figure(figsize=(7, 4))
    plt.plot(xs, mean_series, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    _finalize_plot(show=show, save_path=save_path)


def plot_time_series_mean_with_band(series_list,
                                    ylabel,
                                    title,
                                    show=True,
                                    save_path=None):
    arr = np.array(series_list, dtype=float)
    mean_series = np.nanmean(arr, axis=0)
    std_series = np.nanstd(arr, axis=0)
    xs = np.arange(arr.shape[1])

    plt.figure(figsize=(7, 4))
    plt.plot(xs, mean_series, marker="o")
    plt.fill_between(
        xs,
        mean_series - std_series,
        mean_series + std_series,
        alpha=0.2
    )
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    _finalize_plot(show=show, save_path=save_path)


def plot_party_count_shares(series_list,
                            title="Party count composition over time",
                            show=True,
                            save_path=None):
    arr = np.array(series_list, dtype=int)
    xs = np.arange(arr.shape[1])

    unique_counts = sorted(np.unique(arr))
    shares = {k: np.mean(arr == k, axis=0) for k in unique_counts}

    plt.figure(figsize=(8, 5))
    bottom = np.zeros(arr.shape[1])

    for k in unique_counts:
        plt.bar(xs, shares[k], bottom=bottom, label=f"{k} parties", width=1.0)
        bottom += shares[k]

    plt.xlabel("Iteration")
    plt.ylabel("Share of runs")
    plt.title(title)
    plt.legend()
    _finalize_plot(show=show, save_path=save_path)


# ============================================================
# 4. PROJECT-SPECIFIC PAPER PLOTS
# ============================================================

def plot_final_distance_by_system(fptp_vals,
                                  approval_vals,
                                  irv_vals,
                                  show=True,
                                  save_path=None):
    plot_boxplot(
        {
            "FPTP system": fptp_vals,
            "Approval system": approval_vals,
            "IRV system": irv_vals,
        },
        ylabel="Final mean distance to winner",
        title="Final representation quality by evolving system",
        show=show,
        save_path=save_path
    )


def plot_final_welfare_by_system(fptp_vals,
                                 approval_vals,
                                 irv_vals,
                                 show=True,
                                 save_path=None):
    plot_boxplot(
        {
            "FPTP system": fptp_vals,
            "Approval system": approval_vals,
            "IRV system": irv_vals,
        },
        ylabel="Final welfare",
        title="Final welfare by evolving system",
        show=show,
        save_path=save_path
    )


def plot_final_turnout_by_system(fptp_vals,
                                 approval_vals,
                                 irv_vals,
                                 show=True,
                                 save_path=None):
    plot_boxplot(
        {
            "FPTP system": fptp_vals,
            "Approval system": approval_vals,
            "IRV system": irv_vals,
        },
        ylabel="Final turnout rate",
        title="Final turnout by evolving system",
        show=show,
        save_path=save_path
    )


def plot_final_party_count_by_system(fptp_vals,
                                     approval_vals,
                                     irv_vals,
                                     show=True,
                                     save_path=None):
    plot_boxplot(
        {
            "FPTP system": fptp_vals,
            "Approval system": approval_vals,
            "IRV system": irv_vals,
        },
        ylabel="Final number of parties",
        title="Final party count by evolving system",
        show=show,
        save_path=save_path
    )


def plot_condorcet_match_by_system(fptp_rate,
                                   approval_rate,
                                   irv_rate,
                                   show=True,
                                   save_path=None):
    plot_rate_bars(
        {
            "FPTP system": fptp_rate,
            "Approval system": approval_rate,
            "IRV system": irv_rate,
        },
        ylabel="Condorcet match rate",
        title="Condorcet efficiency by evolving system",
        show=show,
        save_path=save_path
    )


def plot_best_candidate_rate_by_system(fptp_rate,
                                       approval_rate,
                                       irv_rate,
                                       show=True,
                                       save_path=None):
    plot_rate_bars(
        {
            "FPTP system": fptp_rate,
            "Approval system": approval_rate,
            "IRV system": irv_rate,
        },
        ylabel="Best-candidate match rate",
        title="Best-candidate selection by evolving system",
        show=show,
        save_path=save_path
    )


# ============================================================
# 5. RULE-EVALUATION-ON-FINAL-SYSTEM PLOTS
# ============================================================

def plot_rule_performance_on_system(distance_dict,
                                    system_name,
                                    show=True,
                                    save_path=None):
    plot_boxplot(
        distance_dict,
        ylabel="Mean distance to winner",
        title=f"Rule performance on final {system_name} systems",
        show=show,
        save_path=save_path
    )


def plot_rule_welfare_on_system(welfare_dict,
                                system_name,
                                show=True,
                                save_path=None):
    plot_boxplot(
        welfare_dict,
        ylabel="Welfare",
        title=f"Rule welfare on final {system_name} systems",
        show=show,
        save_path=save_path
    )