import os
import pickle
import numpy as np

from plots import (
    plot_boxplot,
    plot_violinplot,
    plot_rate_bars,
    plot_time_series_mean_with_band,
    plot_party_count_shares,
)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "saved_runs_final")


def load_runs():
    runs = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".pkl"):
            with open(os.path.join(DATA_DIR, fname), "rb") as f:
                runs.append(pickle.load(f))
    return runs


def main():
    runs = load_runs()
    print("Runs loaded:", len(runs))

    fptp_final_dist = []
    approval_final_dist = []
    irv_final_dist = []

    fptp_final_party_count = []
    approval_final_party_count = []
    irv_final_party_count = []

    fptp_cond = []
    approval_cond = []
    irv_cond = []

    fptp_party_series = []
    approval_party_series = []
    irv_party_series = []

    fptp_nearest_series = []
    approval_nearest_series = []
    irv_nearest_series = []

    on_fptp_system = {"FPTP": [], "Approval": [], "IRV": []}
    on_approval_system = {"FPTP": [], "Approval": [], "IRV": []}
    on_irv_system = {"FPTP": [], "Approval": [], "IRV": []}

    for run in runs:
        fptp_stats = run["rules"]["fptp"]["iter_stats"]
        approval_stats = run["rules"]["approval"]["iter_stats"]
        irv_stats = run["rules"]["irv"]["iter_stats"]

        fptp_last = fptp_stats[-1]
        approval_last = approval_stats[-1]
        irv_last = irv_stats[-1]

        # final self-consistent outcomes
        fptp_final_dist.append(fptp_last["fptp"]["mean_dist"])
        approval_final_dist.append(approval_last["approval"]["mean_dist"])
        irv_final_dist.append(irv_last["irv"]["mean_dist"])

        fptp_final_party_count.append(fptp_last["num_parties"])
        approval_final_party_count.append(approval_last["num_parties"])
        irv_final_party_count.append(irv_last["num_parties"])

        # use summary_rows for actual condorcet_match
        rows = run["summary_rows"]

        fptp_rows = [r for r in rows if r["system"] == "fptp"]
        approval_rows = [r for r in rows if r["system"] == "approval"]
        irv_rows = [r for r in rows if r["system"] == "irv"]

        fptp_final_row = max(fptp_rows, key=lambda r: r["iteration"])
        approval_final_row = max(approval_rows, key=lambda r: r["iteration"])
        irv_final_row = max(irv_rows, key=lambda r: r["iteration"])

        fptp_cond.append(float(fptp_final_row["condorcet_match"]))
        approval_cond.append(float(approval_final_row["condorcet_match"]))
        irv_cond.append(float(irv_final_row["condorcet_match"]))

        # time series
        fptp_party_series.append([s["num_parties"] for s in fptp_stats])
        approval_party_series.append([s["num_parties"] for s in approval_stats])
        irv_party_series.append([s["num_parties"] for s in irv_stats])

        fptp_nearest_series.append([s["nearest_dist_mean"] for s in fptp_stats])
        approval_nearest_series.append([s["nearest_dist_mean"] for s in approval_stats])
        irv_nearest_series.append([s["nearest_dist_mean"] for s in irv_stats])

        # rule performance on final systems
        on_fptp_system["FPTP"].append(fptp_last["fptp"]["mean_dist"])
        on_fptp_system["Approval"].append(fptp_last["approval"]["mean_dist"])
        on_fptp_system["IRV"].append(fptp_last["irv"]["mean_dist"])

        on_approval_system["FPTP"].append(approval_last["fptp"]["mean_dist"])
        on_approval_system["Approval"].append(approval_last["approval"]["mean_dist"])
        on_approval_system["IRV"].append(approval_last["irv"]["mean_dist"])

        on_irv_system["FPTP"].append(irv_last["fptp"]["mean_dist"])
        on_irv_system["Approval"].append(irv_last["approval"]["mean_dist"])
        on_irv_system["IRV"].append(irv_last["irv"]["mean_dist"])

    fptp_final_dist = np.array(fptp_final_dist, dtype=float)
    approval_final_dist = np.array(approval_final_dist, dtype=float)
    irv_final_dist = np.array(irv_final_dist, dtype=float)

    fptp_final_party_count = np.array(fptp_final_party_count, dtype=float)
    approval_final_party_count = np.array(approval_final_party_count, dtype=float)
    irv_final_party_count = np.array(irv_final_party_count, dtype=float)

    print("Final samples:", len(fptp_final_dist), len(approval_final_dist), len(irv_final_dist))
    print(
        "Condorcet match rates:",
        float(np.mean(fptp_cond)),
        float(np.mean(approval_cond)),
        float(np.mean(irv_cond))
    )

    # 1. final representation quality
    plot_boxplot(
        {
            "FPTP system": fptp_final_dist,
            "Approval system": approval_final_dist,
            "IRV system": irv_final_dist,
        },
        ylabel="Final mean distance to winner",
        title="Final representation quality by evolving system",
    )

    plot_violinplot(
        {
            "FPTP system": fptp_final_dist,
            "Approval system": approval_final_dist,
            "IRV system": irv_final_dist,
        },
        ylabel="Final mean distance to winner",
        title="Distribution of final representation quality",
    )

    # 2. final party counts
    plot_boxplot(
        {
            "FPTP system": fptp_final_party_count,
            "Approval system": approval_final_party_count,
            "IRV system": irv_final_party_count,
        },
        ylabel="Final number of parties",
        title="Final party count by evolving system",
    )

    # 3. condorcet match rates
    plot_rate_bars(
        {
            "FPTP system": float(np.mean(fptp_cond)),
            "Approval system": float(np.mean(approval_cond)),
            "IRV system": float(np.mean(irv_cond)),
        },
        ylabel="Condorcet match rate",
        title="Condorcet efficiency by evolving system",
    )

    # 4. time series party counts
    plot_time_series_mean_with_band(
        fptp_party_series,
        ylabel="Mean number of parties",
        title="FPTP system: average party count over time",
    )

    plot_time_series_mean_with_band(
        approval_party_series,
        ylabel="Mean number of parties",
        title="Approval system: average party count over time",
    )

    plot_time_series_mean_with_band(
        irv_party_series,
        ylabel="Mean number of parties",
        title="IRV system: average party count over time",
    )

    # 5. better party-count composition plots
    plot_party_count_shares(
        fptp_party_series,
        title="FPTP system: party-count composition over time",
    )

    plot_party_count_shares(
        approval_party_series,
        title="Approval system: party-count composition over time",
    )

    plot_party_count_shares(
        irv_party_series,
        title="IRV system: party-count composition over time",
    )

    # 6. time series nearest-party distance
    plot_time_series_mean_with_band(
        fptp_nearest_series,
        ylabel="Mean nearest-party distance",
        title="FPTP system: nearest-party distance over time",
    )

    plot_time_series_mean_with_band(
        approval_nearest_series,
        ylabel="Mean nearest-party distance",
        title="Approval system: nearest-party distance over time",
    )

    plot_time_series_mean_with_band(
        irv_nearest_series,
        ylabel="Mean nearest-party distance",
        title="IRV system: nearest-party distance over time",
    )

    # 7. rule performance on final systems
    plot_boxplot(
        on_fptp_system,
        ylabel="Mean distance to winner",
        title="Rule performance on final FPTP systems",
    )

    plot_boxplot(
        on_approval_system,
        ylabel="Mean distance to winner",
        title="Rule performance on final Approval systems",
    )

    plot_boxplot(
        on_irv_system,
        ylabel="Mean distance to winner",
        title="Rule performance on final IRV systems",
    )


if __name__ == "__main__":
    main()