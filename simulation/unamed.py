# make_presentation_plots.py
import os
import pickle
import numpy as np

from utilities import compute_utilities
from plots import (
    plot_voter_distribution,
    plot_party_evolution,
    plot_party_count_over_time,
    plot_final_distance_by_system,
    plot_final_welfare_by_system,
    plot_condorcet_match_by_system,
    plot_best_candidate_rate_by_system,
    plot_winner_regions,
)

RUNS_DIR = "saved_runs_final"
OUT_DIR = "presentation_plots"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_runs(runs_dir: str) -> list[dict]:
    runs = []
    for fname in sorted(os.listdir(runs_dir)):
        if not fname.endswith(".pkl"):
            continue
        path = os.path.join(runs_dir, fname)
        with open(path, "rb") as f:
            runs.append(pickle.load(f))
    if not runs:
        raise FileNotFoundError(f"No .pkl runs found in {runs_dir}")
    return runs


def safe_mean_utility(voters: np.ndarray, parties: np.ndarray, winner: int) -> float:
    if winner == -1:
        return np.nan
    U = compute_utilities(voters, parties)
    return float(U[:, winner].mean())


def best_candidate_index(voters: np.ndarray, parties: np.ndarray) -> int:
    U = compute_utilities(voters, parties)
    return int(np.argmax(U.mean(axis=0)))


def final_rule_metrics(run: dict, rule: str) -> dict:
    voters = run["voters"]
    parties = run["rules"][rule]["party_history"][-1]
    final_stats = run["rules"][rule]["iter_stats"][-1]

    winner = final_stats[rule]["winner"]
    condorcet = final_stats["condorcet"]
    welfare = safe_mean_utility(voters, parties, winner)
    best_idx = best_candidate_index(voters, parties)

    return {
        "distance": final_stats[rule]["mean_dist"],
        "welfare": welfare,
        "condorcet_match": float(condorcet != -1 and winner == condorcet),
        "best_match": float(winner == best_idx),
        "winner": winner,
        "parties": parties,
        "voters": voters,
        "party_counts": run["rules"][rule]["party_counts"],
        "party_history": run["rules"][rule]["party_history"],
    }


def choose_representative_run(runs: list[dict], rule: str = "approval") -> dict:
    """
    Pick a visually reasonable run:
    choose the run whose final distance is closest to the median for the chosen rule.
    """
    distances = []
    for run in runs:
        final_stats = run["rules"][rule]["iter_stats"][-1]
        distances.append(final_stats[rule]["mean_dist"])

    distances = np.array(distances, dtype=float)
    target = np.nanmedian(distances)
    idx = int(np.nanargmin(np.abs(distances - target)))
    return runs[idx]


def main() -> None:
    ensure_dir(OUT_DIR)
    runs = load_runs(RUNS_DIR)

    # Aggregate final metrics for each evolving system
    fptp_distance = []
    approval_distance = []
    irv_distance = []

    fptp_welfare = []
    approval_welfare = []
    irv_welfare = []

    fptp_condorcet = []
    approval_condorcet = []
    irv_condorcet = []

    fptp_best = []
    approval_best = []
    irv_best = []

    for run in runs:
        mf = final_rule_metrics(run, "fptp")
        ma = final_rule_metrics(run, "approval")
        mi = final_rule_metrics(run, "irv")

        fptp_distance.append(mf["distance"])
        approval_distance.append(ma["distance"])
        irv_distance.append(mi["distance"])

        fptp_welfare.append(mf["welfare"])
        approval_welfare.append(ma["welfare"])
        irv_welfare.append(mi["welfare"])

        fptp_condorcet.append(mf["condorcet_match"])
        approval_condorcet.append(ma["condorcet_match"])
        irv_condorcet.append(mi["condorcet_match"])

        fptp_best.append(mf["best_match"])
        approval_best.append(ma["best_match"])
        irv_best.append(mi["best_match"])

    # 1. Main cross-run result plots
    plot_final_distance_by_system(
        fptp_distance,
        approval_distance,
        irv_distance,
        show=False,
        save_path=os.path.join(OUT_DIR, "01_representation_distance.png"),
    )

    plot_final_welfare_by_system(
        fptp_welfare,
        approval_welfare,
        irv_welfare,
        show=False,
        save_path=os.path.join(OUT_DIR, "02_welfare.png"),
    )

    plot_condorcet_match_by_system(
        float(np.nanmean(fptp_condorcet)),
        float(np.nanmean(approval_condorcet)),
        float(np.nanmean(irv_condorcet)),
        show=False,
        save_path=os.path.join(OUT_DIR, "03_condorcet_match.png"),
    )

    plot_best_candidate_rate_by_system(
        float(np.nanmean(fptp_best)),
        float(np.nanmean(approval_best)),
        float(np.nanmean(irv_best)),
        show=False,
        save_path=os.path.join(OUT_DIR, "04_best_candidate_match.png"),
    )

    # 2. Single-run visuals for explaining the model
    rep_run = choose_representative_run(runs, rule="approval")

    voters = rep_run["voters"]

    approval_history = rep_run["rules"]["approval"]["party_history"]
    approval_counts = rep_run["rules"]["approval"]["party_counts"]
    approval_final = approval_history[-1]

    plot_voter_distribution(
        voters,
        parties=approval_final,
        title="Voter Distribution and Final Party Positions",
        show=False,
        save_path=os.path.join(OUT_DIR, "05_voter_distribution.png"),
    )

    plot_party_evolution(
        voters,
        approval_history,
        title="Party Evolution Over Time (Approval System)",
        show=False,
        save_path=os.path.join(OUT_DIR, "06_party_evolution_approval.png"),
    )

    plot_party_count_over_time(
        approval_counts,
        title="Party Count Over Time (Approval System)",
        show=False,
        save_path=os.path.join(OUT_DIR, "07_party_count_approval.png"),
    )

    plot_winner_regions(
        approval_final,
        rule="fptp",
        show=False,
        save_path=os.path.join(OUT_DIR, "08_winner_regions_fptp_on_approval_run.png"),
    )

    plot_winner_regions(
        approval_final,
        rule="approval",
        show=False,
        save_path=os.path.join(OUT_DIR, "09_winner_regions_approval_on_approval_run.png"),
    )

    plot_winner_regions(
        approval_final,
        rule="irv",
        show=False,
        save_path=os.path.join(OUT_DIR, "10_winner_regions_irv_on_approval_run.png"),
    )

    # 3. Save a text summary for quick reference
    summary_path = os.path.join(OUT_DIR, "plot_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total runs loaded: {len(runs)}\n")
        f.write(f"Mean final distance - FPTP: {np.nanmean(fptp_distance):.3f}\n")
        f.write(f"Mean final distance - Approval: {np.nanmean(approval_distance):.3f}\n")
        f.write(f"Mean final distance - IRV: {np.nanmean(irv_distance):.3f}\n\n")

        f.write(f"Mean welfare - FPTP: {np.nanmean(fptp_welfare):.3f}\n")
        f.write(f"Mean welfare - Approval: {np.nanmean(approval_welfare):.3f}\n")
        f.write(f"Mean welfare - IRV: {np.nanmean(irv_welfare):.3f}\n\n")

        f.write(f"Condorcet match - FPTP: {np.nanmean(fptp_condorcet):.3f}\n")
        f.write(f"Condorcet match - Approval: {np.nanmean(approval_condorcet):.3f}\n")
        f.write(f"Condorcet match - IRV: {np.nanmean(irv_condorcet):.3f}\n\n")

        f.write(f"Best-candidate match - FPTP: {np.nanmean(fptp_best):.3f}\n")
        f.write(f"Best-candidate match - Approval: {np.nanmean(approval_best):.3f}\n")
        f.write(f"Best-candidate match - IRV: {np.nanmean(irv_best):.3f}\n")

    print(f"Saved presentation plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()