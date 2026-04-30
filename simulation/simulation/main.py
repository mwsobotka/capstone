# main.py
import numpy as np

from voters import sample_voters_normal, sample_voters_mixture
from parties import (
    init_parties,
    update_parties_fptp,
    update_parties_approval,
    update_parties_irv,
    merge_close_parties,
)
from utilities import ( 
    compute_utilities,
    ballots_fptp,
    ballots_approval,
    ballots_ranked,
    compute_turnout_probabilistic,
    )
from metrics import iteration_summary
from plots import (
    plot_voter_distribution,
    plot_party_evolution,
    plot_party_count_over_time,
    print_merging_table,
    plot_utility_heatmap,
    plot_winner_regions,
)
import pickle
import os


def run_simulation(
    voter_model: str = "mixture",
    N: int = 2000,
    P_init: int = 4,
    T: int = 15,
    eta: float = 0.3,
    d_merge: float = 0.5,
    seed: int | None = None, ):
    
    if seed is not None:
        np.random.seed(seed)

    excel_rows = []


    # 1. Generate voters
    if voter_model == "normal":
        voters = sample_voters_normal(N, sigma=1.0)
    else:
        voters = sample_voters_mixture(N)

    # 2. Initialize parties
    parties = init_parties(P_init, dim=2, spread=1.0)

    # Separate party systems for each rule
    parties_fptp = parties.copy()
    parties_approval = parties.copy()
    parties_irv = parties.copy()

    # Histories for each rule
    party_history_fptp: list[np.ndarray] = [parties_fptp.copy()]
    party_history_approval: list[np.ndarray] = [parties_approval.copy()]
    party_history_irv: list[np.ndarray] = [parties_irv.copy()]

    # Party counts for each rule
    party_counts_fptp: list[int] = [parties_fptp.shape[0]]
    party_counts_approval: list[int] = [parties_approval.shape[0]]
    party_counts_irv: list[int] = [parties_irv.shape[0]]

    # Stats for each rule
    iter_stats_fptp: list[dict] = []
    iter_stats_approval: list[dict] = []
    iter_stats_irv: list[dict] = []

    # 3. Evolution loop
    for t in range(T):
        # utilities fptp
        U_fptp = compute_utilities(voters, parties_fptp, gamma=0.5)
        turnout_fptp = compute_turnout_probabilistic(voters, parties_fptp, alpha=1.0)
        choices_fptp = ballots_fptp(U_fptp)

        #utilities approval
        U_approval = compute_utilities(voters, parties_approval, gamma=0.5)
        turnout_approval = compute_turnout_probabilistic(voters, parties_approval, alpha=1.0)
        approvals = ballots_approval(U_approval)

        # utilities irv
        U_irv = compute_utilities(voters, parties_irv, gamma=0.5)
        turnout_irv = compute_turnout_probabilistic(voters, parties_irv, alpha=1.0)
        rankings = ballots_ranked(U_irv)

        # update party positions fptp
        parties_before_fptp = parties_fptp.copy()
        P_before_fptp = parties_before_fptp.shape[0]

        parties_fptp = update_parties_fptp(voters, parties_fptp, choices_fptp, eta=eta)
        parties_fptp = merge_close_parties(parties_fptp, d_merge=d_merge)

        P_after_fptp = parties_fptp.shape[0]
        merges_fptp = P_before_fptp - P_after_fptp

        # update party positions approval
        parties_before_approval = parties_approval.copy()
        P_before_approval = parties_before_approval.shape[0]

        parties_approval = update_parties_approval(voters, parties_approval, approvals, eta=eta)
        parties_approval = merge_close_parties(parties_approval, d_merge=d_merge)

        P_after_approval = parties_approval.shape[0]
        merges_approval = P_before_approval - P_after_approval

        # update party positions irv
        parties_before_irv = parties_irv.copy()
        P_before_irv = parties_before_irv.shape[0]

        parties_irv = update_parties_irv(voters, parties_irv, rankings, eta=eta)
        parties_irv = merge_close_parties(parties_irv, d_merge=d_merge)

        P_after_irv = parties_irv.shape[0]
        merges_irv = P_before_irv - P_after_irv

        # save positions
        party_history_fptp.append(parties_fptp.copy())
        party_counts_fptp.append(P_after_fptp)

        party_history_approval.append(parties_approval.copy())
        party_counts_approval.append(P_after_approval)

        party_history_irv.append(parties_irv.copy())
        party_counts_irv.append(P_after_irv)

        # record metrics fptp
        # AFTER merging
        U_fptp_after = compute_utilities(voters, parties_fptp, gamma=0.5)
        turnout_fptp_after = compute_turnout_probabilistic(voters, parties_fptp, alpha=1.0)
        choices_fptp_after = ballots_fptp(U_fptp_after)

        stats_fptp = iteration_summary(
            voters=voters,
            parties_before=parties_before_fptp,
            parties_after=parties_fptp,
            U=U_fptp_after,
            turnout=turnout_fptp_after,
            choices=choices_fptp_after,
            merges_this_round=merges_fptp,
        )
        iter_stats_fptp.append(stats_fptp)

        # record metrics approval
        U_approval_after = compute_utilities(voters, parties_approval, gamma=0.5)
        turnout_approval_after = compute_turnout_probabilistic(voters, parties_approval, alpha=1.0)

        choices_approval_after = ballots_fptp(U_approval_after)

        stats_approval = iteration_summary(
            voters=voters,
            parties_before=parties_before_approval,
            parties_after=parties_approval,
            U=U_approval_after,
            turnout=turnout_approval_after,
            choices=choices_approval_after,
            merges_this_round=merges_approval,
        )
        iter_stats_approval.append(stats_approval)

        # record metrics irv
        choices_irv_for_metrics = rankings[:, 0]

        U_irv_after = compute_utilities(voters, parties_irv, gamma=0.5)
        turnout_irv_after = compute_turnout_probabilistic(voters, parties_irv, alpha=1.0)
        rankings_after = ballots_ranked(U_irv_after)

        choices_irv_after = rankings_after[:, 0]

        stats_irv = iteration_summary(
            voters=voters,
            parties_before=parties_before_irv,
            parties_after=parties_irv,
            U=U_irv_after,
            turnout=turnout_irv_after,
            choices=choices_irv_after,
            merges_this_round=merges_irv,
        )
        iter_stats_irv.append(stats_irv)

        for system, stats, p_after, merges in [
            ("fptp", stats_fptp, P_after_fptp, merges_fptp),
            ("approval", stats_approval, P_after_approval, merges_approval),
            ("irv", stats_irv, P_after_irv, merges_irv),
        ]:
            cw = stats["condorcet"]
            excel_rows.append({
                "iteration": t + 1,
                "system": system,
                "parties": p_after,
                "merges": merges,
                "winner": stats[system]["winner"],
                "mean_distance": stats[system]["mean_dist"],
                "condorcet": cw,
                "condorcet_match": int(cw != -1 and stats[system]["winner"] == cw),
            })

    # Results
    return {
        "metadata": {
            "voter_model": voter_model,
            "N": N,
            "P_init": P_init,
            "T": T,
            "eta": eta,
            "d_merge": d_merge,
            "seed": seed,
        },
        "voters": voters,
        "rules": {
            "fptp": {
                "party_history": party_history_fptp,
                "party_counts": party_counts_fptp,
                "iter_stats": iter_stats_fptp,
            },
            "approval": {
                "party_history": party_history_approval,
                "party_counts": party_counts_approval,
                "iter_stats": iter_stats_approval,
            },
            "irv": {
                "party_history": party_history_irv,
                "party_counts": party_counts_irv,
                "iter_stats": iter_stats_irv,
            },
        },
        "summary_rows": excel_rows,
    }    


def summarize_results(iter_stats: list[dict]):

    T = len(iter_stats)

    print("\n=== SUMMARY OVER ITERATIONS ===")
    print(f"Total iterations: {T}")

    # winners under each rule and Condorcet frequency
    fptp_winners = [s["fptp"]["winner"] for s in iter_stats]
    approval_winners = [s["approval"]["winner"] for s in iter_stats]
    irv_winners = [s["irv"]["winner"] for s in iter_stats]
    condorcet = [s["condorcet"] for s in iter_stats]

    # how often each rule picks the Condorcet winner (when it exists)
    condorcet_exists = [c for c in condorcet if c != -1]
    if condorcet_exists:
        fptp_matches = sum(
            (cw != -1) and (cw == fptp_winners[i])
            for i, cw in enumerate(condorcet)
        )
        approval_matches = sum(
            (cw != -1) and (cw == approval_winners[i])
            for i, cw in enumerate(condorcet)
        )
        irv_matches = sum(
            (cw != -1) and (cw == irv_winners[i])
            for i, cw in enumerate(condorcet)
        )
        total_c = sum(cw != -1 for cw in condorcet)

        print(f"Condorcet exists in {total_c}/{T} iterations.")
        print(f"FPTP chose Condorcet {fptp_matches}/{total_c} times.")
        print(f"Approval chose Condorcet {approval_matches}/{total_c} times.")
        print(f"IRV chose Condorcet {irv_matches}/{total_c} times.")
    else:
        print("No Condorcet winner found in any iteration.")

    # mean voter–winner distance for each rule
    fptp_dists = [s["fptp"]["mean_dist"] for s in iter_stats]
    approval_dists = [s["approval"]["mean_dist"] for s in iter_stats]
    irv_dists = [s["irv"]["mean_dist"] for s in iter_stats]

    def safe_mean(x):
        arr = np.array(x, dtype=float)
        arr = arr[~np.isnan(arr)]
        return arr.mean() if arr.size > 0 else np.nan

    print("\nAverage mean voter–winner distance:")
    print(f"  FPTP:     {safe_mean(fptp_dists):.3f}")
    print(f"  Approval: {safe_mean(approval_dists):.3f}")
    print(f"  IRV:      {safe_mean(irv_dists):.3f}")

# Plots
def run_plots(voters, party_history, party_counts, iter_stats):

    # 1. Voter distribution with final parties
    plot_voter_distribution(
        voters,
        party_history[-1],
        title="Electorate and Final Party Positions",
    )

    # 2. Party evolution and merging over time
    plot_party_evolution(
        voters,
        party_history,
        title="Party Evolution and Merging Over Time",
    )

    # 3. Party count over time
    print_merging_table(party_counts)
    plot_party_count_over_time(party_counts)

    # 4. Utility field for one party 
    final_parties = party_history[-1]
    if final_parties.shape[0] > 0:
        plot_utility_heatmap(
            final_parties,
            party_index=0,
            xlim=(-3, 3),
            ylim=(-3, 3),
        )

        # 5. Winner regions under FPTP (which party wins where in the plane)
        plot_winner_regions(
            final_parties,
            rule="fptp",
            xlim=(-3, 3),
            ylim=(-3, 3),
        )


def main():
    n_runs = 1000
    T = 100
    output_dir = "saved_runs_final"

    os.makedirs(output_dir, exist_ok=True)


    for i in range(n_runs):
        print(f"Running simulation {i+1}/{n_runs}")

        results = run_simulation(
            voter_model="mixture",
            N=1000,
            P_init=4,
            T=T,
            eta=0.3,
            d_merge=0.5,
            seed=i,   
        )

        with open(f"{output_dir}/run_{i:04d}.pkl", "wb") as f:
            pickle.dump(results, f)

    print(f"\nSaved {n_runs} runs to {output_dir}/")


if __name__ == "__main__":
    main()
