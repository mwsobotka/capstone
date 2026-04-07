import os
import pickle
import numpy as np
import scipy.stats

from utilities import compute_utilities

RUNS_DIR = "saved_runs_final"
results = []

for file in os.listdir(RUNS_DIR):
    if not file.endswith(".pkl"):
        continue

    with open(os.path.join(RUNS_DIR, file), "rb") as f:
        run = pickle.load(f)

    summary = run["rules"]["fptp"]["iter_stats"]
    final = summary[-1]

    voters = run["voters"]
    parties = run["rules"]["fptp"]["party_history"][-1]
    U = compute_utilities(voters, parties)

    best_candidate = int(np.argmax(U.mean(axis=0)))

    row = {
        "num_parties_final": final["num_parties"],
        "nearest_dist_final": final["nearest_dist_mean"],
        "condorcet": final["condorcet"],
        "best_candidate": best_candidate,

        "fptp_winner": final["fptp"]["winner"],
        "approval_winner": final["approval"]["winner"],
        "irv_winner": final["irv"]["winner"],

        "fptp_dist": final["fptp"]["mean_dist"],
        "approval_dist": final["approval"]["mean_dist"],
        "irv_dist": final["irv"]["mean_dist"],

        "fptp_tie": final["fptp"]["tie"],
        "approval_tie": final["approval"]["tie"],
        "irv_tie": final["irv"]["tie"],
    }

    # welfare / mean utility of each rule's winner
    row.update({
        "fptp_util": U[:, row["fptp_winner"]].mean() if row["fptp_winner"] != -1 else np.nan,
        "approval_util": U[:, row["approval_winner"]].mean() if row["approval_winner"] != -1 else np.nan,
        "irv_util": U[:, row["irv_winner"]].mean() if row["irv_winner"] != -1 else np.nan,
    })

    # condorcet and best-candidate match
    row.update({
        "fptp_condorcet": (row["fptp_winner"] == row["condorcet"]) if row["condorcet"] != -1 else np.nan,
        "approval_condorcet": (row["approval_winner"] == row["condorcet"]) if row["condorcet"] != -1 else np.nan,
        "irv_condorcet": (row["irv_winner"] == row["condorcet"]) if row["condorcet"] != -1 else np.nan,

        "fptp_best": row["fptp_winner"] == best_candidate,
        "approval_best": row["approval_winner"] == best_candidate,
        "irv_best": row["irv_winner"] == best_candidate,
    })

    num_parties = [s["num_parties"] for s in summary]
    merges = [s["merges"] for s in summary]
    movement = [np.nanmean(s["movement"]) for s in summary]
    nearest = [s["nearest_dist_mean"] for s in summary]

    row.update({
        "num_parties_avg": np.mean(num_parties),
        "num_parties_change": num_parties[-1] - num_parties[0],
        "merges_total": np.sum(merges),
        "movement_avg": np.mean(movement),
        "nearest_dist_avg": np.mean(nearest),
    })

    results.append(row)

# convert to arrays
fptp_dist = np.array([r["fptp_dist"] for r in results])
approval_dist = np.array([r["approval_dist"] for r in results])
irv_dist = np.array([r["irv_dist"] for r in results])

fptp_util = np.array([r["fptp_util"] for r in results])
approval_util = np.array([r["approval_util"] for r in results])
irv_util = np.array([r["irv_util"] for r in results])

num_parties_final = np.array([r["num_parties_final"] for r in results])
num_parties_avg = np.array([r["num_parties_avg"] for r in results])
merges_total = np.array([r["merges_total"] for r in results])
movement_avg = np.array([r["movement_avg"] for r in results])

# stats
def summarize(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    mean = np.mean(x)
    std = np.std(x)
    ci = 1.96 * std / np.sqrt(len(x))
    return mean, std, ci

def rate(condition_list):
    arr = np.array(condition_list, dtype=float)
    arr = arr[~np.isnan(arr)]
    return np.mean(arr)

# RULE PERFORMANCE
def condorcet_match(rule):
    return rate([r[f"{rule}_condorcet"] for r in results])

# tie rates
fptp_tie_rate = rate([r["fptp_tie"] for r in results])
approval_tie_rate = rate([r["approval_tie"] for r in results])
irv_tie_rate = rate([r["irv_tie"] for r in results])

# distance summary
fptp_stats = summarize(fptp_dist)
approval_stats = summarize(approval_dist)
irv_stats = summarize(irv_dist)

# print table
print("\n=== RULE PERFORMANCE ===")
print("Rule      MeanDist   Std     CI     Condorcet%   Tie%")

print(f"FPTP     {fptp_stats[0]:.3f}   {fptp_stats[1]:.3f}   {fptp_stats[2]:.3f}   "
      f"{condorcet_match('fptp'):.3f}   {fptp_tie_rate:.3f}")

print(f"Approval {approval_stats[0]:.3f}   {approval_stats[1]:.3f}   {approval_stats[2]:.3f}   "
      f"{condorcet_match('approval'):.3f}   {approval_tie_rate:.3f}")

print(f"IRV      {irv_stats[0]:.3f}   {irv_stats[1]:.3f}   {irv_stats[2]:.3f}   "
      f"{condorcet_match('irv'):.3f}   {irv_tie_rate:.3f}")

# RULE QUALITY
print("\n=== RULE QUALITY ===")
print("Rule      Welfare    Best%")

print(f"FPTP     {summarize(fptp_util)[0]:.3f}   "
      f"{rate([r['fptp_best'] for r in results]):.3f}")

print(f"Approval {summarize(approval_util)[0]:.3f}   "
      f"{rate([r['approval_best'] for r in results]):.3f}")

print(f"IRV      {summarize(irv_util)[0]:.3f}   "
      f"{rate([r['irv_best'] for r in results]):.3f}")

# SYSTEM OUTCOMES
print("\n=== SYSTEM OUTCOMES ===")
print("Metric              Mean     Std     CI")

for name, arr in [
    ("Final parties", num_parties_final),
    ("Avg parties", num_parties_avg),
    ("Total merges", merges_total),
    ("Avg movement", movement_avg),
]:
    m, s, ci = summarize(arr)
    print(f"{name:18} {m:.3f}   {s:.3f}   {ci:.3f}")

# RULE AGREEMENT
def agreement(a, b):
    return rate([r[a] == r[b] for r in results])

print("\n=== RULE AGREEMENT ===")
print(f"FPTP vs Approval: {agreement('fptp_winner','approval_winner'):.3f}")
print(f"FPTP vs IRV:      {agreement('fptp_winner','irv_winner'):.3f}")
print(f"Approval vs IRV:  {agreement('approval_winner','irv_winner'):.3f}")

# STATISTICAL TESTS
print("\n=== STATISTICAL TESTS (Mean Distance) ===")

def test(a, b, name):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    t, p = scipy.stats.ttest_ind(a, b)
    print(f"{name}: t={t:.3f}, p={p:.6f}")

test(fptp_dist, approval_dist, "FPTP vs Approval")
test(fptp_dist, irv_dist, "FPTP vs IRV")
test(approval_dist, irv_dist, "Approval vs IRV")

print("\n=== STATISTICAL TESTS (Welfare) ===")
test(fptp_util, approval_util, "FPTP vs Approval")
test(fptp_util, irv_util, "FPTP vs IRV")
test(approval_util, irv_util, "Approval vs IRV")