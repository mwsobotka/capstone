import os
import pickle
import numpy as np

folder = "saved_runs_final"

# 1. Check folder exists
if not os.path.exists(folder):
    raise FileNotFoundError(f"Folder '{folder}' does not exist. Run main.py first.")

# 2. Find all pickle files
files = sorted([f for f in os.listdir(folder) if f.endswith(".pkl")])

print("FILES FOUND:")
for f in files:
    print(" ", f)

print("\nTOTAL FILES:", len(files))

if len(files) == 0:
    raise ValueError("No .pkl files found. Simulation may not have saved anything.")

# 3. Check each run file
for file in files:
    path = os.path.join(folder, file)

    print("\n" + "=" * 60)
    print("CHECKING:", file)

    with open(path, "rb") as f:
        data = pickle.load(f)

    # --- top-level structure ---
    print("Top-level keys:", data.keys())

    assert "metadata" in data, "Missing 'metadata'"
    assert "voters" in data, "Missing 'voters'"
    assert "rules" in data, "Missing 'rules'"
    assert "summary_rows" in data, "Missing 'summary_rows'"

    metadata = data["metadata"]
    rules = data["rules"]

    print("Metadata:", metadata)
    print("Rules present:", rules.keys())

    T = metadata["T"]

    # --- rule-level checks ---
    for rule_name in ["fptp", "approval", "irv"]:
        assert rule_name in rules, f"Missing rule '{rule_name}'"

        rule_data = rules[rule_name]

        assert "party_history" in rule_data, f"{rule_name} missing party_history"
        assert "party_counts" in rule_data, f"{rule_name} missing party_counts"
        assert "iter_stats" in rule_data, f"{rule_name} missing iter_stats"

        party_history = rule_data["party_history"]
        party_counts = rule_data["party_counts"]
        iter_stats = rule_data["iter_stats"]

        print(f"\nRULE: {rule_name}")
        print("  party_history length:", len(party_history))
        print("  party_counts length:", len(party_counts))
        print("  iter_stats length:", len(iter_stats))

        # Expected lengths:
        # history and counts include initial state, so T + 1
        # iter_stats are recorded once per iteration, so T
        assert len(party_history) == T + 1, f"{rule_name}: party_history should be T+1"
        assert len(party_counts) == T + 1, f"{rule_name}: party_counts should be T+1"
        assert len(iter_stats) == T, f"{rule_name}: iter_stats should be T"

        # Check shapes
        first_positions = party_history[0]
        last_positions = party_history[-1]

        print("  initial shape:", first_positions.shape)
        print("  final shape:", last_positions.shape)

        assert first_positions.ndim == 2, f"{rule_name}: initial positions should be 2D"
        assert last_positions.ndim == 2, f"{rule_name}: final positions should be 2D"
        assert first_positions.shape[1] == 2, f"{rule_name}: initial positions should have 2 columns"
        assert last_positions.shape[1] == 2, f"{rule_name}: final positions should have 2 columns"

        # Check no NaNs
        assert not np.isnan(first_positions).any(), f"{rule_name}: initial positions contain NaN"
        assert not np.isnan(last_positions).any(), f"{rule_name}: final positions contain NaN"

        # Check counts are sensible
        assert all(p >= 1 for p in party_counts), f"{rule_name}: party count dropped below 1"

        # Check whether simulation changed something
        if first_positions.shape == last_positions.shape:
            changed = not np.allclose(first_positions, last_positions)
            print("  positions changed:", changed)
        else:
            print("  party count changed, so structure changed")

    # --- summary_rows check ---
    expected_summary_rows = 3 * T
    actual_summary_rows = len(data["summary_rows"])
    print("\nsummary_rows length:", actual_summary_rows)
    assert actual_summary_rows == expected_summary_rows, (
        f"summary_rows should be {expected_summary_rows}, got {actual_summary_rows}"
    )

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")