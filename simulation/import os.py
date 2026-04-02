import pickle

# change this to one of your files
path = "saved_runs/run_0000.pkl"

with open(path, "rb") as f:
    results = pickle.load(f)

print("\n--- METADATA ---")
print(results["metadata"])

print("\n--- TOP LEVEL KEYS ---")
print(results.keys())

print("\n--- RULES ---")
print(results["rules"].keys())

# pick one rule
fptp = results["rules"]["fptp"]

print("\n--- FPTP PARTY COUNTS (first 10 iterations) ---")
print(fptp["party_counts"][:10])

print("\n--- NUMBER OF ITERATIONS ---")
print(len(fptp["party_history"]))

print("\n--- FINAL PARTY POSITIONS ---")
print(fptp["party_history"][-1])