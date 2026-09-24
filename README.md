# Simulating Electoral Systems: FPTP vs. IRV vs. Approval Voting

A computational spatial voting model in Python that compares three electoral systems (First-Past-the-Post, Instant Runoff Voting, and Approval Voting) under realistic voter behavior and adaptive party competition.

**Author:** Marris Sobotka, University of Pittsburgh (B.S. Applied Mathematics, B.A. Political Science)
**Recognition:** First Overall Presentation, Undergraduate Research & Creativity Symposium, 2026

---

## Research Question

Which electoral system produces winners that best represent voters' preferences when voters hold complex, multidimensional views and parties respond strategically to the electorate?

## Key Findings

Across 1,000 Monte Carlo trials, **Approval Voting significantly outperformed both Instant Runoff Voting and First-Past-the-Post** (p < 0.001 across all comparisons) on three measures:

- **Voter-winner distance:** how close the winning candidate is to voters' preferences
- **Aggregate welfare:** total voter satisfaction with the outcome
- **Condorcet efficiency:** how often the system elects the candidate who would beat every other candidate head-to-head

## Methodology

**Spatial voting model.** Voters and parties are placed in a multidimensional policy space. Each voter's preference for a party depends on the distance between them.

**Realistic voter behavior.**
- Non-separable utility functions, so a voter's views on one issue can affect how they weigh another
- Probabilistic turnout, so not every voter participates in every election

**Adaptive party competition.** Parties move through the policy space in response to election results, and parties can merge. This makes party positions endogenous rather than fixed.

**Simulation and validation.**
- 1,000 Monte Carlo trials per system
- Results compared with pairwise t-tests
- Condorcet-efficiency rates calculated for each system

## Tech Stack

- **Python**
- **NumPy** for numerical computation and simulation
- **Matplotlib** for data visualization

## Repository Structure

<!-- Replace the file names below with the actual files in this repo -->

```
├── README.md
├── [simulation_file].py     # Core voting model and election logic
├── [analysis_file].py       # Statistical tests and metrics
├── [figures/]               # Output charts
└── [paper].pdf              # Full research paper
```

## How to Run

<!-- Update these commands to match your actual file names -->

```bash
git clone https://github.com/[your-username]/electoral-system-simulation.git
cd electoral-system-simulation
pip install numpy matplotlib
python [simulation_file].py
```

## Full Paper

The complete research paper, including literature review, model specification, and discussion of results, is available here: [link to paper]
