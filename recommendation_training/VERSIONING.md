# Cove experiment and system version map

The labels describe two different axes and must not be treated as one sequence.

| Label | Scope | Role |
|---|---|---|
| V1 | retrieval system | deterministic rules and colour/weather filtering |
| V2 | retrieval system | learned compatibility score with the historical quota-limited catalogue path |
| D2 | compatibility model/data revision | disjoint Polyvore hard-negative head, corrected FITB evaluation and validation-only calibration |
| V3 | retrieval system | D2 head plus abstract Polyvore prototypes, exact bounded search and user-selected clothing-range filtering |

D2 is the learned model deployed inside V3; it is not a fourth retrieval-system
generation. V1/V2 results remain engineering baselines. Because their data and
candidate protocols differ from D2/V3, cross-version metric differences are
descriptive and must not be written as causal improvements. Controlled claims
come from the same-split baselines, three-seed architecture comparison,
structural ablation and exact-versus-quota search experiment.
