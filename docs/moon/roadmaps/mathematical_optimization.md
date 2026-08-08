# Interactive Mathematical Optimization Roadmap

Goal: let visitors manipulate a small optimization problem, watch the algorithm respond, and understand trade-offs through synchronized explanation and visualization.

| ID | Deliverable | Effort | Depends on | Status |
| --- | --- | --- | --- | --- |
| MO1 | PCVRP report backed by reproducible notebook analysis | M | — | ✅ |
| MO2 | Accessible route explorer with depots/customers, day/capacity filters, timeline playback and solution metrics | L | IF3 | 📋 |
| MO3 | Worker-hosted WASM solver spike comparing OR-Tools/HiGHS-compatible options, size, license and static-host constraints | L | IT9 | 📋 |
| MO4 | Small interactive solver lab: edit demand/capacity, solve/cancel, animate incumbent routes and compare baseline | XL | MO2, MO3 | 📋 |
| MO5 | Algorithm-explanation mode showing objective, constraints, feasibility violations, bounds/gap and search progress | L | MO4, DOC5 | 📋 |
| MO6 | Scenario persistence in URL/downloadable JSON with schema validation and deterministic seeded examples | M | MO4 | 📋 |
| MO7 | Performance path using typed arrays/CSC data where profiling shows serialization or model-build bottlenecks | L | MO4, IT6 | 🔬 |
| MO8 | Backend architecture decision for workloads beyond browser limits; separate deployment only if evidence supports it | L | MO4 metrics | 🔬 |

## Acceptance criteria

- MO2 has an SVG/canvas visual view and a synchronized table listing each stop, day, demand, order and route membership.
- Filters and playback are keyboard operable; route identity uses line style/labels as well as color.
- MO3 records compressed solver size, initialization time, solve time, memory, license, worker compatibility and CSP requirements.
- MO4 ships bounded instance limits, input validation, seeded examples, progress, cancellation, timeout and a clear “best known” versus “optimal” distinction.
- Solver work never blocks the main thread. Stale responses cannot overwrite a newer scenario; worker memory is reclaimed after failure/unmount.
- Metrics include distance/cost, vehicles, capacity utilization, feasibility, elapsed time, incumbent/bound and relative gap when supported.
- Every preset has a known feasible result; tests cover infeasible input, cancellation, timeout, invalid schema and deterministic playback.

## Backend decision threshold

Preserve the static site unless representative educational instances cannot finish within 5 seconds or the required solver cannot fit the agreed asset budget. A backend proposal must include cost, abuse prevention, privacy, rate limits, WebSocket reconnect semantics, deployment ownership and a static recorded-result fallback.
