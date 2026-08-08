# Client-side Machine Learning Roadmap

Goal: demonstrate meaningful ML expertise through small, private, reproducible browser experiments—not through a heavyweight model that obscures the portfolio.

| ID | Deliverable | Effort | Depends on | Status |
| --- | --- | --- | --- | --- |
| ML1 | Audio signal-processing report backed by the notebooks workspace | M | — | ✅ |
| ML2 | Feasibility matrix comparing ONNX Runtime Web, Transformers.js, TensorFlow.js, WebGPU/WASM and WebNN status | M | IT6 | 📋 |
| ML3 | Worker-hosted inference runtime with lazy model download, cache/version policy, progress, cancellation and CPU fallback | L | IT9, ML2 | 📋 |
| ML4 | First bounded demo: classify or embed a small supplied text/audio sample and visualize confidence/features | L | ML3, IF3 | 📋 |
| ML5 | Interactive model card: dataset scope, architecture, latency, memory, limitations, bias and privacy | M | ML4, DOC5 | 📋 |
| ML6 | Embedding/project semantic explorer with precomputed vectors and optional local query inference | L | ML3, UI8 | 📋 |
| ML7 | Model optimization pipeline: quantization, graph fusion, operator compatibility and reproducible export checks | L | ML3 | 📋 |
| ML8 | Capability benchmarking across WebGPU/WASM with anonymized, opt-in local display only | M | ML3, IT11 | 🔬 |

## Guardrails and acceptance criteria

- The first demo model is preferably ≤ 15 MB compressed; larger downloads require explicit opt-in and a visible size estimate.
- Input stays on-device. The interface states this plainly and does not persist samples without consent.
- Runtime/model load happens after intent in a worker. Users can cancel; navigation terminates or safely parks work.
- WebGPU is an acceleration path, not a requirement. WASM/CPU fallback and an example result keep the story usable.
- Display warm/cold latency, runtime/provider, approximate memory and model version; never imply scientific validity beyond the model card.
- Tests use a tiny deterministic fixture/model or mock only the runtime boundary; they cover unsupported operator, corrupt cache, cancellation and out-of-memory messaging.
- Notebook/export scripts pin dependencies and reproduce preprocessing exactly; TypeScript preprocessing is checked against known Python outputs.

## Decision gate before ML6+

Proceed only if ML4 remains within route budgets and adds explanatory value. Generative LLMs are out of the default roadmap because their download, memory and energy cost conflicts with progressive enhancement; any future experiment must be separately opt-in.

## R2 research integration

RR4 is a small deterministic training replay with a model card, reward/cost curves, latent explorer, and human correction events. RR5 may use an `AnalyserNode` only as an optional explanatory layer; audio is never presented as model evidence. Static/CPU fixtures remain first-class and all claims identify dataset, preprocessing, runtime, version, and limitations.

**Acceptance:** seeded replay snapshots, accessible table/export, cancellation and unsupported-runtime messaging, and no implication that an illustrative policy is production optimal.
