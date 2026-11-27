# Temporal adaptation in federated intrusion detection

**Author:** Leonardo Oliveira dos Santos

This repository is an organized distribution of the code and scientific documentation associated with a retrospective experiment on temporal adaptation in federated intrusion detection. It accompanies an academic manuscript. The experimental question is how global Page–Hinkley triggered adaptation compares with continuous FedAvg in detection quality and observed costs across TON_IoT and Gotham 2025, using logistic regression and a 64–32 MLP.

The study's approved scope is **descriptive**. The frozen matrix contains 160 positions: two datasets, two models, eight strategies and five seed identifiers. Each window was evaluated before any update using its labels. Logistic regression is deterministic in this implementation; repeating its seed identifier does not create independent replications.

## What is included

- `src/distributed_drift_ids/`: the experimental implementation, causal worker, strategy logic, publication, guards, and descriptive-analysis code.
- `tests/`: artificial fixtures and tests of the code and operational contracts. They do not use the real datasets.
- `configs/runtime/strategy_contract.yaml`: the authored strategy specification and its hash.
- `docs/`: the descriptive analysis plan, findings and technical contracts.
- `artifacts/MANIFEST.sha256`: hashes of the **distributed** configuration files. It does not claim that the external campaign artifacts are present.

The datasets, processed shards, initial model states, detector calibrations, campaign manifests, aggregate analysis archive, reference tables and figures, checkpoints, operational authorizations and entry attestations are **not included**. In particular, TON_IoT-derived and mixed-dataset files are withheld because their redistribution has not been established. The Gotham dataset is also external; its attribution and CC BY 4.0 terms are described below. No historical authorization or attestation is reusable in this new Git repository.

## Install and inspect

Use Windows with CPython 3.11 or 3.12 for the operational code. The recorded reference environment was CPython 3.11.9 on Windows x64. From the repository root, in a fresh virtual environment:

```powershell
python -m pip install .
python -m pip install ".[dev]" -c constraints.txt
drift-analysis --help
drift-campaign --help
drift-analysis verify
```

The two installed commands are `drift-analysis` (`verify`, `report`, `compare`) and `drift-campaign` (`campaign`, `child`). `verify` checks the files distributed in `artifacts/` and `configs/`. `report` and `compare` explain that their required external analysis artifacts are absent. The campaign command is an authorized, supervised execution entry; the files in this repository alone do not make it runnable on real data. An exact new code identity, source bindings, an entry attestation, and a separately approved, single-use GO would be needed. The original campaign's GO and D5 attestation do not transfer to this distribution.

To run selected artificial checks after installing the development dependencies:

```powershell
python -m pytest tests/test_prediction_evidence_synthetic.py tests/test_local_checkpoint.py
```

These 14 checks establish behavior on fixtures only; they do not reproduce the original experiment. Other retained tests bind the campaign manifests, states or contracts that are withheld from this distribution and require those exact external artifacts. Running the entire test directory from this checkout alone is not expected to pass.

## Experiment and results

The main endpoint is global Macro-F1 with fixed classes `[0, 1]` in each window, followed by an unweighted temporal mean. The main contrast is global Page–Hinkley minus continuous FedAvg, paired within dataset × model, seed and common horizon. Per-class errors, supports, coverage and costs are interpreted with their actual availability; a missing rate or cost is never silently changed to zero.

The original campaign processed **160 positions**. **150 runs completed**, ten TON global Page–Hinkley runs had authorized scientific terminations, and the v4 attempt had **zero technical failures**. The ten full-horizon TON main pairs are **not evaluable**; partial windows do not substitute for their full endpoint. In Gotham logistic regression, global Page–Hinkley had lower mean temporal Macro-F1 than FedAvg (0.777686 versus 0.794045). Gotham MLP showed one favorable and four unfavorable paired contrasts, with a descriptive mean of −0.032284. See [the results summary](docs/results.md) and [the analysis plan](docs/analysis_plan.md) for definitions and limitations.

No non-inferiority, confidence interval, preserved safety or generalization to new operating environments is claimed. Recovery after drift and `detection_delay` are not identifiable without independent change ground truth. Neither unfavorable results nor missing full-horizon pairs were imputed away.

## Data and reproducibility

TON_IoT is a retrospective replay of complete flows ordered by recorded `ts`; it does not demonstrate first-packet detection. Gotham uses the strict boundary development `t < T` and confirmation `t ≥ T`, where `T = 1737130976125304000` ns. The contracts in [docs/contracts](docs/contracts) describe the data representations, mappings, temporal controls and outputs. The source datasets and processed shards must be obtained independently under their applicable terms.

This repository supports installation, inspection of the implementation and artificial tests. **It does not currently reproduce the 13 published tables and figures**: the aggregate archive, reference outputs and campaign manifest were withheld as mixed or TON-derived artifacts. Supplying external artifacts requires verifying their provenance and hashes and respecting their distribution terms. This repository does not offer a fresh full execution of the 160 plans. See [artifact availability](docs/artifacts.md) for the exact categories and [provenance](docs/provenance/README.md) for the relationship to the executed version.

## License, attribution and citation

The author-written code, tests, configuration and documentation in this repository are offered under the [MIT License](LICENSE), copyright © 2026 Leonardo Oliveira dos Santos. The MIT license **does not cover either third-party dataset or artifacts derived from them**.

Gotham 2025: Belarbi, O., Spyridopoulos, T., Anthi, E., Rana, O., Carnelli, P. and Khan, A. (2025), *A Device-Level IoT Network Traffic Dataset with Distributed Capture and Non-IID Characteristics*, Zenodo, [doi:10.5281/zenodo.14502760](https://doi.org/10.5281/zenodo.14502760). The dataset is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The research parsed packet fields, converted selected fields to integers, ordered events, split them temporally, fitted preprocessing on development only, formed windows and clients, and derived models and detector statistics. Those adapted artifacts are not distributed here. The dataset creators do not endorse this work.

TON_IoT: the network dataset was obtained from [UNSW Canberra](https://research.unsw.edu.au/projects/toniot-datasets). Its stated terms allow academic research use and request citations; they do not establish redistribution of the derived artifacts. No TON_IoT source data or derived artifact is included here.

Citation metadata for this software is in [CITATION.cff](CITATION.cff). The public distribution is at [Leonardo1952/federated-ids-temporal-adaptation](https://github.com/Leonardo1952/federated-ids-temporal-adaptation). This repository has no DOI. Its URL and the exact commit identify this distribution; it is **derived from**, and not byte-for-byte identical to, the executable and records of the original campaign.
